#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VT 报告收集主程序（优化版）
- 支持多密钥轮换，每个样本尝试所有可用密钥
- 样本失败达到重试上限后自动从 CSV 移除
- 全局临时错误保护，避免因网络问题长时间运行
- 配额用尽时停止当天，不重试
"""

import os
import sys
import time
import json
import csv
import shutil
import zipfile          
import logging
import schedule
import configparser
from datetime import datetime, date
from typing import List, Dict, Tuple, Set, Optional

import requests
import pandas as pd

# ------------------------------- 配置加载 ----------------------------------
CONFIG_FILE = "config.ini"
if not os.path.exists(CONFIG_FILE):
    print(f"错误：配置文件 {CONFIG_FILE} 不存在，请根据模板创建。")
    sys.exit(1)

config = configparser.ConfigParser()
config.read(CONFIG_FILE, encoding='utf-8')

# API 配置
API_KEYS = [k.strip() for k in config.get('api', 'keys').split(',')]
MAX_CALLS_PER_KEY = config.getint('api', 'max_calls_per_key')
REQUEST_INTERVAL = config.getint('api', 'request_interval')

# 路径配置
CSV_FILE = config.get('paths', 'csv_file')
RESULT_FOLDER = config.get('paths', 'result_folder')
ZIP_OUTPUT_DIR = config.get('paths', 'zip_output_dir')
LOG_FILE = config.get('paths', 'log_file')

# 其他设置
ZIP_PREFIX = config.get('settings', 'zip_prefix')
DAILY_TARGET_MIN = config.getint('settings', 'daily_target_min')
MAX_RETRIES_FOR_404 = config.getint('settings', 'max_retries_for_404')
MAX_RETRIES_PER_SAMPLE = config.getint('settings', 'max_retries_per_sample')
MAX_CONSECUTIVE_TEMP_ERRORS = config.getint('settings', 'max_consecutive_temporary_errors')
CLEANUP_OLD_DAYS = config.getint('settings', 'cleanup_old_days')
SCHEDULE_TIME = config.get('settings', 'schedule_time')
RETRY_WAIT_MINUTES = config.getint('settings', 'retry_wait_minutes')

# 确保目录存在
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(ZIP_OUTPUT_DIR, exist_ok=True)

# ------------------------------- 日志配置 ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------- 辅助函数 ----------------------------------
def load_key_usage() -> Dict[str, Dict]:
    usage_file = os.path.join(RESULT_FOLDER, "key_usage.json")
    if os.path.exists(usage_file):
        with open(usage_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_key_usage(usage: Dict[str, Dict]) -> None:
    usage_file = os.path.join(RESULT_FOLDER, "key_usage.json")
    with open(usage_file, 'w', encoding='utf-8') as f:
        json.dump(usage, f, indent=4)

def get_today_date() -> str:
    return datetime.now().strftime('%Y-%m-%d')

def is_today_file(filepath: str) -> bool:
    mtime = os.path.getmtime(filepath)
    return datetime.fromtimestamp(mtime).date() == datetime.now().date()

def read_sha256s_from_csv(csv_file: str) -> List[str]:
    sha256s = []
    if not os.path.exists(csv_file):
        logger.error(f"CSV 文件不存在: {csv_file}")
        return sha256s
    try:
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'sha256' not in reader.fieldnames:
                raise ValueError("CSV 文件必须包含 'sha256' 列")
            for row in reader:
                sha256 = row['sha256'].strip()
                if sha256 and len(sha256) == 64 and all(c in '0123456789abcdefABCDEF' for c in sha256):
                    sha256s.append(sha256)
                else:
                    logger.warning(f"无效 SHA256 格式: {sha256}")
        logger.info(f"从 CSV 读取到 {len(sha256s)} 个有效 SHA256")
        return sha256s
    except Exception as e:
        logger.error(f"读取 CSV 失败: {e}")
        return []

def cleanup_old_no_reports(folder: str) -> int:
    cutoff = datetime.now().timestamp() - CLEANUP_OLD_DAYS * 86400
    cleaned = 0
    for fname in os.listdir(folder):
        if fname.endswith("_no_behaviour_summary.flag") or fname.endswith("_error.flag"):
            path = os.path.join(folder, fname)
            if os.path.getctime(path) < cutoff:
                os.remove(path)
                cleaned += 1
    if cleaned:
        logger.info(f"清理了 {cleaned} 个超过 {CLEANUP_OLD_DAYS} 天的标记文件")
    return cleaned

def get_processed_sha256_from_folder(folder: str, today_only: bool = True) -> Set[str]:
    processed = set()
    for fname in os.listdir(folder):
        if fname.endswith('_behaviour_summary.json'):
            path = os.path.join(folder, fname)
            if today_only and not is_today_file(path):
                continue
            parts = fname.split('_behaviour_summary.json')[0]
            if len(parts) == 64 and all(c in '0123456789abcdefABCDEF' for c in parts):
                processed.add(parts)
    return processed

def remove_processed_from_csv(csv_file: str, processed_sha256s: Set[str]) -> int:
    if not os.path.exists(csv_file):
        return 0
    df = pd.read_csv(csv_file, encoding='utf-8')
    original_len = len(df)
    df_filtered = df[~df['sha256'].isin(processed_sha256s)]
    removed = original_len - len(df_filtered)
    if removed:
        df_filtered.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"从 CSV 中移除了 {removed} 个已处理的 SHA256")
    return removed

def cleanup_failed_entries(folder: str, csv_file: str) -> int:
    """
    清理达到最大重试次数的样本（404 或其他错误）
    删除对应的标记文件，并从 CSV 中移除这些样本
    """
    to_remove = set()
    for fname in os.listdir(folder):
        if fname.endswith('_no_behaviour_summary.flag') or fname.endswith('_error.flag'):
            flag_path = os.path.join(folder, fname)
            # 读取重试次数
            retry_count = 0
            try:
                with open(flag_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('retry_count:'):
                            retry_count = int(line.split(':')[1].strip())
                            break
            except Exception:
                pass
            if retry_count >= MAX_RETRIES_PER_SAMPLE:
                sha256 = fname.split('_')[0]  # 提取 SHA256
                to_remove.add(sha256)
                os.remove(flag_path)
                logger.info(f"标记为删除 {sha256}，已移除标记文件")
    if to_remove:
        df = pd.read_csv(csv_file, encoding='utf-8')
        df_filtered = df[~df['sha256'].isin(to_remove)]
        removed = len(df) - len(df_filtered)
        df_filtered.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"从 CSV 中删除了 {removed} 个多次失败的 SHA256")
        return removed
    return 0

def select_available_key(key_usage: Dict) -> Optional[str]:
    """返回第一个还有剩余配额的密钥，如果全用完则返回 None"""
    for key in API_KEYS:
        if key_usage.get(key, {}).get('count', 0) < MAX_CALLS_PER_KEY:
            return key
    return None

def fetch_vt_data(api_key: str, sha256: str, folder: str, usage: Dict) -> Tuple[bool, int, bool, bool]:
    """
    调用 VT API 获取行为摘要
    返回: (是否需要切换密钥, 调用计数增量, 是否发生错误, 是否为临时错误)
    """
    base_url = "https://www.virustotal.com/api/v3/files/"
    url = f"{base_url}{sha256}/behaviour_summary"
    filename = os.path.join(folder, f"{sha256}_behaviour_summary.json")
    no_report_filename = os.path.join(folder, f"{sha256}_no_behaviour_summary.flag")
    error_filename = os.path.join(folder, f"{sha256}_error.flag")

    # 如果已经成功获取报告，直接跳过
    if os.path.exists(filename):
        logger.debug(f"文件已存在，跳过: {sha256}")
        return False, 0, False, False

    # 如果该样本已经达到最大重试次数，则跳过（但会在 cleanup_failed_entries 中删除）
    if os.path.exists(no_report_filename) or os.path.exists(error_filename):
        # 检查重试次数
        flag_file = no_report_filename if os.path.exists(no_report_filename) else error_filename
        retry_count = 0
        try:
            with open(flag_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('retry_count:'):
                        retry_count = int(line.split(':')[1].strip())
                        break
        except Exception:
            pass
        if retry_count >= MAX_RETRIES_PER_SAMPLE:
            logger.debug(f"达到最大重试次数，跳过: {sha256}")
            return False, 0, False, False

    headers = {"x-apikey": api_key, "Accept": "application/json"}
    try:
        logger.debug(f"请求 VT API: {sha256} (key: {api_key[:8]}...)")
        time.sleep(REQUEST_INTERVAL)
        response = requests.get(url, headers=headers, timeout=60)

        if response.status_code == 200:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=4)
            logger.info(f"成功获取报告: {sha256}")
            # 如果之前有错误标记，删除它
            if os.path.exists(no_report_filename):
                os.remove(no_report_filename)
            if os.path.exists(error_filename):
                os.remove(error_filename)
            return False, 1, False, False

        elif response.status_code == 429:
            logger.warning(f"配额用尽！Key: {api_key[:8]}... 今日配额已达上限")
            usage[api_key]['count'] = MAX_CALLS_PER_KEY
            return True, 0, True, False  # 需要切换密钥，错误，非临时错误

        elif response.status_code == 401:
            logger.error(f"API 密钥无效: {api_key[:8]}...")
            usage[api_key]['count'] = MAX_CALLS_PER_KEY
            return True, 0, True, False  # 需要切换密钥，错误，非临时错误

        elif response.status_code == 404:
            # 无报告，记录重试次数
            retry_count = 0
            if os.path.exists(no_report_filename):
                with open(no_report_filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('retry_count:'):
                            retry_count = int(line.split(':')[1].strip())
                            break
            retry_count += 1
            with open(no_report_filename, 'w', encoding='utf-8') as f:
                f.write(f"checked_at: {datetime.now().isoformat()}\n")
                f.write(f"status_code: 404\n")
                f.write(f"retry_count: {retry_count}\n")
            logger.info(f"VT 无报告: {sha256} (重试 {retry_count}/{MAX_RETRIES_PER_SAMPLE})")
            return False, 0, False, False  # 不切换密钥，无错误（样本特定问题）

        elif 500 <= response.status_code < 600:
            # 服务器错误，视为临时错误
            logger.warning(f"服务器错误 {response.status_code}: {sha256}")
            return False, 0, True, True  # 临时错误，不切换密钥

        else:
            # 其他非预期状态码，也视为临时错误
            logger.error(f"请求失败 {response.status_code}: {sha256}")
            return False, 0, True, True  # 临时错误

    except requests.exceptions.Timeout:
        logger.warning(f"请求超时: {sha256}")
        return False, 0, True, True
    except requests.exceptions.ConnectionError:
        logger.warning(f"连接错误: {sha256}")
        return False, 0, True, True
    except Exception as e:
        logger.error(f"请求异常: {e}")
        return False, 0, True, True

def run_scan(csv_file: str, result_folder: str, key_usage: Dict) -> Tuple[int, int, int, bool]:
    """
    主扫描循环：处理 CSV 中未处理的 SHA256
    返回: (成功数, 总调用数, 总错误数, 是否因配额用尽停止)
    """
    sha256_list = read_sha256s_from_csv(csv_file)
    if not sha256_list:
        return 0, 0, 0, False

    # 加载今日计数
    today = get_today_date()
    for key in API_KEYS:
        if key_usage.get(key, {}).get('date') != today:
            key_usage.setdefault(key, {'date': today, 'count': 0})
            key_usage[key]['date'] = today
            key_usage[key]['count'] = 0
    save_key_usage(key_usage)

    success_count = 0
    total_calls = 0
    total_errors = 0
    quota_exhausted = False
    consecutive_temp_errors = 0  # 连续临时错误计数

    i = 0
    while i < len(sha256_list):
        sha256 = sha256_list[i]

        # 检查是否已有成功报告（可能在之前的循环中已创建）
        if os.path.exists(os.path.join(result_folder, f"{sha256}_behaviour_summary.json")):
            logger.debug(f"文件已存在，跳过: {sha256}")
            i += 1
            continue

        # 尝试所有可用密钥
        tried_keys = set()
        sample_success = False
        while True:
            current_key = select_available_key(key_usage)
            if current_key is None:
                logger.warning("所有密钥今日配额均用尽，停止扫描")
                quota_exhausted = True
                break
            if current_key in tried_keys:
                # 所有密钥都已尝试过，但都失败了
                logger.warning(f"所有可用密钥均尝试失败，跳过样本 {sha256}")
                # 记录错误次数
                error_filename = os.path.join(result_folder, f"{sha256}_error.flag")
                retry_count = 0
                if os.path.exists(error_filename):
                    with open(error_filename, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith('retry_count:'):
                                retry_count = int(line.split(':')[1].strip())
                                break
                retry_count += 1
                with open(error_filename, 'w', encoding='utf-8') as f:
                    f.write(f"checked_at: {datetime.now().isoformat()}\n")
                    f.write(f"retry_count: {retry_count}\n")
                sample_success = False
                break

            tried_keys.add(current_key)
            logger.info(f"处理 [{i+1}/{len(sha256_list)}] {sha256}，使用密钥 {current_key[:8]}...")

            should_switch, call_inc, is_error, is_temporary = fetch_vt_data(
                current_key, sha256, result_folder, key_usage
            )

            total_calls += 1
            if call_inc > 0:
                success_count += call_inc
                key_usage[current_key]['count'] += call_inc
                sample_success = True
                consecutive_temp_errors = 0
                break  # 成功，跳出密钥尝试循环

            # 错误处理
            if should_switch:
                # 密钥不可用，标记为已用尽
                key_usage[current_key]['count'] = MAX_CALLS_PER_KEY
                # 继续循环，尝试下一个密钥
                continue

            if is_error:
                if is_temporary:
                    consecutive_temp_errors += 1
                    logger.warning(f"临时错误，连续临时错误计数: {consecutive_temp_errors}")
                    if consecutive_temp_errors >= MAX_CONSECUTIVE_TEMP_ERRORS:
                        logger.error(f"连续临时错误达到 {MAX_CONSECUTIVE_TEMP_ERRORS} 次，停止扫描")
                        quota_exhausted = True
                        break
                    # 临时错误，不切换密钥，稍等后重试（使用同一密钥）
                    time.sleep(REQUEST_INTERVAL * 2)  # 等待更长时间
                    continue
                else:
                    # 非临时错误（如 429/401），已处理切换，继续尝试下一个密钥
                    continue
            else:
                # 无错误但也没成功（例如404），直接跳过此样本
                # 404 已经记录了重试次数，所以不需要再添加错误标记
                sample_success = False
                break

        if quota_exhausted:
            break

        if not sample_success:
            # 样本失败，移动到下一个样本
            i += 1

        # 定期保存 key_usage
        if total_calls % 10 == 0:
            save_key_usage(key_usage)

    save_key_usage(key_usage)
    logger.info(f"扫描完成：成功 {success_count}，总调用 {total_calls}，总错误 {total_errors}")
    return success_count, total_calls, total_errors, quota_exhausted

def pack_today_reports(result_folder: str, zip_output_dir: str, prefix: str) -> str:
    today_str = datetime.now().strftime('%Y%m%d')
    zip_name = f"{prefix}_{today_str}.zip"
    zip_path = os.path.join(zip_output_dir, zip_name)

    today_files = []
    for fname in os.listdir(result_folder):
        if fname.endswith('_behaviour_summary.json'):
            full_path = os.path.join(result_folder, fname)
            if is_today_file(full_path):
                today_files.append(full_path)
    if not today_files:
        logger.warning("今天没有新的报告文件，跳过打包")
        return ""

    # 修复：使用 zipfile.ZipFile
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in today_files:
            zipf.write(f, arcname=os.path.basename(f))
    logger.info(f"打包完成：{zip_path}，包含 {len(today_files)} 个文件")

    for f in today_files:
        os.remove(f)
        logger.debug(f"已删除源文件：{f}")
    return zip_path

def cleanup_old_zip_files(zip_output_dir: str, keep_days: int = 7) -> None:
    cutoff = datetime.now().timestamp() - keep_days * 86400
    for fname in os.listdir(zip_output_dir):
        if fname.endswith('.zip'):
            path = os.path.join(zip_output_dir, fname)
            if os.path.getctime(path) < cutoff:
                os.remove(path)
                logger.info(f"删除旧 ZIP：{fname}")

def daily_task() -> None:
    logger.info("=" * 60)
    logger.info("开始执行每日任务")
    start_time = time.time()

    # ========== 新增 1：清理历史遗留的已处理样本 ==========
    # 获取 result 文件夹中所有已存在的 JSON 文件（不限日期）
    existing_processed = get_processed_sha256_from_folder(RESULT_FOLDER, today_only=False)
    if existing_processed:
        removed_legacy = remove_processed_from_csv(CSV_FILE, existing_processed)
        if removed_legacy > 0:
            logger.info(f"从 CSV 中移除了 {removed_legacy} 个历史已处理样本（可能来自之前的未打包记录）")
    else:
        logger.info("没有发现历史遗留的已处理样本")
    # ====================================================

    # 1. 清理旧标记
    cleanup_old_no_reports(RESULT_FOLDER)

    # 2. 加载密钥使用记录
    key_usage = load_key_usage()
    for key in API_KEYS:
        if key not in key_usage:
            key_usage[key] = {'date': '', 'count': 0}
    today = get_today_date()
    for key in key_usage:
        if key_usage[key]['date'] != today:
            key_usage[key]['date'] = today
            key_usage[key]['count'] = 0
    save_key_usage(key_usage)

    # 3. 扫描
    success_count, total_calls, total_errors, quota_exhausted = run_scan(
        CSV_FILE, RESULT_FOLDER, key_usage
    )

    # 4. 统计今日总成功数
    existing_today = len(get_processed_sha256_from_folder(RESULT_FOLDER, today_only=True))
    total_today_success = existing_today + success_count
    logger.info(f"今日总成功报告数：{total_today_success}（原有 {existing_today}，新增 {success_count}）")

    # 5. 清理多次失败的条目
    cleanup_failed_entries(RESULT_FOLDER, CSV_FILE)

    # ========== 新增 2：在打包前，从 CSV 中移除今天已成功获取的样本 ==========
    processed_today = get_processed_sha256_from_folder(RESULT_FOLDER, today_only=True)
    if processed_today:
        removed_today = remove_processed_from_csv(CSV_FILE, processed_today)
        logger.info(f"从 CSV 中移除了 {removed_today} 个今天已成功处理的样本（将在打包后删除 JSON）")
    else:
        logger.info("今天没有新成功处理的样本")
    # ====================================================================

    # 6. 打包报告（此函数会删除今天生成的 JSON 文件）
    zip_path = pack_today_reports(RESULT_FOLDER, ZIP_OUTPUT_DIR, ZIP_PREFIX)
    if zip_path:
        logger.info(f"报告已打包为：{zip_path}")

    # 7. 清理旧 ZIP
    cleanup_old_zip_files(ZIP_OUTPUT_DIR, keep_days=7)

    elapsed = time.time() - start_time
    logger.info(f"每日任务执行完毕，耗时 {elapsed:.2f} 秒")

    # 8. 重试判断
    if quota_exhausted:
        logger.info("今日配额已用尽，不再重试，等待明天")
    elif total_today_success < DAILY_TARGET_MIN:
        logger.warning(f"今日成功报告数 {total_today_success} 低于目标 {DAILY_TARGET_MIN}，将在 {RETRY_WAIT_MINUTES} 分钟后重试")
        schedule.every(RETRY_WAIT_MINUTES).minutes.do(daily_task).tag('retry')
    else:
        logger.info("今日任务完成，达到目标")

def main():
    logger.info("VT 扫描主程序启动")
    logger.info(f"配置：CSV={CSV_FILE}, 结果目录={RESULT_FOLDER}, ZIP目录={ZIP_OUTPUT_DIR}")
    logger.info(f"API密钥数量：{len(API_KEYS)}，每日每密钥限额：{MAX_CALLS_PER_KEY}")
    logger.info(f"每日目标最低成功数：{DAILY_TARGET_MIN}，重试间隔：{RETRY_WAIT_MINUTES}分钟")

    schedule.every().day.at(SCHEDULE_TIME).do(daily_task)

    if len(sys.argv) > 1 and sys.argv[1] == '--now':
        daily_task()
    else:
        logger.info(f"等待定时任务，每天 {SCHEDULE_TIME} 执行")

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()