#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import configparser
import mysql.connector
from mysql.connector import Error, pooling
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

# ---------------------- 日志配置（与主程序统一） ----------------------
def setup_logger():
    """配置日志：同时输出到文件和控制台（独立运行时）"""
    logger = logging.getLogger('vue_data')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 避免重复输出到主程序日志

    # 1. 文件日志（与app.py共用log.txt）
    app_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(app_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. 控制台日志（独立运行时显示）
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# 初始化日志
logger = setup_logger()

# ---------------------- 数据库配置与连接 ----------------------
def get_db_config():
    """读取配置文件获取数据库信息（项目根目录的config.ini）"""
    config = configparser.ConfigParser()
    # config.ini在项目根目录,需要向上三级
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '../../../config.ini'
    )
    
    if not os.path.exists(config_path):
        error_msg = f"错误：配置文件 {config_path} 不存在"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
        return None
    
    try:
        config.read(config_path, encoding='utf-8')
        return {
            'host': config.get('mysql', 'host'),
            'database': config.get('mysql', 'db'),
            'user': config.get('mysql', 'user'),
            'password': config.get('mysql', 'passwd'),
            'port': 3306,
            'pool_name': 'mypool',
            'pool_size': 4,
            'autocommit': True
        }
    except Exception as e:
        error_msg = f"读取配置文件出错: {str(e)}"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
        return None

def get_benign_db_config():
    """读取配置文件获取良性样本数据库信息（项目根目录的config.ini）"""
    config = configparser.ConfigParser()
    # config.ini在项目根目录,需要向上三级
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '../../../config.ini'
    )
    
    if not os.path.exists(config_path):
        error_msg = f"错误：配置文件 {config_path} 不存在"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
        return None
    
    try:
        config.read(config_path, encoding='utf-8')
        return {
            'host': config.get('mysql', 'host'),
            'database': config.get('mysql', 'db_benign'),
            'user': config.get('mysql', 'user'),
            'password': config.get('mysql', 'passwd'),
            'port': 3306,
            'pool_name': 'benign_pool',
            'pool_size': 2,
            'autocommit': True
        }
    except Exception as e:
        error_msg = f"读取良性数据库配置出错: {str(e)}"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
        return None

def get_domain_db_config():
    """读取配置文件获取域名数据库信息（db_domain）"""
    config = configparser.ConfigParser()
    # config.ini在项目根目录,需要向上三级
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '../../../config.ini'
    )
    
    if not os.path.exists(config_path):
        error_msg = f"错误：配置文件 {config_path} 不存在"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
        return None
    
    try:
        config.read(config_path, encoding='utf-8')
        return {
            'host': config.get('mysql', 'host'),
            'database': config.get('mysql', 'db_domain'),  # 域名数据库
            'user': config.get('mysql', 'user'),
            'password': config.get('mysql', 'passwd'),
            'port': 3306,
            'pool_name': 'domain_pool',
            'pool_size': 3,
            'autocommit': True
        }
    except Exception as e:
        error_msg = f"读取域名数据库配置出错: {str(e)}"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
        return None

def ensure_directory_exists(path):
    """确保保存目录存在"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"创建目录: {path}")
    return path

def get_save_path():
    """获取保存文件的路径（项目根目录/../vue/src/data）"""
    # 当前文件在new_flask/app/services/data/,需要向上4级到技术路线还原目录,然后进入vue/src/data
    current_dir = os.path.dirname(__file__)
    save_dir = os.path.join(current_dir, '../../../../vue/src/data')
    save_dir = os.path.normpath(save_dir)
    ensure_directory_exists(save_dir)
    logger.info(f"数据保存目录: {save_dir}")
    return save_dir

# 初始化数据库配置
DB_CONFIG = get_db_config()
BENIGN_DB_CONFIG = get_benign_db_config()
DOMAIN_DB_CONFIG = get_domain_db_config()  # 新增域名数据库配置

def create_db_pool():
    """创建主数据库连接池（存储恶意样本）"""
    if not DB_CONFIG:
        error_msg = "主数据库配置无效，无法创建连接池"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
        return None
        
    try:
        pool = mysql.connector.pooling.MySQLConnectionPool(**DB_CONFIG)
        info_msg = f"成功创建主数据库连接池（恶意样本库），大小: {DB_CONFIG['pool_size']}"
        logger.info(info_msg)
        print(f"\033[92m{info_msg}\033[0m")
        return pool
    except Error as e:
        error_msg = f"创建主连接池时发生错误: {str(e)}"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
    return None

def create_benign_db_pool():
    """创建良性样本数据库连接池"""
    if not BENIGN_DB_CONFIG:
        error_msg = "良性数据库配置无效，无法创建连接池"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
        return None
        
    try:
        pool = mysql.connector.pooling.MySQLConnectionPool(**BENIGN_DB_CONFIG)
        info_msg = f"成功创建良性数据库连接池，大小: {BENIGN_DB_CONFIG['pool_size']}"
        logger.info(info_msg)
        print(f"\033[92m{info_msg}\033[0m")
        return pool
    except Error as e:
        error_msg = f"创建良性连接池时发生错误: {str(e)}"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
    return None

def create_domain_db_pool():
    """创建域名数据库（db_domain）连接池"""
    if not DOMAIN_DB_CONFIG:
        error_msg = "域名数据库配置无效，无法创建连接池"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
        return None
        
    try:
        pool = mysql.connector.pooling.MySQLConnectionPool(**DOMAIN_DB_CONFIG)
        info_msg = f"成功创建域名数据库连接池（db_domain），大小: {DOMAIN_DB_CONFIG['pool_size']}"
        logger.info(info_msg)
        print(f"\033[92m{info_msg}\033[0m")
        return pool
    except Error as e:
        error_msg = f"创建域名连接池时发生错误: {str(e)}"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")
    return None

# ---------------------- 新增：db_domain数据库数据读取与统计（修复完整12个月逻辑） ----------------------


def get_domain_stats_data(pool):
    """
    从db_domain数据库读取domain_stats表数据，完成以下统计：
    1. 总数量
    2. 按年汇总（yyyy年 数量）
    3. 近一年数量（完整12个自然月：上个完整月向前推11个月，共12个月）+ 月度明细
    4. 近一个月数量（数据库最新日期向前推30天）+ 每日明细
    """
    try:
        connection = pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        logger.info("成功获取域名数据库连接，开始读取domain_stats数据")

        # 0. 获取数据库中的最新日期
        cursor.execute("SELECT MAX(stat_date) AS latest_date FROM domain_stats")
        latest_result = cursor.fetchone()
        current_date = latest_result['latest_date'] if latest_result and latest_result['latest_date'] else datetime.now().date()
        logger.info(f"数据库最新日期: {current_date}")

        # 1. 获取总数量
        cursor.execute("SELECT SUM(record_count) AS total_domain_count FROM domain_stats")
        total_result = cursor.fetchone()
        total_domain_count = total_result['total_domain_count'] if total_result and total_result['total_domain_count'] else 0
        logger.info(f"域名总数量: {total_domain_count:,}")

        # 2. 按年汇总
        cursor.execute("""
            SELECT YEAR(stat_date) AS year, SUM(record_count) AS yearly_count
            FROM domain_stats
            GROUP BY YEAR(stat_date)
            ORDER BY year ASC
        """)
        yearly_data = cursor.fetchall()
        # 提取日期和数量数组
        year_dates = [f"{item['year']}年" for item in yearly_data]
        year_amounts = [item['yearly_count'] for item in yearly_data]
        
        # 获取年份范围用于名称
        min_year = min([item['year'] for item in yearly_data]) if yearly_data else datetime.now().year
        max_year = max([item['year'] for item in yearly_data]) if yearly_data else datetime.now().year
        year_chart_name = f"{min_year}-{max_year}年恶意域名总数统计"
        
        logger.info(f"域名按年汇总数据: {len(yearly_data)}个年份")

        # 3. 近一年数量（完整12个自然月）
        current_date = datetime.now()
        
        # 计算结束日期：上个月的最后一天
        current_month_first_day = datetime(current_date.year, current_date.month, 1)
        last_month_last_day = current_month_first_day - timedelta(days=1)
        end_year = last_month_last_day.year
        end_month = last_month_last_day.month
        
        # 计算开始日期：结束日期往前推11个月，取那个月的第一天
        # 例如：结束日期是2025-11-30，开始日期是2024-12-01
        if end_month <= 11:
            # 如果结束月份在1-11月，开始月份是结束月份+1，年份减1
            start_year = end_year - 1
            start_month = end_month + 1
        else:
            # 如果结束月份是12月，开始月份是1月，年份不变
            start_year = end_year
            start_month = 1
        
        # 创建开始日期
        start_date = datetime(start_year, start_month, 1)
        
        # 验证月份差
        months_diff = (end_year - start_year) * 12 + (end_month - start_month) + 1
        logger.info(f"月份差验证: {months_diff}个月")
        logger.info(f"开始日期: {start_date.strftime('%Y-%m-%d')}")
        logger.info(f"结束日期: {last_month_last_day.strftime('%Y-%m-%d')}")
        
        # 格式化查询时间范围（示例：2024-12-01 ~ 2025-11-30）
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = last_month_last_day.strftime('%Y-%m-%d')
        logger.info(f"近一年完整12个月查询范围: {start_date_str} ~ {end_date_str}")

        # 查询近一年月度明细：确保12个完整月，GROUP BY和ORDER BY符合only_full_group_by规范
        cursor.execute("""
            SELECT 
                DATE_FORMAT(stat_date, '%Y年%m月') AS month_str, 
                SUM(record_count) AS monthly_count,
                MIN(stat_date) AS sort_date
            FROM domain_stats
            WHERE stat_date > %s AND stat_date <= %s
            GROUP BY month_str
            ORDER BY sort_date ASC
        """, (start_date_str, end_date_str))
        monthly_detail_data = cursor.fetchall()
        
        # 如果查询结果为空，尝试查询所有数据然后筛选最近12个月
        if not monthly_detail_data:
            logger.warning("按日期范围查询无数据，尝试查询所有月份数据并筛选最近12个月...")
            cursor.execute("""
                SELECT 
                    DATE_FORMAT(stat_date, '%Y年%m月') AS month_str, 
                    SUM(record_count) AS monthly_count,
                    MIN(stat_date) AS sort_date
                FROM domain_stats
                GROUP BY month_str
                ORDER BY sort_date ASC
            """)
            all_monthly_data = cursor.fetchall()
            
            # 筛选最近12个月
            if all_monthly_data:
                # 按日期排序，取最后12个月
                sorted_data = sorted(all_monthly_data, key=lambda x: x['sort_date'])
                monthly_detail_data = sorted_data[-12:] if len(sorted_data) >= 12 else sorted_data
        
        # 提取月份和数量数组
        month_dates = [item['month_str'] for item in monthly_detail_data]
        month_amounts = [item['monthly_count'] for item in monthly_detail_data]
        
        # 获取月份范围用于名称
        if monthly_detail_data:
            # 生成名称：使用第一个和最后一个月份
            start_month = monthly_detail_data[0]['month_str']
            end_month = monthly_detail_data[-1]['month_str']
            month_count = len(monthly_detail_data)
            
            # 检查是否跨年
            start_year = int(start_month[:4])
            end_year = int(end_month[:4])
            
            if start_year == end_year:
                month_chart_name = f"{start_year}年每月恶意域名数量统计"
            else:
                month_chart_name = f"{start_month}-{end_month}恶意域名数量统计"
            
            logger.info(f"月度数据：共{month_count}个月，从{start_month}到{end_month}")
        else:
            month_chart_name = f"{end_year}年每月恶意域名数量统计"
            logger.warning("月度数据为空")
        
        # 近一年总数量
        last_year_domain_count = sum(item['monthly_count'] for item in monthly_detail_data) if monthly_detail_data else 0
        logger.info(f"域名近一年数量（{len(monthly_detail_data)}个月）: {last_year_domain_count:,}")

        # 4. 近一个月数量（本日起向前推30天）+ 每日明细
        thirty_days_ago = current_date - timedelta(days=30)
        # 查询近一个月每日明细：符合only_full_group_by规范
        cursor.execute("""
            SELECT 
                DATE_FORMAT(stat_date, '%m月%d日') AS day_str, 
                SUM(record_count) AS daily_count,
                MIN(stat_date) AS sort_date
            FROM domain_stats
            WHERE stat_date > %s AND stat_date <= %s
            GROUP BY day_str
            ORDER BY sort_date ASC
        """, (thirty_days_ago.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d')))
        daily_detail_data = cursor.fetchall()
        
        # 提取日期和数量数组
        day_dates = [item['day_str'] for item in daily_detail_data]
        day_amounts = [item['daily_count'] for item in daily_detail_data]
        
        # 获取月份用于名称
        if daily_detail_data:
            # 尝试从日期字符串中提取月份
            try:
                first_day = daily_detail_data[0]['day_str']
                month_num = int(first_day.split('月')[0])
                day_chart_name = f"{month_num}月每日恶意域名数量统计"
            except:
                day_chart_name = f"{current_date.month}月每日恶意域名数量统计"
        else:
            day_chart_name = f"{current_date.month}月每日恶意域名数量统计"
        
        # 近一个月总数量
        last_30_days_domain_count = sum(item['daily_count'] for item in daily_detail_data) if daily_detail_data else 0
        logger.info(f"域名近一个月数量（30天）: {last_30_days_domain_count:,}，包含{len(daily_detail_data)}天明细")

        return {
            'total_domain_count': total_domain_count,
            'year_summary': yearly_data,
            'last_year_domain_count': last_year_domain_count,
            'month_detail_summary': monthly_detail_data,  # 月度明细
            'last_30_days_domain_count': last_30_days_domain_count,
            'day_detail_summary': daily_detail_data,       # 近30天明细
            # 新增用于图表的数据结构
            'line_chart_data': {
                'total_domain': {
                    'name': year_chart_name,
                    'date_data': year_dates,
                    'amount_data': year_amounts
                },
                'messages': {
                    'name': month_chart_name,
                    'date_data': month_dates,
                    'amount_data': month_amounts
                },
                'purchases': {
                    'name': day_chart_name,
                    'date_data': day_dates,
                    'amount_data': day_amounts
                }
            }
        }
        
    except Error as e:
        error_msg = f"读取domain_stats数据时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\033[91m{error_msg}\033[0m")
        return None
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            logger.info("域名数据库连接已关闭")



            
def get_top10_source(pool):
    """从source_stats表获取count最大的10个source及对应数量"""
    try:
        connection = pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        logger.info("开始读取source_stats表，获取Top10 source")

        cursor.execute("""
            SELECT source, count
            FROM source_stats
            ORDER BY count DESC
            LIMIT 10 OFFSET 1
        """)
        top10_source = cursor.fetchall()
        logger.info(f"成功获取Top10 source数据，共{len(top10_source)}条记录")

        return [{'source': item['source'], 'count': item['count']} for item in top10_source]
        
    except Error as e:
        error_msg = f"读取source_stats数据时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\033[91m{error_msg}\033[0m")
        return None
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()

def get_top10_category(pool):
    """从category_stats表获取count最大的10个category及对应数量"""
    try:
        connection = pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        logger.info("开始读取category_stats表，获取Top10 category")

        cursor.execute("""
            SELECT category, count
            FROM category_stats
            ORDER BY count DESC
            LIMIT 10
        """)
        top10_category = cursor.fetchall()
        logger.info(f"成功获取Top10 category数据，共{len(top10_category)}条记录")

        return [{'category': item['category'], 'count': item['count']} for item in top10_category]
        
    except Error as e:
        error_msg = f"读取category_stats数据时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\033[91m{error_msg}\033[0m")
        return None
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()

# ---------------------- 原有数据读取与处理 ----------------------
def get_benign_sample_count(pool):
    """从db_benign数据库读取sample_counts表的总记录数（良性样本总数）"""
    try:
        connection = pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        logger.info("成功获取良性数据库连接，开始读取sample_counts数据")

        cursor.execute("SELECT SUM(record_count) as total_count FROM sample_counts")
        result = cursor.fetchone()
        total_count = result['total_count'] if result and result['total_count'] is not None else 0
        
        logger.info(f"读取良性样本总数: {total_count:,}")
        return total_count
        
    except Error as e:
        error_msg = f"读取良性样本数据时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\033[91m{error_msg}\033[0m")
        return 0
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            logger.info("良性数据库连接已关闭")

def get_malicious_sample_data(pool):
    """从主数据库读取恶意样本的统计数据（年度、月度、分类等）"""
    try:
        connection = pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        logger.info("成功获取主数据库连接（恶意样本库），开始读取统计数据")

        # 1. 年度统计数据（恶意样本）
        cursor.execute("SELECT year, total_samples FROM sample_yearly_stats ORDER BY year")
        year_data = cursor.fetchall()
        year_counts = {item['year']: item['total_samples'] for item in year_data}
        malicious_total = sum(item['total_samples'] for item in year_data)  # 恶意样本总数
        logger.info(f"读取恶意样本年度数据: {len(year_counts)}个年份，恶意样本总数: {malicious_total:,}")

        # 2. 月度统计数据（近12个月恶意样本）
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # 计算近12个月的起始年月（从当前月份往前推12个月）
        # 例如：现在是2026年2月，近一年应该是2025年3月到2026年2月
        start_year = current_year - 1
        start_month = current_month + 1
        if start_month > 12:
            start_month = 1
            start_year = current_year
        
        # 查询近12个月的数据
        cursor.execute("""
            SELECT year, month, total_samples 
            FROM sample_monthly_stats 
            WHERE (year > %s OR (year = %s AND month >= %s))
               AND (year < %s OR (year = %s AND month <= %s))
            ORDER BY year, month
        """, (start_year, start_year, start_month, current_year, current_year, current_month))
        monthly_data = cursor.fetchall()
        monthly_counts = {f"{item['year']}-{item['month']:02d}": item['total_samples'] for item in monthly_data}
        recent_year_malicious = sum(item['total_samples'] for item in monthly_data)  # 近一年恶意样本数
        logger.info(f"读取近12个月恶意样本月度数据: {len(monthly_counts)}个月份，样本数: {recent_year_malicious:,}")

        # 3. 分类统计（恶意样本）
        cursor.execute("SELECT category, count FROM sample_category_stats")
        category_data = cursor.fetchall()
        category_counter = Counter({item['category']: item['count'] for item in category_data})
        logger.info(f"读取恶意样本Category数据: {len(category_counter)}个类别")

        # 4. 平台统计（恶意样本）
        cursor.execute("SELECT platform, count FROM sample_platform_stats")
        platform_data = cursor.fetchall()
        platform_counter = Counter({item['platform']: item['count'] for item in platform_data})
        logger.info(f"读取恶意样本Platform数据: {len(platform_counter)}个类别")

        # 5. 家族统计（恶意样本）
        cursor.execute("SELECT family, count FROM sample_family_stats")
        family_data = cursor.fetchall()
        family_counter = Counter({item['family']: item['count'] for item in family_data})
        logger.info(f"读取恶意样本Family数据: {len(family_counter)}个类别")

        # 6. 行为统计【关键修改：适配新列名】
        cursor.execute("""
            SELECT has_vt_1, has_vt_summary_1, has_vt_mitre_1 
            FROM sample_behavior_stats 
            LIMIT 1
        """)
        behavior_data = cursor.fetchone() or {
            'has_vt_1': 0,
            'has_vt_summary_1': 0,
            'has_vt_mitre_1': 0
        }
        logger.info(f"读取恶意样本行为统计数据: has_vt={behavior_data['has_vt_1']}, has_vt_summary={behavior_data['has_vt_summary_1']}")

        # 7. 文件类型统计（恶意样本）
        cursor.execute("SELECT filetype, count FROM sample_filetype_stats")
        filetype_data = cursor.fetchall()
        filetype_counter = {item['filetype']: item['count'] for item in filetype_data}
        logger.info(f"读取恶意样本文件类型数据: {len(filetype_counter)}种类型")

        # 处理文件类型与卡巴结果统计
        filetype_kaspersky = {
            'exe32': {'total': 0, 'has_result': 0, 'no_result': 0},
            'exe64': {'total': 0, 'has_result': 0, 'no_result': 0},
            'dll32': {'total': 0, 'has_result': 0, 'no_result': 0},
            'dll64': {'total': 0, 'has_result': 0, 'no_result': 0},
            'total': {'total': 0, 'has_result': 0, 'no_result': 0}
        }
        for ft, count in filetype_counter.items():
            if ft.lower() in filetype_kaspersky:
                filetype_kaspersky[ft.lower()]['total'] = count
                filetype_kaspersky[ft.lower()]['has_result'] = count
                filetype_kaspersky['total']['total'] += count
        logger.info("恶意样本文件类型统计数据处理完成")

        return {
            'year_counts': year_counts,
            'monthly_counts': monthly_counts,
            'category_counter': category_counter,
            'platform_counter': platform_counter,
            'family_counter': family_counter,
            'malicious_total': malicious_total,  # 恶意样本总数
            'recent_year_malicious': recent_year_malicious,  # 近一年恶意样本数
            'filetype_kaspersky': filetype_kaspersky,
            # 行为统计字段（新列名映射）
            'detection_1': behavior_data['has_vt_1'],
            'behaviour_summary_1': behavior_data['has_vt_summary_1'],
            'behaviour_mitre_trees_1': behavior_data['has_vt_mitre_1'],
            'current_year': current_year
        }
        
    except Error as e:
        error_msg = f"读取恶意样本统计数据时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\033[91m{error_msg}\033[0m")
        return None
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            logger.info("主数据库连接（恶意样本库）已关闭")

def process_pie_top10(counter):
    """处理饼图数据：取Top10，其余合并为Others"""
    all_items = counter.most_common()
    if len(all_items) <= 10:
        return [{'name': name, 'value': count} for name, count in all_items]
    top10 = all_items[:10]
    others_count = sum(count for _, count in all_items[10:])
    top10_with_others = [{'name': name, 'value': count} for name, count in top10]
    top10_with_others.append({'name': 'Others', 'value': others_count})
    return top10_with_others

def process_full_pie(counter):
    """处理完整分类数据（所有类别）"""
    return [{'name': name, 'value': count} for name, count in counter.most_common()]

def prepare_chart_data(malicious_data, benign_total_count=0, domain_stats_data=None, top10_source=None, top10_category=None):
    """准备图表所需的数据格式（匹配前端要求，新增域名相关统计数据）"""
    logger.info("开始准备图表数据格式")

    # 1. 年份折线图数据（恶意样本）
    # 确保包含当前年份,即使没有数据也显示为0
    current_year = datetime.now().year
    year_items = sorted(malicious_data['year_counts'].items(), key=lambda x: x[0])
    
    # 检查是否包含当前年份,如果没有则添加
    if year_items and year_items[-1][0] < current_year:
        year_items.append((current_year, 0))
    
    year_chart_data = {
        'date_data': [f"{item[0]}年" for item in year_items],
        'amount_data': [item[1] for item in year_items]
    }

    # 2. 近一年月度折线图数据（近12个月恶意样本）
    # 生成完整的12个月,没有数据的月份显示为0
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # 计算起始年月（从当前月份往前推12个月）
    start_year = current_year - 1
    start_month = current_month + 1
    if start_month > 12:
        start_month = 1
        start_year = current_year
    
    # 生成完整的12个月列表
    all_months = []
    year, month = start_year, start_month
    for _ in range(12):
        all_months.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    # 填充数据,没有数据的月份设为0
    full_monthly = {month: 0 for month in all_months}
    for month, count in malicious_data['monthly_counts'].items():
        if month in full_monthly:
            full_monthly[month] = count
    
    sorted_monthly = sorted(full_monthly.items())
    monthly_chart_data = {
        'date_data': [f"{item[0].split('-')[0]}年{item[0].split('-')[1]}月" for item in sorted_monthly],
        'amount_data': [item[1] for item in sorted_monthly]
    }

    # 3. 饼图数据（恶意样本分类）
    category_top10 = process_pie_top10(malicious_data['category_counter'])
    category_full = process_full_pie(malicious_data['category_counter'])
    platform_top10 = process_pie_top10(malicious_data['platform_counter'])
    platform_full = process_full_pie(malicious_data['platform_counter'])
    family_top10 = process_pie_top10(malicious_data['family_counter'])
    family_full = process_full_pie(malicious_data['family_counter'])

    # 整理文件类型统计数据（恶意样本）
    filetype_stats = {}
    for filetype in ['exe32', 'exe64', 'dll32', 'dll64', 'total']:
        data = malicious_data['filetype_kaspersky'][filetype]
        percentage = round(
            (data['has_result'] / data['total'] * 100) 
            if data['total'] > 0 else 0, 2
        )
        filetype_stats[filetype] = {
            'total': data['total'],
            'hasResult': data['has_result'],
            'noResult': data['no_result'],
            'percentage': percentage
        }

    # 行为统计数据【无需修改：前端仍用旧字段名，后端映射新值】
    behavior_stats = {
        'detection1': malicious_data['detection_1'],
        'behaviourSummary1': malicious_data['behaviour_summary_1'],
        'behaviourMitreTrees1': malicious_data['behaviour_mitre_trees_1']
    }

    # 处理域名数据
    domain_line_chart_data = {}
    if domain_stats_data and 'line_chart_data' in domain_stats_data:
        domain_line_chart_data = domain_stats_data['line_chart_data']
    else:
        # 如果没有域名数据，提供空结构
        domain_line_chart_data = {
            'total_domain': {
                'name': '年度域名统计',
                'date_data': [],
                'amount_data': []
            },
            'messages': {
                'name': '月度域名统计',
                'date_data': [],
                'amount_data': []
            },
            'purchases': {
                'name': '每日域名统计',
                'date_data': [],
                'amount_data': []
            }
        }

    # 整合所有数据【核心修正：总数=恶意+良性，恶意样本数直接使用主库统计值，新增域名相关数据】
    chart_data = {
        'lineChartData': {
            'total_amount': year_chart_data,  # 年度恶意样本趋势
            'year_amount': monthly_chart_data  # 近一年恶意样本趋势
        },
        'pieTop10Data': {
            'category': category_top10,
            'platform': platform_top10,
            'family': family_top10
        },
        'pieFullData': {
            'category': category_full,
            'platform': platform_full,
            'family': family_full
        },
        'filetypeStats': filetype_stats,
        'behaviorStats': behavior_stats,
        # 新增域名相关统计数据（含12个完整月明细）
        'domainStats': domain_stats_data or {},
        'top10Source': top10_source or [],
        'top10Category': top10_category or [],
        # 新增：域名折线图数据，格式符合Vue前端要求
        'lineChartDataDomain': domain_line_chart_data,
        'summary': {
            'total_samples': malicious_data['malicious_total'] + benign_total_count,  # 总样本数=恶意+良性
            'benign_samples': benign_total_count,  # 良性样本数
            'malicious_samples': malicious_data['malicious_total'],  # 恶意样本数（直接使用主库统计）
            'recent_year_samples': malicious_data['recent_year_malicious'],  # 近一年样本数（恶意）
            'current_year': malicious_data['current_year'],
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    logger.info("图表数据格式准备完成")
    return chart_data

def convert_decimal_to_int(obj):
    """将对象中的Decimal转换为整数"""
    if hasattr(obj, 'to_eng_string'):
        # 如果是Decimal对象
        try:
            return int(obj)
        except:
            # 如果无法转换为整数，尝试转换为浮点数
            try:
                return float(obj)
            except:
                return str(obj)
    elif isinstance(obj, list):
        return [convert_decimal_to_int(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_decimal_to_int(value) for key, value in obj.items()}
    else:
        return obj

# ---------------------- 文件生成（新增域名数据+12个月明细写入） ----------------------
def save_to_js(data, filename='chart_data.js'):
    """生成Vue可用的JS文件（完全匹配要求的格式，新增域名相关数据）"""
    try:
        save_dir = get_save_path()
        file_path = os.path.join(save_dir, filename)
        
        # 转换Decimal为普通数字
        data_converted = {
            'lineChartData': convert_decimal_to_int(data['lineChartData']),
            'pieTop10Data': convert_decimal_to_int(data['pieTop10Data']),
            'pieFullData': convert_decimal_to_int(data['pieFullData']),
            'filetypeStats': convert_decimal_to_int(data['filetypeStats']),
            'behaviorStats': convert_decimal_to_int(data['behaviorStats']),
            'lineChartDataDomain': convert_decimal_to_int(data['lineChartDataDomain']),
            'top10Source': convert_decimal_to_int(data['top10Source']),
            'top10Category': convert_decimal_to_int(data['top10Category']),
            'summary': convert_decimal_to_int(data['summary'])
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入固定开头
            f.write("export default {\n")
            
            # 1. 折线图数据
            f.write("  lineChartData: {\n")
            total_amount = data_converted['lineChartData']['total_amount']
            year_amount = data_converted['lineChartData']['year_amount']
            f.write(f"    total_amount: {{'date_data': {json.dumps(total_amount['date_data'])}, 'amount_data': {json.dumps(total_amount['amount_data'])}}},\n")
            f.write(f"    year_amount: {{'date_data': {json.dumps(year_amount['date_data'])}, 'amount_data': {json.dumps(year_amount['amount_data'])}}}\n")
            f.write("  },\n")
            
            # 2. 饼图Top10数据
            f.write("  pieTop10Data: {\n")
            pie_data = data_converted['pieTop10Data']
            f.write(f"    category: {json.dumps(pie_data['category'], ensure_ascii=False)},\n")
            f.write(f"    platform: {json.dumps(pie_data['platform'], ensure_ascii=False)},\n")
            f.write(f"    family: {json.dumps(pie_data['family'], ensure_ascii=False)}\n")
            f.write("  },\n")
            
            # 3. 完整饼图数据
            f.write("  pieFullData: {\n")
            pie_full_data = data_converted['pieFullData']
            f.write(f"    category: {json.dumps(pie_full_data['category'], ensure_ascii=False)},\n")
            f.write(f"    platform: {json.dumps(pie_full_data['platform'], ensure_ascii=False)},\n")
            f.write(f"    family: {json.dumps(pie_full_data['family'], ensure_ascii=False)}\n")
            f.write("  },\n")
            
            # 4. 文件类型统计
            f.write("  filetypeStats: {\n")
            filetype_data = data_converted['filetypeStats']
            for ft in ['exe32', 'exe64', 'dll32', 'dll64', 'total']:
                stats = filetype_data[ft]
                f.write(f"    {ft}: {{\n")
                f.write(f"      total: {stats['total']},\n")
                f.write(f"      hasResult: {stats['hasResult']},\n")
                f.write(f"      noResult: {stats['noResult']},\n")
                f.write(f"      percentage: {stats['percentage']}\n")
                f.write(f"    }},\n")
            f.write("  },\n")
            
            # 5. 行为统计
            f.write("  behaviorStats: {\n")
            behavior_data = data_converted['behaviorStats']
            f.write(f"    detection1: {behavior_data['detection1']},\n")
            f.write(f"    behaviourSummary1: {behavior_data['behaviourSummary1']},\n")
            f.write(f"    behaviourMitreTrees1: {behavior_data['behaviourMitreTrees1']}\n")
            f.write("  },\n")
            
            # 6. 新增：域名折线图数据（符合Vue前端格式要求）
            f.write("  lineChartDataDomain: {\n")
            domain_line_data = data_converted['lineChartDataDomain']
            
            # 格式化total_domain数据
            total_domain = domain_line_data['total_domain']
            f.write(f"    total_domain: {{'name': '{total_domain['name']}', 'date_data': {json.dumps(total_domain['date_data'], ensure_ascii=False)}, 'amount_data': {json.dumps(total_domain['amount_data'])}}},\n")
            
            # 格式化messages数据
            messages = domain_line_data['messages']
            f.write(f"    messages: {{'name': '{messages['name']}', 'date_data': {json.dumps(messages['date_data'], ensure_ascii=False)}, 'amount_data': {json.dumps(messages['amount_data'])}}},\n")
            
            # 格式化purchases数据
            purchases = domain_line_data['purchases']
            f.write(f"    purchases: {{'name': '{purchases['name']}', 'date_data': {json.dumps(purchases['date_data'], ensure_ascii=False)}, 'amount_data': {json.dumps(purchases['amount_data'])}}}\n")
            f.write("  },\n")
            
            # 7. 新增：Top10 Source
            top10_source_data = data_converted['top10Source']
            f.write("  top10Source: [")
            for i, item in enumerate(top10_source_data):
                source = item['source'].replace('\r', '\\r')
                f.write(f'{{"source": "{source}", "count": {item["count"]}}}')
                if i < len(top10_source_data) - 1:
                    f.write(", ")
            f.write("],\n")
            
            # 8. 新增：Top10 Category
            top10_category_data = data_converted['top10Category']
            f.write("  top10Category: [")
            for i, item in enumerate(top10_category_data):
                f.write(f'{{"category": "{item["category"]}", "count": {item["count"]}}}')
                if i < len(top10_category_data) - 1:
                    f.write(", ")
            f.write("],\n")
            
            # 9. 汇总信息
            summary = data_converted['summary']
            f.write("  summary: {\n")
            f.write(f"    total_samples: {summary['total_samples']},\n")
            f.write(f"    benign_samples: {summary['benign_samples']},\n")
            f.write(f"    malicious_samples: {summary['malicious_samples']},\n")
            f.write(f"    recent_year_samples: {summary['recent_year_samples']},\n")
            f.write(f"    current_year: {summary['current_year']},\n")
            f.write(f"    generated_at: '{summary['generated_at']}'\n")
            f.write("  }\n")
            
            # 结尾
            f.write("};\n")
        
        success_msg = f"Vue可用JS文件已保存到 {file_path}"
        logger.info(success_msg)
        print(f"\033[92m{success_msg}\033[0m")
        
    except Exception as e:
        error_msg = f"保存JS文件时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\033[91m{error_msg}\033[0m")

def save_to_txt(data, filename='stats_summary.txt'):
    """生成TXT统计文件（保留原有功能，新增域名12个完整月明细）"""
    try:
        save_dir = get_save_path()
        file_path = os.path.join(save_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # 1. 统计概览【已修正：总数=恶意+良性】
            f.write("="*50 + "\n")
            f.write("统计概览\n")
            f.write("-"*50 + "\n")
            f.write(f"总样本数: {data['summary']['total_samples']:,}\n")
            f.write(f"恶意样本数: {data['summary']['malicious_samples']:,}\n")  # 调整顺序，突出恶意样本
            f.write(f"良性样本数: {data['summary']['benign_samples']:,}\n")
            f.write(f"{data['summary']['current_year']}年恶意样本数: {data['summary']['recent_year_samples']:,}\n")
            f.write(f"数据生成时间: {data['summary']['generated_at']}\n")
            f.write("\n"*2)
            
            # 新增：域名统计概览
            f.write("="*50 + "\n")
            f.write("域名（db_domain）统计概览\n")
            f.write("-"*50 + "\n")
            domain_data = data['domainStats']
            f.write(f"域名总数量: {domain_data.get('total_domain_count', 0):,}\n")
            f.write(f"域名近一年数量（12个完整自然月）: {domain_data.get('last_year_domain_count', 0):,}\n")
            f.write(f"域名近一个月数量（30天）: {domain_data.get('last_30_days_domain_count', 0):,}\n")
            f.write("\n")
            
            # 域名按年汇总
            f.write("域名按年汇总\n")
            f.write("-"*50 + "\n")
            f.write(f"{'年份':<10} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            year_summary = domain_data.get('year_summary', [])
            for item in year_summary:
                year_str = f"{item['year']}年"
                f.write(f"{year_str:<10} | {item['yearly_count']:15,}\n")
            f.write("\n"*2)
            
            # 新增：近一年域名月度明细（12个完整自然月，按时间升序）
            f.write("="*50 + "\n")
            f.write("域名近一年月度明细（12个完整自然月，按时间升序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'月份（yyyy年mm月）':<15} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            month_detail = domain_data.get('month_detail_summary', [])
            for item in month_detail:
                f.write(f"{item['month_str']:<15} | {item['monthly_count']:15,}\n")
            f.write("\n"*2)
            
            # 新增：近一个月域名每日明细（mm月dd日 + 数量）
            f.write("="*50 + "\n")
            f.write("域名近一个月每日明细（30天内，按时间升序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'日期（mm月dd日）':<15} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            day_detail = domain_data.get('day_detail_summary', [])
            for item in day_detail:
                f.write(f"{item['day_str']:<15} | {item['daily_count']:15,}\n")
            f.write("\n"*2)
            
            # 新增：Top10 Source统计
            f.write("="*50 + "\n")
            f.write("Source Top10统计（按count降序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Source名称':<30} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for item in data['top10Source']:
                short_source = item['source'][:27] + "..." if len(item['source']) > 30 else item['source']
                f.write(f"{short_source:<30} | {item['count']:15,}\n")
            f.write("\n"*2)
            
            # 新增：Top10 Category统计
            f.write("="*50 + "\n")
            f.write("Category Top10统计（按count降序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Category名称':<30} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for item in data['top10Category']:
                short_category = item['category'][:27] + "..." if len(item['category']) > 30 else item['category']
                f.write(f"{short_category:<30} | {item['count']:15,}\n")
            f.write("\n"*2)
            
            # 2. 文件类型统计（恶意样本）
            f.write("="*50 + "\n")
            f.write("恶意样本文件类型统计\n")
            f.write("-"*50 + "\n")
            f.write(f"{'文件类型':<10} | {'总数':<10} | {'有结果':<10} | {'无结果':<10} | {'占比(%)':<8}\n")
            f.write("-"*50 + "\n")
            for filetype in ['exe32', 'exe64', 'dll32', 'dll64']:
                stats = data['filetypeStats'][filetype]
                f.write(f"{filetype:<10} | {stats['total']:<10,} | {stats['hasResult']:<10,} | {stats['noResult']:<10,} | {stats['percentage']:<8}\n")
            f.write("-"*50 + "\n")
            total = data['filetypeStats']['total']
            f.write(f"{'总计':<10} | {total['total']:<10,} | {total['hasResult']:<10,} | {total['noResult']:<10,} | {total['percentage']:<8}\n")
            f.write("\n"*2)
            
            # 3. 行为特征统计【备注新列名映射】
            f.write("="*50 + "\n")
            f.write("恶意样本行为特征统计（值为1的数量）\n")
            f.write("-"*50 + "\n")
            bs = data['behaviorStats']
            f.write(f"has_vt (原detection): {bs['detection1']:,}\n")
            f.write(f"has_vt_summary (原behaviour_summary): {bs['behaviourSummary1']:,}\n")
            f.write(f"has_vt_mitre (原behaviour_mitre_trees): {bs['behaviourMitreTrees1']:,}\n")
            f.write("\n"*2)
            
            # 4. 恶意样本年份分布
            f.write("="*50 + "\n")
            f.write("恶意样本年份分布\n")
            f.write("-"*50 + "\n")
            f.write(f"{'年份':<10} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for year, amount in zip(
                data['lineChartData']['total_amount']['date_data'],
                data['lineChartData']['total_amount']['amount_data']
            ):
                f.write(f"{year:<10} | {amount:15,}\n")
            f.write("\n"*2)
            
            # 5. 近一年恶意样本月度分布
            f.write("="*50 + "\n")
            f.write(f"{data['summary']['current_year']}年恶意样本月度分布\n")
            f.write("-"*50 + "\n")
            f.write(f"{'月份':<12} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for month, amount in zip(
                data['lineChartData']['year_amount']['date_data'],
                data['lineChartData']['year_amount']['amount_data']
            ):
                f.write(f"{month:<12} | {amount:15,}\n")
            f.write("\n"*2)
            
            # 6. Category完整分类（恶意样本）
            f.write("="*50 + "\n")
            f.write("恶意样本Category完整分类（按数量降序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'类别':<30} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for item in data['pieFullData']['category']:
                short_name = item['name'][:27] + "..." if len(item['name']) > 30 else item['name']
                f.write(f"{short_name:<30} | {item['value']:15,}\n")
            f.write("\n"*2)
            
            # 7. Platform完整分类（恶意样本）
            f.write("="*50 + "\n")
            f.write("恶意样本Platform完整分类（按数量降序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'类别':<30} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for item in data['pieFullData']['platform']:
                short_name = item['name'][:27] + "..." if len(item['name']) > 30 else item['name']
                f.write(f"{short_name:<30} | {item['value']:15,}\n")
            f.write("\n"*2)
            
            # 8. Family完整分类（恶意样本）
            f.write("="*50 + "\n")
            f.write("恶意样本Family完整分类（按数量降序）\n")
            f.write("-"*50 + "\n")
            f.write(f"{'类别':<30} | {'数量':<15}\n")
            f.write("-"*50 + "\n")
            for item in data['pieFullData']['family']:
                short_name = item['name'][:27] + "..." if len(item['name']) > 30 else item['name']
                f.write(f"{short_name:<30} | {item['value']:15,}\n")
        
        success_msg = f"TXT统计文件已保存到 {file_path}"
        logger.info(success_msg)
        print(f"\033[92m{success_msg}\033[0m")
    except Exception as e:
        error_msg = f"保存TXT文件时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\033[91m{error_msg}\033[0m")

# ---------------------- 对外暴露的核心函数 ----------------------
def generate_frontend_data():
    """
    生成前端数据的核心入口函数
    返回：True（成功）/ False（失败）
    """
    start_time = datetime.now()
    logger.info("="*60)
    logger.info("开始执行前端数据生成任务")
    logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    try:
        # 1. 检查数据库配置
        if not DB_CONFIG:
            logger.error("主数据库（恶意样本库）配置无效，数据生成任务终止")
            return False

        # 2. 创建数据库连接池
        malicious_pool = create_db_pool()
        if not malicious_pool:
            logger.error("主连接池（恶意样本库）创建失败，数据生成任务终止")
            return False

        # 3. 创建良性数据库连接池（可选）
        benign_pool = create_benign_db_pool()
        benign_total_count = 0
        if benign_pool:
            benign_total_count = get_benign_sample_count(benign_pool)
            logger.info(f"良性样本总数: {benign_total_count:,}")
        else:
            logger.warning("未获取到良性样本数据，良性样本数按0计算")

        # 4. 创建域名数据库连接池并读取相关数据（含12个完整月明细）
        domain_pool = create_domain_db_pool()
        domain_stats_data = None
        top10_source = None
        top10_category = None
        if domain_pool:
            domain_stats_data = get_domain_stats_data(domain_pool)
            top10_source = get_top10_source(domain_pool)
            top10_category = get_top10_category(domain_pool)
            logger.info("域名数据库相关数据（含12个完整月明细）读取完成")
        else:
            logger.warning("未获取到域名数据库数据，域名相关统计按空值计算")

        # 5. 读取恶意样本统计数据
        malicious_data = get_malicious_sample_data(malicious_pool)
        if not malicious_data:
            logger.error("恶意样本统计数据读取失败，数据生成任务终止")
            return False

        # 6. 准备图表数据（新增域名12个完整月明细）
        chart_data = prepare_chart_data(malicious_data, benign_total_count, domain_stats_data, top10_source, top10_category)

        # 7. 保存JS和TXT文件（含12个完整月明细）
        save_to_js(chart_data)
        save_to_txt(chart_data)

        # 8. 输出执行结果【同步修正统计逻辑，新增域名12个月数据展示】
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info("="*60)
        logger.info("前端数据生成任务执行成功！")
        logger.info(f"总样本量: {chart_data['summary']['total_samples']:,}")
        logger.info(f"恶意样本量: {chart_data['summary']['malicious_samples']:,}")
        logger.info(f"良性样本量: {chart_data['summary']['benign_samples']:,}")
        logger.info(f"{chart_data['summary']['current_year']}年恶意样本量: {chart_data['summary']['recent_year_samples']:,}")
        logger.info(f"年份数量: {len(malicious_data['year_counts'])}")
        logger.info(f"category类别数量: {len(malicious_data['category_counter'])}")
        logger.info(f"platform类别数量: {len(malicious_data['platform_counter'])}")
        logger.info(f"family类别数量: {len(malicious_data['family_counter'])}")
        if domain_stats_data:
            logger.info(f"域名总数量: {domain_stats_data['total_domain_count']:,}")
            logger.info(f"域名近一年数量: {domain_stats_data['last_year_domain_count']:,}（含{len(domain_stats_data.get('month_detail_summary', []))}个完整月明细）")
            logger.info(f"域名近30天数量: {domain_stats_data['last_30_days_domain_count']:,}（含{len(domain_stats_data.get('day_detail_summary', []))}天明细）")
        logger.info(f"执行时间: {elapsed:.2f} 秒")
        logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

        # 控制台提示
        print("\n" + "="*60)
        print(f"\033[92m前端数据生成任务执行成功！\033[0m")
        print(f"总样本量: {chart_data['summary']['total_samples']:,}")
        print(f"恶意样本量: {chart_data['summary']['malicious_samples']:,}")
        print(f"良性样本量: {chart_data['summary']['benign_samples']:,}")
        if domain_stats_data:
            print(f"域名总数量: {domain_stats_data['total_domain_count']:,}")
            print(f"域名近一年数量: {domain_stats_data['last_year_domain_count']:,}（含{len(domain_stats_data.get('month_detail_summary', []))}个完整月明细）")
            print(f"域名近30天数量: {domain_stats_data['last_30_days_domain_count']:,}（含{len(domain_stats_data.get('day_detail_summary', []))}天明细）")
        print(f"执行时间: {elapsed:.2f} 秒")
        print("="*60)

        return True

    except Exception as e:
        error_msg = f"数据生成任务异常终止: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\n\033[91m{error_msg}\033[0m")
        return False

# ---------------------- 独立运行入口 ----------------------
if __name__ == "__main__":
    """独立运行时执行数据生成"""
    print("="*60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 独立执行前端数据生成任务")
    print("="*60)
    
    result = generate_frontend_data()
    
    print("\n" + "="*60)
    if result:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 执行结果：\033[92m成功\033[0m")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 执行结果：\033[91m失败\033[0m")
    print("="*60)