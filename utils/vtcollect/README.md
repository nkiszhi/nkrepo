========================================
VT 报告收集器（优化版） 使用说明
========================================

【简介】
本程序用于通过 VirusTotal API 批量获取样本的行为摘要（behaviour summary）报告。支持多 API 密钥轮换、自动重试、配额管理、失败样本自动清理以及每日报告打包。新增功能：自动同步 CSV 与已处理样本，避免重复请求；修复打包时的 ZipFile 错误；ZIP 文件永久保留，不会自动删除。

【主要特性】
- 多密钥轮换：每个样本尝试所有可用 API 密钥，直至成功或全部尝试失败。
- 配额管理：记录每个密钥每日调用次数，达到上限后自动切换密钥，所有密钥用尽则停止当天任务。
- 临时错误保护：网络超时、服务器 5xx 等临时错误会重试，连续错误达到阈值则停止扫描，避免无效循环。
- 失败样本自动清理：
  * 对返回 404（无报告）或持续错误的样本记录重试次数，达到上限后自动从 CSV 中移除。
  * 成功获取报告的样本，会在当天打包前从 CSV 中删除，确保次日不重复处理。
- 历史遗留清理：每日任务开始时，检查 result 文件夹中所有已存在的 JSON 报告，自动从 CSV 中移除对应的样本（用于修复之前未打包导致的数据残留）。
- 每日报告打包：将当天生成的 *_behaviour_summary.json 文件打包为 ZIP，并删除源文件。ZIP 文件永久保留，不会自动删除。
- 定时任务：支持按配置时间每天自动执行，也可通过 --now 参数立即运行。
- 灵活配置：所有参数均通过 config.ini 配置，无需修改代码。

【环境要求】
- Python 3.7+
- 依赖包：requests, pandas, schedule
- 安装命令：pip install requests pandas schedule

【文件结构】
项目目录/
├── vt_scan_main.py     # 主程序
├── config.ini          # 配置文件（需自行创建）
├── result/             # 存放 JSON 报告及临时标记文件（程序自动创建）
├── zip/                # 存放每日 ZIP 打包文件（程序自动创建）
├── logs/               # 日志文件（程序自动创建）
└── list.csv            # 样本 SHA256 列表（需自行准备）

【配置说明】
config.ini 示例：

[api]
keys = YOUR_API_KEY1, YOUR_API_KEY2, YOUR_API_KEY3
max_calls_per_key = 1000
request_interval = 1

[paths]
csv_file = list.csv
result_folder = result
zip_output_dir = zip
log_file = vt_scan.log

[settings]
zip_prefix = vt_reports
daily_target_min = 100
max_retries_for_404 = 3
max_retries_per_sample = 5
max_consecutive_temporary_errors = 10
cleanup_old_days = 7
schedule_time = 08:00
retry_wait_minutes = 30

各参数含义：
- keys：API 密钥，逗号分隔
- max_calls_per_key：每个密钥每日最大调用次数（根据 VT 套餐设置）
- request_interval：API 请求间隔（秒），避免触发速率限制
- csv_file：样本列表 CSV 文件，必须包含 sha256 列
- result_folder：存放 JSON 报告和标记文件的目录
- zip_output_dir：打包后的 ZIP 存放目录
- log_file：日志文件路径
- zip_prefix：ZIP 文件名前缀（最终格式：前缀_YYYYMMDD.zip）
- daily_target_min：每日成功报告数最低目标，未达到时会在 retry_wait_minutes 分钟后重试
- max_retries_for_404：404 错误重试次数（当前未使用，保留字段）
- max_retries_per_sample：单个样本最大重试次数（达到后从 CSV 删除）
- max_consecutive_temporary_errors：最大连续临时错误次数，超过则停止当天扫描
- cleanup_old_days：清理超过 N 天的标记文件
- schedule_time：每日定时执行时间（24 小时制，如 08:00）
- retry_wait_minutes：未达到每日目标时的重试等待分钟数

【运行方法】
1. 直接运行（立即执行一次）：
   python vt_scan_main.py --now

2. 后台定时运行：
   python vt_scan_main.py
   程序将在后台按 schedule_time 配置的每天时间执行，并保持运行状态。

3. 停止运行：按 Ctrl+C 终止。

【工作流程】
1. 启动：加载配置，初始化日志。
2. 每日定时触发（或 --now 触发）：
   - 清理历史遗留：从 CSV 中移除所有 result 文件夹中已有 JSON 对应的样本（防止重复）。
   - 清理过期的标记文件。
   - 加载密钥使用记录，重置今日计数。
   - 开始扫描 CSV 中的样本（仅处理未成功、未达重试上限的样本）。
   - 每个样本依次尝试所有可用密钥，直到成功或全部失败。
   - 成功获取报告后写入 JSON 文件，并从密钥计数中扣除调用次数。
   - 遇到临时错误（超时、5xx 等）会重试，连续错误达阈值则终止当天任务。
   - 遇到配额用尽或密钥无效，自动切换下一个密钥。
   - 扫描结束后，统计今日成功数。
   - 清理达到重试上限的失败样本（删除标记文件并从 CSV 移除）。
   - 从 CSV 中移除今天已成功获取的样本（基于 result 文件夹中当天的 JSON 文件）。
   - 打包当天生成的 JSON 文件为 ZIP，并删除源文件。ZIP 文件永久保留，不会自动删除。
   - 若当日成功数未达 daily_target_min 且配额未用尽，则在 retry_wait_minutes 分钟后重试当日任务。
3. 循环等待：保持程序运行，等待下一个定时任务。

【注意事项】
- CSV 文件格式：必须包含 sha256 列，每行一个 64 位十六进制哈希值。
- API 密钥配额：每个密钥每日调用次数应正确配置 max_calls_per_key，否则可能提前用尽或浪费。
- 日志：程序会记录详细信息，便于追踪错误和配额使用情况。
- 数据保留：成功报告打包后即删除 JSON，ZIP 文件永久保留，请手动管理（如需清理可自行删除旧 ZIP）。
- 安全：API 密钥应妥善保管，避免泄露。

【故障排查】
- 若程序无法启动，请检查 config.ini 是否存在且格式正确。
- 若出现 zipfile 相关错误，请确保已使用最新版代码（已修复 shutil.ZipFile 问题）。
- 若发现重复请求样本，请检查 result 文件夹中是否存在对应 JSON 文件，或 CSV 中是否残留已处理样本。新版程序已自动处理。
- 日志中若显示“所有密钥今日配额均用尽”，说明当日调用次数已达上限，次日自动重置。

【更新日志】
- v2.1：移除自动删除 ZIP 文件功能，ZIP 文件永久保留。
- v2.0：新增自动同步 CSV 功能；修复打包时 shutil.ZipFile 错误；增加历史遗留样本清理。
- v1.0：初始版本，支持多密钥轮换、失败清理、定时打包。