# sample_stats 数据库操作说明

## 需要先改的配置

在服务器 `web/fastapi/config.ini` 的 `[mysql]` 下增加：

```ini
malicious_dbs = malicious_elf,malicious_pe,malicious_others
benign_dbs = benign_elf,benign_pe,benign_others
db_stats = sample_stats
```

`web/fastapi/config.ini` 被 `.gitignore` 忽略，不会上传到 GitHub。服务器拉取最新代码后，需要在服务器本地按本节配置手工确认一次。

同时确认 `[paths]`：

```ini
sample_root = ../../data/samples
web_upload_dir = ../../data/web_upload_file
zips_dir = ../../data/zips
```

确认已有 MySQL 连接项：

```ini
host = 10.134.13.242
user = 你的用户
passwd = 你的密码
port = 3306
charset = utf8
```

`webdatadb` 不需要调整。

## 第一次建库并全量刷新

在服务器项目目录执行：

```bash
cd /home/nkamg/nkrepo/web/fastapi
python app/services/data/sample_stats_refresh.py --mode full
```

脚本会自动创建：

- `sample_stats` 数据库
- 联动汇总表
- 表级缓存表
- 刷新日志表

不需要你手工写 `CREATE TABLE`，除非 MySQL 用户没有建库权限。

## 后续有新数据后的刷新

如果只是新增或更新部分样本，只需要手动刷新样本联动表：

```bash
cd /home/nkamg/nkrepo/web/fastapi
python app/services/data/sample_stats_refresh.py --mode incremental
```

增量模式会检查每个 `sample_00` 到 `sample_ff` 分表的：

- `COUNT(*)`
- `MAX(updated_at)`

只有数量或最大更新时间变化的分表才会重新统计。最后脚本会重新汇总全局联动表。

如果样本从 `others` 调整到 `pe` 或 `elf`，通常会表现为旧库分表数量减少、新库分表数量增加，增量刷新会重算对应分表并更新汇总。因此新增、删除、库间迁移都能通过下一次 `sample_stats_refresh.py --mode incremental` 反映到联动表。大批量迁移后仍建议跑一次 `--mode full` 做全量校准。

FastAPI 后端启动时会自动读取 `sample_stats` 和域名联动表，并继续生成前端使用的：

- `web/vue/src/data/chart_data.js`
- `web/vue/src/data/stats_summary.txt`

后端运行期间还会每天 `08:30` 自动重新读取数据库联动表并生成上述前端数据文件。这个定时任务只读取联动表，不会自动刷新 `sample_stats`；样本联动表刷新仍由你手动执行 `sample_stats_refresh.py`。

## 如果数据迁移很大

大批量迁移后建议直接跑全量：

```bash
python app/services/data/sample_stats_refresh.py --mode full
```

全量模式会清空表级缓存后重算全部六个库。

## 核心联动表

常用查询表：

- `sample_total_stats`
  - 按 `malicious/benign + elf/pe/others` 保存总数。
- `sample_behavior_stats`
  - 统计恶意样本 `has_vt = 1`、`has_vt_summary = 1`、`has_vt_mitre = 1` 的数量。
- `sample_yearly_stats`
  - 只写恶意样本年度统计。
- `sample_monthly_stats`
  - 只写恶意样本月度统计。
- `sample_category_stats`
  - 只写恶意样本 category 统计。
- `sample_platform_stats`
  - 只写恶意样本 platform 统计。
- `sample_family_stats`
  - 只写恶意样本 family 统计。
- `sample_filetype_stats`
  - 只写恶意样本 filetype 统计。
- `sample_stats_refresh_log`
  - 每次刷新结果。

## 验证 SQL

查看三类恶意样本总量：

```sql
SELECT sample_class, file_kind, total_samples
FROM sample_stats.sample_total_stats
WHERE sample_class = 'malicious'
ORDER BY file_kind;
```

查看三列 VT 相关字段为 1 的数量：

```sql
SELECT
  file_kind,
  total_samples,
  has_vt_1,
  has_vt_summary_1,
  has_vt_mitre_1
FROM sample_stats.sample_behavior_stats
WHERE sample_class = 'malicious'
ORDER BY file_kind;
```

查看恶意样本三类合计：

```sql
SELECT
  SUM(total_samples) AS malicious_total,
  SUM(has_vt_1) AS has_vt_1_total,
  SUM(has_vt_summary_1) AS has_vt_summary_1_total,
  SUM(has_vt_mitre_1) AS has_vt_mitre_1_total
FROM sample_stats.sample_behavior_stats
WHERE sample_class = 'malicious';
```

查看最近一次刷新结果：

```sql
SELECT *
FROM sample_stats.sample_stats_refresh_log
ORDER BY id DESC
LIMIT 1;
```

## 权限要求

执行脚本的 MySQL 用户至少需要：

```sql
CREATE, SELECT, INSERT, UPDATE, DELETE, DROP
```

其中 `DROP` 是为了 `TRUNCATE TABLE` 使用。如果不想给 `DROP`，可以把脚本里的 `TRUNCATE TABLE` 改成 `DELETE FROM`，但全量刷新会慢一些。
