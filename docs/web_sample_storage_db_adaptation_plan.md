# Web 样本存储与数据库拆分适配计划

## 背景

当前服务器样本存储已从原来的恶意/白样本单独目录，调整为统一 `data/samples` 根目录下按样本性质和文件类型拆分：

- `data/samples/malicious/elf/{0-f}/{0-f}/{0-f}/{0-f}/{0-f}/{sha256}`
- `data/samples/malicious/pe/{0-f}/{0-f}/{0-f}/{0-f}/{0-f}/{sha256}`
- `data/samples/malicious/others/{0-f}/{0-f}/{0-f}/{0-f}/{0-f}/{sha256}`
- `data/samples/benign/elf/{0-f}/{0-f}/{0-f}/{0-f}/{0-f}/{sha256}`
- `data/samples/benign/pe/{0-f}/{0-f}/{0-f}/{0-f}/{0-f}/{sha256}`
- `data/samples/benign/others/{0-f}/{0-f}/{0-f}/{0-f}/{0-f}/{sha256}`

上传目录仍保持：

- `data/web_upload_file/{sha256}`

报告文件仍跟随样本目录保存，例如：

- `{sample_dir}/{sha256}.json`
- `{sample_dir}/{sha256}_behaviour_summary.json`

数据库也从单库拆成六个样本库：

- 恶意样本：`malicious_elf`、`malicious_pe`、`malicious_others`
- 白样本：`benign_elf`、`benign_pe`、`benign_others`

每个库仍为 `sample_00` 到 `sample_ff` 256 张分表，表名由 `sha256[:2]` 决定。

## 当前代码影响面

### 1. 配置项

当前 `web/fastapi/config.ini` 仍使用单库配置：

- `[mysql].db = malicious`
- `[mysql].db_benign = benign`
- `[paths].sample_root = ../../../sample-share/`
- `[paths].web_upload_dir = ../../data/web_upload_file`

需要改为相对路径和多库配置。服务器上建议：

```ini
[paths]
sample_root = ../../data/samples
web_upload_dir = ../../data/web_upload_file
zips_dir = ../../data/zips

[mysql]
malicious_dbs = malicious_elf,malicious_pe,malicious_others
benign_dbs = benign_elf,benign_pe,benign_others
db_stats = sample_stats
db_web = webdatadb
```

如果后续需要按文件类型精确映射，也可以使用：

```ini
malicious_db_elf = malicious_elf
malicious_db_pe = malicious_pe
malicious_db_others = malicious_others
benign_db_elf = benign_elf
benign_db_pe = benign_pe
benign_db_others = benign_others
```

### 2. 样本路径定位

当前受影响位置：

- `web/fastapi/app/api/detect.py`
  - `_load_sample_roots()`
  - `get_existing_sample_path()`
- `web/fastapi/app/api/query.py`
  - `_load_query_path_roots()`
  - `get_file_path_and_zip()`

当前逻辑假设样本路径是：

```text
{sample_root}/{sha256[0]}/{sha256[1]}/{sha256[2]}/{sha256[3]}/{sha256[4]}/{sha256}
```

新逻辑需要支持：

```text
{sample_root}/{malicious|benign}/{elf|pe|others}/{sha256[0]}/{sha256[1]}/{sha256[2]}/{sha256[3]}/{sha256[4]}/{sha256}
```

建议新增统一样本定位模块，例如：

- `web/fastapi/app/services/sample_locator.py`

职责：

- 从配置读取 `sample_root` 和 `web_upload_dir`。
- 根据数据库命中结果确定 `sample_class` 和 `file_kind`。
- 在必要时按六个候选目录探测文件。
- 始终先找正式样本目录，再找 `web_upload_file`。
- 返回 `sample_dir_path`、`sample_file_path`、`sample_class`、`file_kind`、`source_db`。

这样 `detect.py`、`query.py`、VT 报告保存、模型检测、下载压缩都共用同一套路径逻辑。

### 3. SHA256 查询与上传入库

当前核心类：

- `web/fastapi/app/utils/flask_mysql.py`

当前问题：

- `Databaseoperation.mysqlsha256()` 只查 `[mysql].db`，即原 `malicious`。
- `Databaseoperation.mysqlsha256s()` 只查原恶意库和 `webdatadb`。
- `Databaseoperation.update_db()` 只更新原恶意库和 `webdatadb`。
- 查询结果虽然用了 `DictCursor`，但部分 fallback 仍按旧恶意样本字段下标处理。
- 白样本表结构不同，不能用恶意样本的字段下标和详情字段完全套用。

计划改法：

- 新增配置读取：
  - `self.malicious_dbs`
  - `self.benign_dbs`
  - `self.sample_dbs = malicious_dbs + benign_dbs`
- 查询顺序建议：
  1. 六个正式样本库。
  2. `webdatadb` 上传库。
- 返回结果统一包装为内部结构：

```python
{
    "record": row,
    "source_db": "malicious_pe",
    "sample_class": "malicious",
    "file_kind": "pe",
    "is_upload": False
}
```

- 对前端输出再做字段归一化：
  - 恶意样本：`name`、`length`、`category`、`platform`、`family`、`kav_result`、`defender_result`、`filetype`、`packer`
  - 白样本：`software_name`、`software_type`、`file_name`、`length`、`file_path`、`filetype`、`platform`、`packer`、`os_versions`
- `update_db()` 应更新命中的正式库；如果不知道命中库，则按六个库加 `webdatadb` 逐个更新，命中后停止。
- 上传的新文件仍写入 `webdatadb` 和 `data/web_upload_file`，不自动归入六个正式样本库。

### 4. 前端详情展示兼容

当前接口在这些位置把查询结果转为前端详情：

- `web/fastapi/app/api/detect.py`
  - `upload_file()`
  - `detect_by_sha256()`
- `web/fastapi/app/api/query.py`
  - `query_sha256()`
  - `get_detail_common()`

需要增加白样本详情字段映射，避免白样本只显示空的 `类型/平台/家族`。

建议接口仍保持原字段，额外补充：

- `样本性质`: `恶意样本` / `白样本` / `上传样本`
- `文件类别`: `elf` / `pe` / `others`
- `软件名称`
- `软件类型`
- `原始文件名`
- `适用系统`
- `来源库`

前端如果目前只是遍历对象展示，可以无需大改；如果某些页面写死字段名，再做小范围兼容。

### 5. 下载、VT 和报告读取

当前 VT 逻辑：

- `web/fastapi/app/api/detect.py`
  - `get_detection_API()`
  - `get_behaviour_API()`
- `web/fastapi/app/services/external/api_vt.py`
  - 把报告写到传入的 `sample_dir_path`

`api_vt.py` 本身可以不改，关键是 `sample_dir_path` 必须由新的样本定位模块返回。

下载逻辑：

- `web/fastapi/app/api/query.py:get_file_path_and_zip()`

需要改为调用统一定位模块，不能再只找旧五级根目录。

## 统计与联动表计划

### 当前统计链路

首页样本库图表不是实时接口查询，而是由脚本生成静态前端数据：

- 生成脚本：`web/fastapi/app/services/data/vue_data_service.py`
- 输出文件：
  - `web/vue/src/data/chart_data.js`
  - `web/vue/src/data/stats_summary.txt`
- 前端读取：`web/vue/src/data/chart_data.js`

当前脚本假设：

- 一个恶意库：`[mysql].db`
- 一个白样本库：`[mysql].db_benign`
- 恶意统计表在恶意库：
  - `sample_yearly_stats`
  - `sample_monthly_stats`
  - `sample_category_stats`
  - `sample_platform_stats`
  - `sample_family_stats`
  - `sample_behavior_stats`
  - `sample_filetype_stats`
- 白样本总数来自白样本库：
  - `sample_counts`

数据库拆分后，这套脚本需要变成多库聚合。

### 已确认统计口径

- 白样本不做年度/月度趋势展示。
- 首页“近一年样本数”只统计恶意样本。
- `webdatadb` 上传库保持现状，本次不拆分。
- `category`、`family`、`platform` 三个反查库保持现状，本次不拆分。
- 恶意样本 `has_vt`、`has_vt_summary`、`has_vt_mitre` 三列需要统计值为 `1` 的数量，并按 `elf`、`pe`、`others` 拆分后再可汇总。

### 推荐联动表设计

建议不要让 Web 页面实时扫 6 个库 * 256 张原始表。应继续走“定时聚合表 + 前端生成数据”的方式。

已经新增脚本：

- `web/fastapi/app/services/data/sample_stats_refresh.py`

脚本会创建 `sample_stats` 统计库，并集中保存联动表。

```sql
CREATE TABLE sample_db_registry (
  id INT AUTO_INCREMENT PRIMARY KEY,
  sample_class VARCHAR(16) NOT NULL,
  file_kind VARCHAR(16) NOT NULL,
  db_name VARCHAR(64) NOT NULL,
  storage_subdir VARCHAR(64) NOT NULL,
  enabled TINYINT(1) NOT NULL DEFAULT 1,
  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_sample_db (sample_class, file_kind)
);
```

```sql
CREATE TABLE sample_total_stats (
  sample_class VARCHAR(16) NOT NULL,
  file_kind VARCHAR(16) NOT NULL,
  total_samples BIGINT NOT NULL DEFAULT 0,
  has_vt_count BIGINT NOT NULL DEFAULT 0,
  has_vt_summary_count BIGINT NOT NULL DEFAULT 0,
  has_vt_mitre_count BIGINT NOT NULL DEFAULT 0,
  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (sample_class, file_kind)
);
```

```sql
CREATE TABLE sample_yearly_stats (
  sample_class VARCHAR(16) NOT NULL,
  file_kind VARCHAR(16) NOT NULL,
  year INT NOT NULL,
  total_samples BIGINT NOT NULL DEFAULT 0,
  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (sample_class, file_kind, year)
);
```

月度表使用 `year`、`month` 两列，方便兼容现有前端数据生成脚本。

```sql
CREATE TABLE sample_category_stats (
  sample_class VARCHAR(16) NOT NULL,
  file_kind VARCHAR(16) NOT NULL,
  category VARCHAR(255) NOT NULL,
  count BIGINT NOT NULL DEFAULT 0,
  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (sample_class, file_kind, category)
);
```

```sql
CREATE TABLE sample_platform_stats (
  sample_class VARCHAR(16) NOT NULL,
  file_kind VARCHAR(16) NOT NULL,
  platform VARCHAR(255) NOT NULL,
  count BIGINT NOT NULL DEFAULT 0,
  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (sample_class, file_kind, platform)
);
```

```sql
CREATE TABLE sample_family_stats (
  sample_class VARCHAR(16) NOT NULL,
  file_kind VARCHAR(16) NOT NULL,
  family VARCHAR(255) NOT NULL,
  count BIGINT NOT NULL DEFAULT 0,
  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (sample_class, file_kind, family)
);
```

```sql
CREATE TABLE sample_filetype_stats (
  sample_class VARCHAR(16) NOT NULL,
  file_kind VARCHAR(16) NOT NULL,
  filetype VARCHAR(255) NOT NULL,
  count BIGINT NOT NULL DEFAULT 0,
  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (sample_class, file_kind, filetype)
);
```

```sql
CREATE TABLE sample_stats_refresh_log (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  started_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at TIMESTAMP NULL,
  status VARCHAR(16) NOT NULL,
  message TEXT NULL
);
```

### 年/月统计口径

恶意样本表有 `date` 字段，也新增了 `created_at`、`updated_at`。建议图表口径明确为：

- 样本发现/归档时间：优先用恶意表 `date`。
- 如果 `date` 缺失，再使用 `created_at`。
- 白样本本次不参与年度/月度趋势统计。

这样原来的“恶意样本年分布、近一年月分布”语义基本不变，同时能兼容新字段。

### 聚合脚本计划

已新增脚本：

- `web/fastapi/app/services/data/sample_stats_refresh.py`

职责：

- 读取六个样本库配置。
- 遍历每个库的 `sample_00` 到 `sample_ff`。
- 分库分类型聚合：
  - 总数
  - 年度
  - 月度
  - category
  - platform
  - family
  - filetype
  - has_vt / has_vt_summary / has_vt_mitre
- 写入 `sample_stats` 联动表，使用 `INSERT ... ON DUPLICATE KEY UPDATE`。
- 刷新前先清理本次涉及的维度，避免旧分类残留。
- 记录刷新日志。
- 第一次使用 `--mode full` 全量刷新。
- 后续可用 `--mode incremental`，脚本会用每张分表的 `COUNT(*)` 和 `MAX(updated_at)` 判断是否需要重算该分表。

`vue_data_service.py` 后续只读 `sample_stats`，不再直接读单个 `malicious` / `benign` 库的旧统计表。

### 前端图表兼容

短期保持 `chart_data.js` 结构不变：

- `summary.total_samples`
- `summary.benign_samples`
- `summary.malicious_samples`
- `summary.recent_year_samples`
- `lineChartData.total_amount`
- `lineChartData.year_amount`
- `pieTop10Data.category/platform/family`

这样 `web/vue/src/views/dashboard/sample` 基本不用改。

中期可以增加筛选维度：

- 全部 / 恶意 / 白样本
- 全部 / PE / ELF / Others

但这不是本次适配的第一优先级。

## 实施顺序

1. 配置改造
   - 增加六库配置。
   - `sample_root` 改为相对路径 `../../data/samples`。
   - 保留 `web_upload_dir` 不变。

2. 新增样本定位模块
   - 支持六个新目录。
   - 支持上传目录。
   - 替换 `detect.py` 和 `query.py` 中旧路径拼接逻辑。

3. 改造数据库查询层
   - `flask_mysql.py` 支持六个正式样本库。
   - 查询结果增加来源库和样本类型。
   - 统一恶意/白样本字段映射。
   - 更新 VT 状态时能更新命中库。

4. 改造详情接口
   - `query_sha256()`、`get_detail_common()`、`detect_by_sha256()` 支持白样本字段。
   - 前端保持兼容展示。

5. 改造下载、VT、模型检测链路
   - 全部调用统一样本定位模块。
   - 确认 VT 报告继续写入样本所在目录。

6. 建联动统计表
   - 建议新增 `sample_stats` 统计库。
   - 创建 registry、total、yearly、monthly、category、platform、family、filetype、refresh_log 表。

7. 聚合脚本改造
   - 从六个库聚合到 `sample_stats`。
   - `vue_data_service.py` 改为读取 `sample_stats`。
   - 继续输出原 `chart_data.js` 格式。

8. 服务器验证
   - 用一个已知恶意 PE 样本测试 SHA 查询、下载、VT 报告。
   - 用一个已知白样本测试 SHA 查询和下载。
   - 用一个上传样本测试 `web_upload_file` 路径仍可用。
   - 跑统计刷新，核对总数等于六个库分表总和。

## 已确认问题

1. 白样本不展示年度/月度趋势。
2. `webdatadb` 上传库保持当前结构，后续再说。
3. 首页“近一年样本数”只统计恶意样本。
4. `category`、`family`、`platform` 三个反查库先不拆分。
