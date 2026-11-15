# Mini SQL — 单机列存批向量化 SQL 查询引擎

一个为「几十 GB 临时 CSV 查询」设计的单机迷你 SQL 引擎。**只用 Python 标准库 +
numpy**，不依赖任何现成数据库或数据框（pandas / polars / sqlite / duckdb 均不被
引擎本身导入；duckdb 仅在基准测试里作为独立的造数器与对照库使用）。

配套一个零前端构建的网页查询台，以及一套以 TPC-H 为基准、与成熟数据库逐行比对的
验证框架。

---

## 1. 它能做什么

### SQL 能力

| 类别 | 支持内容 |
| --- | --- |
| 查询结构 | `SELECT`（含表达式/别名/`*`）、`WHERE`、`GROUP BY`、`HAVING`、`ORDER BY`（ASC/DESC、NULLS FIRST/LAST）、`LIMIT`/`OFFSET` |
| 连接 | 内连接、左外连接（多表、等值连接，支持 AND 复合键） |
| 聚合 | `COUNT(*)`、`COUNT(col)`、`SUM`、`AVG`、`MIN`、`MAX`（均正确忽略 NULL） |
| 表达式 | `+ - * /`、字符串拼接 `||`、比较、`AND/OR/NOT`、`LIKE`、`IN (...)`、`BETWEEN`、`CASE WHEN`、`COALESCE`、`SUBSTRING`、`EXTRACT(YEAR/MONTH/DAY)`、`UPPER/LOWER/LENGTH/TRIM`、`DATE '...'` 字面量 |
| 注释 | `--` 行注释与 `/* ... */` 块注释 |

不支持子查询、右/全外连接、`DISTINCT` 聚合与窗口函数（TPC-H 中依赖这些的查询在
报告里如实标注为「引擎可表达子集之外」）。

### 引擎设计要点（对应需求逐条落地）

1. **按列批向量化执行** — 数据以 `Batch`（一批等长的 `Col` 列）在算子间拉取式流动，
   默认每批最多 65,536 行，表达式在 numpy 数组上整批求值，绝不逐行解释。
2. **谓词下推 + 投影下推** — 单表谓词下推到扫描（外连接可空侧的谓词不会错误下推），
   扫描只解析上游真正需要的列。`EXPLAIN` 与网页计划树里能看到每个算子的
   「进入行数 / 出行数 / 批次数」。
3. **超内存溢写** — 排序（外部排序 + n 路归并）、哈希聚合（分区部分状态落盘后归并）、
   哈希连接（Grace 分区连接）在驻留内存超过预算阈值（默认预算的 75%）时自动溢写到
   临时目录，再归并；所有聚合状态都满足结合律，部分状态可精确合并。
4. **CSV 并行分块解析** — 文件按字节切分并对齐换行边界，进程池并行解析；列类型靠
   采样多数投票推断；脏行（字段数不符或违反列类型）**整行原子跳过并计数**，校验覆盖
   所有列而非仅投影列。
5. **NULL 三值逻辑** — TRUE/FALSE/UNKNOWN 的 Kleene 逻辑，`NOT(UNKNOWN)=UNKNOWN`，
   连接键为 NULL 永不匹配，左连接未匹配行补 NULL。
6. **字符串与整数语义** — 字符串按码点比较与排序；整数算术按 bigint 做溢出检查，
   越界（含一元负号、字面量超界）抛 SQL 标准的数据异常；除零抛异常而非产生 inf。
7. **可取消 + 临时文件清理** — 每个查询一个 `CancelToken`，在批边界检查；临时目录在
   正常结束、出错、取消三种路径下都确定性删除，启动时还会清扫崩溃残留。
8. **字符串字典编码** — 低基数文本列（日期、标志位、状态码、名称）在读取侧去重，
   数据列只存 int32 编码，显著降低内存占用并加速分组/比较。

---

## 2. 目录结构

```
mini_sql/
  types.py        列类型、可空列(含字典编码)、三值逻辑、整数溢出检查
  csv_reader.py   并行分块 CSV 解析、采样猜类型、脏行计数
  parser.py       SQL tokenizer + 递归下降解析器
  expr.py         向量化表达式求值（三值逻辑、函数、类型转换）
  plan.py         逻辑计划、名称绑定、谓词/投影下推
  operators.py    扫描/过滤/哈希连接/外部排序/投影/Limit 物理算子
  aggregate.py    哈希聚合（含分区溢写与状态合并）
  spill.py        内存核算 MemTracker、临时文件 TempManager、溢写读写
  engine.py       引擎主体：目录、计划装配、执行驱动、统计
web/
  app.py          标准库 http.server 异步查询服务（作业/取消/下载）
  static_site.py  单页查询台（五个功能区）
bench/
  generate_data.py  DuckDB TPC-H 造数 + CSV 导出 + 脏行注入
  run_benchmark.py  双引擎逐行比对框架
queries/          引擎可表达的 9 条 TPC-H 查询（同一 SQL 两边跑）
tests/            5 套测试 + 统一入口 run_all.py
run_web.py        网页台启动脚本
```

---

## 3. 快速开始

```powershell
# 依赖：Python 3.11+、numpy；对照/造数另需 duckdb（引擎本身不需要）
py -m pip install numpy duckdb

# 跑全部测试（无需联网、无需造数）
py -I tests/run_all.py

# 启动网页查询台
py -I run_web.py --memory-mb 2048 --parallel 4 --port 8765
# 也可启动时直接挂载
py -I run_web.py --mount orders=D:/data/orders.csv --mount customer=D:/data/customer.csv
```

浏览器打开 `http://127.0.0.1:8765/`，页面五个功能区：

1. **数据源挂载** — 填表名、CSV 路径、分隔符，挂载后展示推断出的列名/类型；
2. **SQL 编辑** — 写 SQL、设返回行数上限、执行 / 取消 / 查看计划；
3. **执行计划** — 逻辑计划文本（标注下推）+ 每个算子的进出行数树；
4. **算子耗时** — 各算子耗时条、内存峰值/预算、溢写读写量；
5. **结果下载** — 表格展示（NULL 显式标注），一键导出 CSV。

### 作为库使用

```python
from mini_sql import Engine

eng = Engine(memory_limit_bytes=2 * 1024**3, parallel=4)
eng.mount("orders", "D:/data/orders.csv")
eng.mount("customer", "D:/data/customer.csv")

result = eng.run("""
    SELECT c.c_name, COUNT(*) AS n
    FROM customer AS c
    JOIN orders AS o ON c.c_custkey = o.o_custkey
    WHERE o.o_orderdate >= DATE '1995-01-01'
    GROUP BY c.c_name
    ORDER BY n DESC
    LIMIT 20
""")
print(result.columns)
for row in result.rows:
    print(row)
print(result.stats.plan)          # 每算子进出行数
print(result.stats.memory)        # 内存预算/峰值/阈值
```

---

## 4. TPC-H 对照验证

### 造数（DuckDB 仅充当生成器与对照库）

```powershell
# 小规模冒烟（约 10MB CSV）
py -I bench/generate_data.py --sf 0.01 --out data/tpch_sf001

# 正式 10GB：scale factor 10
py -I bench/generate_data.py --sf 10 --out data/tpch_sf10

# 可选：注入 ~0.1% 脏行，验证跳过与计数
py -I bench/generate_data.py --sf 10 --out data/tpch_sf10_dirty --dirty 0.001
```

### 双引擎逐行比对

```powershell
# 内存上限 2GB、4 个解析进程；同一份 SQL 在 DuckDB 与本引擎各跑一遍
py -I bench/run_benchmark.py --sf 10 --data-dir data/tpch_sf10 `
    --memory-mb 2048 --parallel 4 --out reports/sf10_2gb.json

# 强制溢写：把预算压到 64MB，验证磁盘路径结果不变
py -I bench/run_benchmark.py --sf 10 --data-dir data/tpch_sf10 `
    --memory-mb 64 --parallel 4 --out reports/sf10_spill64mb.json
```

比对规则：整数/字符串/日期/NULL **精确相等**；浮点（DECIMAL 以 float64 承载）按
相对容差 1e-6 + 绝对容差 1e-2 比较。所有支持的查询都带确定性 `ORDER BY`，因此按位置
逐行对齐。输出 JSON 报告含每条查询的状态、行数、两边耗时、内存峰值、溢写字节。

### 可表达的 TPC-H 子集

22 条查询中，**9 条无需子查询即可完整表达并已纳入逐行比对**：

| 查询 | 主题 | 覆盖的能力 |
| --- | --- | --- |
| Q1 | 定价汇总 | 单表分组、8 个聚合、日期谓词 |
| Q3 | 运输优先级 | 3 表内连接、分组聚合、ORDER BY 别名、LIMIT |
| Q5 | 本地供应商额 | 6 表连接、分组聚合 |
| Q6 | 预测收入变化 | 标量聚合、BETWEEN、复合谓词 |
| Q7 | 货运量 | 多表连接、`EXTRACT(YEAR)`、OR/AND |
| Q9 | 产品利润 | 多表连接、`LIKE`、复合算术、双分组键 |
| Q10 | 退货报告 | 4 表连接、多列分组、LIMIT |
| Q12 | 运输方式 | 连接 + `CASE` 条件聚合 |
| Q19 | 折扣收入 | 单表多条件、IN、BETWEEN、标量聚合 |

其余 13 条（Q2、Q4、Q8、Q11、Q13–Q18、Q20–Q22）依赖相关子查询、`EXISTS/NOT EXISTS`、
`IN (子查询)` 或 `UNION`，超出当前解析器范围，框架会以 `unsupported` 如实记录而非
冒充通过。这是诚实的能力边界，而不是静默跳过。

### 当前验证结果（本仓库内已执行）

| 数据集 | 内存预算 | 结果 |
| --- | --- | --- |
| sf=0.01（~10MB） | 256MB | 9/9 与 DuckDB 逐行一致 |
| sf=0.1（~105MB） | 2GB | 9/9 一致 |
| sf=0.1（~105MB） | 64MB（强制溢写） | 9/9 一致，Q7/Q9/Q12 触发溢写 |
| sf=0.01 + 脏行 | 256MB | 173 个脏行全部跳过计数，干净行结果一致 |

> 在 Windows 上使用并行解析（`--parallel > 1`）时，入口脚本需保留
> `if __name__ == "__main__":` 保护（spawn 启动方式要求），仓库内脚本均已遵守。

---

## 5. 内存与溢写说明

- `MemTracker` 核算引擎**显式持有**的物化数据：批次列（含字典编码的字符串净荷）、
  哈希表、分组状态、排序缓冲区、溢写分区。阈值默认取预算的 75%。
- 核算不重复计入 Python/numpy 解释器内部对象与进程池 worker 的瞬时反序列化开销，
  因此操作系统层面的 RSS 会高于自核算峰值——这是核算边界，不是泄漏；查询结束后
  临时目录与句柄都会释放。
- 字典编码让低基数字符串列的引擎驻留占用接近同宽整数列；批次行数会根据预算与投影
  列宽自适应，避免单个宽批次就击穿预算。
- 溢写文件为引擎私有 `.npz`（字符串/字典数组需要 `allow_pickle`，仅读取自身写出的
  可信临时文件），位于每查询独立的临时根目录下。

---

## 6. 运行测试

```powershell
py -I tests/run_all.py        # 一次跑全部 5 套
# 或单独跑
py -I tests/test_smoke.py     # 基础 SQL/三值逻辑/取消/清理
py -I tests/test_edge.py      # 外连接 NULL、字符串序、空关系、整数溢出
py -I tests/test_spill.py     # 外部排序/聚合/连接溢写 + 并行解析
py -I tests/test_web.py       # 网页台端到端 HTTP
py -I tests/test_dirty.py     # TPC-H 脏行跳过与计数
```
