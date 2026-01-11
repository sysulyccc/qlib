# Factor Backtest

一个独立的因子回测框架，与 qlib 完全解耦，支持快速进行单因子分析和 TopK 策略回测。

## 功能特点

- **完全解耦**: 不依赖 qlib 运行时，只需要预先提取的市场数据（parquet 格式）
- **OOP 架构**: 模块化设计，包含 Position、Exchange、Account、Strategy 等核心组件
- **流式回测**: 逐日流式执行，完整模拟真实交易流程
- **TopK 策略**: 实现 TopkDropoutStrategy，支持持仓轮换和涨跌停检查
- **完整分析**: 
  - 因子统计信息（均值、标准差、分布等）
  - IC/ICIR 分析（日度、月度热力图）
  - 十分组分析（超额收益、累积 PnL、Sharpe）
- **PDF 报告**: 自动生成美观的 PDF 报告

## 目录结构

```
factor_backtest/
├── core/                          # 核心回测组件
│   ├── __init__.py
│   ├── position.py                # Position 类 - 持仓管理
│   ├── exchange.py                # Exchange 类 - 市场数据和交易规则
│   ├── account.py                 # Account 类 - 账户和订单执行
│   ├── strategy.py                # TopkDropoutStrategy 类 - 交易策略
│   └── backtest.py                # BacktestEngine 类 - 回测引擎
├── analysis/                      # 因子分析模块
│   ├── __init__.py
│   └── factor_analysis.py         # IC/分组分析
├── report/                        # 报告生成模块
│   ├── __init__.py
│   └── generator.py               # PDF 报告生成
├── data/                          # 市场数据目录
│   ├── csi500_stock_pool.txt      # 中证500股票池
│   ├── csi500_daily_price.parquet # 日频价格数据（含 change 字段）
│   ├── csi500_daily_returns.parquet # 日频收益率数据
│   └── csi500_benchmark.parquet   # 基准指数数据
├── output/                        # 输出目录（报告等）
├── scripts/                       # 辅助脚本
│   └── extract_qlib_data.py       # 从 qlib 提取数据的脚本
├── factor_backtester.py           # 主入口
├── requirements.txt               # 依赖包
└── README.md                      # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备市场数据

如果你有 qlib 环境，可以使用提供的脚本提取数据：

```bash
cd scripts
python extract_qlib_data.py --output_dir ../data --start_date 2015-01-01 --end_date 2024-12-31
```

或者手动准备以下文件：

- `data/csi500_stock_pool.txt`: 每行一个股票代码
- `data/csi500_daily_returns.parquet`: 包含 `[date, symbol, return_1d]` 列
- `data/csi500_benchmark.parquet`: 包含 `[date, symbol, close, return]` 列

### 3. 准备因子数据

因子数据格式要求：
- 文件格式: parquet
- 必须包含列: `date`, `symbol`
- 第三列为因子值（列名任意，会自动识别）

示例：
```
| date       | symbol    | my_factor |
|------------|-----------|-----------|
| 2020-01-02 | SH600000  | 0.523     |
| 2020-01-02 | SH600001  | -0.234    |
| ...        | ...       | ...       |
```

### 4. 运行回测

```bash
python factor_backtester.py /path/to/your/factor.parquet
```

完整参数：

```bash
python factor_backtester.py factor.parquet \
    --data_dir data \
    --output_dir output \
    --n_quantiles 10
```

### 5. 查看报告

报告将保存到 `output/` 目录，文件名格式为 `{factor_name}_backtest_report.pdf`。

## 报告内容

生成的 PDF 报告包含以下内容：

1. **Factor Statistics 表格**: 总行数、有效行数、均值、标准差、最小/最大值
2. **Global IC 表格**: Mean IC、ICIR、IC>0 比例、IC 数量、平均截面大小
3. **Factor Distribution**: 因子值分布直方图
4. **IC Time Series**: IC 时序图（20日滚动均值）
5. **Decile Excess Returns**: 十分组超额收益柱状图
6. **Monthly IC Heatmap**: 月度 IC 热力图
7. **Decile PnL Curves**: 十分组累积 PnL 曲线
8. **Cumulative IC**: 累积 IC 曲线

## 使用示例

### Python API

```python
from factor_backtester import FactorBacktester

# 初始化
backtester = FactorBacktester(
    data_dir="data",
    output_dir="output",
    n_quantiles=10,
)

# 运行因子分析
results = backtester.run_analysis("path/to/factor.parquet")

# 生成因子分析报告
report_path = backtester.generate_report(results)
print(f"Factor report saved to: {report_path}")

# 生成 TopK 策略回测报告
backtest_path = backtester.generate_backtest_report(
    results,
    topk=50,
    n_drop=5,
    open_cost=0.0005,
    close_cost=0.0015,
)
print(f"Backtest report saved to: {backtest_path}")

# 访问详细结果
print(f"Mean IC: {results['ic_stats']['mean_ic']:.4f}")
print(f"ICIR: {results['ic_stats']['icir']:.4f}")
print(f"Long Sharpe: {results['decile_analysis']['long_sharpe']:.4f}")
```

### 核心组件使用

```python
from core import BacktestEngine, TopkDropoutStrategy, Exchange, Account, Position

# 直接使用核心组件进行自定义回测
engine = BacktestEngine(
    close_series=close_s,
    factor_series=factor_s,
    change_series=change_s,
    benchmark_series=bench_s,
    signal_series=sig_s,
    trade_unit=100,
    limit_threshold=0.095,
)

strategy = TopkDropoutStrategy(topk=50, n_drop=5)
report_df = engine.run(strategy, start_time, end_time)
```

## 与 rolling_train.py 配合使用

`custom_exp/rolling_train.py` 生成的预测文件可以直接用于回测：

```bash
python factor_backtester.py ../custom_exp/model_pred/lightgbm_csi500_topk/full_pred.parquet
```

## 核心组件说明

| 组件 | 说明 |
|------|------|
| `Position` | 持仓管理，跟踪现金、股票数量、持仓天数等 |
| `Exchange` | 市场数据访问，涨跌停检查，交易数量取整 |
| `Account` | 账户管理，订单执行，指标计算 |
| `TopkDropoutStrategy` | TopK 轮换策略，生成买卖决策 |
| `BacktestEngine` | 回测引擎，流式执行每日交易 |
| `FactorAnalyzer` | IC/分组分析 |
| `ReportGenerator` | PDF 报告生成 |

## 回测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `topk` | 50 | 持有股票数量 |
| `n_drop` | 5 | 每日轮换数量 |
| `open_cost` | 0.0005 | 买入成本率 |
| `close_cost` | 0.0015 | 卖出成本率 |
| `min_cost` | 5.0 | 最低交易成本 |
| `trade_unit` | 100 | 交易单位（手） |
| `risk_degree` | 0.95 | 资金使用比例 |
| `hold_thresh` | 1 | 最短持仓天数 |
| `limit_threshold` | 0.095 | 涨跌停阈值 |
| `init_cash` | 1e8 | 初始资金 |

## 注意事项

1. 因子值和收益率会自动进行 inner join，只保留同时存在的数据
2. 十分组分析使用截面中性化（减去每日平均收益）
3. Sharpe Ratio 按年化计算（sqrt(252)）
4. 因子分布图会裁剪 1%/99% 分位数的极值
5. 涨跌停检查需要 price 数据中包含 `change` 字段
6. 回测结果与 qlib 原生 TopkDropoutStrategy 完全一致

## License

MIT
