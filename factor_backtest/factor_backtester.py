"""
Factor Backtester - A standalone factor analysis and backtesting framework.
Completely decoupled from qlib, only requires parquet files with factor values.
"""
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core import BacktestEngine, TopkDropoutStrategy
from analysis import FactorAnalyzer
from report import ReportGenerator


class FactorBacktester:
    """
    Factor backtesting framework.
    
    Input: parquet file with columns [date, symbol, factor_value]
    Output: comprehensive factor analysis report (PDF)
    """

    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "output",
        n_quantiles: int = 10,
        benchmark_file: str = "csi500_benchmark.parquet",
        price_file: str = "csi500_daily_price.parquet",
        returns_file: str = "csi500_daily_returns.parquet",
        stock_pool_file: str = "csi500_stock_pool.txt",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_quantiles = n_quantiles

        self.benchmark_df = self._load_benchmark(benchmark_file)
        self.price_df = self._load_price(price_file)
        self.returns_df = self._load_returns(returns_file)
        self.stock_pool = self._load_stock_pool(stock_pool_file)

        self._analyzer = FactorAnalyzer(n_quantiles=n_quantiles)
        self._report_gen = ReportGenerator(output_dir=output_dir, n_quantiles=n_quantiles)

        logger.info(f"Loaded benchmark: {len(self.benchmark_df)} rows")
        logger.info(f"Loaded price: {len(self.price_df)} rows")
        logger.info(f"Loaded returns: {len(self.returns_df)} rows")
        logger.info(f"Loaded stock pool: {len(self.stock_pool)} stocks")

    def _load_benchmark(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_price(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Price file not found: {path}")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_returns(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Returns file not found: {path}")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_stock_pool(self, filename: str) -> List[str]:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Stock pool file not found: {path}")
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def load_factor(self, factor_path: str) -> Tuple[pd.DataFrame, str]:
        """
        Load factor data from parquet file.
        Expects columns: [date, symbol, <factor_column>]
        Returns: (factor_df, factor_name)
        """
        path = Path(factor_path)
        if not path.exists():
            raise FileNotFoundError(f"Factor file not found: {path}")

        df = pd.read_parquet(path)

        required_cols = {"date", "symbol"}
        existing_cols = set(df.columns)

        if not required_cols.issubset(existing_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}, Got: {existing_cols}")

        factor_cols = [c for c in df.columns if c not in required_cols]
        if len(factor_cols) == 0:
            raise ValueError("No factor column found")

        factor_name = factor_cols[0]
        logger.info(f"Detected factor column: {factor_name}")

        df = df[["date", "symbol", factor_name]].copy()
        df.columns = ["date", "symbol", "factor"]
        df["date"] = pd.to_datetime(df["date"])

        return df, factor_name

    def _merge_factor_with_returns(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(
            factor_df,
            self.returns_df[["date", "symbol", "return_1d"]],
            on=["date", "symbol"],
            how="inner"
        )
        return merged.dropna(subset=["factor", "return_1d"])

    def run_backtest(
        self,
        factor_df: pd.DataFrame,
        topk: int = 50,
        n_drop: int = 5,
        open_cost: float = 0.0005,
        close_cost: float = 0.0015,
        min_cost: float = 5.0,
        trade_unit: int = 100,
        risk_degree: float = 0.95,
        hold_thresh: int = 1,
        limit_threshold: float = 0.095,
        init_cash: float = 1e8,
    ) -> pd.DataFrame:
        """
        Run TopK dropout backtest.
        
        Returns: DataFrame with columns [return, cost, bench, turnover]
        """
        sig = factor_df.loc[:, ["date", "symbol", "factor"]].copy()
        sig["date"] = pd.to_datetime(sig["date"])

        price_cols = ["date", "symbol", "close", "adj_factor"]
        if "change" in self.price_df.columns:
            price_cols.append("change")
        price = self.price_df.loc[:, price_cols].copy()
        price["date"] = pd.to_datetime(price["date"])
        price = price.drop_duplicates(subset=["date", "symbol"], keep="last")

        close_s = price.set_index(["date", "symbol"])["close"]
        factor_s = price.set_index(["date", "symbol"])["adj_factor"]
        change_s = price.set_index(["date", "symbol"])["change"] if "change" in price.columns else None

        bench = self.benchmark_df.loc[:, ["date", "return"]].copy()
        bench["date"] = pd.to_datetime(bench["date"])
        bench_s = bench.set_index("date")["return"].sort_index().fillna(0.0)

        start_time = sig["date"].min()
        end_time = sig["date"].max()

        price_in_range = price[(price["date"] >= start_time) & (price["date"] <= end_time)]
        trade_w_adj_price = (price_in_range["adj_factor"].isna() & price_in_range["close"].notna()).any()
        trade_unit_effective = None if trade_w_adj_price else trade_unit

        sig_s = sig.set_index(["date", "symbol"])["factor"].sort_index()

        engine = BacktestEngine(
            close_series=close_s,
            factor_series=factor_s,
            change_series=change_s,
            benchmark_series=bench_s,
            signal_series=sig_s,
            trade_unit=trade_unit_effective,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            init_cash=init_cash,
        )

        strategy = TopkDropoutStrategy(
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            risk_degree=risk_degree,
        )

        return engine.run(strategy, start_time, end_time)

    def run_analysis(self, factor_path: str) -> Dict[str, Any]:
        """
        Run complete factor analysis.
        
        Returns:
            Dictionary with all analysis results
        """
        start_time = datetime.now()
        logger.info(f"Starting factor analysis: {factor_path}")

        factor_df, factor_name = self.load_factor(factor_path)
        logger.info(f"Loaded factor data: {len(factor_df)} rows")

        merged_df = self._merge_factor_with_returns(factor_df)
        logger.info(f"Merged with returns: {len(merged_df)} rows")

        factor_stats = self._analyzer.calculate_factor_stats(factor_df)
        logger.info("Factor stats calculated")

        ic_df = self._analyzer.calculate_ic_series(merged_df)
        ic_stats = self._analyzer.calculate_ic_stats(ic_df)
        ic_stats["avg_xs_size"] = merged_df.groupby("date").size().mean()
        logger.info(f"IC stats: Mean IC={ic_stats['mean_ic']:.4f}, ICIR={ic_stats['icir']:.4f}")

        decile_analysis = self._analyzer.calculate_decile_analysis(merged_df)
        if "error" not in decile_analysis:
            logger.info(f"Decile analysis: Long (D{self.n_quantiles}) Sharpe={decile_analysis['long_sharpe']:.4f}")

        monthly_ic = self._analyzer.calculate_monthly_ic(ic_df)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {elapsed:.2f}s")

        return {
            "factor_name": factor_name,
            "factor_path": factor_path,
            "factor_df": factor_df,
            "merged_df": merged_df,
            "factor_stats": factor_stats,
            "ic_df": ic_df,
            "ic_stats": ic_stats,
            "decile_analysis": decile_analysis,
            "monthly_ic": monthly_ic,
            "elapsed_time": elapsed,
        }

    def generate_report(
        self,
        results: Dict[str, Any],
        output_name: Optional[str] = None,
    ) -> str:
        """Generate factor analysis report PDF."""
        path = self._report_gen.generate_factor_report(results, output_name)
        logger.info(f"Factor report saved to {path}")
        return path

    def generate_backtest_report(
        self,
        results: Dict[str, Any],
        topk: int = 50,
        n_drop: int = 5,
        open_cost: float = 0.0005,
        close_cost: float = 0.0015,
        output_name: Optional[str] = None,
    ) -> str:
        """Generate backtest report PDF."""
        factor_df = results["factor_df"]
        factor_name = results["factor_name"]

        report_df = self.run_backtest(
            factor_df=factor_df,
            topk=topk,
            n_drop=n_drop,
            open_cost=open_cost,
            close_cost=close_cost,
        )

        path = self._report_gen.generate_backtest_report(report_df, factor_name, output_name)
        logger.info(f"Backtest report saved to {path}")
        return path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Factor Backtester")
    parser.add_argument("factor_file", type=str, help="Path to factor parquet file")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--n_quantiles", type=int, default=10, help="Number of quantile groups")
    parser.add_argument("--topk", type=int, default=50, help="TopK for backtest")
    parser.add_argument("--n_drop", type=int, default=5, help="N drop for TopkDropoutStrategy")
    parser.add_argument("--open_cost", type=float, default=0.0005, help="Open cost ratio")
    parser.add_argument("--close_cost", type=float, default=0.0015, help="Close cost ratio")

    args = parser.parse_args()

    backtester = FactorBacktester(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_quantiles=args.n_quantiles,
    )

    results = backtester.run_analysis(args.factor_file)

    report_path1 = backtester.generate_report(results)
    print(f"Factor analysis report saved to: {report_path1}")

    report_path2 = backtester.generate_backtest_report(
        results,
        topk=args.topk,
        n_drop=args.n_drop,
        open_cost=args.open_cost,
        close_cost=args.close_cost,
    )
    print(f"Backtest report saved to: {report_path2}")


if __name__ == "__main__":
    main()
