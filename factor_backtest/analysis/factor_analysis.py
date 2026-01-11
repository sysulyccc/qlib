"""
Factor analysis utilities including IC calculation and decile analysis.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
import polars as pl


class FactorAnalyzer:
    """
    Factor analyzer for IC and decile analysis.
    """

    def __init__(self, n_quantiles: int = 10):
        self._n_quantiles = n_quantiles

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles

    def calculate_ic_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily IC series.
        
        Args:
            df: DataFrame with columns [date, symbol, factor, return_1d]
            
        Returns:
            DataFrame with columns [date, ic, rank_ic]
        """
        ic_list = []

        for date, group in df.groupby("date"):
            if len(group) < 10:
                continue

            ic = group["factor"].corr(group["return_1d"])
            rank_ic = group["factor"].corr(group["return_1d"], method="spearman")

            ic_list.append({
                "date": date,
                "ic": ic,
                "rank_ic": rank_ic,
            })

        return pd.DataFrame(ic_list)

    def calculate_factor_stats(self, factor_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate factor statistics."""
        factor_vals = factor_df["factor"].dropna()

        return {
            "total_rows": len(factor_df),
            "valid_rows": len(factor_vals),
            "mean": factor_vals.mean(),
            "std": factor_vals.std(),
            "min": factor_vals.min(),
            "max": factor_vals.max(),
        }

    def calculate_ic_stats(self, ic_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate IC statistics."""
        ic_series = ic_df["rank_ic"].dropna()

        return {
            "mean_ic": ic_series.mean(),
            "icir": ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            "ic_gt0_ratio": (ic_series > 0).mean(),
            "ic_count": len(ic_series),
        }

    def calculate_decile_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate decile (quantile) group analysis."""
        df_pl = pl.from_pandas(df)

        df_clean = df_pl.drop_nulls(["factor", "return_1d"])

        if df_clean.height == 0:
            return {"error": "No valid data for decile analysis"}

        df_with_avg = df_clean.with_columns([
            pl.col("return_1d").mean().over(["date"]).alias("avg_return_by_date")
        ])

        xs_returns = (
            df_with_avg.group_by(["date"]).agg([
                pl.col("avg_return_by_date").first().alias("xs_ret"),
                pl.col("return_1d").count().alias("xs_size"),
            ])
            .sort(["date"])
        )

        df_with_decile = df_with_avg.with_columns([
            pl.col("factor")
            .rank(method='random')
            .qcut(self._n_quantiles, labels=[str(i+1) for i in range(self._n_quantiles)], allow_duplicates=True)
            .over(["date"])
            .alias("decile")
        ])

        decile_returns = df_with_decile.group_by(["date", "decile"]).agg([
            (pl.col("return_1d") - pl.col("avg_return_by_date")).mean().alias("mean_ex_return"),
            pl.col("return_1d").mean().alias("mean_return"),
            pl.col("return_1d").count().alias("count")
        ]).sort(["date", "decile"])

        decile_pnl = decile_returns.with_columns([
            pl.col("mean_ex_return").cum_sum().over("decile").alias("cumulative_pnl")
        ])

        sharpe_stats = decile_returns.group_by("decile").agg([
            (pl.col("mean_ex_return").mean() / pl.col("mean_ex_return").std() * np.sqrt(252)).alias("sharpe"),
            pl.col("mean_ex_return").mean().alias("avg_return"),
            pl.col("mean_ex_return").std().alias("return_std"),
            pl.col("count").sum().alias("total_count"),
        ]).sort("decile")

        long_decile = str(self._n_quantiles)
        short_decile = "1"

        long_returns = decile_returns.filter(pl.col("decile") == long_decile)["mean_ex_return"]
        short_returns = decile_returns.filter(pl.col("decile") == short_decile)["mean_ex_return"]

        def safe_sharpe(returns: pl.Series) -> float:
            if returns.len() == 0:
                return 0.0
            mean_ret = float(returns.mean())
            std_ret = float(returns.std())
            if std_ret <= 0:
                return 0.0
            return mean_ret / std_ret * np.sqrt(252)

        long_sharpe = safe_sharpe(long_returns)

        return {
            "decile_returns": decile_returns.to_pandas(),
            "decile_pnl": decile_pnl.to_pandas(),
            "decile_stats": sharpe_stats.to_pandas(),
            "xs_returns": xs_returns.to_pandas(),
            "long_sharpe": long_sharpe,
        }

    def calculate_monthly_ic(self, ic_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly IC heatmap data."""
        ic_df = ic_df.copy()
        ic_df["year"] = ic_df["date"].dt.year
        ic_df["month"] = ic_df["date"].dt.month

        monthly_ic = ic_df.groupby(["year", "month"])["rank_ic"].mean().reset_index()
        monthly_pivot = monthly_ic.pivot(index="year", columns="month", values="rank_ic")

        return monthly_pivot
