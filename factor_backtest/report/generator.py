"""
Report generation utilities.
"""
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.io as pio

from qlib_plot import report_graph


class ReportGenerator:
    """
    Generate PDF reports for factor analysis and backtest results.
    """

    def __init__(self, output_dir: str = "output", n_quantiles: int = 10):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._n_quantiles = n_quantiles

    def generate_factor_report(
        self,
        results: Dict[str, Any],
        output_name: Optional[str] = None,
    ) -> str:
        """Generate PDF report with factor analysis results."""
        factor_name = results["factor_name"]
        output_name = output_name or f"{factor_name}_backtest_report.pdf"
        output_path = self._output_dir / output_name

        fig = plt.figure(figsize=(12, 16))
        gs = gridspec.GridSpec(4, 2, height_ratios=[0.8, 1.2, 1.2, 2.5], hspace=0.35, wspace=0.25)

        fig.suptitle(f"Factor Backtest Report: {factor_name}", fontsize=14, fontweight='bold', y=0.98)

        subtitle = f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    Backtest Time: {results['elapsed_time']:.2f}s"
        fig.text(0.5, 0.955, subtitle, ha='center', fontsize=10)

        ax_factor_stats = fig.add_subplot(gs[0, 0])
        self._plot_factor_stats_table(ax_factor_stats, results["factor_stats"])

        ax_ic_stats = fig.add_subplot(gs[0, 1])
        self._plot_ic_stats_table(ax_ic_stats, results["ic_stats"])

        ax_dist = fig.add_subplot(gs[1, 0])
        self._plot_factor_distribution(ax_dist, results["factor_df"])

        ax_ic_ts = fig.add_subplot(gs[1, 1])
        self._plot_ic_time_series(ax_ic_ts, results["ic_df"])

        ax_decile_bar = fig.add_subplot(gs[2, 0])
        self._plot_decile_returns(ax_decile_bar, results["decile_analysis"])

        ax_heatmap = fig.add_subplot(gs[2, 1])
        self._plot_monthly_ic_heatmap(ax_heatmap, results["monthly_ic"])

        ax_pnl = fig.add_subplot(gs[3, :])
        self._plot_decile_pnl_curves(ax_pnl, results["decile_analysis"])

        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(output_path)

    def generate_backtest_report(
        self,
        report_df: pd.DataFrame,
        factor_name: str,
        output_name: Optional[str] = None,
    ) -> str:
        """Generate qlib-style backtest report PDF."""
        output_name = output_name or f"{factor_name}_qlib_backtest.pdf"
        output_path = self._output_dir / output_name

        report_df = report_df.copy()
        report_df.index.name = "date"

        fig_list = report_graph(report_df)

        if fig_list and len(fig_list) > 0:
            pio.write_image(fig_list[0], str(output_path), format="pdf", width=1200, height=1400)

        return str(output_path)

    def _plot_factor_stats_table(self, ax, stats: Dict[str, float]):
        ax.axis('off')

        table_data = [
            ["Metric", "Value"],
            ["Total Rows", f"{stats['total_rows']:,}"],
            ["Valid Rows", f"{stats['valid_rows']:,}"],
            ["Mean", f"{stats['mean']:.4f}"],
            ["Std", f"{stats['std']:.4f}"],
            ["Min", f"{stats['min']:.4f}"],
            ["Max", f"{stats['max']:.4f}"],
        ]

        table = ax.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.5, 0.5],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

    def _plot_ic_stats_table(self, ax, stats: Dict[str, float]):
        ax.axis('off')

        mean_ic = stats['mean_ic']
        ic_color = '#00B050' if mean_ic > 0.03 else ('#FFC000' if mean_ic > 0 else '#FF0000')

        table_data = [
            ["Global IC", "Value"],
            ["Mean IC", f"{mean_ic:.4f}"],
            ["ICIR", f"{stats['icir']:.4f}"],
            ["IC>0 Ratio", f"{stats['ic_gt0_ratio']*100:.1f}%"],
            ["IC Count", f"{stats['ic_count']:,}"],
            ["Avg XS Size", f"{stats['avg_xs_size']:.1f}"],
        ]

        table = ax.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.5, 0.5],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        table[(1, 1)].set_text_props(color=ic_color, fontweight='bold')

    def _plot_factor_distribution(self, ax, factor_df: pd.DataFrame):
        factor_vals = factor_df["factor"].dropna()

        lower, upper = factor_vals.quantile([0.01, 0.99])
        factor_clipped = factor_vals.clip(lower, upper)

        ax.hist(factor_clipped, bins=50, color='#5B9BD5', edgecolor='white', alpha=0.8)
        ax.set_xlabel("Factor Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_title("Factor Distribution", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_ic_time_series(self, ax, ic_df: pd.DataFrame):
        ic_df = ic_df.copy()
        ic_df["cum_rank_ic"] = ic_df["rank_ic"].cumsum()

        ax.fill_between(ic_df["date"], 0, ic_df["cum_rank_ic"],
                       alpha=0.3, color='#4472C4')
        ax.plot(ic_df["date"], ic_df["cum_rank_ic"], color='#4472C4', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Cumulative RankIC", fontsize=9)
        ax.set_title("Cumulative RankIC", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_decile_returns(self, ax, decile_analysis: Dict[str, Any]):
        if "error" in decile_analysis:
            ax.text(0.5, 0.5, "Decile analysis unavailable", ha='center', va='center')
            return

        stats = decile_analysis["decile_stats"]

        colors = ['#FF6B6B' if i in [0, self._n_quantiles-1] else '#5B9BD5'
                  for i in range(self._n_quantiles)]

        ax.bar(range(1, self._n_quantiles + 1), stats["avg_return"] * 10000,
               color=colors, edgecolor='white', alpha=0.8)

        ax.set_xlabel("Decile", fontsize=9)
        ax.set_ylabel("Avg Excess Return (bps)", fontsize=9)
        ax.set_title("Factor Decile Excess Returns", fontsize=10, fontweight='bold')
        ax.set_xticks(range(1, self._n_quantiles + 1))
        ax.set_xticklabels([f"D{i}" for i in range(1, self._n_quantiles + 1)], fontsize=8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_monthly_ic_heatmap(self, ax, monthly_ic: pd.DataFrame):
        sns.heatmap(
            monthly_ic,
            ax=ax,
            cmap='RdYlBu_r',
            center=0,
            annot=False,
            fmt='.2f',
            cbar_kws={'label': 'RankIC', 'shrink': 0.8}
        )
        ax.set_xlabel("Month", fontsize=9)
        ax.set_ylabel("Year", fontsize=9)
        ax.set_title("Monthly RankIC Heatmap", fontsize=10, fontweight='bold')

    def _plot_decile_pnl_curves(self, ax, decile_analysis: Dict[str, Any]):
        if "error" in decile_analysis:
            ax.text(0.5, 0.5, "Decile analysis unavailable", ha='center', va='center')
            return

        pnl_df = decile_analysis["decile_pnl"]

        colors = plt.cm.RdYlGn(np.linspace(0, 1, self._n_quantiles))

        for i in range(self._n_quantiles):
            decile = str(i + 1)
            data = pnl_df[pnl_df["decile"] == decile].copy()
            if len(data) > 0:
                lw = 2.5 if i == self._n_quantiles - 1 else 1.0
                ax.plot(data["date"], data["cumulative_pnl"] * 100,
                       color=colors[i], linewidth=lw, label=f"D{decile}")

        long_sharpe = decile_analysis["long_sharpe"]
        ax.text(0.02, 0.98, f"D{self._n_quantiles} Sharpe: {long_sharpe:.3f}",
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Cumulative Excess PnL (%)", fontsize=9)
        ax.set_title("Decile Excess PnL Curves (vs Cross-Sectional Mean)", fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', ncol=5, fontsize=7)
        ax.grid(True, alpha=0.3)
