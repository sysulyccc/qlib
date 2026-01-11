"""
Backtest engine for running streaming backtest.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger

from .position import Position
from .exchange import Exchange
from .account import Account
from .strategy import TopkDropoutStrategy


class BacktestEngine:
    """
    Backtest engine runs the main backtest loop.
    Streams through trade dates and executes strategy decisions.
    """

    def __init__(
        self,
        close_series: pd.Series,
        factor_series: pd.Series,
        change_series: Optional[pd.Series],
        benchmark_series: pd.Series,
        signal_series: pd.Series,
        trade_unit: int = 100,
        limit_threshold: float = 0.095,
        open_cost: float = 0.0005,
        close_cost: float = 0.0015,
        min_cost: float = 5.0,
        init_cash: float = 1e8,
    ):
        self._close_s = close_series
        self._factor_s = factor_series
        self._change_s = change_series
        self._bench_s = benchmark_series
        self._signal_s = signal_series

        self._exchange = Exchange(
            close_series=close_series,
            factor_series=factor_series,
            change_series=change_series,
            trade_unit=trade_unit,
            limit_threshold=limit_threshold,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )

        self._account = Account(
            exchange=self._exchange,
            init_cash=init_cash,
        )

        self._trade_unit = trade_unit
        self._init_cash = init_cash

    def run(
        self,
        strategy: TopkDropoutStrategy,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Run backtest from start_time to end_time.
        
        Returns: DataFrame with columns [return, cost, bench, turnover]
        """
        full_calendar = self._bench_s.index
        trade_dates = [d for d in full_calendar if start_time <= d <= end_time]

        if len(trade_dates) == 0:
            raise ValueError("No trade dates in the given range")

        results: List[Dict[str, Any]] = []

        for i, trade_date in enumerate(trade_dates):
            pred_date = self._get_pred_date(trade_date, full_calendar)
            pred_score = self._get_pred_score(pred_date)

            if pred_score is None or len(pred_score) == 0:
                self._account.update_bar_end(trade_date)
                bench_return = float(self._bench_s.get(trade_date, 0.0))
                metrics = self._account.calculate_metrics(trade_date, bench_return)
                results.append(metrics)
                continue

            sell_list, buy_list = strategy.generate_trade_decision(
                pred_score=pred_score,
                account=self._account,
                exchange=self._exchange,
                trade_date=trade_date,
            )

            strategy.execute_trades(
                sell_list=sell_list,
                buy_list=buy_list,
                account=self._account,
                exchange=self._exchange,
                trade_date=trade_date,
            )

            self._account.update_bar_end(trade_date)

            bench_return = float(self._bench_s.get(trade_date, 0.0))
            metrics = self._account.calculate_metrics(trade_date, bench_return)
            results.append(metrics)

        report_df = pd.DataFrame(results).set_index("date")
        report_df.index = pd.to_datetime(report_df.index)
        report_df = report_df.sort_index()
        return report_df.loc[:, ["return", "cost", "bench", "turnover"]]

    def _get_pred_date(self, trade_date: pd.Timestamp, calendar: pd.Index) -> Optional[pd.Timestamp]:
        pred_idx = calendar.get_indexer([trade_date])[0] - 1
        if pred_idx >= 0:
            return calendar[pred_idx]
        return None

    def _get_pred_score(self, pred_date: Optional[pd.Timestamp]) -> Optional[pd.Series]:
        if pred_date is None:
            return None
        try:
            return self._signal_s.loc[pred_date]
        except KeyError:
            return None
