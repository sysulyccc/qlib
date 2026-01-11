"""
Exchange class for market data access and trade execution logic.
"""
from typing import Optional
import numpy as np
import pandas as pd


class Exchange:
    """
    Exchange provides market data access and trade execution utilities.
    Handles price queries, tradability checks, and order rounding.
    """

    def __init__(
        self,
        close_series: pd.Series,
        factor_series: pd.Series,
        change_series: Optional[pd.Series] = None,
        trade_unit: int = 100,
        limit_threshold: float = 0.095,
        open_cost: float = 0.0005,
        close_cost: float = 0.0015,
        min_cost: float = 5.0,
    ):
        self._close_s = close_series
        self._factor_s = factor_series
        self._change_s = change_series
        self._trade_unit = trade_unit
        self._limit_threshold = limit_threshold
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    @property
    def open_cost(self) -> float:
        return self._open_cost

    @property
    def close_cost(self) -> float:
        return self._close_cost

    @property
    def min_cost(self) -> float:
        return self._min_cost

    def get_deal_price(self, stock_id: str, trade_date: pd.Timestamp) -> float:
        try:
            return float(self._close_s.loc[(trade_date, stock_id)])
        except KeyError:
            return np.nan

    def get_factor(self, stock_id: str, trade_date: pd.Timestamp) -> float:
        try:
            return float(self._factor_s.loc[(trade_date, stock_id)])
        except KeyError:
            return np.nan

    def check_stock_suspended(self, stock_id: str, trade_date: pd.Timestamp) -> bool:
        try:
            px = self._close_s.loc[(trade_date, stock_id)]
            return pd.isna(px) or float(px) <= 1e-12
        except KeyError:
            return True

    def check_stock_limit(
        self, stock_id: str, trade_date: pd.Timestamp, direction: Optional[int] = None
    ) -> bool:
        """
        Check if stock hits limit up/down.
        direction: 1=buy, 0=sell, None=check both (forbid_all_trade_at_limit)
        Returns True if stock is at limit (cannot trade in given direction)
        """
        if self._change_s is None or self._limit_threshold is None:
            return False
        try:
            change = self._change_s.loc[(trade_date, stock_id)]
            if pd.isna(change):
                return False
            limit_buy = change >= self._limit_threshold
            limit_sell = change <= -self._limit_threshold
            if direction == 1:
                return limit_buy
            elif direction == 0:
                return limit_sell
            else:
                return limit_buy or limit_sell
        except KeyError:
            return False

    def is_stock_tradable(
        self, stock_id: str, trade_date: pd.Timestamp, direction: Optional[int] = None
    ) -> bool:
        if self.check_stock_suspended(stock_id, trade_date):
            return False
        if self.check_stock_limit(stock_id, trade_date, direction):
            return False
        return True

    def round_amount_by_trade_unit(self, amount: float, factor: Optional[float]) -> float:
        if self._trade_unit is None:
            return amount
        if amount <= 0:
            return 0.0
        if factor is None or pd.isna(factor) or float(factor) <= 1e-12:
            return amount
        return float(
            (amount * float(factor) + 0.1) // self._trade_unit * self._trade_unit / float(factor)
        )

    def get_buy_amount_by_cash_limit(
        self, trade_price: float, cash: float
    ) -> float:
        max_trade_amount = 0.0
        if cash >= self._min_cost:
            critical_price = self._min_cost / self._open_cost + self._min_cost
            if cash >= critical_price:
                max_trade_amount = cash / (1 + self._open_cost) / trade_price
            else:
                max_trade_amount = (cash - self._min_cost) / trade_price
        return max_trade_amount
