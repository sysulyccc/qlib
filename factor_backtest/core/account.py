"""
Account class for managing trading account state and metrics.
"""
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd

from .position import Position
from .exchange import Exchange


class Account:
    """
    Account manages the trading account including position, cash, and accumulated metrics.
    Handles order execution and portfolio metrics calculation.
    """

    def __init__(
        self,
        exchange: Exchange,
        init_cash: float = 1e8,
    ):
        self._exchange = exchange
        self._position = Position(init_cash)
        self._accum_turnover = 0.0
        self._accum_cost = 0.0
        self._last_account_value = float(init_cash)
        self._last_total_turnover = 0.0
        self._last_total_cost = 0.0

    @property
    def position(self) -> Position:
        return self._position

    @property
    def cash(self) -> float:
        return self._position.cash

    @property
    def accum_turnover(self) -> float:
        return self._accum_turnover

    @property
    def accum_cost(self) -> float:
        return self._accum_cost

    def get_stock_list(self) -> List[str]:
        return self._position.get_stock_list()

    def get_stock_amount(self, code: str) -> float:
        return self._position.get_stock_amount(code)

    def get_stock_count(self, code: str, bar: str = "day") -> int:
        return self._position.get_stock_count(code, bar)

    def deal_sell_order(
        self,
        stock_id: str,
        sell_amount: float,
        trade_date: pd.Timestamp,
    ) -> Tuple[float, float, float]:
        """
        Execute sell order.
        Returns: (trade_val, trade_cost, trade_price)
        """
        if not self._exchange.is_stock_tradable(stock_id, trade_date, direction=None):
            return 0.0, 0.0, np.nan

        trade_price = self._exchange.get_deal_price(stock_id, trade_date)
        if np.isnan(trade_price) or trade_price <= 1e-12:
            return 0.0, 0.0, np.nan

        factor = self._exchange.get_factor(stock_id, trade_date)
        current_amount = self._position.get_stock_amount(stock_id)

        deal_amount = min(sell_amount, current_amount)

        if not self._position.is_amount_close(stock_id, deal_amount):
            deal_amount = self._exchange.round_amount_by_trade_unit(deal_amount, factor)

        if deal_amount <= 0:
            return 0.0, 0.0, np.nan

        trade_val = deal_amount * trade_price
        trade_cost = max(trade_val * self._exchange.close_cost, self._exchange.min_cost) if trade_val > 1e-5 else 0.0

        if self._position.cash + trade_val < trade_cost:
            return 0.0, 0.0, np.nan

        if self._position.is_amount_close(stock_id, deal_amount):
            self._position.remove_stock(stock_id)
        else:
            self._position.reduce_stock(stock_id, deal_amount)

        self._position.cash += trade_val - trade_cost
        self._accum_turnover += trade_val
        self._accum_cost += trade_cost

        return trade_val, trade_cost, trade_price

    def deal_buy_order(
        self,
        stock_id: str,
        buy_amount: float,
        trade_date: pd.Timestamp,
    ) -> Tuple[float, float, float]:
        """
        Execute buy order.
        Returns: (trade_val, trade_cost, trade_price)
        """
        if not self._exchange.is_stock_tradable(stock_id, trade_date, direction=None):
            return 0.0, 0.0, np.nan

        trade_price = self._exchange.get_deal_price(stock_id, trade_date)
        if np.isnan(trade_price) or trade_price <= 1e-12:
            return 0.0, 0.0, np.nan

        factor = self._exchange.get_factor(stock_id, trade_date)
        cash = self._position.cash
        cost_ratio = self._exchange.open_cost
        min_cost = self._exchange.min_cost

        deal_amount = buy_amount
        trade_val = deal_amount * trade_price

        if cash < max(trade_val * cost_ratio, min_cost):
            return 0.0, 0.0, np.nan
        elif cash < trade_val + max(trade_val * cost_ratio, min_cost):
            max_buy_amount = self._exchange.get_buy_amount_by_cash_limit(trade_price, cash)
            deal_amount = self._exchange.round_amount_by_trade_unit(
                min(max_buy_amount, deal_amount), factor
            )
        else:
            deal_amount = self._exchange.round_amount_by_trade_unit(deal_amount, factor)

        if deal_amount <= 0:
            return 0.0, 0.0, np.nan

        trade_val = deal_amount * trade_price
        trade_cost = max(trade_val * cost_ratio, min_cost) if trade_val > 1e-5 else 0.0

        if trade_val <= 1e-5:
            return 0.0, 0.0, np.nan

        self._position.add_stock(stock_id, deal_amount, trade_price)
        self._position.cash -= trade_val + trade_cost
        self._accum_turnover += trade_val
        self._accum_cost += trade_cost

        return trade_val, trade_cost, trade_price

    def update_bar_end(self, trade_date: pd.Timestamp) -> None:
        """Update position at bar end."""
        for code in self._position.get_stock_list():
            if self._exchange.is_stock_tradable(code, trade_date):
                px = self._exchange.get_deal_price(code, trade_date)
                if not np.isnan(px):
                    self._position.update_stock_price(code, px)
        self._position.add_count_all()

    def calculate_metrics(self, trade_date: pd.Timestamp, bench_return: float) -> Dict[str, Any]:
        """Calculate daily metrics."""
        def get_price(code):
            return self._exchange.get_deal_price(code, trade_date)

        now_account_value = self._position.calculate_total_value(get_price)
        now_earning = now_account_value - self._last_account_value
        now_cost = self._accum_cost - self._last_total_cost
        now_turnover = self._accum_turnover - self._last_total_turnover

        metrics = {
            "date": trade_date,
            "return": (now_earning + now_cost) / self._last_account_value,
            "turnover": now_turnover / self._last_account_value,
            "cost": now_cost / self._last_account_value,
            "bench": float(bench_return),
        }

        self._last_account_value = now_account_value
        self._last_total_cost = self._accum_cost
        self._last_total_turnover = self._accum_turnover

        return metrics
