"""
Trading strategy implementations.
"""
from typing import List, Tuple
import pandas as pd

from .account import Account
from .exchange import Exchange


class TopkDropoutStrategy:
    """
    TopK Dropout Strategy - maintains top K stocks by signal score.
    
    Each trading day:
    1. Get signal scores from previous day
    2. Determine stocks to sell (bottom n_drop from current holdings)
    3. Determine stocks to buy (top candidates not in holdings)
    4. Execute sell orders first, then buy orders
    """

    def __init__(
        self,
        topk: int = 50,
        n_drop: int = 5,
        hold_thresh: int = 1,
        risk_degree: float = 0.95,
    ):
        self._topk = topk
        self._n_drop = n_drop
        self._hold_thresh = hold_thresh
        self._risk_degree = risk_degree

    def generate_trade_decision(
        self,
        pred_score: pd.Series,
        account: Account,
        exchange: Exchange,
        trade_date: pd.Timestamp,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate trade decision based on prediction scores.
        
        Returns: (sell_list, buy_list)
        """
        if pred_score is None or len(pred_score) == 0:
            return [], []

        pred_score = pred_score.sort_values(ascending=False)
        current_stock_list = account.get_stock_list()

        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        today_cand = pred_score[~pred_score.index.isin(last)].index
        today = list(today_cand[: (self._n_drop + self._topk - len(last))])

        comb_idx = last.union(pd.Index(today))
        comb = pred_score.reindex(comb_idx).sort_values(ascending=False).index

        if self._n_drop <= 0:
            sell = pd.Index([])
        else:
            sell = last[last.isin(comb[-self._n_drop:])]

        buy = today[: (len(sell) + self._topk - len(last))]

        return list(sell), list(buy)

    def execute_trades(
        self,
        sell_list: List[str],
        buy_list: List[str],
        account: Account,
        exchange: Exchange,
        trade_date: pd.Timestamp,
    ) -> None:
        """Execute sell and buy orders."""
        for code in sell_list:
            if not exchange.is_stock_tradable(code, trade_date, direction=None):
                continue
            if not account.position.has_stock(code):
                continue
            if account.get_stock_count(code, "day") < self._hold_thresh:
                continue

            sell_amount = account.get_stock_amount(code)
            account.deal_sell_order(code, sell_amount, trade_date)

        buy_list = [
            x for x in buy_list
            if not account.position.has_stock(x)
        ]

        cash = account.cash
        value = cash * self._risk_degree / len(buy_list) if len(buy_list) > 0 else 0.0

        for code in buy_list:
            if not exchange.is_stock_tradable(code, trade_date, direction=None):
                continue

            trade_price = exchange.get_deal_price(code, trade_date)
            if pd.isna(trade_price) or trade_price <= 1e-12:
                continue

            buy_amount = value / trade_price
            account.deal_buy_order(code, buy_amount, trade_date)
