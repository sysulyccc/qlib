"""
Position class for tracking portfolio holdings.
"""
from typing import Dict, Any, List
import numpy as np


class Position:
    """
    Position tracks the current portfolio state including cash and stock holdings.
    Each stock holding contains: amount, price, and count (holding days).
    """

    def __init__(self, init_cash: float = 1e8):
        self._cash: float = float(init_cash)
        self._holdings: Dict[str, Dict[str, Any]] = {}

    @property
    def cash(self) -> float:
        return self._cash

    @cash.setter
    def cash(self, value: float) -> None:
        self._cash = value

    def get_stock_list(self) -> List[str]:
        return list(self._holdings.keys())

    def get_stock_amount(self, code: str) -> float:
        if code not in self._holdings:
            return 0.0
        return self._holdings[code].get("amount", 0.0)

    def get_stock_price(self, code: str) -> float:
        if code not in self._holdings:
            return 0.0
        return self._holdings[code].get("price", 0.0)

    def get_stock_count(self, code: str, bar: str = "day") -> int:
        if code not in self._holdings:
            return 0
        return self._holdings[code].get(f"count_{bar}", 0)

    def has_stock(self, code: str) -> bool:
        return code in self._holdings

    def add_stock(self, code: str, amount: float, price: float) -> None:
        if code in self._holdings:
            self._holdings[code]["amount"] += amount
        else:
            self._holdings[code] = {
                "amount": amount,
                "price": price,
                "weight": 0,
            }

    def remove_stock(self, code: str) -> None:
        if code in self._holdings:
            del self._holdings[code]

    def reduce_stock(self, code: str, amount: float) -> None:
        if code in self._holdings:
            self._holdings[code]["amount"] -= amount
            if self._holdings[code]["amount"] <= 0:
                del self._holdings[code]

    def update_stock_price(self, code: str, price: float) -> None:
        if code in self._holdings:
            self._holdings[code]["price"] = price

    def add_count_all(self, bar: str = "day") -> None:
        for code in self._holdings:
            key = f"count_{bar}"
            if key in self._holdings[code]:
                self._holdings[code][key] += 1
            else:
                self._holdings[code][key] = 1

    def calculate_stock_value(self, get_price_func=None) -> float:
        value = 0.0
        for code, holding in self._holdings.items():
            if get_price_func is not None:
                px = get_price_func(code)
                if np.isnan(px):
                    px = holding.get("price", 0.0)
            else:
                px = holding.get("price", 0.0)
            value += holding["amount"] * float(px)
        return value

    def calculate_total_value(self, get_price_func=None) -> float:
        return self.calculate_stock_value(get_price_func) + self._cash

    def is_amount_close(self, code: str, amount: float) -> bool:
        if code not in self._holdings:
            return False
        return np.isclose(self._holdings[code]["amount"], amount)
