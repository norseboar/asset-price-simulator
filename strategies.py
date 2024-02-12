from collections import deque
import math

from utilities import cond_print


class Strategy:
    name = "Base Strategy"

    def __init__(self, seed, starting_money, print_details):
        self.buy_turns = []
        self.shares = 0
        self.peak_price = 0
        self.last_price = 0
        self.buy_count = 0
        self.peak_buy_count = 0

        self.seed = seed
        self.money = starting_money
        self.print_details = print_details

    def _update_data(self, price):
        self.last_price = price
        if price > self.peak_price:
            self.peak_price = price

    def _should_buy(self, _):
        return True

    def assess_and_buy(self, price, turn):
        self._update_data(price)
        if self._should_buy(price):
            share_count = math.floor(self.money / price)
            if share_count > 0:
                cond_print(
                    self.print_details,
                    f"{self.name} buying at {price} with {self.money}",
                )
                self.buy_count += 1
                if price >= self.peak_price:
                    self.peak_buy_count += 1
                self.buy_turns.append(turn)
            self.money -= price * share_count
            self.shares += share_count

    def get_net_worth(self):
        return self.shares * self.last_price + self.money

    def __repr__(self) -> str:
        return f"{self.name}: {self.shares} shares, {self.money} money"


class NeverBuy(Strategy):
    name = "Never Buy"

    def _should_buy(self, _):
        return False


class BuyRegularly(Strategy):
    name = "Buy Regularly"


class BuyDipThreshold(Strategy):
    name = "Buy Dip at Threshold"

    def __init__(self, threshold, window, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prices = deque()
        self.threshold = threshold
        self.window = window

    def _update_data(self, price):
        super()._update_data(price)
        self.prices.append(price)
        if len(self.prices) > self.window:
            self.prices.popleft()
        cond_print(
            self.print_details,
            f"{self.name} 'real value' is {sum(self.prices)/len(self.prices)}",
        )

    def _should_buy(self, price):
        return (sum(self.prices) / len(self.prices)) * self.threshold >= price

    def __repr__(self) -> str:
        return f"Shares: {self.shares}, Money: {self.money}, Prices: {self.prices}"


class BuyDipTrend(Strategy):
    name = "Buy Dip after Trend"

    def __init__(self, trend_length, **kwargs):
        super().__init__(**kwargs)
        self.trend_length = trend_length
        self.trend_count = 0

    def _update_data(self, price):
        if price < self.last_price:
            self.trend_count += 1
        else:
            self.trend_count = 0
        super()._update_data(price)

    def _should_buy(self, _):
        return self.trend_count >= self.trend_length
