import math
import random
from textwrap import dedent

import numpy as np
from tabulate import tabulate

rng = np.random.default_rng()


class Actor:
    shares = 0
    money = 120000

    peak_price = 0
    buy_count = 0
    peak_buy_count = 0

    def should_buy(self, price):
        if price > self.peak_price:
            self.peak_price = price
        return True

    def buy(self, price):
        self.buy_count += 1
        if price >= self.peak_price:
            self.peak_buy_count += 1
        share_count = math.floor(self.money / price)
        self.money -= price * share_count
        self.shares += share_count

    def get_net_worth(self, price):
        return self.shares * price + self.money

    def __repr__(self) -> str:
        return f"Shares: {self.shares}, Money: {self.money}"


class DipActor(Actor):
    prices = []
    buy_pct = 0.95
    buy_window = 30

    def __init__(self) -> None:
        super().__init__()
        self.prices = []

    def should_buy(self, price):
        super().should_buy(price)
        self.prices.append(price)
        if len(self.prices) > self.buy_window:
            self.prices.pop(0)
        if (sum(self.prices) / len(self.prices)) * self.buy_pct >= price:
            return True
        else:
            return False

    def __repr__(self) -> str:
        return f"Shares: {self.shares}, Money: {self.money}, Prices: {self.prices}"


def update_price(price):
    variance = rng.normal(0.002, 0.01)
    return price + (price * variance)


def run_trial(
    turns,
    starting_price=100,
    buy_pct=0.95,
    buy_window=30,
    print_summary=False,
    print_details=False,
):
    regactor = Actor()
    dipactor = DipActor()
    cond_print(print_summary, f"regactor: {regactor}, dipactor: {dipactor}")
    dipactor.buy_pct = buy_pct
    dipactor.buy_window = buy_window

    price = starting_price
    turn_count = 0

    while turn_count < turns:
        if turn_count % 14 == 0:
            regactor.money += 10000
            dipactor.money += 10000
        price = update_price(price)
        cond_print(print_details, f"price is {price}")
        if regactor.should_buy(price):
            cond_print(
                print_details, f"regactor buying at {price} with {regactor.money}"
            )
            regactor.buy(price)
        if dipactor.should_buy(price):
            cond_print(
                print_details, f"dipactor buying at {price} with {dipactor.money}"
            )
            dipactor.buy(price)
        cond_print(
            print_details,
            f"dipactor avg was {sum(dipactor.prices)/len(dipactor.prices)}",
        )
        turn_count += 1

    cond_print(
        print_summary,
        dedent(
            f"""
            -------------------------------------
            After {turns} turns:

            reg_actor has {regactor.shares} shares and {regactor.money} money
            Net worth: {regactor.get_net_worth(price)}

            dip_actor has {dipactor.shares} shares and {dipactor.money} money
            Net worth: {dipactor.get_net_worth(price)}

            reg_actor performed {regactor.get_net_worth(price) / dipactor.get_net_worth(price)}
            times better than dip_actor
            """
        ),
    )

    return (
        regactor.get_net_worth(price),
        dipactor.get_net_worth(price),
        regactor.buy_count,
        dipactor.buy_count,
        regactor.peak_buy_count,
        dipactor.peak_buy_count,
        price,
    )


def run_many_trials(
    trials,
    turns,
    starting_price=100,
    buy_pct=0.95,
    buy_window=30,
    print_summary=False,
    print_details=False,
):
    regactor_networths = []
    dipactor_networths = []
    regactor_buy_counts = []
    dipactor_buy_counts = []
    regactor_peak_buy_counts = []
    dipactor_peak_buy_counts = []
    prices = []
    for _ in range(trials):
        (
            r_net_worth,
            d_net_worth,
            r_buy_count,
            d_buy_count,
            r_peak_buy_count,
            d_peak_buy_count,
            price,
        ) = run_trial(
            turns,
            starting_price=starting_price,
            buy_pct=buy_pct,
            buy_window=buy_window,
            print_summary=print_summary,
            print_details=print_details,
        )
        regactor_networths.append(r_net_worth)
        dipactor_networths.append(d_net_worth)
        regactor_buy_counts.append(r_buy_count)
        dipactor_buy_counts.append(d_buy_count)
        regactor_peak_buy_counts.append(r_peak_buy_count)
        dipactor_peak_buy_counts.append(d_peak_buy_count)
        prices.append(price)
    ratios = [r / d for r, d in zip(regactor_networths, dipactor_networths)]

    ratio_data = [
        "Reg:Dip Net Worth Ratio",
        np.percentile(ratios, 5),
        np.percentile(ratios, 50),
        np.percentile(ratios, 95),
    ]
    price_data = [
        "Stock Price",
        np.percentile(prices, 5),
        np.percentile(prices, 50),
        np.percentile(prices, 95),
    ]
    r_net_worth_data = [
        "Reg Net Worth",
        np.percentile(regactor_networths, 5),
        np.percentile(regactor_networths, 50),
        np.percentile(regactor_networths, 95),
    ]
    d_net_worth_data = [
        "Dip Net Worth",
        np.percentile(dipactor_networths, 5),
        np.percentile(dipactor_networths, 50),
        np.percentile(dipactor_networths, 95),
    ]

    r_buy_count_data = [
        "Reg Buy Count",
        np.percentile(regactor_buy_counts, 5),
        np.percentile(regactor_buy_counts, 50),
        np.percentile(regactor_buy_counts, 95),
    ]

    d_buy_count_data = [
        "Dip Buy Count",
        np.percentile(dipactor_buy_counts, 5),
        np.percentile(dipactor_buy_counts, 50),
        np.percentile(dipactor_buy_counts, 95),
    ]

    r_peak_buy_count_data = [
        "Reg Peak Buy Count",
        np.percentile(regactor_peak_buy_counts, 5),
        np.percentile(regactor_peak_buy_counts, 50),
        np.percentile(regactor_peak_buy_counts, 95),
    ]

    d_peak_buy_count_data = [
        "Dip Peak Buy Count",
        np.percentile(dipactor_peak_buy_counts, 5),
        np.percentile(dipactor_peak_buy_counts, 50),
        np.percentile(dipactor_peak_buy_counts, 95),
    ]

    r_peak_buy_pct_data = [
        "Reg Peak Buy Pct",
        np.percentile(regactor_peak_buy_counts, 5)
        / np.percentile(regactor_buy_counts, 5),
        np.percentile(regactor_peak_buy_counts, 50)
        / np.percentile(regactor_buy_counts, 50),
        np.percentile(regactor_peak_buy_counts, 95)
        / np.percentile(regactor_buy_counts, 95),
    ]

    d_peak_buy_pct_data = [
        "Dip Peak Buy Pct",
        np.percentile(dipactor_peak_buy_counts, 5)
        / np.percentile(dipactor_buy_counts, 5),
        np.percentile(dipactor_peak_buy_counts, 50)
        / np.percentile(dipactor_buy_counts, 50),
        np.percentile(dipactor_peak_buy_counts, 95)
        / np.percentile(dipactor_buy_counts, 95),
    ]

    print(f"After {trials} trials of {turns} turns each:")
    print(
        tabulate(
            [
                ratio_data,
                price_data,
                r_net_worth_data,
                r_buy_count_data,
                r_peak_buy_count_data,
                r_peak_buy_pct_data,
                d_net_worth_data,
                d_buy_count_data,
                d_peak_buy_count_data,
                d_peak_buy_pct_data,
            ],
            headers=["Metric", "5th Percentile", "50th Percentile", "95th Percentile"],
        )
    )


def cond_print(should_print, *args):
    if should_print:
        print(*args)
