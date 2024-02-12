from collections import defaultdict
import math
import random
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate, SEPARATING_LINE

from strategies import BuyRegularly, BuyDipThreshold, BuyDipTrend, NeverBuy
from utilities import cond_print


def update_price(rng, price, midpoint, stddev):
    variance = rng.normal(midpoint, stddev)
    return max(price + (price * variance), 0)


def run_trial(
    turns,
    seed=None,
    starting_price=100,
    starting_money=10000,
    salary=100,
    salary_interval=1,
    growth_midpoint=0.002,
    growth_stddev=0.01,
    bdthreshold_threshold=0.95,
    bdthreshold_window=30,
    bdtrend_trend_length=3,
    print_summary=False,
    print_details=False,
    show_chart=None,
):
    if seed == None:
        seed = random.randint(0, 99999999)
    rng = np.random.default_rng(seed)

    strategy_kwargs = {
        "seed": seed,
        "starting_money": starting_money,
        "print_details": print_details,
    }
    reg_strategy = BuyRegularly(**strategy_kwargs)
    dip_threshold_strategy = BuyDipThreshold(
        **strategy_kwargs,
        threshold=bdthreshold_threshold,
        window=bdthreshold_window,
    )
    dip_trend_strategy = BuyDipTrend(
        **strategy_kwargs, trend_length=bdtrend_trend_length
    )
    never_buy = NeverBuy(**strategy_kwargs)

    strategies = [reg_strategy, dip_threshold_strategy, dip_trend_strategy, never_buy]
    for s in strategies:
        cond_print(print_summary, s)
        s.money = starting_money

    all_prices = []

    price = starting_price
    turn_count = 0

    while turn_count < turns:
        if turn_count % salary_interval == 0:
            for s in strategies:
                s.money += salary

        new_price = update_price(rng, price, growth_midpoint, growth_stddev)
        cond_print(
            print_details,
            f"Price changed by {new_price - price}. New price is {new_price}",
        )

        price = new_price
        all_prices.append(price)

        for s in strategies:
            s.assess_and_buy(price, turn_count)

        turn_count += 1

    if show_chart:
        colors = ["b", "g", "r", "c", "m", "y", "k"]

        chart_strategies = [s for s in strategies if s.name in show_chart]
        for s, c in zip(chart_strategies, colors):
            x = range(len(all_prices))
            y = all_prices
            plt.plot(x, y)
            for t in s.buy_turns:
                plt.axvline(t, linewidth=0.5, color=c)
        plt.xlabel("Turn")
        plt.ylabel("Price")
        plt.show()

    return strategies


def run_many_trials(
    trials,
    turns,
    show_seed=False,
    starting_money=10000,
    salary=100,
    salary_interval=1,
    starting_price=100,
    growth_midpoint=0.002,
    growth_stddev=0.01,
    bdthreshold_threshold=0.95,
    bdthreshold_window=30,
    bdtrend_trend_length=3,
    print_summary=False,
    print_details=False,
    show_chart=False,
):
    print(
        dedent(
            f"""
            Starting {trials} trials, each running for {turns} turns, with the following parameters:
            Starting Money: {starting_money:,}
            Salary: {salary}
            Salary Interval: {salary_interval}
            Starting Price: {starting_price}
            Growth Midpoint: {growth_midpoint}
            Growth Stddev: {growth_stddev}
            Buy Dip Threshold Threshold: {bdthreshold_threshold}
            Buy Dip Threshold Window: {bdthreshold_window}
            Buy Dip Trend Trend Length: {bdtrend_trend_length}
        """
        )
    )
    strategy_map = defaultdict(list)
    for _ in range(trials):
        for s in run_trial(
            turns,
        ):
            strategy_map[s.name].append(s)

    ratios = [
        r.get_net_worth() / d.get_net_worth() if d.get_net_worth() > 0 else math.inf
        for r, d in zip(
            strategy_map[BuyRegularly.name], strategy_map[BuyDipThreshold.name]
        )
    ]

    ratio_output = [
        f"{BuyRegularly.name} vs {BuyDipThreshold.name}",
        "Net Worth Ratio",
        np.percentile(ratios, 5),
        np.percentile(ratios, 50),
        np.percentile(ratios, 95),
    ]

    output_lines = [ratio_output]

    for strategy_name, l in strategy_map.items():
        l.sort(key=lambda s: s.get_net_worth())
        l.sort(key=lambda s: s.get_net_worth())

        s_5pct = l[math.floor(len(l) / 20)]
        s_50pct = l[math.floor(len(l) / 2)]
        s_95pct = l[math.floor(len(l) / 20) * 19]

        output_lines.append(SEPARATING_LINE)

        output_lines.append(
            [
                strategy_name,
                "Net Worth",
                s_5pct.get_net_worth(),
                s_50pct.get_net_worth(),
                s_95pct.get_net_worth(),
            ]
        )

        output_lines.append(
            [
                strategy_name,
                "Last Price",
                s_5pct.last_price,
                s_50pct.last_price,
                s_95pct.last_price,
            ]
        )

        output_lines.append(
            [
                strategy_name,
                "Days with Buy",
                s_5pct.buy_count,
                s_50pct.buy_count,
                s_95pct.buy_count,
            ]
        )

        output_lines.append(
            [
                strategy_name,
                "Days with Buy at Peak Price",
                s_5pct.peak_buy_count,
                s_50pct.peak_buy_count,
                s_95pct.peak_buy_count,
            ]
        )

        output_lines.append(
            [
                strategy_name,
                "% of Days with Buy at Peak Price",
                s_5pct.peak_buy_count / s_5pct.buy_count if s_5pct.buy_count > 0 else 0,
                (
                    s_50pct.peak_buy_count / s_50pct.buy_count
                    if s_50pct.buy_count > 0
                    else 0
                ),
                (
                    s_95pct.peak_buy_count / s_95pct.buy_count
                    if s_95pct.buy_count > 0
                    else 0
                ),
            ]
        )
        if show_seed:
            output_lines.append(
                [strategy_name, "Seed", s_5pct.seed, s_50pct.seed, s_95pct.seed]
            )

    print(f"Results:")
    print(
        tabulate(
            output_lines,
            headers=[
                "Strategy",
                "Metric",
                "5th Percentile",
                "50th Percentile",
                "95th Percentile",
            ],
            numalign="right",
            floatfmt=",.3f",
        )
    )
