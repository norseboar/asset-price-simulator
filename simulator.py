from collections import defaultdict
import math
from multiprocessing import cpu_count, Pool
import random
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import prettytable
import scipy.stats as st

from strategies import BuyRegularly, BuyDipThreshold, NeverBuy
from utilities import cond_print


def update_price(rng, price, midpoint, stddev):
    variance = rng.normal(midpoint, stddev)
    return max(price + (price * variance), 1)


def run_trial(
    turns,
    seed=None,
    starting_price=100,
    starting_money=10000,
    salary=100,
    salary_interval=1,
    growth_midpoint=0.002,
    growth_stddev=0.01,
    dip_threshold=0.95,
    dip_window=30,
    print_summary=False,
    print_details=False,
    show_chart=None,
):
    if seed is None:
        seed = random.randint(0, 999999999)
    rng = np.random.default_rng(seed)

    strategy_kwargs = {
        "seed": seed,
        "starting_money": starting_money,
        "print_details": print_details,
    }
    reg_strategy = BuyRegularly(**strategy_kwargs)
    buy_dip_strategy = BuyDipThreshold(
        **strategy_kwargs,
        threshold=dip_threshold,
        window=dip_window,
    )
    never_buy = NeverBuy(**strategy_kwargs)

    strategies = [reg_strategy, buy_dip_strategy, never_buy]

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
        x = range(len(all_prices))
        plt.plot(x, all_prices, label="Price", color="k")
        plt.plot(x, buy_dip_strategy.buy_thresholds, "--r", label="Thresholds")

        for t in buy_dip_strategy.buy_turns:
            plt.axvline(t, linewidth=0.5, color="b")

        plt.xlabel("Turn")
        plt.ylabel("Price")
        plt.show()

    return strategies


def run_trial_map_wrapper(kwargs):
    return run_trial(**kwargs)


def run_many_trials(
    trials,
    turns=365 * 3,
    show_headline=True,
    show_results_table=True,
    starting_money=10000,
    salary=100,
    salary_interval=1,
    starting_price=100,
    growth_midpoint=0.0006,
    growth_stddev=0.0094,
    dip_threshold=0.95,
    dip_window=30,
    **kwargs,
):
    if show_headline:
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
                Buy Dip Threshold: {dip_threshold}
                Buy Dip Window: {dip_window}
            """
            )
        )

    strategy_map = defaultdict(list)

    with Pool(cpu_count()) as pool:
        trials = pool.map(
            run_trial_map_wrapper,
            [
                dict(
                    turns=turns,
                    starting_money=starting_money,
                    salary=salary,
                    salary_interval=salary_interval,
                    starting_price=starting_price,
                    growth_midpoint=growth_midpoint,
                    growth_stddev=growth_stddev,
                    dip_threshold=dip_threshold,
                    dip_window=dip_window,
                    **kwargs,
                )
            ]
            * trials,
        )

    for trial in trials:
        for s in trial:
            strategy_map[s.name].append(s)

    strategy_pairs = sorted(
        zip(strategy_map[BuyRegularly.name], strategy_map[BuyDipThreshold.name]),
        key=lambda pair: pair[0].get_net_worth() / pair[1].get_net_worth()
        if pair[1].get_net_worth() > 0
        else math.inf,
    )

    pair_5pct = strategy_pairs[len(strategy_pairs) // 20]
    pair_25pct = strategy_pairs[len(strategy_pairs) // 4]
    pair_50pct = strategy_pairs[len(strategy_pairs) // 2]
    pair_75pct = strategy_pairs[(len(strategy_pairs) // 4) * 3]
    pair_95pct = strategy_pairs[(len(strategy_pairs) // 20) * 19]

    ratios = [
        r.get_net_worth() / d.get_net_worth() if d.get_net_worth() > 0 else math.inf
        for r, d in strategy_pairs
    ]

    mean_ratio = np.mean(ratios)
    confidence_interval = st.norm.ppf(0.95) * st.sem(ratios)

    if show_headline:
        print("Results:")
        summary_table = prettytable.PrettyTable()
        summary_table.field_names = [
            f"{BuyRegularly.name} vs {BuyDipThreshold.name}: Mean Net Worth Ratio (95% Confidence)"
        ]
        summary_table.add_row([f"{mean_ratio:.2f} ± {confidence_interval:.2f}"])
        summary_table.set_style(prettytable.DOUBLE_BORDER)
        print(summary_table)

    if show_results_table:
        table = prettytable.PrettyTable()
        table.field_names = [
            "Strategy",
            "Metric",
            "5th Percentile",
            "25th Percentile",
            "50th Percentile",
            "75th Percentile",
            "95th Percentile",
        ]

        table.add_row(
            [
                "Reg vs Dip Ratio",
                "Net Worth",
                *map(
                    lambda p: f"{p[0].get_net_worth() / p[1].get_net_worth():.2f}",
                    [pair_5pct, pair_25pct, pair_50pct, pair_75pct, pair_95pct],
                ),
            ]
        )

        table.add_row(
            [
                "Reg vs Dip Ratio",
                "Price Paid",
                *map(
                    lambda p: f"{p[0].get_avg_price()/p[1].get_avg_price() if p[1].get_avg_price() > 0 else math.inf:,.2f}",
                    [pair_5pct, pair_25pct, pair_50pct, pair_75pct, pair_95pct],
                ),
            ]
        )

        table.add_row(
            [
                "Reg vs Dip Ratio",
                "Days with Buy",
                *map(
                    lambda p: f"{p[0].buy_count / p[1].buy_count if p[1].buy_count > 0 else math.inf:,.2f}",
                    [pair_5pct, pair_25pct, pair_50pct, pair_75pct, pair_95pct],
                ),
            ]
        )

        table.add_row(
            [
                "",
                "% days at peak price",
                *map(
                    lambda p: f"{p[0].peak_count / turns:.2%}",
                    [pair_5pct, pair_25pct, pair_50pct, pair_75pct, pair_95pct],
                ),
            ]
        )
        table.add_row(
            [
                "",
                "Seed",
                *map(
                    lambda p: p[0].seed,
                    [pair_5pct, pair_25pct, pair_50pct, pair_75pct, pair_95pct],
                ),
            ],
            divider=True,
        )

        for i, strategy_name in enumerate([BuyRegularly.name, BuyDipThreshold.name]):
            s_5pct = pair_5pct[i]
            s_25pct = pair_25pct[i]
            s_50pct = pair_50pct[i]
            s_75pct = pair_75pct[i]
            s_95pct = pair_95pct[i]

            table.add_row(
                [
                    strategy_name,
                    "Net Worth",
                    *map(
                        lambda s: f"{s.get_net_worth():,.2f}",
                        [s_5pct, s_25pct, s_50pct, s_75pct, s_95pct],
                    ),
                ]
            )

            table.add_row(
                [
                    strategy_name,
                    "Average Price Paid",
                    *map(
                        lambda s: f"{s.get_avg_price():,.2f}",
                        [s_5pct, s_25pct, s_50pct, s_75pct, s_95pct],
                    ),
                ]
            )

            table.add_row(
                [
                    strategy_name,
                    "Days with Buy",
                    *map(
                        lambda s: f"{s.buy_count}",
                        [s_5pct, s_25pct, s_50pct, s_75pct, s_95pct],
                    ),
                ],
                divider=True,
            )

        table.align = "r"
        table.vrules = prettytable.FRAME
        print(table)

    return mean_ratio, confidence_interval


def optimal_walker(
    growth_midpoint,
    growth_stddev,
    starting_threshold=0.95,
    starting_window=30,
    starting_trials=1000,
    turns=365 * 3,
    **kwargs,
):
    trials = starting_trials
    dip_threshold = starting_threshold
    dip_window = starting_window

    window_max = turns

    mean_ratio, ci = try_params(
        trials,
        dip_threshold,
        dip_window,
        growth_midpoint=growth_midpoint,
        growth_stddev=growth_stddev,
        turns=turns,
        **kwargs,
    )

    step_multiplier = 1

    while True:
        if 0.99 < mean_ratio < 1.01:
            print("Found tipping point")
            break
        if mean_ratio > 1:
            direction = -1
        else:
            direction = 1

        print(
            f"=================== Best params so far: Threshold {dip_threshold:.3f}, window {dip_window} with a mean of {mean_ratio:.2f} ± {ci:.2f} ==================="
        )

        dip_t_up = min(1 - ((1 - dip_threshold) / (2 * step_multiplier)), 0.99)
        dip_t_up_ratio, dip_t_up_ci = try_params(
            trials,
            dip_t_up,
            dip_window,
            growth_midpoint=growth_midpoint,
            growth_stddev=growth_stddev,
            turns=turns,
            **kwargs,
        )

        dip_t_down = max(1 - ((1 - dip_threshold) * (1 + (1 / step_multiplier))), 0)
        dip_t_down_ratio, dip_t_down_ci = try_params(
            trials,
            dip_t_down,
            dip_window,
            growth_midpoint=growth_midpoint,
            growth_stddev=growth_stddev,
            turns=turns,
            **kwargs,
        )

        dip_w_up = min(math.floor(dip_window * (1 + (1 / step_multiplier))), window_max)
        dip_w_up_ratio, dip_w_up_ci = try_params(
            trials,
            dip_threshold,
            dip_w_up,
            growth_midpoint=growth_midpoint,
            growth_stddev=growth_stddev,
            turns=turns,
            **kwargs,
        )

        dip_w_down = max(dip_window // (1 + (1 / step_multiplier)), 2)
        dip_w_down_ratio, dip_w_down_ci = try_params(
            trials,
            dip_threshold,
            dip_w_down,
            growth_midpoint=growth_midpoint,
            growth_stddev=growth_stddev,
            turns=turns,
            **kwargs,
        )

        best_option = "current"
        best_ratio = mean_ratio
        if check_ratio(dip_t_up_ratio, direction, best_ratio):
            best_option = "dip_t_up"
            best_ratio = dip_t_up_ratio
        if check_ratio(dip_t_down_ratio, direction, best_ratio):
            best_option = "dip_t_down"
            best_ratio = dip_t_down_ratio
        if check_ratio(dip_w_up_ratio, direction, best_ratio):
            best_option = "dip_w_up"
            best_ratio = dip_w_up_ratio
        if check_ratio(dip_w_down_ratio, direction, best_ratio):
            best_option = "dip_w_down"
            best_ratio = dip_w_down_ratio

        if best_option == "current":
            print("Breaking on current")
            break

        if best_option == "dip_t_up":
            dip_threshold = dip_t_up
            mean_ratio = dip_t_up_ratio
            ci = dip_t_up_ci
        elif best_option == "dip_t_down":
            dip_threshold = dip_t_down
            mean_ratio = dip_t_down_ratio
            ci = dip_t_down_ci
        elif best_option == "dip_w_up":
            dip_window = dip_w_up
            mean_ratio = dip_w_up_ratio
            ci = dip_w_up_ci
        elif best_option == "dip_w_down":
            dip_window = dip_w_down
            mean_ratio = dip_w_down_ratio
            ci = dip_w_down_ci

        # If we overshoot the mark, double the step size so that we take smaller steps
        if (mean_ratio > 1 and direction == 1) or (mean_ratio < 1 and direction == -1):
            step_multiplier *= 2

        direction = 1 if mean_ratio < 1 else -1

    print(
        dedent(
            f"""Optimal params with ratio {mean_ratio:.2f} ± {ci:.2f}:
            Growth midpoint: {growth_midpoint}
            Growth stddev: {growth_stddev}
            Dip threshold: {dip_threshold:.3f}
            Dip window: {dip_window}
        """
        )
    )


def try_params(trials, dip_threshold, dip_window, **kwargs):
    # Temp starting variables that will fail first iteration of while loop
    mean_ratio = 0
    confidence_interval = 1

    while abs(mean_ratio) - confidence_interval < 0:
        mean_ratio, confidence_interval = run_many_trials(
            trials,
            dip_threshold=dip_threshold,
            dip_window=dip_window,
            show_results_table=False,
            **kwargs,
        )
        trials *= 2

    return mean_ratio, confidence_interval


def check_ratio(ratio, direction, best_ratio):
    if direction == -1:
        if ratio > 1:
            if ratio < best_ratio:
                return True
        else:
            if ratio > best_ratio:
                return True
    else:
        if ratio < 1:
            if ratio > best_ratio:
                return True
        else:
            if ratio < best_ratio:
                return True
    return False
