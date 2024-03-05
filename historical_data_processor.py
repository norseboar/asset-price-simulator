import os
import numpy as np

import prettytable


def get_pct_moves(filename):
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # Skip the header
        pct_moves = []
        for i, line in enumerate(lines[1:]):  # Skip the first open price
            prev_open_price = float(lines[i - 1].split(",")[1])
            open_price = float(line.split(",")[1])
            pct_moves.append((open_price - prev_open_price) / prev_open_price * 100)
    return pct_moves


def get_log_moves(filename):
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # Skip the header
        log_moves = []
        for i, line in enumerate(lines[1:]):  # Skip the first open price
            prev_open_price = float(lines[i - 1].split(",")[1])
            open_price = float(line.split(",")[1])
            log_moves.append(np.log(open_price / prev_open_price))
    return log_moves


def show_summary_data(filename):
    pct_moves = get_pct_moves(filename)
    print(f"Mean: {np.mean(pct_moves):.3f}")
    print(f"Standard deviation: {np.std(pct_moves):.3f}")
    print(
        f"P5: {np.percentile(pct_moves, 5):.4f} \
        P25: {np.percentile(pct_moves, 25):.4f} \
        P50: {np.percentile(pct_moves, 50):.4f} \
        P75: {np.percentile(pct_moves, 75):.4f} \
        P95: {np.percentile(pct_moves, 95):.4f}"
    )


def get_data_row(filename):
    log_moves = get_log_moves(filename)
    pct_moves = get_pct_moves(filename)
    return [
        os.path.basename(filename).split(".")[0],
        f"{np.percentile(pct_moves, 5):.4f}",
        f"{np.percentile(pct_moves, 50):.4f}",
        f"{np.percentile(pct_moves, 95):.4f}",
        f"{np.mean(log_moves):.6f}",
        f"{np.std(log_moves):.4f}",
    ]


def show_summary_data_dir(dir_path):
    table = prettytable.PrettyTable()
    table.field_names = [
        "Name",
        "P5 Daily Growth %",
        "P50 Daily Growth %",
        "P95 Daily Growth %",
        "Mean Log Growth",
        "Std Log Growth",
    ]
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            table.add_row(get_data_row(os.path.join(dir_path, filename)))
    print(table)
