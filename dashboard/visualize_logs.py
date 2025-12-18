from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Phase 1/2 simulator CSV logs with Plotly.")
    p.add_argument("--log", type=str, required=True, help="Path to phase1_orders_*.csv")
    p.add_argument("--outdir", type=str, default="data/logs", help="Directory to write HTML plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    df_del = df[df["stockout"] == 0].copy()

    # Delay distribution
    fig_delay = px.histogram(
        df_del,
        x="delay",
        nbins=30,
        title="Delivery delay distribution (delivered orders only)",
    )
    delay_out = outdir / f"{log_path.stem}_delay_hist.html"
    fig_delay.write_html(delay_out)

    # Time series
    fig_ts = px.scatter(
        df_del,
        x="created_time",
        y="delay",
        color="vehicle_id",
        title="Delay vs order created_time (colored by vehicle)",
    )
    ts_out = outdir / f"{log_path.stem}_delay_timeseries.html"
    fig_ts.write_html(ts_out)

    # Cost vs distance sanity check
    fig_cost = px.scatter(
        df_del,
        x="travel_distance",
        y="cost",
        title="Cost vs travel distance (delivered orders only)",
    )
    cost_out = outdir / f"{log_path.stem}_cost_vs_distance.html"
    fig_cost.write_html(cost_out)

    print(f"wrote={delay_out}")
    print(f"wrote={ts_out}")
    print(f"wrote={cost_out}")


if __name__ == "__main__":
    main()


