#!/usr/bin/env python3
"""
visualize.py

CLI tool to save visualizations of synthetic traffic data.

By default, all visualizations are exported as interactive HTML.
If you prefer static PNGs (requires Kaleido), use `--format png`.

Usage:
  python visualize.py --input_file path/to/data.csv --viz <type> [--format html|png]

Available viz types:
  map               Interactive map (always HTML)
  speed_hist        Histogram of Speed_kmh
  event_count       Bar chart of Event_Type counts
  time_series       Time-series of event counts by hour
  density_heatmap   Density heatmap of Traffic_Density on map
  severity_pie      Pie chart of Severity distribution
  speed_vs_density  Scatter plot of Speed_kmh vs Traffic_Density

Requirements:
    pip install pandas plotly kaleido matplotlib
"""

import argparse
from pathlib import Path

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import pandas as pd
import plotly.express as px
import plotly.io as pio

# default figure size
DEFAULT_WIDTH, DEFAULT_HEIGHT = 800, 800

def main():
    parser = argparse.ArgumentParser(
        description="Visualize synthetic traffic data",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_file', required=True,
                        help="Path to synthetic traffic CSV")
    parser.add_argument('--viz', required=True,
                        choices=[
                            "map", "speed_hist", "event_count",
                            "time_series", "density_heatmap",
                            "severity_pie", "speed_vs_density"
                        ],
                        help="Visualization type")
    parser.add_argument('--format', choices=['html', 'png'], default='html',
                        help="Export format for non-map viz (default: html)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_file, parse_dates=["Timestamp"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Prepare output directory one level deeper
    base = Path("experiment/visualizations/data_insights") / args.viz
    base.mkdir(parents=True, exist_ok=True)

    # Map is always HTML and uses full square size
    if args.viz == "map":
        fig = px.scatter_mapbox(
            df, lat="Latitude", lon="Longitude",
            color="Event_Type", size="Traffic_Density",
            hover_data=["Event_ID", "Vehicle_Type", "Speed_kmh", "Severity", "Traffic_Density"],
            zoom=10,
            width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
            mapbox_style="open-street-map",
            title="Traffic Events Map"
        )
        out_file = base / "map.html"
        fig.write_html(str(out_file))
        print(f"✅ Saved interactive map to {out_file}")
        return

    # Other visualizations
    if args.viz == "speed_hist":
        fig = px.histogram(
            df, x="Speed_kmh", nbins=50, title="Speed Distribution",
            width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT
        )
    elif args.viz == "event_count":
        counts = df["Event_Type"].value_counts().reset_index()
        counts.columns = ["Event_Type", "Count"]
        fig = px.bar(
            counts, x="Event_Type", y="Count", title="Event Type Counts",
            width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT
        )
    elif args.viz == "time_series":
        ts = df.set_index("Timestamp").resample("h").size().reset_index(name="Count")
        fig = px.line(
            ts, x="Timestamp", y="Count", title="Events per Hour",
            width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT
        )
    elif args.viz == "density_heatmap":
        fig = px.density_mapbox(
            df, lat="Latitude", lon="Longitude", z="Traffic_Density",
            radius=10, zoom=10,
            width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
            mapbox_style="open-street-map",
            title="Traffic Density Heatmap"
        )
    elif args.viz == "severity_pie":
        sev = df["Severity"].value_counts().reset_index()
        sev.columns = ["Severity", "Count"]
        fig = px.pie(
            sev, names="Severity", values="Count", title="Severity Distribution",
            width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT
        )
    elif args.viz == "speed_vs_density":
        fig = px.scatter(
            df, x="Traffic_Density", y="Speed_kmh",
            color="Event_Type",
            hover_data=["Event_ID", "Vehicle_Type", "Severity"],
            title="Speed vs Traffic Density",
            width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT
        )
    else:
        raise ValueError(f"Unknown viz type: {args.viz}")

    # Export
    if args.format == 'html':
        out_file = base / f"{args.viz}.html"
        fig.write_html(str(out_file))
        print(f"✅ Saved interactive HTML plot to {out_file}")
    else:
        out_file = base / f"{args.viz}.png"
        pio.write_image(fig, str(out_file), engine="kaleido")
        print(f"✅ Saved static PNG plot to {out_file}")

if __name__ == "__main__":
    main()
