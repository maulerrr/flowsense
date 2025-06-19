#!/usr/bin/env python3
"""
visualize_traffic.py

CLI tool to interactively visualize synthetic traffic data.
Usage:
  python visualize_traffic.py --input_file path/to/data.csv --viz <type>

Available viz types:
  map               Interactive map with hover details
  speed_hist        Histogram of Speed_kmh
  event_count       Bar chart of Event_Type counts
  time_series       Time-series of event counts by hour
  density_heatmap   Density heatmap of Traffic_Density on map
  severity_pie      Pie chart of Severity distribution
  speed_vs_density  Scatter plot of Speed_kmh vs Traffic_Density

Requirements:
    pip install pandas plotly
"""

import argparse
import pandas as pd
import plotly.express as px

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
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_file, parse_dates=["Timestamp"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Dispatch visualizations
    if args.viz == "map":
        fig = px.scatter_mapbox(
            df, lat="Latitude", lon="Longitude",
            color="Event_Type", size="Traffic_Density",
            hover_data=["Event_ID", "Vehicle_Type", "Speed_kmh", "Severity", "Traffic_Density"],
            zoom=10, height=600, mapbox_style="open-street-map",
            title="Traffic Events Map"
        )

    elif args.viz == "speed_hist":
        fig = px.histogram(
            df, x="Speed_kmh", nbins=50,
            title="Speed Distribution"
        )

    elif args.viz == "event_count":
        counts = df["Event_Type"].value_counts().reset_index()
        counts.columns = ["Event_Type", "Count"]
        fig = px.bar(
            counts, x="Event_Type", y="Count",
            title="Event Type Counts"
        )

    elif args.viz == "time_series":
        ts = df.set_index("Timestamp").resample("H").size().reset_index(name="Count")
        fig = px.line(
            ts, x="Timestamp", y="Count",
            title="Events per Hour"
        )

    elif args.viz == "density_heatmap":
        fig = px.density_mapbox(
            df, lat="Latitude", lon="Longitude", z="Traffic_Density",
            radius=10, zoom=10, height=600, mapbox_style="open-street-map",
            title="Traffic Density Heatmap"
        )

    elif args.viz == "severity_pie":
        sev = df["Severity"].value_counts().reset_index()
        sev.columns = ["Severity", "Count"]
        fig = px.pie(
            sev, names="Severity", values="Count",
            title="Severity Distribution"
        )

    elif args.viz == "speed_vs_density":
        fig = px.scatter(
            df, x="Traffic_Density", y="Speed_kmh",
            color="Event_Type",
            hover_data=["Event_ID", "Vehicle_Type", "Severity"],
            title="Speed vs Traffic Density"
        )

    else:
        raise ValueError(f"Unknown viz type: {args.viz}")

    fig.show()


if __name__ == "__main__":
    main()