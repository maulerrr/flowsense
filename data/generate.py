#!/usr/bin/env python3
"""
Enhanced synthetic traffic data generator with realistic road-based coordinates,
rush-hour timestamp patterns, incident clustering at busy intersections,
per-event traffic density estimates, and severity tied to density.

Requirements:
    pip install osmnx networkx shapely numpy pandas
"""

import argparse
import math
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString
from datetime import datetime, timedelta

# --- Default distributions (used if no sample_file is provided) ---
DEFAULT_EVENT_TYPES    = ['Normal', 'Accident', 'Congestion', 'Sudden De-celeration']
DEFAULT_EVENT_WEIGHTS  = [0.7, 0.1, 0.1, 0.1]
DEFAULT_SPEED_RANGES   = {etype: (0, 120) for etype in DEFAULT_EVENT_TYPES}

# Default center (Astana city center)
ASTANA_LAT = 51.1605
ASTANA_LON = 71.4704


def parse_vehicle_args(v_types: str, v_weights: str):
    types = [v.strip() for v in v_types.split(',')]
    if v_weights:
        w = np.array([float(x) for x in v_weights.split(',')], dtype=float)
        if len(w) != len(types):
            raise ValueError("Length of --vehicle_weights must match --vehicle_types")
        w /= w.sum()
        return types, w
    return types, np.ones(len(types), dtype=float) / len(types)


def haversine_point(lat0, lon0, r_km, size):
    R = 6371.0  # Earth radius in km
    lat0_r, lon0_r = math.radians(lat0), math.radians(lon0)
    u, v = np.random.rand(size), np.random.rand(size)
    w = (r_km / R) * np.sqrt(u)
    t = 2 * math.pi * v
    lat_r = np.arcsin(np.sin(lat0_r)*np.cos(w) +
                      np.cos(lat0_r)*np.sin(w)*np.cos(t))
    lon_r = lon0_r + np.arctan2(
        np.sin(t)*np.sin(w)*np.cos(lat0_r),
        np.cos(w) - np.sin(lat0_r)*np.sin(lat_r)
    )
    return np.degrees(lat_r), np.degrees(lon_r)


def infer_distributions(df: pd.DataFrame):
    et_counts = df['Event_Type'].value_counts(normalize=True)
    types   = et_counts.index.tolist()
    weights = et_counts.values.tolist()
    speed_ranges = {
        et: (int(sub.Speed_kmh.min()), int(sub.Speed_kmh.max()))
        for et, sub in df.groupby('Event_Type')
    }
    return types, weights, speed_ranges


def random_timestamps(n: int, start: datetime, end: datetime):
    dates = (end.date() - start.date()).days
    ts_list = []
    for _ in range(n):
        day_offset = np.random.randint(0, dates + 1)
        date = start.date() + timedelta(days=int(day_offset))
        if np.random.rand() < 0.3:
            window_start = np.random.choice([8*3600, 17*3600])
            sec = window_start + np.random.uniform(0, 2*3600)
        else:
            sec = np.random.uniform(0, 24*3600)
        ts = datetime.combine(date, datetime.min.time()) + timedelta(seconds=sec)
        ts_list.append(ts)
    return sorted(ts_list)


def compute_base_density(ts: datetime):
    seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
    frac = seconds / 86400.0
    d1 = math.sin(2 * math.pi * frac)
    d2 = math.sin(4 * math.pi * frac + 1)
    base = 50 + 30 * d1 + 20 * d2
    return max(5, base)


def main():
    p = argparse.ArgumentParser(description="Generate realistic synthetic traffic data")
    p.add_argument('--sample_file',    help="path to sample CSV for infer_distributions")
    p.add_argument('--n_rows',         type=int,   default=30000)
    p.add_argument('--incident_ratio', type=float, default=0.5)
    p.add_argument('--vehicle_types',  type=str,
                   default='Car,Truck,Bus,Motorcycle,Bicycle')
    p.add_argument('--vehicle_weights',type=str)
    p.add_argument('--center_lat',     type=float, default=ASTANA_LAT)
    p.add_argument('--center_lon',     type=float, default=ASTANA_LON)
    p.add_argument('--radius_km',      type=float, default=10.0)
    p.add_argument('--use_osm',        action='store_true')
    p.add_argument('--output_file',    default='synthetic_traffic.csv')
    args = p.parse_args()

    # 1) Vehicle distribution
    v_types, v_weights = parse_vehicle_args(args.vehicle_types, args.vehicle_weights)

    # 2) Event distributions & speed ranges
    if args.sample_file:
        df = pd.read_csv(args.sample_file, parse_dates=['Timestamp'])
        et_types, et_weights, speed_ranges = infer_distributions(df)
        t_min, t_max = df.Timestamp.min(), df.Timestamp.max()
    else:
        et_types, et_weights = DEFAULT_EVENT_TYPES, DEFAULT_EVENT_WEIGHTS
        speed_ranges        = DEFAULT_SPEED_RANGES
        t_max, t_min        = pd.Timestamp.now(), pd.Timestamp.now() - pd.Timedelta(days=365)

    # 3) Adjust weights by incident_ratio
    idx_norm    = et_types.index('Normal') if 'Normal' in et_types else None
    total_inc_w = sum(w for i,w in enumerate(et_weights) if i != idx_norm)
    p_norm      = 1 - args.incident_ratio
    p_inc_scale = args.incident_ratio / total_inc_w if total_inc_w > 0 else 0
    weights     = [
        p_norm if et=='Normal' else et_weights[i]*p_inc_scale
        for i,et in enumerate(et_types)
    ]
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    # 4) Sample core columns
    event_types   = np.random.choice(et_types,   size=args.n_rows, p=weights)
    vehicle_types = np.random.choice(v_types,    size=args.n_rows, p=v_weights)
    timestamps    = random_timestamps(
                        args.n_rows,
                        t_min.to_pydatetime(),
                        t_max.to_pydatetime()
                    )

    # 5) Speeds only
    speeds = np.empty(args.n_rows, int)
    for i, et in enumerate(event_types):
        lo, hi     = speed_ranges.get(et, (0,120))
        speeds[i]  = np.random.randint(lo, hi+1) if lo < hi else lo

    # 6) Geo coords & clustering
    if args.use_osm:
        graph    = ox.graph_from_point(
                       (args.center_lat, args.center_lon),
                       dist=args.radius_km*1000,
                       network_type='drive'
                   )
        G_undir  = graph.to_undirected()
        cc       = max(nx.connected_components(G_undir), key=len)
        Gsub     = G_undir.subgraph(cc)
        node_ids = list(Gsub.nodes)
        busy     = [n for n in node_ids if Gsub.degree[n] >= 3]

        def sample_route_coords():
            u, v = np.random.choice(node_ids, 2, replace=False)
            path = nx.shortest_path(Gsub, u, v, weight='length')
            return [(Gsub.nodes[n]['y'], Gsub.nodes[n]['x']) for n in path]

        def sample_point_on_route(rc):
            line = LineString([(lon, lat) for lat, lon in rc])
            pt   = line.interpolate(np.random.rand() * line.length)
            return pt.y, pt.x

        coords     = [sample_point_on_route(sample_route_coords())
                      for _ in range(args.n_rows)]
        lats, lons = map(list, zip(*coords))

        for i, et in enumerate(event_types):
            if et != 'Normal':
                nid           = np.random.choice(busy)
                lat0, lon0    = Gsub.nodes[nid]['y'], Gsub.nodes[nid]['x']
                lat_arr, lon_arr = haversine_point(lat0, lon0, 0.05, 1)
                lats[i], lons[i] = lat_arr[0], lon_arr[0]
    else:
        lats, lons = haversine_point(
            args.center_lat, args.center_lon,
            args.radius_km, args.n_rows
        )

    # 7) Traffic density & severity
    densities, severities = [], []
    for ts, et, spd in zip(timestamps, event_types, speeds):
        base = compute_base_density(ts)
        if et == 'Congestion':
            dens = base + np.random.uniform(50, 100)
        elif et == 'Accident':
            dens = base + np.random.uniform(20, 60)
        elif et == 'Sudden De-celeration':
            dens = base + np.random.uniform(10, 40)
        else:
            dens = base + np.random.uniform(-5, 5)
        dens = max(0, dens)
        densities.append(dens)

        # severity based on density thresholds
        if et != 'Normal':
            if dens >= 75:
                sev = 'High'
            elif dens >= 40:
                sev = 'Medium'
            else:
                sev = 'Low'
        else:
            sev = 'Low'
        severities.append(sev)

    # 8) Assemble & save
    df_out = pd.DataFrame({
        'Event_ID':        np.arange(1, args.n_rows+1),
        'Timestamp':       [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
        'Vehicle_Type':    vehicle_types,
        'Speed_kmh':       speeds,
        'Latitude':        np.round(lats, 6),
        'Longitude':       np.round(lons, 6),
        'Event_Type':      event_types,
        'Severity':        severities,
        'Traffic_Density': np.round(densities, 2)
    })

    df_out.to_csv(args.output_file, index=False)
    print(f"✅ Wrote {args.n_rows} rows → {args.output_file}")


if __name__ == '__main__':
    main()
