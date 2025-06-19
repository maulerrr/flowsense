# Synthetic Traffic Data Generator

A standalone Python script to generate large volumes of synthetic traffic-event records matching the schema:

```
Event_ID, Timestamp, Vehicle_Type, Speed_kmh, Latitude, Longitude, Event_Type, Severity
```

You can optionally infer distributions from a small “sample” CSV, or use built-in defaults. Geo coordinates are sampled within a configurable radius around a city center (Astana by default).

---

## 📦 Prerequisites

* Python 3.7+
* pip (for installing dependencies)

---

## ⚙️ Installation

1. Copy or clone this repository (or place `generate_synthetic_traffic.py` in your working directory).
2. Install required packages:

   ```bash
   pip install pandas numpy
   ```

---

## 🚀 Usage

```bash
python generate_synthetic_traffic.py [OPTIONS]
```

### Required

* `--center_lat`, `--center_lon`
  Coordinates of your city center. Defaults to Astana (latitude 51.1605, longitude 71.4704) if omitted.

### Optional Flags

| Flag                      | Type   | Default                            | Description                                                                                                  |
| ------------------------- | ------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--sample_file <path>`    | string | *none*                             | Path to sample CSV with columns `Timestamp,Event_Type,Speed_kmh,Severity` to infer real-world distributions. |
| `--n_rows <int>`          | int    | `30000`                            | Total number of synthetic records to generate.                                                               |
| `--incident_ratio <flt>`  | float  | `0.5`                              | Fraction of non-`Normal` events (incidents).                                                                 |
| `--vehicle_types <str>`   | string | `Car,Truck,Bus,Motorcycle,Bicycle` | Comma-separated list of vehicle categories.                                                                  |
| `--vehicle_weights <str>` | string | *equal split*                      | Comma-separated weights matching `--vehicle_types`. Sum is normalized to 1.                                  |
| `--radius_km <flt>`       | float  | `10.0`                             | Sampling radius (km) around the center point for latitude/longitude.                                         |
| `--output_file <path>`    | string | `synthetic_traffic.csv`            | File path for the generated CSV output.                                                                      |

---

## 🛠️ Examples

1. **Infer distributions** from your sample, generate 30 000 rows, 40 % incidents, within 5 km of Astana:

   ```bash
   python generate_synthetic_traffic.py \
     --sample_file sample_traffic_data.csv \
     --n_rows 30000 \
     --incident_ratio 0.4 \
     --radius_km 5 \
     --output_file astana_synth.csv
   ```

2. **No sample file** (uses built-in defaults), customize vehicle mix, generate 10 000 rows:

   ```bash
   python generate_synthetic_traffic.py \
     --vehicle_types Car,Bus,Truck \
     --vehicle_weights 0.6,0.3,0.1 \
     --n_rows 10000 \
     --incident_ratio 0.2 \
     --center_lat 51.1284 \
     --center_lon 71.4306 \
     --radius_km 8 \
     --output_file default_synth.csv
   ```

3. **Use Astana defaults** (center lat/lon need not be passed):

   ```bash
   python generate_synthetic_traffic.py \
     --n_rows 5000 \
     --incident_ratio 0.3 \
     --output_file small_astana.csv
   ```

---

## 🔧 Customization

* **Speed ranges** and **severity distributions**

  * With `--sample_file`, these are inferred per event type.
  * Without, defaults are uniform 0–120 km/h speeds and equal Low/Medium/High severities.

* **Timestamp window**

  * With `--sample_file`, timestamps fall between the sample’s min/max.
  * Without, a one-year window ending today is used.

* **Extending the script**
  Feel free to add CLI flags to override:

  * Per-event-type speed ranges
  * Custom severity categories and probabilities
  * A different timestamp sampling distribution

---

## 📝 Schema

| Column         | Type    | Description                                               |
| -------------- | ------- | --------------------------------------------------------- |
| `Event_ID`     | integer | Sequential ID (1…N)                                       |
| `Timestamp`    | string  | `YYYY-MM-DD HH:MM:SS`                                     |
| `Vehicle_Type` | string  | Category (e.g., Car, Truck)                               |
| `Speed_kmh`    | integer | Speed in km/h                                             |
| `Latitude`     | float   | Decimal degrees (within your city radius)                 |
| `Longitude`    | float   | Decimal degrees (within your city radius)                 |
| `Event_Type`   | string  | “Normal” or incident subtype (Accident, Congestion, etc.) |
| `Severity`     | string  | “Low”, “Medium”, or “High” (or as inferred)               |

---

## 📜 License

MIT License – feel free to adapt for your experiments!
