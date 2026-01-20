"""
Combine the 3 Minard datasets (troops route, temperature, cities) into ONE visualization.

What it draws:
- Troop movements as a polyline on lon/lat with line thickness ~ sqrt(survivors)
- Cities as markers + labels
- Temperature as a dotted curve embedded below the map (still x = longitude)

Run:
    python minard_combined_plot.py

Outputs:
    minard_combined_visualization_final.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop the common "Unnamed: 0" column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    # ---- Update these paths if your filenames/locations differ ----
    troops_path = "Data_from_Minard_s_famous_graphic_map_of_Napoleon_s_march_on_Moscow_900_85.csv"
    temp_path = "Data_from_Minard_s_famous_graphic_map_of_Napoleon_s_march_on_Moscow_899_95.csv"
    cities_path = "Data_from_Minard_s_famous_graphic_map_of_Napoleon_s_march_on_Moscow_898_15.csv"

    troops = load_csv(troops_path)
    temp = load_csv(temp_path)
    cities = load_csv(cities_path)

    # Ensure numeric columns are numeric
    troops = to_numeric(troops, ["X", "long", "lat", "survivors", "group"])
    temp = to_numeric(temp, ["X", "long", "temp", "days"])
    cities = to_numeric(cities, ["X", "long", "lat"])

    # Sort for clean plotting
    if set(["group", "direction", "X"]).issubset(troops.columns):
        troops = troops.sort_values(["group", "direction", "X"]).reset_index(drop=True)
    elif "X" in troops.columns:
        troops = troops.sort_values(["X"]).reset_index(drop=True)

    if "X" in temp.columns:
        temp = temp.sort_values("X").reset_index(drop=True)
    if "X" in cities.columns:
        cities = cities.sort_values("X").reset_index(drop=True)

    # Basic checks
    needed_troops = {"long", "lat", "survivors"}
    needed_temp = {"long", "temp"}
    needed_cities = {"long", "lat", "city"}

    if not needed_troops.issubset(troops.columns):
        raise ValueError(f"Troops CSV must include columns: {sorted(needed_troops)}")
    if not needed_temp.issubset(temp.columns):
        raise ValueError(f"Temp CSV must include columns: {sorted(needed_temp)}")
    if not needed_cities.issubset(cities.columns):
        raise ValueError(f"Cities CSV must include columns: {sorted(needed_cities)}")

    max_surv = float(troops["survivors"].max())
    min_lat = float(troops["lat"].min())
    max_lat = float(troops["lat"].max())

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Build legend handles first (so legend line styles match) ---
    ax.set_prop_cycle(None)
    adv_handle, = ax.plot([], [], linestyle="-", linewidth=3, alpha=0.85, label="Advance (solid)")
    ax.set_prop_cycle(None)
    ret_handle, = ax.plot([], [], linestyle="--", linewidth=3, alpha=0.85, label="Retreat (dashed)")

    # --- Troop movement route ---
    # If direction/group exist, keep them; else just draw sequential segments
    if set(["group", "direction"]).issubset(troops.columns):
        troop_groups = troops.groupby(["group", "direction"])
        for (_, direction), gdf in troop_groups:
            gdf = gdf.sort_values("X") if "X" in gdf.columns else gdf
            for i in range(len(gdf) - 1):
                p1 = gdf.iloc[i]
                p2 = gdf.iloc[i + 1]
                surv = float((p1["survivors"] + p2["survivors"]) / 2.0)

                # linewidth proportional to sqrt(survivors)
                lw = 0.5 + 11.5 * np.sqrt(surv / max_surv)

                # A=advance, R=retreat (dataset uses 'A'/'R')
                ls = "-" if str(direction).upper().startswith("A") else "--"

                # Reset cycle so all segments share the default line color
                ax.set_prop_cycle(None)
                ax.plot([p1["long"], p2["long"]], [p1["lat"], p2["lat"]],
                        linewidth=lw, linestyle=ls, alpha=0.85)
    else:
        troops_sorted = troops.sort_values("X") if "X" in troops.columns else troops
        for i in range(len(troops_sorted) - 1):
            p1 = troops_sorted.iloc[i]
            p2 = troops_sorted.iloc[i + 1]
            surv = float((p1["survivors"] + p2["survivors"]) / 2.0)
            lw = 0.5 + 11.5 * np.sqrt(surv / max_surv)
            ax.set_prop_cycle(None)
            ax.plot([p1["long"], p2["long"]], [p1["lat"], p2["lat"]],
                    linewidth=lw, linestyle="-", alpha=0.85)

    # --- Cities ---
    ax.set_prop_cycle(None)
    city_handle = ax.scatter(cities["long"], cities["lat"], marker="^", s=30, alpha=0.95, label="Cities")
    for _, row in cities.iterrows():
        ax.text(float(row["long"]) + 0.08, float(row["lat"]) + 0.05, str(row["city"]), fontsize=8)

    # --- Temperature curve embedded below the map ---
    temp_min, temp_max = float(temp["temp"].min()), float(temp["temp"].max())

    # Place the temperature curve below the minimum troop latitude
    y_base = min_lat - 2.3
    y_span = 1.4  # vertical range used to map temperatures into latitude space

    y_temp = y_base + (temp["temp"] - temp_min) / (temp_max - temp_min) * y_span

    ax.set_prop_cycle(None)
    temp_handle, = ax.plot(temp["long"], y_temp, marker="o", linestyle=":", linewidth=1.5,
                           alpha=0.95, label="Temperature (dotted)")

    # Label each temperature point (uses date column if present)
    for _, row in temp.iterrows():
        yt = y_base + (float(row["temp"]) - temp_min) / (temp_max - temp_min) * y_span
        date_label = str(row["date"]) if "date" in temp.columns else ""
        ax.text(float(row["long"]), yt - 0.10, f'{int(row["temp"])}° {date_label}'.strip(),
                fontsize=7, ha="center")

    # --- Final formatting ---
    ax.set_title("Napoleon’s March on Moscow — Combined view (Troops + Cities + Temperature)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude (temperature curve embedded below route)")

    ax.legend(handles=[adv_handle, ret_handle, city_handle, temp_handle],
              loc="upper right", frameon=True)

    ax.set_xlim(float(troops["long"].min()) - 0.5, float(troops["long"].max()) + 0.5)
    ax.set_ylim(y_base - 0.5, max_lat + 0.8)
    ax.grid(True, alpha=0.2)

    out_file = "minard_combined_visualization_final.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
