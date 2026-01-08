import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import geopandas as gpd
import pandas as pd
import contextily as ctx

st.title("🌿 Analyse Drone Output – Segmentatie-maskers")

st.info("Segmentatie-analyse: kies eerst de eigenschap die je wilt analyseren.")

# Vraag gebruiker welke eigenschap
property_choice = st.selectbox(
    label="📈 Kies eigenschap om te analyseren:",
    options=["Hoogte", "Diameter", "ExG", "ExR"]
)

# Upload GeoPackage
gpkg_file = st.file_uploader("📁 Upload GeoPackage (.gpkg)", type="gpkg")

if gpkg_file is not None:
    # --- Laad en bewaar in session_state ---
    if "gdf" not in st.session_state:
        load_bar = st.progress(0)
        load_status = st.empty()

        load_status.text("📥 Bestand ontvangen, starten met inlezen...")
        time.sleep(0.2)
        load_bar.progress(0.2)

        gdf = gpd.read_file(gpkg_file)
        time.sleep(0.2)
        load_bar.progress(0.6)
        load_status.text(f"📂 {len(gdf)} segmenten ingelezen...")

        # Optionele berekening
        if property_choice in ["ExG", "ExR"]:
            pass
        time.sleep(0.2)
        load_bar.progress(1.0)
        load_status.text(f"✅ {len(gdf)} segmenten geladen en verwerkt.")

        st.session_state.gdf = gdf
        st.success(f"✅ Klaar! {len(gdf)} segmenten beschikbaar voor analyse.")
    else:
        gdf = st.session_state.gdf

    # --- Kolom mapping ---
    col_mapping = {
        "Hoogte": "height_p95",
        "Diameter": "diameter",
        "ExG": "ExG_median",
        "ExR": "ExR_median"
    }
    col_to_plot = col_mapping[property_choice]
    values = gdf[col_to_plot].values

    # --- Histogram ---
    st.subheader(f"📊 Histogram van {property_choice}")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(values, bins=50, color="gray", edgecolor="black")
    ax_hist.set_xlabel(property_choice)
    ax_hist.set_ylabel("Aantal segmenten")
    st.pyplot(fig_hist)

    # --- Filter sliders ---
    st.subheader("🎚️ Filter segmenten")
    min_val, max_val = float(values.min()), float(values.max())
    lower = st.slider("Ondergrens", min_val, max_val, min_val)
    upper = st.slider("Bovengrens", min_val, max_val, max_val)

    filtered = gdf[(gdf[col_to_plot] >= lower) & (gdf[col_to_plot] <= upper)]

    # --- Progress bar bij kaartweergave ---
    map_bar = st.progress(0)
    map_status = st.empty()

    st.subheader(f"🗺️ Kaartweergave van gefilterde segmenten ({len(filtered)})")

    fig_map, ax_map = plt.subplots(figsize=(20, 10), dpi=150)

    try:
        map_status.text("🗺️ Voorbereiden van kaartdata...")
        map_bar.progress(0.2)
        time.sleep(0.1)

        # Controleer CRS en stel in als dat ontbreekt
        if filtered.crs is None:
            st.warning("Geen CRS gevonden, stel in op EPSG:32631.")
            filtered = filtered.set_crs(epsg=32631)

        # Herprojecteer naar Web Mercator (EPSG:3857)
        filtered_3857 = filtered.to_crs(epsg=3857)
        map_bar.progress(0.5)

        filtered_points = filtered_3857.copy()
        filtered_points["geometry"] = filtered_points.centroid

        filtered_points.plot(
            ax=ax_map,
            color='red',
            markersize=10,
            alpha=0.8
        )
        map_bar.progress(0.7)

        # Voeg satellietachtergrond toe
        map_status.text("🛰️ Basemap laden...")
        ctx.add_basemap(
            ax_map,
            source=ctx.providers.Esri.WorldImagery,
            crs=filtered_3857.crs.to_string(),
            zoom=20
        )
        map_bar.progress(0.9)

        ax_map.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig_map)
        map_bar.progress(1.0)
        map_status.text("✅ Kaart succesvol weergegeven.")
        plt.close(fig_map)

    except Exception as e:
        st.warning(f"Kaartweergave mislukt: {e}")
        filtered.plot(column=col_to_plot, ax=ax_map, cmap="terrain", legend=True)
        ax_map.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig_map)
        plt.close(fig_map)
