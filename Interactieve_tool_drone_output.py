import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import geopandas as gpd
import pandas as pd
import contextily as ctx
from pathlib import Path

st.title("🌿 Analyse Drone Output")

# Kies analyse type
analysis_type = st.radio("🔍 Kies type analyse:", ["Orthomosaic", "Segmentatie-maskers"])

if analysis_type == "Orthomosaic":
    file_path = st.text_input("📁 Pad naar orthomosaic (.tif):", "")
    index_choice = st.selectbox("📈 Kies index:", ["Excess Green (ExG)", "Excess Red (ExR)"])
    preview_mode = st.checkbox("Gebruik preview (sneller, lagere resolutie)", value=True)

    # check of de berekening al is gedaan
    if "index" not in st.session_state:
        st.session_state.index = None

    # --- Berekening starten ---
    if st.button("🚀 Start berekening"):
        if not os.path.exists(file_path):
            st.error("Bestandspad bestaat niet!")
        else:
            with rasterio.open(file_path) as src:
                width, height = src.width, src.height
                st.write(f"Afmetingen: {width} x {height}")
                progress_bar = st.progress(0)
                status_text = st.empty()

                start = time.time()

                if preview_mode:
                    # Preview op lagere resolutie
                    step = 20
                    img = src.read(
                        out_shape=(src.count, height // step, width // step),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    R, G, B = img[0].astype(np.float32), img[1].astype(np.float32), img[2].astype(np.float32)
                    sumRGB = R+G+B
                    sumRGB = np.where(sumRGB == 0, 1e-6, sumRGB)
                    if index_choice == "Excess Green (ExG)":
                        index = (2 * G - R - B)/sumRGB
                    else:
                        index = (1.4 * R - G)/sumRGB
                    progress_bar.progress(1.0)
                    status_text.text("✅ Preview berekend.")
                else:
                    # Volledige berekening 
                    profile = src.profile
                    index = np.zeros((height, width), dtype=np.float32)
                    total_blocks = sum(1 for _ in src.block_windows(1))
                    for i, (ji, window) in enumerate(src.block_windows(1)):
                        img = src.read(window=window)
                        R, G, B = img[0].astype(np.float32), img[1].astype(np.float32), img[2].astype(np.float32)
                        sumRGB = R+G+B
                        sumRGB = np.where(sumRGB == 0, 1e-6, sumRGB)
                        if index_choice == "Excess Green (ExG)":
                            chunk = (2 * G - R - B)/sumRGB
                        else:
                            chunk = (1.4 * R - G)/sumRGB
                        row_off, col_off = window.row_off, window.col_off
                        index[row_off:row_off+window.height, col_off:col_off+window.width] = chunk
                        progress = (i + 1) / total_blocks
                        progress_bar.progress(progress)
                        elapsed = time.time() - start
                        est_total = elapsed / (i + 1) * total_blocks
                        remaining = est_total - elapsed
                        status_text.text(f"{progress*100:.1f}% voltooid – nog ~{remaining/60:.1f} min")

                    status_text.text("✅ Berekening voltooid!")

            # Sla het resultaat op in session_state zodat het blijft bestaan
            st.session_state.index = index
            st.success("✅ Berekening klaar – je kunt nu de sliders gebruiken.")

    # --- Visualisatie ---
    if st.session_state.index is not None:
        index = st.session_state.index

        st.subheader(f"🖼️ {index_choice} – Heatmap")
        fig, ax = plt.subplots()
        cax = ax.imshow(index, cmap="viridis")
        fig.colorbar(cax, ax=ax, label=index_choice)
        ax.axis("off")
        st.pyplot(fig)

        st.subheader("📊 Histogram van indexwaarden")
        fig2, ax2 = plt.subplots()
        ax2.hist(index.flatten(), bins=100, color="gray", edgecolor="black")
        ax2.set_xlabel("Indexwaarde")
        ax2.set_ylabel("Aantal pixels")
        st.pyplot(fig2)

        st.subheader("🎚️ Filter outliers")
        min_val, max_val = float(np.min(index)), float(np.max(index))
        lower = st.slider("Ondergrens", min_val, max_val, min_val)
        upper = st.slider("Bovengrens", min_val, max_val, max_val)

        filtered = np.clip(index, lower, upper)

        fig3, ax3 = plt.subplots()
        cax2 = ax3.imshow(filtered, cmap="viridis")
        fig3.colorbar(cax2, ax=ax3, label=f"{index_choice} (gefilterd)")
        ax3.axis("off")
        st.pyplot(fig3)

elif analysis_type == "Segmentatie-maskers":
    st.info("Segmentatie-analyse: kies eerst de eigenschap die je wilt analyseren.")

    # Kies eigenschap
    property_choice = st.selectbox(
        label="📈 Kies eigenschap om te analyseren:",
        options=["Hoogte", "Diameter", "ExG", "ExR"]
    )

    # Pad normalisatie helper
    def normalize_path(path_str: str) -> str:
        path_str = path_str.strip().strip('"').strip("'")
        return str(Path(path_str))

    # Pad naar GeoPackage i.p.v. upload
    gpkg_path_input = st.text_input(
        "📁 Pad naar GeoPackage (.gpkg):",
        placeholder=r"C:\data\segmentatie.gpkg of /data/segmentatie.gpkg"
    )

    if gpkg_path_input:
        gpkg_path = normalize_path(gpkg_path_input)

        st.caption(f"📂 Gebruikt pad:\n{gpkg_path}")

        if not Path(gpkg_path).exists():
            st.error("❌ Bestandspad bestaat niet.")
            st.stop()

        # --- Laad en cache GeoDataFrame ---
        if "gdf" not in st.session_state or st.session_state.get("gdf_path") != gpkg_path:
            load_bar = st.progress(0)
            load_status = st.empty()

            load_status.text("📥 GeoPackage inlezen...")
            load_bar.progress(0.2)

            gdf = gpd.read_file(gpkg_path)

            load_bar.progress(0.8)
            load_status.text(f"📂 {len(gdf)} segmenten ingelezen")

            st.session_state.gdf = gdf
            st.session_state.gdf_path = gpkg_path

            load_bar.progress(1.0)
            load_status.text("✅ Bestand succesvol geladen")
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

        if col_to_plot not in gdf.columns:
            st.error(f"❌ Kolom '{col_to_plot}' niet gevonden in GeoPackage.")
            st.stop()

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

        st.subheader(f"🗺️ Kaartweergave van gefilterde segmenten ({len(filtered)})")

        fig_map, ax_map = plt.subplots(figsize=(20, 10), dpi=150)

        try:
            # CRS check
            if filtered.crs is None:
                st.warning("Geen CRS gevonden, stel in op EPSG:32631.")
                filtered = filtered.set_crs(epsg=32631)

            # Projecteer naar Web Mercator
            filtered_3857 = filtered.to_crs(epsg=3857)

            # Gebruik centroiden voor performance
            filtered_points = filtered_3857.copy()
            filtered_points["geometry"] = filtered_points.centroid

            filtered_points.plot(
                ax=ax_map,
                color="red",
                markersize=10,
                alpha=0.8
            )

            ctx.add_basemap(
                ax=ax_map,
                source=ctx.providers.Esri.WorldImagery,
                crs=filtered_3857.crs.to_string(),
                zoom=20
            )

            ax_map.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig_map)
            plt.close(fig_map)

        except Exception as e:
            st.warning(f"Kaartweergave mislukt: {e}")
            filtered.plot(column=col_to_plot, ax=ax_map, cmap="terrain", legend=True)
            ax_map.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig_map)
            plt.close(fig_map)







