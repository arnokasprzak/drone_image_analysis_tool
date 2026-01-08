import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import time
import tempfile
from pathlib import Path
import os

st.set_page_config(layout="wide")
st.title("🌿 Analyse Drone Output")

# =========================================================
# Helpers
# =========================================================

def save_uploaded_file(uploaded_file, subdir="uploads"):
    """Schrijft uploaded bestand veilig weg naar disk"""
    base_dir = Path(tempfile.gettempdir()) / "streamlit_drone_app" / subdir
    base_dir.mkdir(parents=True, exist_ok=True)

    file_path = base_dir / uploaded_file.name

    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    return str(file_path)

# =========================================================
# UI
# =========================================================

analysis_type = st.radio(
    "🔍 Kies type analyse:",
    ["Orthomosaic", "Segmentatie-maskers"]
)

# =========================================================
# ORTHOMOSAIC ANALYSE
# =========================================================

if analysis_type == "Orthomosaic":

    uploaded_file = st.file_uploader(
        "📁 Upload orthomosaic (.tif)",
        type=["tif", "tiff"]
    )

    index_choice = st.selectbox(
        "📈 Kies index:",
        ["Excess Green (ExG)", "Excess Red (ExR)"]
    )

    preview_mode = st.checkbox(
        "Gebruik preview (sneller, lagere resolutie)",
        value=True
    )

    if "index" not in st.session_state:
        st.session_state.index = None

    if uploaded_file is not None:
        tif_path = save_uploaded_file(uploaded_file, subdir="tifs")
        st.caption(f"📂 Bestand opgeslagen als: `{tif_path}`")

    # -----------------------------------------------------
    # Berekening
    # -----------------------------------------------------

    if st.button("🚀 Start berekening", disabled=uploaded_file is None):

        with rasterio.open(tif_path) as src:

            if src.count < 3:
                st.error("❌ TIFF heeft minder dan 3 banden (geen RGB).")
                st.stop()

            width, height = src.width, src.height
            st.write(f"📐 Afmetingen: {width} × {height}")

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            start = time.time()

            # ================= PREVIEW ======================
            if preview_mode:
                step = 20
                out_h = max(1, height // step)
                out_w = max(1, width // step)

                img = src.read(
                    out_shape=(src.count, out_h, out_w),
                    resampling=rasterio.enums.Resampling.bilinear
                )

                R = img[0].astype(np.float32)
                G = img[1].astype(np.float32)
                B = img[2].astype(np.float32)

                sumRGB = R + G + B
                sumRGB = np.where(sumRGB == 0, 1e-6, sumRGB)

                if index_choice == "Excess Green (ExG)":
                    index = (2 * G - R - B) / sumRGB
                else:
                    index = (1.4 * R - G) / sumRGB

                progress_bar.progress(1.0)
                status_text.text("✅ Preview berekend.")

            # ================= VOLLEDIG =====================
            else:
                index = np.zeros((height, width), dtype=np.float32)
                total_blocks = sum(1 for _ in src.block_windows(1))

                for i, (_, window) in enumerate(src.block_windows(1)):
                    img = src.read(window=window)

                    R = img[0].astype(np.float32)
                    G = img[1].astype(np.float32)
                    B = img[2].astype(np.float32)

                    sumRGB = R + G + B
                    sumRGB = np.where(sumRGB == 0, 1e-6, sumRGB)

                    if index_choice == "Excess Green (ExG)":
                        chunk = (2 * G - R - B) / sumRGB
                    else:
                        chunk = (1.4 * R - G) / sumRGB

                    r0, c0 = window.row_off, window.col_off
                    index[r0:r0 + window.height, c0:c0 + window.width] = chunk

                    progress = (i + 1) / total_blocks
                    progress_bar.progress(progress)

                    elapsed = time.time() - start
                    est_total = elapsed / (i + 1) * total_blocks
                    remaining = est_total - elapsed

                    status_text.text(
                        f"{progress*100:.1f}% voltooid – nog ~{remaining/60:.1f} min"
                    )

                status_text.text("✅ Berekening voltooid.")

        st.session_state.index = index
        st.success("🎉 Berekening klaar!")

    # -----------------------------------------------------
    # Visualisatie
    # -----------------------------------------------------

    if st.session_state.index is not None:
        index = st.session_state.index

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"🖼️ {index_choice} – Heatmap")
            fig, ax = plt.subplots()
            im = ax.imshow(index, cmap="viridis")
            fig.colorbar(im, ax=ax, label=index_choice)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("📊 Histogram")
            fig2, ax2 = plt.subplots()
            ax2.hist(index.flatten(), bins=100, color="gray", edgecolor="black")
            ax2.set_xlabel("Indexwaarde")
            ax2.set_ylabel("Aantal pixels")
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("🎚️ Filter outliers")
        min_val, max_val = float(index.min()), float(index.max())

        lower, upper = st.slider(
            "Bereik",
            min_val,
            max_val,
            (min_val, max_val)
        )

        filtered = np.clip(index, lower, upper)

        fig3, ax3 = plt.subplots()
        im2 = ax3.imshow(filtered, cmap="viridis")
        fig3.colorbar(im2, ax=ax3, label=f"{index_choice} (gefilterd)")
        ax3.axis("off")
        st.pyplot(fig3)
        plt.close(fig3)

# =========================================================
# SEGMENTATIE MASKERS
# =========================================================

elif analysis_type == "Segmentatie-maskers":

    uploaded_gpkg = st.file_uploader(
        "📁 Upload GeoPackage (.gpkg)",
        type=["gpkg"]
    )

    property_choice = st.selectbox(
        "📈 Kies eigenschap:",
        ["Hoogte", "Diameter", "ExG", "ExR"]
    )

    if uploaded_gpkg is not None:

        gpkg_path = save_uploaded_file(uploaded_gpkg, subdir="gpkg")

        @st.cache_data(show_spinner=True)
        def load_gpkg(path):
            return gpd.read_file(path)

        gdf = load_gpkg(gpkg_path)

        col_mapping = {
            "Hoogte": "height_p95",
            "Diameter": "diameter",
            "ExG": "ExG_median",
            "ExR": "ExR_median"
        }

        col = col_mapping[property_choice]

        if col not in gdf.columns:
            st.error(f"❌ Kolom '{col}' niet gevonden.")
            st.stop()

        values = gdf[col].values

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Histogram")
            fig, ax = plt.subplots()
            ax.hist(values, bins=50, color="gray", edgecolor="black")
            ax.set_xlabel(property_choice)
            ax.set_ylabel("Aantal segmenten")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("🎚️ Filter")
            min_val, max_val = float(values.min()), float(values.max())
            lower, upper = st.slider(
                "Bereik",
                min_val,
                max_val,
                (min_val, max_val)
            )

        filtered = gdf[(gdf[col] >= lower) & (gdf[col] <= upper)]

        st.subheader(f"🗺️ Kaart ({len(filtered)} segmenten)")
        fig_map, ax_map = plt.subplots(figsize=(20, 10), dpi=150)

        if filtered.crs is None:
            filtered = filtered.set_crs(epsg=32631)

        filtered_3857 = filtered.to_crs(epsg=3857)
        points = filtered_3857.copy()
        points["geometry"] = points.centroid

        points.plot(ax=ax_map, color="red", markersize=8, alpha=0.8)

        ctx.add_basemap(
            ax=ax_map,
            source=ctx.providers.Esri.WorldImagery,
            crs=filtered_3857.crs.to_string(),
            zoom=20
        )

        ax_map.set_axis_off()
        st.pyplot(fig_map)
        plt.close(fig_map)
