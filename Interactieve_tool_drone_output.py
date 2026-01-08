import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import tempfile
from pathlib import Path
import os
import time

st.set_page_config(layout="wide")
st.title("🌿 Analyse Drone Output")

# =========================================================
# CONFIG — HIER MAG JE AAN ZITTEN
# =========================================================

MAX_TIF_SIZE_MB = 300   # <<< HARD LIMIET (veilig voor Streamlit Cloud)
PREVIEW_FACTOR = 25

# =========================================================
# HELPERS
# =========================================================

def save_uploaded_file(uploaded_file, subdir):
    base = Path(tempfile.gettempdir()) / "streamlit_drone_app" / subdir
    base.mkdir(parents=True, exist_ok=True)
    path = base / uploaded_file.name
    if not path.exists():
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return path

# =========================================================
# UI
# =========================================================

analysis_type = st.radio(
    "🔍 Kies type analyse:",
    ["Orthomosaic", "Segmentatie-maskers"]
)

# =========================================================
# ORTHOMOSAIC — CRASH-PROOF
# =========================================================

if analysis_type == "Orthomosaic":

    uploaded_file = st.file_uploader(
        "📁 Upload orthomosaic (.tif)",
        type=["tif", "tiff"]
    )

    if uploaded_file is None:
        st.info("Upload een orthomosaic om te starten.")
        st.stop()

    # -------------------------------
    # FILESIZE CHECK (KRITIEK)
    # -------------------------------

    size_mb = uploaded_file.size / (1024 * 1024)

    st.caption(f"📦 Bestandsgrootte: {size_mb:.1f} MB")

    if size_mb > MAX_TIF_SIZE_MB:
        st.error(
            f"❌ Bestand is te groot voor web-analyse ({size_mb:.0f} MB).\n\n"
            f"Maximum toegestaan: {MAX_TIF_SIZE_MB} MB.\n\n"
            "➡️ Gebruik preview-export of voer analyse lokaal uit."
        )
        st.stop()

    # -------------------------------
    # PAS NU OPSLAAN & OPENEN
    # -------------------------------

    tif_path = save_uploaded_file(uploaded_file, "tifs")

    index_choice = st.selectbox(
        "📈 Kies index:",
        ["Excess Green (ExG)", "Excess Red (ExR)"]
    )

    if st.button("🚀 Bereken preview-index"):

        with rasterio.open(tif_path) as src:

            if src.count < 3:
                st.error("❌ TIFF heeft minder dan 3 banden.")
                st.stop()

            width, height = src.width, src.height
            st.caption(f"📐 Resolutie: {width} × {height}")

            out_h = max(1, height // PREVIEW_FACTOR)
            out_w = max(1, width // PREVIEW_FACTOR)

            start = time.time()

            img = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear
            )

            R = img[0].astype(np.float32)
            G = img[1].astype(np.float32)
            B = img[2].astype(np.float32)

            sumRGB = R + G + B
            sumRGB[sumRGB == 0] = 1e-6

            if index_choice == "Excess Green (ExG)":
                index = (2 * G - R - B) / sumRGB
            else:
                index = (1.4 * R - G) / sumRGB

            st.session_state.index = index
            elapsed = time.time() - start

            st.success(f"✅ Preview berekend in {elapsed:.2f} s")

    # -------------------------------
    # VISUALISATIE
    # -------------------------------

    if "index" in st.session_state:

        index = st.session_state.index

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            im = ax.imshow(index, cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.hist(index.flatten(), bins=100, color="gray")
            st.pyplot(fig2)
            plt.close(fig2)


# =========================================================
# SEGMENTATIE MASKERS (ONGEWIJZIGD & STABIEL)
# =========================================================

elif analysis_type == "Segmentatie-maskers":

    uploaded_gpkg = st.file_uploader(
        "📁 Upload GeoPackage (.gpkg)",
        type=["gpkg"]
    )

    if uploaded_gpkg is None:
        st.stop()

    gpkg_path = save_uploaded_file(uploaded_gpkg, "gpkg")

    @st.cache_data(show_spinner=True)
    def load_gpkg(path):
        return gpd.read_file(path)

    gdf = load_gpkg(gpkg_path)

    property_choice = st.selectbox(
        "📈 Kies eigenschap:",
        ["Hoogte", "Diameter", "ExG", "ExR"]
    )

    col_mapping = {
        "Hoogte": "height_p95",
        "Diameter": "diameter",
        "ExG": "ExG_median",
        "ExR": "ExR_median"
    }

    col = col_mapping[property_choice]

    if col not in gdf.columns:
        st.error(f"Kolom '{col}' niet gevonden.")
        st.stop()

    values = gdf[col].values

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(values, bins=50, color="gray")
        ax.set_xlabel(property_choice)
        ax.set_ylabel("Aantal segmenten")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        lower, upper = st.slider(
            "Filter",
            float(values.min()),
            float(values.max()),
            (float(values.min()), float(values.max()))
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

