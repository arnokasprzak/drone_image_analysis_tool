import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import tempfile
from pathlib import Path
import time

st.set_page_config(layout="wide")
st.title("🌿 Analyse Drone Output")

# =========================================================
# Helpers
# =========================================================

def save_uploaded_file(uploaded_file, subdir):
    base = Path(tempfile.gettempdir()) / "streamlit_drone_app" / subdir
    base.mkdir(parents=True, exist_ok=True)
    path = base / uploaded_file.name
    if not path.exists():
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return str(path)

# =========================================================
# UI
# =========================================================

analysis_type = st.radio(
    "🔍 Kies type analyse:",
    ["Orthomosaic", "Segmentatie-maskers"]
)

# =========================================================
# ORTHOMOSAIC (PREVIEW-ONLY, STABIEL)
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

    preview_factor = st.slider(
        "🔽 Downsampling factor (lager = scherper, hoger = sneller)",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

    if uploaded_file is None:
        st.info("Upload een orthomosaic om te starten.")
        st.stop()

    tif_path = save_uploaded_file(uploaded_file, "tifs")

    if st.button("🚀 Bereken index (preview)"):
        with rasterio.open(tif_path) as src:

            if src.count < 3:
                st.error("❌ TIFF heeft minder dan 3 banden.")
                st.stop()

            width, height = src.width, src.height
            st.caption(f"📐 Originele resolutie: {width} × {height}")

            # --- VEILIGE PREVIEW RESOLUTIE ---
            out_h = max(1, height // preview_factor)
            out_w = max(1, width // preview_factor)

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
            st.session_state.preview_shape = (out_h, out_w)

            elapsed = time.time() - start
            st.success(f"✅ Preview berekend in {elapsed:.2f} s")

    # =====================================================
    # VISUALISATIE
    # =====================================================

    if "index" in st.session_state:

        index = st.session_state.index
        out_h, out_w = st.session_state.preview_shape

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"🖼️ {index_choice} – Preview ({out_w}×{out_h})")
            fig, ax = plt.subplots()
            im = ax.imshow(index, cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("📊 Histogram")
            fig2, ax2 = plt.subplots()
            ax2.hist(index.flatten(), bins=100, color="gray")
            ax2.set_xlabel("Indexwaarde")
            ax2.set_ylabel("Aantal pixels")
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("🎚️ Filter outliers")

        min_val = float(np.nanmin(index))
        max_val = float(np.nanmax(index))

        lower, upper = st.slider(
            "Bereik",
            min_val,
            max_val,
            (min_val, max_val)
        )

        filtered = np.clip(index, lower, upper)

        fig3, ax3 = plt.subplots()
        im2 = ax3.imshow(filtered, cmap="viridis")
        fig3.colorbar(im2, ax=ax3)
        ax3.axis("off")
        st.pyplot(fig3)
        plt.close(fig3)

        st.info(
            "ℹ️ Dit is een **preview-analyse**.\n\n"
            "Volledige resolutie verwerking wordt bewust geblokkeerd\n"
            "om crashes en 502-errors te voorkomen."
        )

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
