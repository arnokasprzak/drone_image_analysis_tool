import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import geopandas as gpd
import contextily as ctx
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Drone Output Analysis")

# Path normalization helper
def normalize_path(path_str: str) -> str:
    """
    Cleans and normalizes file paths:
    - removes quotes
    - removes hidden CR/LF characters
    - converts to OS-correct path format
    """
    if not path_str:
        return ""
    path_str = path_str.strip().strip('"').strip("'")
    path_str = path_str.replace("\r", "").replace("\n", "")
    return os.path.normpath(path_str)


analysis_type = st.radio("Select analysis type:", ["Orthomosaic", "Segmentation masks"])


# Orthomosaic analysis
if analysis_type == "Orthomosaic":

    file_path_input = st.text_input("Path to orthomosaic (.tif):", "")
    file_path = normalize_path(file_path_input)

    index_choice = st.selectbox("Select index:", ["Excess Green (ExG)", "Excess Red (ExR)"])
    preview_mode = st.checkbox("Use preview mode (lower resolution, faster)", value=True)

    if "index" not in st.session_state:
        st.session_state.index = None

    if st.button("Start calculation"):
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            st.stop()

        with rasterio.open(file_path) as src:
            width, height = src.width, src.height
            st.write(f"Dimensions: {width} x {height}")

            progress_bar = st.progress(0)
            status_text = st.empty()
            start = time.time()

            if preview_mode:
                step = 20
                img = src.read(
                    out_shape=(src.count, height // step, width // step),
                    resampling=rasterio.enums.Resampling.bilinear
                )
                R, G, B = img[0].astype(np.float32), img[1].astype(np.float32), img[2].astype(np.float32)
                sumRGB = np.where(R + G + B == 0, 1e-6, R + G + B)

                if index_choice == "Excess Green (ExG)":
                    index = (2 * G - R - B) / sumRGB
                else:
                    index = (1.4 * R - G) / sumRGB

                progress_bar.progress(1.0)
                status_text.text("Preview calculated.")

            else:
                index = np.zeros((height, width), dtype=np.float32)
                total_blocks = sum(1 for _ in src.block_windows(1))

                for i, (ji, window) in enumerate(src.block_windows(1)):
                    img = src.read(window=window)
                    R, G, B = img[0].astype(np.float32), img[1].astype(np.float32), img[2].astype(np.float32)
                    sumRGB = np.where(R + G + B == 0, 1e-6, R + G + B)

                    if index_choice == "Excess Green (ExG)":
                        chunk = (2 * G - R - B) / sumRGB
                    else:
                        chunk = (1.4 * R - G) / sumRGB

                    index[
                        window.row_off:window.row_off + window.height,
                        window.col_off:window.col_off + window.width
                    ] = chunk

                    progress = (i + 1) / total_blocks
                    progress_bar.progress(progress)

                    elapsed = time.time() - start
                    est_total = elapsed / (i + 1) * total_blocks
                    remaining = est_total - elapsed
                    status_text.text(f"{progress*100:.1f}% completed, approximately {remaining/60:.1f} minutes remaining")

                status_text.text("Calculation completed.")

        st.session_state.index = index
        st.success("Calculation finished. You can now use the visualization tools.")

    if st.session_state.index is not None:
        index = st.session_state.index

        st.subheader(f"{index_choice} Heatmap")
        fig, ax = plt.subplots()
        cax = ax.imshow(index, cmap="viridis")
        fig.colorbar(cax, ax=ax)
        ax.axis("off")
        st.pyplot(fig)

        st.subheader("Histogram of index values")
        fig2, ax2 = plt.subplots()
        ax2.hist(index.flatten(), bins=100, color="gray", edgecolor="black")
        st.pyplot(fig2)

        st.subheader("Filter outliers")
        min_val, max_val = float(index.min()), float(index.max())
        lower = st.slider("Lower bound", min_val, max_val, min_val)
        upper = st.slider("Upper bound", min_val, max_val, max_val)

        filtered = np.clip(index, lower, upper)

        fig3, ax3 = plt.subplots()
        cax2 = ax3.imshow(filtered, cmap="viridis")
        fig3.colorbar(cax2, ax=ax3)
        ax3.axis("off")
        st.pyplot(fig3)


# Segmentation mask analysis
elif analysis_type == "Segmentation masks":

    st.info("Select the property you want to analyze.")

    property_choice = st.selectbox(
        "Select property:",
        ["Height", "Diameter", "ExG", "ExR"]
    )

    gpkg_input = st.text_input(
        "Path to GeoPackage (.gpkg):",
        placeholder="C:/data/segmentation.gpkg"
    )

    if gpkg_input:
        gpkg_path = normalize_path(gpkg_input)
        st.caption(f"Using path: {gpkg_path}")

        if not Path(gpkg_path).exists():
            st.error("File path does not exist.")
            st.stop()

        if "gdf" not in st.session_state or st.session_state.get("gdf_path") != gpkg_path:
            load_bar = st.progress(0)
            load_status = st.empty()

            load_status.text("Loading GeoPackage...")
            load_bar.progress(0.3)

            gdf = gpd.read_file(gpkg_path)

            load_bar.progress(0.9)
            load_status.text(f"{len(gdf)} segments loaded")

            st.session_state.gdf = gdf
            st.session_state.gdf_path = gpkg_path

            load_bar.progress(1.0)
        else:
            gdf = st.session_state.gdf

        col_mapping = {
            "Height": "height_p95",
            "Diameter": "diameter",
            "ExG": "ExG_median",
            "ExR": "ExR_median"
        }

        col = col_mapping[property_choice]

        if col not in gdf.columns:
            st.error(f"Column '{col}' not found in GeoPackage.")
            st.stop()

        values = gdf[col].astype(float)

        st.subheader(f"Histogram of {property_choice}")
        fig, ax = plt.subplots()
        ax.hist(values, bins=50, color="gray", edgecolor="black")
        st.pyplot(fig)

        st.subheader("Filter segments")
        lower = st.slider("Lower bound", float(values.min()), float(values.max()), float(values.min()))
        upper = st.slider("Upper bound", float(values.min()), float(values.max()), float(values.max()))

        filtered = gdf[(values >= lower) & (values <= upper)]

        st.subheader(f"Map view of filtered segments ({len(filtered)})")

        fig_map, ax_map = plt.subplots(figsize=(20, 10), dpi=150)

        try:
            if filtered.crs is None:
                filtered = filtered.set_crs(epsg=32631)

            filtered_3857 = filtered.to_crs(epsg=3857)
            filtered_points = filtered_3857.copy()
            filtered_points["geometry"] = filtered_points.centroid

            filtered_points.plot(ax=ax_map, color="red", markersize=10, alpha=0.8)

            ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, crs=filtered_3857.crs.to_string(), zoom=20)

            ax_map.set_axis_off()
            st.pyplot(fig_map)

        except Exception as e:
            st.warning(f"Basemap failed: {e}")
            filtered.plot(column=col, ax=ax_map, cmap="terrain", legend=True)
            ax_map.set_axis_off()
            st.pyplot(fig_map)
