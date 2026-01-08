import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import tempfile
from pathlib import Path
import numpy as np

st.set_page_config(layout="wide")
st.title("🌿 Analyse Segmentatie-maskers")

# =========================================================
# Helper: upload veilig opslaan
# =========================================================

def save_uploaded_file(uploaded_file):
    base = Path(tempfile.gettempdir()) / "streamlit_segmentatie"
    base.mkdir(parents=True, exist_ok=True)
    path = base / uploaded_file.name

    if not path.exists():
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    return str(path)

# =========================================================
# Upload GeoPackage
# =========================================================

uploaded_gpkg = st.file_uploader(
    "📁 Upload GeoPackage (.gpkg)",
    type=["gpkg"]
)

if uploaded_gpkg is None:
    st.info("Upload een GeoPackage om te starten.")
    st.stop()

gpkg_path = save_uploaded_file(uploaded_gpkg)

# =========================================================
# Laad GeoDataFrame (gecached)
# =========================================================

@st.cache_data(show_spinner=True)
def load_gpkg(path):
    return gpd.read_file(path)

gdf = load_gpkg(gpkg_path)

st.success(f"✅ {len(gdf)} segmenten geladen")

# =========================================================
# Kies eigenschap
# =========================================================

property_choice = st.selectbox(
    "📈 Kies eigenschap om te analyseren:",
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
    st.error(f"❌ Kolom '{col}' niet gevonden in GeoPackage.")
    st.stop()

values = gdf[col].dropna().values

# =========================================================
# Histogram
# =========================================================

st.subheader(f"📊 Histogram van {property_choice}")

fig_hist, ax_hist = plt.subplots()
ax_hist.hist(values, bins=50, color="gray", edgecolor="black")
ax_hist.set_xlabel(property_choice)
ax_hist.set_ylabel("Aantal segmenten")
st.pyplot(fig_hist)
plt.close(fig_hist)

# =========================================================
# Filters
# =========================================================

st.subheader("🎚️ Filter segmenten")

min_val, max_val = float(values.min()), float(values.max())

lower, upper = st.slider(
    "Selecteer bereik",
    min_val,
    max_val,
    (min_val, max_val)
)

filtered = gdf[(gdf[col] >= lower) & (gdf[col] <= upper)]

st.caption(f"🔎 {len(filtered)} van {len(gdf)} segmenten geselecteerd")

# =========================================================
# Kaartweergave
# =========================================================

st.subheader("🗺️ Kaartweergave")

fig_map, ax_map = plt.subplots(figsize=(20, 10), dpi=150)

try:
    # CRS check
    if filtered.crs is None:
        st.warning("Geen CRS gevonden — ingesteld op EPSG:32631.")
        filtered = filtered.set_crs(epsg=32631)

    # Projecteer naar Web Mercator
    filtered_3857 = filtered.to_crs(epsg=3857)

    # Gebruik centroiden voor performance
    points = filtered_3857.copy()
    points["geometry"] = points.centroid

    points.plot(
        ax=ax_map,
        color="red",
        markersize=8,
        alpha=0.8
    )

    ctx.add_basemap(
        ax=ax_map,
        source=ctx.providers.Esri.WorldImagery,
        crs=filtered_3857.crs.to_string(),
        zoom=20
    )

    ax_map.set_axis_off()
    st.pyplot(fig_map)
    plt.close(fig_map)

except Exception as e:
    st.warning(f"⚠️ Basemap mislukt: {e}")

    filtered.plot(
        column=col,
        ax=ax_map,
        cmap="terrain",
        legend=True
    )

    ax_map.set_axis_off()
    st.pyplot(fig_map)
    plt.close(fig_map)

