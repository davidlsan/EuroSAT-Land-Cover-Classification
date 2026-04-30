from pathlib import Path
import sys

import folium
import streamlit as st
from branca.element import MacroElement, Template
from folium.plugins import Draw, MeasureControl
from streamlit_folium import st_folium

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.model_utils import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    load_rgb_model,
    predict_topk,
)
from app.tile_utils import (  # noqa: E402
    TileFetchError,
    bbox_scale_status,
    bbox_size_meters,
    choose_zoom_level,
    extract_bbox_from_geojson,
    fetch_bbox_image,
    size_warning_for_bbox,
)


ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Farmlands starter pos, good for examples right of the bat
DEFAULT_MAP_CENTER = [50, 10]
DEFAULT_MAP_ZOOM = 15


class SingleRectangleLimiter(MacroElement):
    _template = Template(
        """
        {% macro script(this, kwargs) %}
        {{ this.map_name }}.on('draw:created', function(e) {
            {{ this.drawn_items_name }}.clearLayers();
            {{ this.drawn_items_name }}.addLayer(e.layer);
        });
        {% endmacro %}
        """
    )

    def __init__(self, map_name: str, drawn_items_name: str):
        super().__init__()
        self._name = "SingleRectangleLimiter"
        self.map_name = map_name
        self.drawn_items_name = drawn_items_name


@st.cache_resource(show_spinner="Loading RGB ResNet-50 model...")
def get_model():
    return load_rgb_model(DEFAULT_CHECKPOINT_PATH)


def build_map(
    drawing: dict | None = None,
    center: list[float] | None = None,
    zoom: int = DEFAULT_MAP_ZOOM,
) -> folium.Map:
    fmap = folium.Map(
        location=center or DEFAULT_MAP_CENTER,
        zoom_start=zoom,
        min_zoom=13,
        max_zoom=18,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=ESRI_WORLD_IMAGERY,
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and GIS User Community",
        name="Esri World Imagery",
        overlay=False,
        control=True,
    ).add_to(fmap)
    draw_control = Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": {
                "shapeOptions": {
                    "color": "#ff7800",
                    "weight": 2,
                    "fillOpacity": 0.05,
                }
            },
        },
        edit_options={"edit": True, "remove": True},
    )
    draw_control.add_to(fmap)
    SingleRectangleLimiter(
        map_name=fmap.get_name(),
        drawn_items_name=f"drawnItems_{draw_control.get_name()}",
    ).add_to(fmap)
    MeasureControl(
        position="bottomleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="sqmeters",
    ).add_to(fmap)

    if drawing:
        folium.GeoJson(
            drawing,
            name="Last selected rectangle",
            style_function=lambda _: {
                "color": "#00bcd4",
                "weight": 2,
                "fillOpacity": 0.04,
            },
        ).add_to(fmap)

    return fmap


def render_sidebar() -> None:
    st.sidebar.header("How to use")
    st.sidebar.markdown(
        "1. Pan and zoom to a land area.\n"
        "2. Select the rectangle tool on the map.\n"
        "3. Use the map scale bar or measure tool as a guide.\n"
        "4. Draw a near-square box roughly 500m-1km on a side.\n"
        "5. Review the fetched image and top predictions."
    )
    st.sidebar.warning(
        "This model was trained on EuroSAT-RGB tiles (~64x64 pixels, ~640m on a side). "
        "Predictions on arbitrary map regions are illustrative; for best results, draw "
        "a rectangle of roughly 500m-1km on a side over land."
    )
    st.sidebar.header("EuroSAT Classes")
    for class_name in CLASS_NAMES:
        st.sidebar.write(f"- {class_name}")


def render_prediction(drawing) -> None:
    try:
        bbox = extract_bbox_from_geojson(drawing)
    except ValueError as exc:
        st.error(str(exc))
        return

    width_m, height_m = bbox_size_meters(bbox)
    scale_state, scale_message = bbox_scale_status(bbox)

    metric_col, scale_col, zoom_col = st.columns(3)
    metric_col.metric("Rectangle width", format_meters(width_m))
    scale_col.metric("Rectangle height", format_meters(height_m))
    zoom_col.metric("Tile zoom", choose_zoom_level(bbox))

    warning = size_warning_for_bbox(bbox)
    if warning:
        st.warning(warning)
        return
    if scale_state == "invalid":
        st.warning(scale_message)
        return
    if scale_state == "good":
        st.success(scale_message)
    else:
        st.warning(scale_message)

    try:
        with st.spinner("Fetching Esri imagery tiles..."):
            image = fetch_bbox_image(bbox)
    except TileFetchError as exc:
        st.error(f"Could not fetch satellite imagery for this rectangle. {exc}")
        return

    try:
        model = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    with st.spinner("Running RGB land cover inference..."):
        top_predictions = predict_topk(model, image, top_k=3)

    preview_col, prediction_col = st.columns([1, 1])
    with preview_col:
        st.subheader("Fetched Tile Preview")
        st.image(image, caption="Cropped Esri World Imagery", width='stretch')

    with prediction_col:
        st.subheader("Prediction")
        best_class, best_prob = top_predictions[0]
        st.metric("Predicted class", best_class, f"{best_prob:.1%}")
        st.write("Top-3 class probabilities")
        st.bar_chart(
            {"Probability": {name: prob for name, prob in top_predictions}},
            horizontal=True,
        )


def format_meters(value: float) -> str:
    if value >= 1_000:
        return f"{value / 1_000:.2f} km"
    return f"{value:.0f} m"


def get_drawing(data: dict | None) -> dict | None:
    incoming_drawing = data.get("last_active_drawing") if data else None
    current_drawing = st.session_state.get("last_drawing")
    if incoming_drawing and incoming_drawing != current_drawing:
        st.session_state["last_drawing"] = incoming_drawing
        st.session_state["map_center"] = drawing_center(incoming_drawing)
    return st.session_state.get("last_drawing")


def drawing_center(drawing: dict) -> list[float]:
    bbox = extract_bbox_from_geojson(drawing)
    return [
        (bbox.south + bbox.north) / 2.0,
        (bbox.west + bbox.east) / 2.0,
    ]


def reset_map() -> None:
    st.session_state["map_version"] = st.session_state.get("map_version", 0) + 1


def clear_selection() -> None:
    st.session_state.pop("last_drawing", None)
    st.session_state.pop("map_center", None)
    reset_map()


def main() -> None:
    st.set_page_config(
        page_title="EuroSAT RGB Land Cover Classifier",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("EuroSAT Land Cover Classifier (RGB Model)")
    st.markdown(
        "This demo classifies RGB satellite imagery into the 10 EuroSAT land cover "
        "classes using a ResNet-50."
        """
        <style>
        html {
            overflow-y: scroll;
        }
        .block-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_sidebar()

    previous_drawing = st.session_state.get("last_drawing")
    map_version = st.session_state.get("map_version", 0)
    map_center = st.session_state.get("map_center", DEFAULT_MAP_CENTER)
    data = st_folium(
        build_map(previous_drawing, center=map_center),
        key=f"eurosat-rgb-map-{map_version}",
        height=600,
        width='stretch',
        returned_objects=["last_active_drawing"],
    )

    drawing = get_drawing(data)
    if drawing:
        st.button("Reset rectangle", on_click=clear_selection)
        render_prediction(drawing)
    else:
        st.info(
            "Draw a near-square rectangle on the map to fetch imagery and run the classifier. "
            "Aim for 500m-1km on each side, similar to the original EuroSAT-RGB tiles."
        )


if __name__ == "__main__":
    main()
