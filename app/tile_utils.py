import math
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import requests
import streamlit as st
from PIL import Image


ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TILE_SIZE = 256
USER_AGENT = "eurosat-rgb-streamlit-demo/1.0"
EUROSAT_TARGET_MIN_M = 500
EUROSAT_TARGET_MAX_M = 1_000
EUROSAT_ACCEPTABLE_MIN_M = 250
EUROSAT_ACCEPTABLE_MAX_M = 1_500
EUROSAT_MAX_ASPECT_RATIO = 2.0


class TileFetchError(RuntimeError):
    """Raised when an Esri imagery tile cannot be fetched."""


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float


@dataclass(frozen=True)
class TileRange:
    zoom: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int


def extract_bbox_from_geojson(drawing: dict[str, Any]) -> BBox:
    """Extract a lon/lat bbox from a Folium Draw GeoJSON rectangle."""
    geometry = drawing.get("geometry", {})
    coordinates = geometry.get("coordinates")
    if geometry.get("type") != "Polygon" or not coordinates:
        raise ValueError("Expected a drawn rectangle polygon.")

    ring = coordinates[0]
    lons = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)
    if west == east or south == north:
        raise ValueError("The drawn rectangle has no area.")

    return BBox(west=west, south=south, east=east, north=north)


def lonlat_to_tile_fraction(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Convert lon/lat to fractional XYZ tile coordinates.

    Uses the OpenStreetMap slippy-map convention:
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    XYZ y coordinates start at 0 at the northern edge of the world.
    """
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (
        1.0
        - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi
    ) / 2.0 * n
    return x, y


def bbox_to_tile_range(bbox: BBox, zoom: int) -> TileRange:
    """Return the inclusive XYZ tile range covering a lon/lat bbox."""
    max_tile = (2**zoom) - 1
    x_west, y_north = lonlat_to_tile_fraction(bbox.west, bbox.north, zoom)
    x_east, y_south = lonlat_to_tile_fraction(bbox.east, bbox.south, zoom)

    x_min = max(0, min(max_tile, math.floor(x_west)))
    x_max = max(0, min(max_tile, math.floor(x_east)))
    y_min = max(0, min(max_tile, math.floor(y_north)))
    y_max = max(0, min(max_tile, math.floor(y_south)))

    return TileRange(
        zoom=zoom,
        x_min=min(x_min, x_max),
        x_max=max(x_min, x_max),
        y_min=min(y_min, y_max),
        y_max=max(y_min, y_max),
    )


def choose_zoom_level(bbox: BBox) -> int:
    """Choose a tile zoom; EuroSAT-scale rectangles use zoom 14-15."""
    width_m, height_m = bbox_size_meters(bbox)
    max_side_m = max(width_m, height_m)
    if max_side_m <= 1_000:
        return 15
    if max_side_m <= 5_000:
        return 14
    return 13


def bbox_size_meters(bbox: BBox) -> tuple[float, float]:
    """Approximate bbox width and height in meters."""
    mid_lat = (bbox.north + bbox.south) / 2.0
    width_m = _haversine_meters(bbox.west, mid_lat, bbox.east, mid_lat)
    height_m = _haversine_meters(bbox.west, bbox.south, bbox.west, bbox.north)
    return width_m, height_m


def size_warning_for_bbox(bbox: BBox) -> str | None:
    """Return a user-facing warning for rectangles outside the demo range."""
    width_m, height_m = bbox_size_meters(bbox)
    min_side_m = min(width_m, height_m)
    max_side_m = max(width_m, height_m)
    if min_side_m < 50:
        return "This rectangle is very small. Draw at least about 50m on a side."
    if max_side_m > 5_000:
        return "This rectangle is very large. Draw at most about 5km on a side."
    return None


def bbox_scale_status(bbox: BBox) -> tuple[str, str]:
    """Classify whether a bbox is close enough to EuroSAT-RGB tile scale."""
    width_m, height_m = bbox_size_meters(bbox)
    min_side_m = min(width_m, height_m)
    max_side_m = max(width_m, height_m)
    aspect_ratio = max_side_m / min_side_m

    if min_side_m < EUROSAT_ACCEPTABLE_MIN_M:
        return (
            "invalid",
            "This rectangle is too small for a useful EuroSAT-style prediction. "
            "Draw closer to 500m-1km on each side.",
        )
    if max_side_m > EUROSAT_ACCEPTABLE_MAX_M:
        return (
            "invalid",
            "This rectangle is too large for this EuroSAT-style demo. "
            "Zoom in and draw closer to 500m-1km on each side.",
        )
    if aspect_ratio > EUROSAT_MAX_ASPECT_RATIO:
        return (
            "invalid",
            "This rectangle is too stretched. Draw a more square region, like the original EuroSAT tiles.",
        )
    if (
        EUROSAT_TARGET_MIN_M <= min_side_m
        and max_side_m <= EUROSAT_TARGET_MAX_M
    ):
        return (
            "good",
            "Great scale: this is close to the original EuroSAT-RGB tile footprint.",
        )
    return (
        "usable",
        "Usable, but not ideal. For the most trustworthy demo result, draw 500m-1km on each side.",
    )


def fetch_bbox_image(bbox: BBox, zoom: int | None = None) -> Image.Image:
    """Fetch Esri XYZ tiles for a bbox, stitch them, and crop to the bbox."""
    zoom = choose_zoom_level(bbox) if zoom is None else zoom
    tile_range = bbox_to_tile_range(bbox, zoom)

    stitched = Image.new(
        "RGB",
        (
            (tile_range.x_max - tile_range.x_min + 1) * TILE_SIZE,
            (tile_range.y_max - tile_range.y_min + 1) * TILE_SIZE,
        ),
    )

    for x in range(tile_range.x_min, tile_range.x_max + 1):
        for y in range(tile_range.y_min, tile_range.y_max + 1):
            tile = fetch_esri_tile(zoom, x, y)
            stitched.paste(
                tile,
                (
                    (x - tile_range.x_min) * TILE_SIZE,
                    (y - tile_range.y_min) * TILE_SIZE,
                ),
            )
            time.sleep(0.05)

    crop_box = _bbox_crop_box(bbox, tile_range, stitched.size)
    cropped = stitched.crop(crop_box)
    if cropped.width <= 0 or cropped.height <= 0:
        raise TileFetchError("The fetched imagery crop was empty.")
    return cropped


@st.cache_data(show_spinner=False)
def fetch_esri_tile(zoom: int, x: int, y: int) -> Image.Image:
    """Download one Esri World Imagery XYZ tile."""
    url = ESRI_TILE_URL.format(z=zoom, x=x, y=y)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TileFetchError(f"Could not download imagery tile z{zoom}/{x}/{y}.") from exc

    try:
        return Image.open(BytesIO(response.content)).convert("RGB")
    except OSError as exc:
        raise TileFetchError(f"Downloaded imagery tile z{zoom}/{x}/{y} was invalid.") from exc


def _bbox_crop_box(
    bbox: BBox, tile_range: TileRange, stitched_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    zoom = tile_range.zoom
    west_px, north_px = _lonlat_to_global_pixel(bbox.west, bbox.north, zoom)
    east_px, south_px = _lonlat_to_global_pixel(bbox.east, bbox.south, zoom)
    origin_x = tile_range.x_min * TILE_SIZE
    origin_y = tile_range.y_min * TILE_SIZE

    left = math.floor(west_px - origin_x)
    top = math.floor(north_px - origin_y)
    right = math.ceil(east_px - origin_x)
    bottom = math.ceil(south_px - origin_y)

    width, height = stitched_size
    return (
        max(0, min(width, left)),
        max(0, min(height, top)),
        max(0, min(width, right)),
        max(0, min(height, bottom)),
    )


def _lonlat_to_global_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    x_tile, y_tile = lonlat_to_tile_fraction(lon, lat, zoom)
    return x_tile * TILE_SIZE, y_tile * TILE_SIZE


def _haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
