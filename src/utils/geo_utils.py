"""
Geographic utilities for Islamabad Smog Detection System.

This module provides functions for handling geographic coordinates,
projections, bounding boxes, and spatial calculations specific to
the Islamabad region.
"""

import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import pyproj
from pyproj import CRS, Transformer
import logging

logger = logging.getLogger(__name__)


class GeoUtils:
    """Geographic utility functions for smog detection system."""

    # Constants
    EARTH_RADIUS_KM = 6371.0
    DEG_TO_RAD = math.pi / 180.0
    RAD_TO_DEG = 180.0 / math.pi

    @staticmethod
    def create_bounding_box(center_lat: float, center_lon: float,
                          buffer_km: float) -> Dict[str, float]:
        """
        Create a bounding box around a center point with specified buffer.

        Args:
            center_lat: Center latitude in decimal degrees
            center_lon: Center longitude in decimal degrees
            buffer_km: Buffer distance in kilometers

        Returns:
            Dictionary with bounding box coordinates (north, south, east, west)
        """
        # Calculate buffer in degrees
        lat_buffer = buffer_km / 111.32  # Approximate km per degree latitude
        lon_buffer = buffer_km / (111.32 * math.cos(center_lat * GeoUtils.DEG_TO_RAD))

        return {
            'north': center_lat + lat_buffer,
            'south': center_lat - lat_buffer,
            'east': center_lon + lon_buffer,
            'west': center_lon - lon_buffer
        }

    @staticmethod
    def create_utm_transformer(lon: float, lat: float, utm_zone: Optional[int] = None) -> Transformer:
        """
        Create coordinate transformer from WGS84 to UTM.

        Args:
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees
            utm_zone: UTM zone (calculated if None)

        Returns:
            PyProj transformer object
        """
        if utm_zone is None:
            utm_zone = int((lon + 180) / 6) + 1

        # Determine hemisphere for UTM zone
        hemisphere = 'north' if lat >= 0 else 'south'
        if hemisphere == 'south':
            utm_crs = f"EPSG:{32600 + utm_zone}"  # North
        else:
            utm_crs = f"EPSG:{32700 + utm_zone}"  # South

        wgs84 = CRS("EPSG:4326")
        utm = CRS(utm_crs)

        return Transformer.from_crs(wgs84, utm, always_xy=True)

    @staticmethod
    def transform_coordinates(coords: List[Tuple[float, float]],
                           transformer: Transformer) -> List[Tuple[float, float]]:
        """
        Transform a list of coordinates using the provided transformer.

        Args:
            coords: List of (x, y) coordinate tuples
            transformer: PyProj transformer

        Returns:
            List of transformed coordinates
        """
        return [transformer.transform(x, y) for x, y in coords]

    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance between two points using Haversine formula.

        Args:
            lat1, lon1: First point coordinates (decimal degrees)
            lat2, lon2: Second point coordinates (decimal degrees)

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = lat1 * GeoUtils.DEG_TO_RAD
        lon1_rad = lon1 * GeoUtils.DEG_TO_RAD
        lat2_rad = lat2 * GeoUtils.DEG_TO_RAD
        lon2_rad = lon2 * GeoUtils.DEG_TO_RAD

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))

        return GeoUtils.EARTH_RADIUS_KM * c

    @staticmethod
    def create_polygon_from_bbox(bbox: Dict[str, float]) -> Polygon:
        """
        Create a Shapely polygon from bounding box coordinates.

        Args:
            bbox: Dictionary with north, south, east, west coordinates

        Returns:
            Shapely Polygon object
        """
        return box(bbox['west'], bbox['south'], bbox['east'], bbox['north'])

    @staticmethod
    def create_islamabad_region(buffer_km: float = 50) -> Dict[str, Any]:
        """
        Create Islamabad region definition with bounding box and polygon.

        Args:
            buffer_km: Buffer distance around Islamabad center point

        Returns:
            Dictionary with region information
        """
        center_lat = 33.6844
        center_lon = 73.0479

        bbox = GeoUtils.create_bounding_box(center_lat, center_lon, buffer_km)
        polygon = GeoUtils.create_polygon_from_bbox(bbox)

        # Create transformer for UTM Zone 43N (Pakistan)
        transformer = GeoUtils.create_utm_transformer(center_lon, center_lat, utm_zone=43)

        return {
            'name': 'Islamabad',
            'center': {'lat': center_lat, 'lon': center_lon},
            'buffer_km': buffer_km,
            'bounding_box': bbox,
            'polygon': polygon,
            'utm_transformer': transformer,
            'utm_zone': 43,
            'area_km2': GeoUtils.calculate_polygon_area_km2(polygon)
        }

    @staticmethod
    def calculate_polygon_area_km2(polygon: Polygon) -> float:
        """
        Calculate area of polygon in square kilometers.

        Args:
            polygon: Shapely Polygon object in WGS84 coordinates

        Returns:
            Area in square kilometers
        """
        # Transform to equal area projection for accurate area calculation
        wgs84 = CRS("EPSG:4326")

        # Use Mollweide projection for global area calculations
        mollweide = CRS("ESRI:54009")
        transformer = Transformer.from_crs(wgs84, mollweide, always_xy=True)

        # Transform polygon coordinates
        coords = list(polygon.exterior.coords)
        transformed_coords = [transformer.transform(x, y) for x, y in coords]
        transformed_polygon = Polygon(transformed_coords)

        # Calculate area and convert from m² to km²
        area_m2 = transformed_polygon.area
        return area_m2 / 1_000_000

    @staticmethod
    def create_grid_points(bbox: Dict[str, float], resolution_km: float) -> List[Point]:
        """
        Create a regular grid of points within bounding box.

        Args:
            bbox: Bounding box coordinates
            resolution_km: Grid resolution in kilometers

        Returns:
            List of Point objects
        """
        # Convert resolution to degrees
        center_lat = (bbox['north'] + bbox['south']) / 2
        lat_resolution = resolution_km / 111.32
        lon_resolution = resolution_km / (111.32 * math.cos(center_lat * GeoUtils.DEG_TO_RAD))

        points = []
        lat = bbox['south']
        while lat <= bbox['north']:
            lon = bbox['west']
            while lon <= bbox['east']:
                points.append(Point(lon, lat))
                lon += lon_resolution
            lat += lat_resolution

        return points

    @staticmethod
    def point_in_polygon(point: Point, polygon: Polygon) -> bool:
        """
        Check if a point is within a polygon.

        Args:
            point: Point object
            polygon: Polygon object

        Returns:
            True if point is within polygon
        """
        return polygon.contains(point) or polygon.touches(point)

    @staticmethod
    def filter_points_by_region(points: List[Point], region_polygon: Polygon) -> List[Point]:
        """
        Filter points to keep only those within region polygon.

        Args:
            points: List of Point objects
            region_polygon: Region boundary polygon

        Returns:
            List of points within the region
        """
        return [p for p in points if GeoUtils.point_in_polygon(p, region_polygon)]

    @staticmethod
    def create_buffer_around_point(point: Point, buffer_km: float) -> Polygon:
        """
        Create a circular buffer around a point.

        Args:
            point: Center point
            buffer_km: Buffer radius in kilometers

        Returns:
            Circular buffer polygon
        """
        # Convert buffer distance to degrees (approximate)
        lat = point.y
        buffer_deg = buffer_km / 111.32

        # Create buffer in projected coordinate system for accuracy
        transformer = GeoUtils.create_utm_transformer(point.x, point.y)
        transformer_inv = Transformer.from_crs(
            transformer.target_crs, transformer.source_crs, always_xy=True
        )

        # Transform point to UTM
        x_utm, y_utm = transformer.transform(point.x, point.y)
        utm_point = Point(x_utm, y_utm)

        # Create buffer in UTM (meters)
        buffer_polygon = utm_point.buffer(buffer_km * 1000)

        # Transform back to WGS84
        coords = list(buffer_polygon.exterior.coords)
        wgs84_coords = [transformer_inv.transform(x, y) for x, y in coords]

        return Polygon(wgs84_coords)

    @staticmethod
    def calculate_wind_vector(u_wind: float, v_wind: float) -> Dict[str, float]:
        """
        Calculate wind speed and direction from U and V components.

        Args:
            u_wind: U component of wind (m/s, eastward)
            v_wind: V component of wind (m/s, northward)

        Returns:
            Dictionary with wind_speed and wind_direction (degrees from north)
        """
        wind_speed = math.sqrt(u_wind**2 + v_wind**2)

        # Calculate direction (degrees from north, clockwise)
        if u_wind == 0 and v_wind == 0:
            wind_direction = 0
        else:
            wind_direction = math.atan2(u_wind, v_wind) * GeoUtils.RAD_TO_DEG
            if wind_direction < 0:
                wind_direction += 360

        return {
            'wind_speed': wind_speed,
            'wind_direction': wind_direction
        }

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """
        Validate latitude and longitude coordinates.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            True if coordinates are valid
        """
        return -90 <= lat <= 90 and -180 <= lon <= 180

    @staticmethod
    def normalize_longitude(lon: float) -> float:
        """
        Normalize longitude to range [-180, 180].

        Args:
            lon: Longitude in decimal degrees

        Returns:
            Normalized longitude
        """
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
        return lon

    @staticmethod
    def create_geodataframe_from_points(points: List[Point],
                                     data: Optional[List[Dict]] = None) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame from a list of points.

        Args:
            points: List of Point objects
            data: Optional list of dictionaries with attributes for each point

        Returns:
            GeoDataFrame with points
        """
        if data is None:
            data = [{}] * len(points)

        gdf = gpd.GeoDataFrame(data, geometry=points, crs="EPSG:4326")
        return gdf

    @staticmethod
    def get_epsg_code_for_utm_zone(utm_zone: int, hemisphere: str = 'north') -> str:
        """
        Get EPSG code for UTM zone.

        Args:
            utm_zone: UTM zone number (1-60)
            hemisphere: 'north' or 'south'

        Returns:
            EPSG code as string
        """
        if hemisphere.lower() == 'north':
            return f"EPSG:{32600 + utm_zone}"
        else:
            return f"EPSG:{32700 + utm_zone}"