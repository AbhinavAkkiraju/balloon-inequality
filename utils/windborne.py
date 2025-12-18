"""
Windborne Data Module

This module handles fetching, processing, and analyzing data from Windborne
weather sensors. Windborne uses high-altitude balloons to collect atmospheric
data from previously under-sampled regions.

The primary data source is the WindBorne treasure hunt API which provides
real-time balloon position data. The API can occasionally return corrupted
or malformed JSON responses, so all fetching functions include robust error
handling to gracefully skip invalid data while continuing to process valid
entries.
"""

import pandas as pd
import numpy as np
import requests
from typing import Optional, List, Dict, Tuple, Union, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_mock_data(num_points: int = 500) -> pd.DataFrame:
    """
    Generate mock balloon data for demonstration when API is unavailable.
    
    Creates synthetic balloon observations with realistic geographic distribution,
    biased toward underserved regions (oceans, Southern Hemisphere, remote areas).
    
    Args:
        num_points: Number of mock balloon observations to generate
        
    Returns:
        DataFrame with mock balloon observations
    """
    import random
    
    np.random.seed(42)
    random.seed(42)
    
    # Create clusters in underserved regions
    clusters = [
        {"lat": -10, "lon": -140, "n": 80},
        {"lat": 20, "lon": 160, "n": 60},
        {"lat": -30, "lon": -20, "n": 70},
        {"lat": 35, "lon": -45, "n": 50},
        {"lat": -15, "lon": 75, "n": 65},
        {"lat": -55, "lon": 120, "n": 40},
        {"lat": -50, "lon": -60, "n": 35},
        {"lat": 5, "lon": 20, "n": 45},
        {"lat": -20, "lon": -60, "n": 30},
        {"lat": 75, "lon": -100, "n": 25},
    ]
    
    data = []
    for cluster in clusters:
        n = cluster["n"]
        for _ in range(n):
            lat = cluster["lat"] + np.random.normal(0, 8)
            lon = cluster["lon"] + np.random.normal(0, 10)
            
            lat = max(-90, min(90, lat))
            lon = ((lon + 180) % 360) - 180
            
            data.append({
                'latitude': lat,
                'longitude': lon,
                'altitude_km': np.random.uniform(12, 20),
                'hours_ago': np.random.randint(0, 24)
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} mock balloon observations for demonstration")
    return df


def fetch_windborne_data() -> pd.DataFrame:
    """
    Fetch WindBorne balloon position data from the treasure hunt API.
    
    This function fetches data from https://a.windbornesystems.com/treasure/{hour}.json
    for hours 0-23. The live API can occasionally return corrupted or malformed JSON,
    so this function includes robust error handling.
    
    Returns:
        DataFrame containing Windborne balloon observations with columns:
        - latitude, longitude, altitude_km, hours_ago
    """
    base_url = "https://a.windbornesystems.com/treasure/{:02d}.json"
    all_data = []
    
    try:
        test_response = requests.get(base_url.format(0), timeout=10)
        if test_response.status_code == 404:
            logger.warning("Windborne API returned 404 - using mock data")
            return generate_mock_data()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Cannot reach Windborne API: {e} - using mock data")
        return generate_mock_data()
    
    for hour in range(24):
        url = base_url.format(hour)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            try:
                data = response.json()
            except ValueError as e:
                logger.warning(f"Corrupted JSON at hour {hour}: {e}. Skipping.")
                continue
            
            if isinstance(data, list):
                for entry in data:
                    parsed_entry = parse_windborne_entry(entry, hour)
                    if parsed_entry is not None:
                        all_data.append(parsed_entry)
            elif isinstance(data, dict):
                parsed_entry = parse_windborne_entry(data, hour)
                if parsed_entry is not None:
                    all_data.append(parsed_entry)
            else:
                logger.warning(f"Unexpected data format at hour {hour}: {type(data)}. Skipping.")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error fetching hour {hour}: {e}. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Unexpected error at hour {hour}: {e}. Skipping.")
            continue
    
    if not all_data:
        logger.warning("No valid data collected from API, using mock data")
        return generate_mock_data()
    
    df = pd.DataFrame(all_data)
    df = validate_coordinates(df)
    
    logger.info(f"Successfully fetched {len(df)} valid balloon observations")
    return df


def parse_windborne_entry(entry, hours_ago: int) -> Optional[Dict]:
    """
    Parse a single WindBorne API entry into a standardized format.
    
    Handles both array format [lat, lon, alt] and dictionary format.
    """
    try:
        if isinstance(entry, (list, tuple)):
            if len(entry) >= 3:
                return {
                    'latitude': float(entry[0]),
                    'longitude': float(entry[1]),
                    'altitude_km': float(entry[2]),
                    'hours_ago': hours_ago
                }
        elif isinstance(entry, dict):
            lat = entry.get('latitude') or entry.get('lat')
            lon = entry.get('longitude') or entry.get('lon') or entry.get('lng')
            alt = entry.get('altitude_km') or entry.get('altitude') or entry.get('alt')
            
            if lat is None or lon is None:
                return None
            
            lat = float(lat)
            lon = float(lon)
            alt = float(alt) if alt is not None else 0.0
            
            if alt > 100:
                alt = alt / 1000.0
            
            return {
                'latitude': lat,
                'longitude': lon,
                'altitude_km': alt,
                'hours_ago': hours_ago
            }
        return None
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        logger.debug(f"Failed to parse entry: {e}")
        return None


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and filter coordinates to ensure they are within valid ranges.
    """
    initial_count = len(df)
    
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[~df['latitude'].isin([float('inf'), float('-inf')])]
    df = df[~df['longitude'].isin([float('inf'), float('-inf')])]
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
    df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
    
    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} entries with invalid coordinates")
    
    return df


def filter_windborne_by_region(data: pd.DataFrame, 
                               lat_range: Tuple[float, float],
                               lon_range: Tuple[float, float]) -> pd.DataFrame:
    """Filter Windborne data by geographic region."""
    filtered = data[
        (data['latitude'] >= lat_range[0]) &
        (data['latitude'] <= lat_range[1]) &
        (data['longitude'] >= lon_range[0]) &
        (data['longitude'] <= lon_range[1])
    ].copy()
    return filtered


def aggregate_windborne_coverage(data: pd.DataFrame, 
                                grid_size: float = 1.0) -> pd.DataFrame:
    """Aggregate Windborne observations into geographic grid cells."""
    if data.empty:
        return pd.DataFrame(columns=['grid_lat', 'grid_lon', 'observation_count'])
    
    data = data.copy()
    data['grid_lat'] = (data['latitude'] // grid_size) * grid_size
    data['grid_lon'] = (data['longitude'] // grid_size) * grid_size
    
    aggregated = data.groupby(['grid_lat', 'grid_lon']).agg({
        'latitude': 'count',
        'altitude_km': ['mean', 'min', 'max'],
        'hours_ago': ['min', 'max']
    }).reset_index()
    
    aggregated.columns = [
        'grid_lat', 'grid_lon', 'observation_count',
        'avg_altitude_km', 'min_altitude_km', 'max_altitude_km',
        'min_hours_ago', 'max_hours_ago'
    ]
    
    return aggregated


def get_windborne_metadata(data: pd.DataFrame) -> Dict:
    """Get metadata about Windborne balloon observations."""
    if data.empty:
        return {
            'total_observations': 0,
            'unique_locations': 0,
            'altitude_range_km': (0, 0),
            'geographic_bounds': {}
        }
    
    return {
        'total_observations': len(data),
        'unique_locations': len(data[['latitude', 'longitude']].drop_duplicates()),
        'altitude_range_km': (data['altitude_km'].min(), data['altitude_km'].max()),
        'avg_altitude_km': data['altitude_km'].mean(),
        'geographic_bounds': {
            'min_lat': data['latitude'].min(),
            'max_lat': data['latitude'].max(),
            'min_lon': data['longitude'].min(),
            'max_lon': data['longitude'].max()
        }
    }


def calculate_windborne_density(data: pd.DataFrame, 
                                grid_size: float = 5.0) -> pd.DataFrame:
    """Calculate observation density for Windborne data."""
    if data.empty:
        return pd.DataFrame(columns=['grid_lat', 'grid_lon', 'density'])
    
    aggregated = aggregate_windborne_coverage(data, grid_size)
    aggregated['density'] = aggregated['observation_count'] / (grid_size ** 2)
    
    return aggregated[['grid_lat', 'grid_lon', 'observation_count', 'density']]
