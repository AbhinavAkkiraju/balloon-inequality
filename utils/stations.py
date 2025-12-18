"""
Weather Stations Data Module

This module handles fetching, processing, and analyzing data from traditional
weather stations (e.g., NOAA, METAR, synoptic stations). Used for comparing
coverage with Windborne sensors.
"""

import pandas as pd
import requests
import os
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOAA_ISD_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
CACHE_DIR = "data"
CACHE_FILE = os.path.join(CACHE_DIR, "stations.csv")


def fetch_station_metadata(force_download: bool = False) -> pd.DataFrame:
    """
    Download and load NOAA ISD station metadata.
    
    Downloads from https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv
    and caches locally in data/stations.csv.
    """
    if os.path.exists(CACHE_FILE) and not force_download:
        logger.info(f"Loading cached station data from {CACHE_FILE}")
        return load_cached_stations()
    
    logger.info(f"Downloading station metadata from {NOAA_ISD_URL}")
    
    try:
        response = requests.get(NOAA_ISD_URL, timeout=30)
        response.raise_for_status()
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        with open(CACHE_FILE, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Cached station data to {CACHE_FILE}")
        return load_cached_stations()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download station metadata: {e}")
        raise


def load_cached_stations() -> pd.DataFrame:
    """Load and process cached station metadata from local file."""
    try:
        df = pd.read_csv(CACHE_FILE, low_memory=False)
        df.columns = df.columns.str.strip().str.upper()
        
        required_cols = {}
        if 'LAT' in df.columns:
            required_cols['latitude'] = 'LAT'
        elif 'LATITUDE' in df.columns:
            required_cols['latitude'] = 'LATITUDE'
        
        if 'LON' in df.columns:
            required_cols['longitude'] = 'LON'
        elif 'LONGITUDE' in df.columns:
            required_cols['longitude'] = 'LONGITUDE'
        
        if 'CTRY' in df.columns:
            required_cols['country_code'] = 'CTRY'
        elif 'COUNTRY' in df.columns:
            required_cols['country_code'] = 'COUNTRY'
        
        if 'latitude' not in required_cols or 'longitude' not in required_cols:
            raise ValueError("Could not find latitude/longitude columns in station data")
        
        stations = pd.DataFrame()
        stations['latitude'] = pd.to_numeric(df[required_cols['latitude']], errors='coerce')
        stations['longitude'] = pd.to_numeric(df[required_cols['longitude']], errors='coerce')
        
        if 'country_code' in required_cols:
            stations['country_code'] = df[required_cols['country_code']].astype(str)
        else:
            stations['country_code'] = 'UNKNOWN'
        
        initial_count = len(stations)
        stations = stations.dropna(subset=['latitude', 'longitude'])
        
        # Filter out invalid coordinates
        stations = stations[
            (stations['latitude'] >= -90) & 
            (stations['latitude'] <= 90) &
            (stations['longitude'] >= -180) & 
            (stations['longitude'] <= 180)
        ]
        
        # Filter out (0, 0) coordinates which are placeholder/invalid data
        # Many stations with missing data are set to exactly (0.000, 0.000)
        stations = stations[
            ~((stations['latitude'] == 0.0) & (stations['longitude'] == 0.0))
        ]
        
        # Fix country code data quality issues in NOAA data
        # Multiple country code errors have been identified in the source data
        
        # Fix 1: NI (Nicaragua) incorrectly used for Nigerian stations
        # Nicaragua is in Central America (longitude < -80), Nigeria is in Africa (longitude > 0)
        nigeria_mask = (stations['country_code'] == 'NI') & (stations['longitude'] > 0)
        if nigeria_mask.any():
            logger.info(f"Correcting {nigeria_mask.sum()} stations: NI (Nicaragua) -> NG (Nigeria) based on coordinates")
            stations.loc[nigeria_mask, 'country_code'] = 'NG'
        
        # Fix 2: NG (Nigeria) incorrectly used for Niger stations
        # Nigeria is generally below 14°N, Niger extends much further north
        niger_mask = (stations['country_code'] == 'NG') & (stations['latitude'] > 14.0)
        if niger_mask.any():
            logger.info(f"Correcting {niger_mask.sum()} stations: NG (Nigeria) -> NE (Niger) based on coordinates")
            stations.loc[niger_mask, 'country_code'] = 'NE'
        
        # Fix 3: NE (Niger) incorrectly used for Niue stations (Pacific island)
        # Niger is in Africa (0-16°E), Niue is in Pacific (around -170°E)
        niue_mask = (stations['country_code'] == 'NE') & (stations['longitude'] < -100)
        if niue_mask.any():
            logger.info(f"Correcting {niue_mask.sum()} stations: NE (Niger) -> NU (Niue) based on coordinates")
            stations.loc[niue_mask, 'country_code'] = 'NU'
        
        # Fix 4: CH is used for both Switzerland and China
        # Switzerland is in Europe (45-48°N, 5-11°E), China is in Asia (18-54°N, 73-135°E)
        china_mask = (stations['country_code'] == 'CH') & (stations['longitude'] > 70)
        if china_mask.any():
            logger.info(f"Correcting {china_mask.sum()} stations: CH (Switzerland) -> CN (China) based on coordinates")
            stations.loc[china_mask, 'country_code'] = 'CN'
        
        removed_count = initial_count - len(stations)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} stations with invalid coordinates")
        
        logger.info(f"Loaded {len(stations)} valid weather stations")
        return stations.reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Failed to load cached station data: {e}")
        raise


def calculate_station_density(stations: pd.DataFrame, 
                              grid_size: float = 5.0) -> pd.DataFrame:
    """Calculate station density per grid cell globally."""
    if stations.empty:
        return pd.DataFrame(columns=['grid_lat', 'grid_lon', 'station_count', 'density', 'country_codes'])
    
    df = stations.copy()
    df['grid_lat'] = (df['latitude'] // grid_size) * grid_size + (grid_size / 2)
    df['grid_lon'] = (df['longitude'] // grid_size) * grid_size + (grid_size / 2)
    
    density = df.groupby(['grid_lat', 'grid_lon']).agg({
        'latitude': 'count',
        'country_code': lambda x: ','.join(sorted(set(str(c) for c in x if pd.notna(c))))
    }).reset_index()
    
    density.columns = ['grid_lat', 'grid_lon', 'station_count', 'country_codes']
    density['density'] = density['station_count'] / (grid_size ** 2)
    
    return density


def filter_stations_by_region(stations: pd.DataFrame,
                              lat_range: Tuple[float, float],
                              lon_range: Tuple[float, float]) -> pd.DataFrame:
    """Filter weather stations by geographic region."""
    filtered = stations[
        (stations['latitude'] >= lat_range[0]) &
        (stations['latitude'] <= lat_range[1]) &
        (stations['longitude'] >= lon_range[0]) &
        (stations['longitude'] <= lon_range[1])
    ].copy()
    
    logger.info(f"Filtered to {len(filtered)} stations in region")
    return filtered


def filter_stations_by_country(stations: pd.DataFrame,
                               country_codes: List[str]) -> pd.DataFrame:
    """Filter weather stations by country code(s)."""
    filtered = stations[stations['country_code'].isin(country_codes)].copy()
    logger.info(f"Filtered to {len(filtered)} stations in {len(country_codes)} countries")
    return filtered


def identify_underserved_regions(stations: pd.DataFrame,
                                grid_size: float = 5.0,
                                threshold: int = 1) -> pd.DataFrame:
    """Identify geographic regions with insufficient weather station coverage."""
    density = calculate_station_density(stations, grid_size)
    underserved = density[density['station_count'] < threshold].copy()
    underserved = underserved.sort_values('station_count')
    
    logger.info(f"Found {len(underserved)} underserved grid cells (< {threshold} stations)")
    return underserved


def get_station_network_stats(stations: pd.DataFrame) -> Dict:
    """Get summary statistics about the weather station network."""
    if stations.empty:
        return {'total_stations': 0, 'countries': 0}
    
    country_counts = stations['country_code'].value_counts().head(10).to_dict()
    earth_land_area_mkm2 = 149.0
    overall_density = len(stations) / earth_land_area_mkm2
    
    return {
        'total_stations': len(stations),
        'countries': stations['country_code'].nunique(),
        'geographic_bounds': {
            'min_lat': float(stations['latitude'].min()),
            'max_lat': float(stations['latitude'].max()),
            'min_lon': float(stations['longitude'].min()),
            'max_lon': float(stations['longitude'].max())
        },
        'stations_by_country': country_counts,
        'coverage_density': overall_density
    }


def compare_with_windborne(stations: pd.DataFrame,
                           windborne_data: pd.DataFrame,
                           grid_size: float = 5.0) -> pd.DataFrame:
    """Compare station coverage with Windborne balloon coverage."""
    station_density = calculate_station_density(stations, grid_size)
    
    if not windborne_data.empty:
        wb = windborne_data.copy()
        wb['grid_lat'] = (wb['latitude'] // grid_size) * grid_size + (grid_size / 2)
        wb['grid_lon'] = (wb['longitude'] // grid_size) * grid_size + (grid_size / 2)
        
        windborne_density = wb.groupby(['grid_lat', 'grid_lon']).size().reset_index()
        windborne_density.columns = ['grid_lat', 'grid_lon', 'windborne_count']
    else:
        windborne_density = pd.DataFrame(columns=['grid_lat', 'grid_lon', 'windborne_count'])
    
    comparison = pd.merge(
        station_density[['grid_lat', 'grid_lon', 'station_count', 'density']],
        windborne_density,
        on=['grid_lat', 'grid_lon'],
        how='outer'
    ).fillna(0)
    
    comparison = comparison.rename(columns={'density': 'station_density'})
    comparison['has_station'] = comparison['station_count'] > 0
    comparison['has_windborne'] = comparison['windborne_count'] > 0
    comparison['coverage_type'] = 'none'
    comparison.loc[comparison['has_station'] & ~comparison['has_windborne'], 'coverage_type'] = 'station_only'
    comparison.loc[~comparison['has_station'] & comparison['has_windborne'], 'coverage_type'] = 'windborne_only'
    comparison.loc[comparison['has_station'] & comparison['has_windborne'], 'coverage_type'] = 'both'
    
    return comparison
