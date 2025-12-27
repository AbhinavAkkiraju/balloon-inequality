"""
Analysis Module

This module contains functions for analyzing inequality in weather data
coverage between Windborne sensors and traditional weather stations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Country code to name mapping
COUNTRY_NAMES = {
    'AF': 'Afghanistan', 'AL': 'Albania', 'DZ': 'Algeria', 'AO': 'Angola',
    'AR': 'Argentina', 'AM': 'Armenia', 'AU': 'Australia', 'AT': 'Austria',
    'AZ': 'Azerbaijan', 'BH': 'Bahrain', 'BD': 'Bangladesh', 'BY': 'Belarus',
    'BE': 'Belgium', 'BZ': 'Belize', 'BJ': 'Benin', 'BT': 'Bhutan',
    'BO': 'Bolivia', 'BA': 'Bosnia', 'BW': 'Botswana', 'BR': 'Brazil',
    'BN': 'Brunei', 'BG': 'Bulgaria', 'BF': 'Burkina Faso', 'KH': 'Cambodia',
    'CM': 'Cameroon', 'CA': 'Canada', 'TD': 'Chad', 'CL': 'Chile',
    'CN': 'China', 'CO': 'Colombia', 'CR': 'Costa Rica', 'CI': 'Ivory Coast',
    'HR': 'Croatia', 'CU': 'Cuba', 'CY': 'Cyprus', 'CZ': 'Czechia',
    'CD': 'DRC', 'DK': 'Denmark', 'DJ': 'Djibouti', 'DO': 'Dominican Rep.',
    'EC': 'Ecuador', 'EG': 'Egypt', 'SV': 'El Salvador', 'GQ': 'Eq. Guinea',
    'ER': 'Eritrea', 'EE': 'Estonia', 'ET': 'Ethiopia', 'FJ': 'Fiji',
    'FI': 'Finland', 'FR': 'France', 'GA': 'Gabon', 'GM': 'Gambia',
    'GE': 'Georgia', 'DE': 'Germany', 'GH': 'Ghana', 'GR': 'Greece',
    'GT': 'Guatemala', 'GN': 'Guinea', 'GY': 'Guyana', 'HT': 'Haiti',
    'HN': 'Honduras', 'HU': 'Hungary', 'IS': 'Iceland', 'IN': 'India',
    'ID': 'Indonesia', 'IR': 'Iran', 'IQ': 'Iraq', 'IE': 'Ireland',
    'IL': 'Israel', 'IT': 'Italy', 'JM': 'Jamaica', 'JP': 'Japan',
    'JO': 'Jordan', 'KZ': 'Kazakhstan', 'KE': 'Kenya', 'KP': 'North Korea',
    'KR': 'South Korea', 'KW': 'Kuwait', 'KG': 'Kyrgyzstan', 'LA': 'Laos',
    'LV': 'Latvia', 'LB': 'Lebanon', 'LS': 'Lesotho', 'LY': 'Libya',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'MK': 'North Macedonia', 'MG': 'Madagascar',
    'MW': 'Malawi', 'MY': 'Malaysia', 'MV': 'Maldives', 'ML': 'Mali',
    'MT': 'Malta', 'MR': 'Mauritania', 'MU': 'Mauritius', 'MX': 'Mexico',
    'MD': 'Moldova', 'MN': 'Mongolia', 'ME': 'Montenegro', 'MA': 'Morocco',
    'MZ': 'Mozambique', 'MM': 'Myanmar', 'NA': 'Namibia', 'NP': 'Nepal',
    'NL': 'Netherlands', 'NZ': 'New Zealand', 'NI': 'Nicaragua', 'NE': 'Niger',
    'NG': 'Nigeria', 'NO': 'Norway', 'OM': 'Oman', 'PK': 'Pakistan',
    'PA': 'Panama', 'PG': 'Papua New Guinea', 'PY': 'Paraguay', 'PE': 'Peru',
    'PH': 'Philippines', 'PL': 'Poland', 'PT': 'Portugal', 'QA': 'Qatar',
    'RO': 'Romania', 'RU': 'Russia', 'RW': 'Rwanda', 'SA': 'Saudi Arabia',
    'SN': 'Senegal', 'RS': 'Serbia', 'SL': 'Sierra Leone', 'SG': 'Singapore',
    'SK': 'Slovakia', 'SI': 'Slovenia', 'SB': 'Solomon Islands', 'SO': 'Somalia',
    'ZA': 'South Africa', 'SS': 'South Sudan', 'ES': 'Spain', 'LK': 'Sri Lanka',
    'SD': 'Sudan', 'SR': 'Suriname', 'SZ': 'Eswatini', 'SE': 'Sweden',
    'CH': 'Switzerland', 'SY': 'Syria', 'TJ': 'Tajikistan', 'TZ': 'Tanzania',
    'TH': 'Thailand', 'TG': 'Togo', 'TT': 'Trinidad', 'TN': 'Tunisia',
    'TR': 'Turkey', 'TM': 'Turkmenistan', 'UG': 'Uganda', 'UA': 'Ukraine',
    'AE': 'UAE', 'GB': 'UK', 'US': 'USA', 'UY': 'Uruguay', 'UZ': 'Uzbekistan',
    'VE': 'Venezuela', 'VN': 'Vietnam', 'YE': 'Yemen', 'ZM': 'Zambia',
    'ZW': 'Zimbabwe', 'RE': 'Réunion', 'GD': 'Grenada', 'BB': 'Barbados',
    'BS': 'Bahamas', 'BM': 'Bermuda', 'PR': 'Puerto Rico', 'GU': 'Guam',
    'VI': 'US Virgin Is.', 'KY': 'Cayman Is.', 'MQ': 'Martinique',
    'GP': 'Guadeloupe', 'NC': 'New Caledonia', 'PF': 'French Polynesia',
    'AS': 'American Samoa', 'GF': 'French Guiana', 'AW': 'Aruba',
    'CW': 'Curaçao', 'IM': 'Isle of Man', 'JE': 'Jersey', 'GG': 'Guernsey',
    'FO': 'Faroe Is.', 'GL': 'Greenland', 'PM': 'St Pierre', 'WF': 'Wallis & Futuna',
    'TK': 'Tokelau', 'NU': 'Niue', 'CK': 'Cook Is.', 'WS': 'Samoa',
    'TO': 'Tonga', 'TV': 'Tuvalu', 'KI': 'Kiribati', 'MH': 'Marshall Is.',
    'FM': 'Micronesia', 'PW': 'Palau', 'NR': 'Nauru', 'VU': 'Vanuatu'
}

# Country population data (2024 estimates in millions)
COUNTRY_POPULATIONS = {
    'CN': 1425.0, 'IN': 1428.0, 'US': 340.0, 'ID': 277.0, 'PK': 240.0,
    'BR': 216.0, 'NG': 223.0, 'BD': 173.0, 'RU': 144.0, 'MX': 128.0,
    'JP': 123.0, 'ET': 126.0, 'PH': 117.0, 'EG': 112.0, 'VN': 98.0,
    'CD': 102.0, 'TR': 85.0, 'IR': 89.0, 'DE': 84.0, 'TH': 71.0,
    'GB': 68.0, 'FR': 65.0, 'IT': 59.0, 'ZA': 60.0, 'TZ': 67.0,
    'MM': 54.0, 'KE': 56.0, 'KR': 52.0, 'CO': 52.0, 'ES': 47.0,
    'AR': 46.0, 'DZ': 45.0, 'SD': 48.0, 'UG': 48.0, 'UA': 37.0,
    'CA': 39.0, 'PL': 37.0, 'MA': 37.0, 'IQ': 44.0, 'AF': 42.0,
    'PE': 34.0, 'MY': 34.0, 'UZ': 35.0, 'SA': 36.0, 'VE': 28.0,
    'NP': 30.0, 'GH': 34.0, 'YE': 34.0, 'MZ': 33.0, 'MG': 30.0,
    'AU': 26.0, 'KP': 26.0, 'CM': 28.0, 'CI': 28.0, 'NE': 27.0,
    'LK': 22.0, 'BF': 23.0, 'ML': 23.0, 'CL': 19.0, 'RO': 19.0,
    'MW': 20.0, 'KZ': 19.0, 'SY': 23.0, 'ZM': 20.0, 'EC': 18.0,
    'NL': 17.0, 'SN': 18.0, 'KH': 17.0, 'TD': 18.0, 'SO': 18.0,
    'GT': 18.0, 'ZW': 16.0, 'RW': 14.0, 'GN': 14.0, 'BJ': 13.0,
    'BO': 12.0, 'TN': 12.0, 'BE': 11.8, 'HT': 11.7, 'CU': 11.2,
    'JO': 11.0, 'CZ': 10.5, 'GR': 10.3, 'SE': 10.5, 'DO': 11.3,
    'AZ': 10.2, 'PT': 10.3, 'HU': 9.6, 'BY': 9.2, 'TJ': 10.1,
    'AE': 10.0, 'IL': 9.7, 'HN': 10.4, 'CH': 8.8, 'AT': 9.0,
    'PG': 10.1, 'RS': 6.7, 'TG': 9.0, 'SI': 2.1, 'LB': 5.5,
    'NZ': 5.2, 'LY': 6.9, 'PY': 6.9, 'SV': 6.3, 'LA': 7.5,
    'NI': 7.0, 'KG': 7.0, 'DK': 5.9, 'FI': 5.6, 'NO': 5.5,
    'SK': 5.4, 'TM': 6.4, 'SG': 6.0, 'CR': 5.2, 'ER': 3.7,
    'IE': 5.1, 'PA': 4.4, 'OM': 4.6, 'KW': 4.3, 'MR': 4.9,
    'GE': 3.7, 'UY': 3.4, 'BA': 3.2, 'MN': 3.4, 'AM': 2.8,
    'AL': 2.8, 'JM': 2.8, 'LT': 2.7, 'QA': 2.7, 'NA': 2.6,
    'BW': 2.6, 'GM': 2.7, 'LS': 2.3, 'MK': 2.0, 'SI': 2.1,
    'LV': 1.8, 'BH': 1.5, 'GQ': 1.7, 'TT': 1.5, 'EE': 1.4,
    'MU': 1.3, 'CY': 1.3, 'SZ': 1.2, 'DJ': 1.1, 'FJ': 0.9,
    'RE': 0.9, 'GY': 0.8, 'BT': 0.8, 'ME': 0.6, 'SB': 0.7,
    'LU': 0.7, 'SR': 0.6, 'MT': 0.5, 'MV': 0.5, 'BN': 0.5,
    'IS': 0.4, 'BZ': 0.4, 'BS': 0.4, 'GD': 0.1, 'BB': 0.3
}


def format_country_names(country_codes: str) -> str:
    """Convert country codes to readable country names."""
    if not country_codes or country_codes == 'Unknown/Ocean':
        return 'Unknown/Ocean'
    
    codes = [c.strip() for c in country_codes.split(',') if c.strip()]
    names = [COUNTRY_NAMES.get(code, code) for code in codes]
    return ', '.join(names)


def estimate_grid_population(country_codes: str, num_cells_in_country: Dict[str, int]) -> float:
    """Estimate population for a grid cell based on country codes."""
    if not country_codes or country_codes == 'Unknown/Ocean':
        return 0.0
    
    total_pop = 0.0
    countries = [c.strip() for c in country_codes.split(',') if c.strip()]
    
    for country in countries:
        if country in COUNTRY_POPULATIONS and country in num_cells_in_country:
            country_pop = COUNTRY_POPULATIONS[country]
            cells = num_cells_in_country[country]
            if cells > 0:
                total_pop += country_pop / cells
        elif country in COUNTRY_POPULATIONS:
            total_pop += COUNTRY_POPULATIONS[country] / 100
    
    return total_pop


def compare_coverage(balloon_data: pd.DataFrame,
                    station_data: pd.DataFrame,
                    grid_size: float = 5.0) -> pd.DataFrame:
    """
    Compare coverage between Windborne balloons and weather stations.
    
    This function bins the Earth into a latitude-longitude grid and computes:
    - Balloon observation count per cell (from last 24 hours)
    - Station count per cell
    - Observation gap score: balloon_count / (station_count + 1)
    
    The observation gap score identifies regions where balloons are providing
    significant coverage in areas with few traditional weather stations.
    """
    # Create grid assignments for balloons
    balloon_grid = pd.DataFrame()
    if not balloon_data.empty:
        balloon_grid = balloon_data.copy()
        balloon_grid['grid_lat'] = (balloon_grid['latitude'] // grid_size) * grid_size + (grid_size / 2)
        balloon_grid['grid_lon'] = (balloon_grid['longitude'] // grid_size) * grid_size + (grid_size / 2)
        
        balloon_counts = balloon_grid.groupby(['grid_lat', 'grid_lon']).size().reset_index()
        balloon_counts.columns = ['grid_lat', 'grid_lon', 'balloon_count']
    else:
        balloon_counts = pd.DataFrame(columns=['grid_lat', 'grid_lon', 'balloon_count'])
        logger.warning("No balloon data provided")
    
    # Create grid assignments for stations
    station_grid = pd.DataFrame()
    if not station_data.empty:
        station_grid = station_data.copy()
        station_grid['grid_lat'] = (station_grid['latitude'] // grid_size) * grid_size + (grid_size / 2)
        station_grid['grid_lon'] = (station_grid['longitude'] // grid_size) * grid_size + (grid_size / 2)
        
        station_agg = station_grid.groupby(['grid_lat', 'grid_lon']).agg({
            'latitude': 'count',
            'country_code': lambda x: ','.join(sorted(set(str(c) for c in x if pd.notna(c))))
        }).reset_index()
        station_agg.columns = ['grid_lat', 'grid_lon', 'station_count', 'country_codes']
    else:
        station_agg = pd.DataFrame(columns=['grid_lat', 'grid_lon', 'station_count', 'country_codes'])
        logger.warning("No station data provided")
    
    # Merge balloon and station data
    comparison = pd.merge(
        balloon_counts,
        station_agg,
        on=['grid_lat', 'grid_lon'],
        how='outer'
    )
    
    # Fill missing values
    comparison['balloon_count'] = comparison['balloon_count'].fillna(0).astype(int)
    comparison['station_count'] = comparison['station_count'].fillna(0).astype(int)
    comparison['country_codes'] = comparison['country_codes'].fillna('Unknown/Ocean')
    comparison.loc[comparison['country_codes'] == '', 'country_codes'] = 'Unknown/Ocean'
    
    # Calculate observation gap score
    comparison['observation_gap_score'] = (
        comparison['balloon_count'] / (comparison['station_count'] + 1)
    )
    
    # Add population estimates
    cells_per_country = {}
    for codes in comparison['country_codes'].dropna():
        if codes and codes != 'Unknown/Ocean':
            for country in codes.split(','):
                country = country.strip()
                if country:
                    cells_per_country[country] = cells_per_country.get(country, 0) + 1
    
    comparison['estimated_population_millions'] = comparison['country_codes'].apply(
        lambda x: estimate_grid_population(x, cells_per_country)
    )
    
    # Calculate population-weighted gap score
    comparison['population_weighted_gap_score'] = (
        comparison['observation_gap_score'] * 
        np.log1p(comparison['estimated_population_millions'])
    )
    
    # Add readable country names
    comparison['country_names'] = comparison['country_codes'].apply(format_country_names)
    
    # Sort by population-weighted gap score
    comparison = comparison.sort_values('population_weighted_gap_score', ascending=False)
    
    logger.info(f"Analyzed {len(comparison)} grid cells")
    logger.info(f"Cells with balloons: {(comparison['balloon_count'] > 0).sum()}")
    logger.info(f"Cells with stations: {(comparison['station_count'] > 0).sum()}")
    logger.info(f"Total estimated population: {comparison['estimated_population_millions'].sum():.1f}M")
    
    return comparison.reset_index(drop=True)


def identify_high_gap_regions(balloon_data: pd.DataFrame,
                              station_data: pd.DataFrame,
                              grid_size: float = 5.0,
                              min_gap_score: float = 2.0,
                              top_n: Optional[int] = None) -> pd.DataFrame:
    """Identify regions with high observation gap scores."""
    comparison = compare_coverage(balloon_data, station_data, grid_size)
    high_gap = comparison[comparison['balloon_count'] > 0].copy()
    high_gap = high_gap[high_gap['observation_gap_score'] >= min_gap_score]
    
    if top_n is not None:
        high_gap = high_gap.head(top_n)
    
    logger.info(f"Found {len(high_gap)} high-gap regions (score >= {min_gap_score})")
    return high_gap.reset_index(drop=True)


def calculate_gini_coefficient(coverage_data: pd.DataFrame,
                               metric_column: str) -> float:
    """Calculate Gini coefficient to measure inequality in coverage distribution."""
    if coverage_data.empty or metric_column not in coverage_data.columns:
        logger.warning(f"Cannot calculate Gini: empty data or missing column")
        return 0.0
    
    values = coverage_data[metric_column].dropna().values
    if len(values) == 0:
        return 0.0
    
    values = np.abs(values)
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    cumsum = np.cumsum(sorted_values)
    total = cumsum[-1] if len(cumsum) > 0 else 0
    
    if total == 0:
        return 0.0
    
    weighted_sum = np.sum((np.arange(1, n + 1)) * sorted_values)
    gini = (2.0 * weighted_sum) / (n * total) - (n + 1) / n
    
    return float(gini)


def calculate_inequality_metrics(comparison: pd.DataFrame) -> Dict:
    """Calculate various inequality metrics for the coverage data."""
    metrics = {
        'gini_stations': calculate_gini_coefficient(comparison, 'station_count'),
        'gini_balloons': calculate_gini_coefficient(comparison, 'balloon_count'),
        'gini_gap_score': calculate_gini_coefficient(
            comparison[comparison['balloon_count'] > 0], 
            'observation_gap_score'
        ),
        'total_cells': len(comparison),
        'cells_with_stations': (comparison['station_count'] > 0).sum(),
        'cells_with_balloons': (comparison['balloon_count'] > 0).sum(),
        'cells_with_both': (
            (comparison['station_count'] > 0) & 
            (comparison['balloon_count'] > 0)
        ).sum(),
        'cells_with_neither': (
            (comparison['station_count'] == 0) & 
            (comparison['balloon_count'] == 0)
        ).sum(),
        'avg_stations_per_cell': comparison['station_count'].mean(),
        'avg_balloons_per_cell': comparison['balloon_count'].mean(),
        'max_gap_score': comparison['observation_gap_score'].max(),
        'avg_gap_score': comparison[
            comparison['balloon_count'] > 0
        ]['observation_gap_score'].mean() if (comparison['balloon_count'] > 0).any() else 0.0
    }
    
    return metrics


def identify_coverage_gaps(balloon_data: pd.DataFrame,
                          station_data: pd.DataFrame,
                          grid_size: float = 5.0,
                          threshold: int = 1) -> pd.DataFrame:
    """Identify regions with insufficient coverage from both balloons and stations."""
    comparison = compare_coverage(balloon_data, station_data, grid_size)
    comparison['total_coverage'] = comparison['balloon_count'] + comparison['station_count']
    gaps = comparison[comparison['total_coverage'] <= threshold].copy()
    gaps = gaps.sort_values('total_coverage')
    
    logger.info(f"Found {len(gaps)} grid cells with <= {threshold} total observations")
    return gaps.reset_index(drop=True)


def analyze_regional_disparities(comparison: pd.DataFrame,
                                regions: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """Analyze coverage disparities across geographic regions."""
    if regions is None:
        comparison_copy = comparison.copy()
        comparison_copy = comparison_copy[comparison_copy['country_codes'] != '']
        
        results = []
        unique_countries = set()
        for codes in comparison_copy['country_codes']:
            unique_countries.update(codes.split(','))
        
        for country in sorted(unique_countries):
            country_cells = comparison_copy[
                comparison_copy['country_codes'].str.contains(country, regex=False)
            ]
            
            if len(country_cells) > 0:
                results.append({
                    'region': country,
                    'cells': len(country_cells),
                    'total_stations': country_cells['station_count'].sum(),
                    'total_balloons': country_cells['balloon_count'].sum(),
                    'avg_gap_score': country_cells['observation_gap_score'].mean(),
                    'max_gap_score': country_cells['observation_gap_score'].max()
                })
        
        return pd.DataFrame(results).sort_values('total_stations', ascending=False)
    else:
        results = []
        for region_name, country_codes in regions.items():
            region_cells = comparison[
                comparison['country_codes'].apply(
                    lambda x: any(c in x for c in country_codes)
                )
            ]
            
            if len(region_cells) > 0:
                results.append({
                    'region': region_name,
                    'cells': len(region_cells),
                    'total_stations': region_cells['station_count'].sum(),
                    'total_balloons': region_cells['balloon_count'].sum(),
                    'avg_gap_score': region_cells['observation_gap_score'].mean(),
                    'max_gap_score': region_cells['observation_gap_score'].max()
                })
        
        return pd.DataFrame(results)


def calculate_coverage_improvement(balloon_data: pd.DataFrame,
                                  station_data: pd.DataFrame,
                                  grid_size: float = 5.0) -> Dict:
    """Calculate how much Windborne improves overall coverage."""
    comparison = compare_coverage(balloon_data, station_data, grid_size)
    
    baseline_coverage = (comparison['station_count'] > 0).sum()
    combined_coverage = (
        (comparison['station_count'] > 0) | 
        (comparison['balloon_count'] > 0)
    ).sum()
    new_coverage = (
        (comparison['station_count'] == 0) & 
        (comparison['balloon_count'] > 0)
    ).sum()
    
    improvement_pct = 0.0
    if baseline_coverage > 0:
        improvement_pct = (new_coverage / baseline_coverage) * 100
    
    return {
        'baseline_cells_covered': int(baseline_coverage),
        'combined_cells_covered': int(combined_coverage),
        'new_cells_from_balloons': int(new_coverage),
        'improvement_percentage': float(improvement_pct),
        'total_cells_analyzed': len(comparison)
    }


def create_summary_report(balloon_data: pd.DataFrame,
                         station_data: pd.DataFrame,
                         grid_size: float = 5.0) -> Dict:
    """Create comprehensive summary report of the analysis."""
    comparison = compare_coverage(balloon_data, station_data, grid_size)
    inequality = calculate_inequality_metrics(comparison)
    improvement = calculate_coverage_improvement(balloon_data, station_data, grid_size)
    high_gap = identify_high_gap_regions(balloon_data, station_data, grid_size, top_n=10)
    
    report = {
        'data_summary': {
            'total_balloon_observations': len(balloon_data),
            'total_stations': len(station_data),
            'grid_size_degrees': grid_size
        },
        'inequality_metrics': inequality,
        'coverage_improvement': improvement,
        'top_high_gap_regions': high_gap[
            ['grid_lat', 'grid_lon', 'balloon_count', 'station_count', 'observation_gap_score']
        ].to_dict('records') if not high_gap.empty else []
    }
    
    return report


def calculate_global_equality_score(coverage_comparison: pd.DataFrame) -> Dict:
    """
    Calculate a comprehensive global equality score (0-100).
    
    Higher scores indicate more equal distribution of weather observations globally.
    Considers: Gini coefficient, station density inequality, coverage adequacy.
    """
    if coverage_comparison.empty:
        return {
            'score': 0, 'grade': 'F', 'interpretation': 'No data',
            'components': {'gini_score': 0, 'density_score': 0, 'adequacy_score': 0},
            'population_impact': {
                'total_population_millions': 0,
                'adequately_covered_millions': 0,
                'underserved_millions': 0
            }
        }
    
    land_cells = coverage_comparison[coverage_comparison['country_codes'] != 'Unknown/Ocean'].copy()
    
    if land_cells.empty:
        return {
            'score': 0, 'grade': 'F', 'interpretation': 'No data',
            'components': {'gini_score': 0, 'density_score': 0, 'adequacy_score': 0},
            'population_impact': {
                'total_population_millions': 0,
                'adequately_covered_millions': 0,
                'underserved_millions': 0
            }
        }
    
    # Calculate component scores
    gini_stations = calculate_gini_coefficient(land_cells, 'station_count')
    
    # Invert Gini (0 = perfect inequality, 1 = perfect equality)
    gini_score = (1 - gini_stations) * 100
    
    # Calculate station density score
    # Good coverage = at least 2 stations per grid cell (2° × 2° = ~500km × 500km)
    # Reality check: most cells have 0-1 stations, which is poor coverage
    cells_with_good_coverage = (land_cells['station_count'] >= 2).sum()
    density_score = (cells_with_good_coverage / len(land_cells) * 100)
    
    # Calculate adequacy score based on realistic thresholds
    # Adequate = at least 3 stations OR (1+ station AND 3+ balloon observations)
    # This is a high bar, as it should be
    land_cells['adequate_coverage'] = (
        (land_cells['station_count'] >= 3) | 
        ((land_cells['station_count'] >= 1) & (land_cells['balloon_count'] >= 3))
    )
    
    # Calculate population estimates (more realistic approach)
    # Assume uniform population distribution within each country across its grid cells
    total_global_pop = sum(COUNTRY_POPULATIONS.values())
    
    # For population impact, we need to account for ALL land area, not just cells with data
    # Approximate: use actual population in cells vs total population possible
    cells_with_data_pop = land_cells['estimated_population_millions'].sum()
    
    # More realistic: many populated areas have NO stations
    adequately_covered_pop = land_cells[land_cells['adequate_coverage']]['estimated_population_millions'].sum()
    
    # Adequacy score based on population
    adequacy_score = (adequately_covered_pop / cells_with_data_pop * 100) if cells_with_data_pop > 0 else 0
    
    # Composite score (weighted average emphasizing adequacy)
    # Gini shows distribution inequality, density shows coverage gaps, adequacy shows quality
    score = (gini_score * 0.2) + (density_score * 0.3) + (adequacy_score * 0.5)
    
    # Assign grade
    if score >= 80:
        grade = 'A'
        interpretation = 'Very good equality'
    elif score >= 70:
        grade = 'B'
        interpretation = 'Good equality'
    elif score >= 60:
        grade = 'C'
        interpretation = 'Moderate inequality'
    elif score >= 50:
        grade = 'D'
        interpretation = 'Significant inequality'
    elif score >= 40:
        grade = 'E'
        interpretation = 'Severe inequality'
    else:
        grade = 'F'
        interpretation = 'Critical inequality'
    
    return {
        'score': float(score),
        'grade': grade,
        'interpretation': interpretation,
        'components': {
            'gini_score': float(gini_score),
            'density_score': float(density_score),
            'adequacy_score': float(adequacy_score)
        },
        'population_impact': {
            'total_population_millions': float(cells_with_data_pop),
            'adequately_covered_millions': float(adequately_covered_pop),
            'underserved_millions': float(cells_with_data_pop - adequately_covered_pop)
        }
    }


def calculate_population_impact(coverage_comparison: pd.DataFrame) -> Dict:
    """Calculate detailed population impact metrics with realistic categorization."""
    land_cells = coverage_comparison[coverage_comparison['country_codes'] != 'Unknown/Ocean'].copy()
    
    if land_cells.empty:
        return {
            'total_population': 0,
            'no_coverage': 0,
            'poor_coverage': 0,
            'adequate_coverage': 0,
            'good_coverage': 0,
            'balloon_critical': 0
        }
    
    # Define coverage quality levels more realistically
    # No coverage: 0 stations and 0 balloons
    no_coverage = land_cells[
        (land_cells['station_count'] == 0) & (land_cells['balloon_count'] == 0)
    ]['estimated_population_millions'].sum()
    
    # Poor coverage: Only 1 station OR only balloons (no permanent infrastructure)
    poor_coverage = land_cells[
        ((land_cells['station_count'] == 1) & (land_cells['balloon_count'] == 0)) |
        ((land_cells['station_count'] == 0) & (land_cells['balloon_count'] > 0))
    ]['estimated_population_millions'].sum()
    
    # Adequate coverage: 2-3 stations OR 1 station + balloons
    adequate_coverage = land_cells[
        ((land_cells['station_count'] >= 2) & (land_cells['station_count'] < 4)) |
        ((land_cells['station_count'] == 1) & (land_cells['balloon_count'] > 0))
    ]['estimated_population_millions'].sum()
    
    # Good coverage: 4+ stations
    good_coverage = land_cells[
        land_cells['station_count'] >= 4
    ]['estimated_population_millions'].sum()
    
    # Balloon critical: Areas where balloons are critically important
    # This includes:
    # 1. Areas with ONLY balloon observations (no stations)
    # 2. Areas where balloons dramatically outnumber stations (5:1 ratio or higher)
    # 3. Areas with very high observation gap scores (> 3.0), indicating heavy balloon reliance
    land_cells['balloon_critical_flag'] = (
        # Case 1: Only balloons, no stations
        ((land_cells['station_count'] == 0) & (land_cells['balloon_count'] > 0)) |
        # Case 2: Balloons outnumber stations by 5:1 or more
        ((land_cells['station_count'] > 0) & 
         (land_cells['balloon_count'] >= land_cells['station_count'] * 5)) |
        # Case 3: Very high gap score (balloons doing most of the work)
        (land_cells['observation_gap_score'] > 3.0)
    )
    
    balloon_critical = land_cells[
        land_cells['balloon_critical_flag']
    ]['estimated_population_millions'].sum()
    
    # Also track "balloon only" separately for comparison
    balloon_only = land_cells[
        (land_cells['station_count'] == 0) & (land_cells['balloon_count'] > 0)
    ]['estimated_population_millions'].sum()
    
    # High gap regions: Balloons significantly outnumber stations
    high_gap = land_cells[
        land_cells['observation_gap_score'] > 2.0
    ]['estimated_population_millions'].sum()
    
    total = land_cells['estimated_population_millions'].sum()
    
    # Calculate underserved (no coverage + poor coverage)
    underserved = no_coverage + poor_coverage
    
    return {
        'total_population': float(total),
        'no_coverage': float(no_coverage),
        'poor_coverage': float(poor_coverage),
        'adequate_coverage': float(adequate_coverage),
        'good_coverage': float(good_coverage),
        'balloon_critical': float(balloon_critical),
        'balloon_only': float(balloon_only),
        'high_gap_regions': float(high_gap),
        'underserved_total': float(underserved),
        'percentages': {
            'no_coverage_pct': float(no_coverage / total * 100) if total > 0 else 0,
            'poor_coverage_pct': float(poor_coverage / total * 100) if total > 0 else 0,
            'adequate_coverage_pct': float(adequate_coverage / total * 100) if total > 0 else 0,
            'good_coverage_pct': float(good_coverage / total * 100) if total > 0 else 0,
            'underserved_pct': float(underserved / total * 100) if total > 0 else 0,
            'balloon_critical_pct': float(balloon_critical / total * 100) if total > 0 else 0,
            'balloon_only_pct': float(balloon_only / total * 100) if total > 0 else 0
        }
    }


def compare_countries(coverage_comparison: pd.DataFrame,
                     country_code1: str,
                     country_code2: str) -> Dict:
    """Compare coverage metrics between two countries."""
    
    def get_country_metrics(code: str) -> Dict:
        country_cells = coverage_comparison[
            coverage_comparison['country_codes'].str.contains(code, na=False, regex=False)
        ]
        
        if country_cells.empty:
            return {
                'name': COUNTRY_NAMES.get(code, code),
                'cells': 0,
                'stations': 0,
                'balloons': 0,
                'population': 0,
                'avg_gap_score': 0,
                'coverage_quality': 'No data'
            }
        
        total_stations = int(country_cells['station_count'].sum())
        total_balloons = int(country_cells['balloon_count'].sum())
        population = float(country_cells['estimated_population_millions'].sum())
        avg_gap = float(country_cells['observation_gap_score'].mean())
        
        # Determine coverage quality
        stations_per_million = total_stations / population if population > 0 else 0
        if stations_per_million > 10:
            quality = 'Excellent'
        elif stations_per_million > 5:
            quality = 'Good'
        elif stations_per_million > 1:
            quality = 'Adequate'
        else:
            quality = 'Poor'
        
        return {
            'name': COUNTRY_NAMES.get(code, code),
            'cells': len(country_cells),
            'stations': total_stations,
            'balloons': total_balloons,
            'population': population,
            'avg_gap_score': avg_gap,
            'stations_per_million': float(stations_per_million),
            'coverage_quality': quality
        }
    
    country1 = get_country_metrics(country_code1)
    country2 = get_country_metrics(country_code2)
    
    return {
        'country1': country1,
        'country2': country2,
        'comparison': {
            'station_ratio': country1['stations'] / country2['stations'] if country2['stations'] > 0 else float('inf'),
            'balloon_ratio': country1['balloons'] / country2['balloons'] if country2['balloons'] > 0 else float('inf'),
            'better_covered': country1['name'] if country1['stations_per_million'] > country2['stations_per_million'] else country2['name']
        }
    }
