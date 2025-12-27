"""
Windborne Inequality - Main Streamlit Application

This is the main entry point for the Streamlit web application that analyzes
inequality in weather data coverage between Windborne sensors and traditional
weather stations.
"""

import streamlit as st
import pandas as pd
import pydeck as pdk
from utils import windborne, stations, analysis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page settings
st.set_page_config(
    page_title="Can the Sky Reveal Global Inequality?",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(ttl=3600)
def load_balloon_data():
    """Load and cache Windborne balloon data."""
    try:
        with st.spinner("Fetching Windborne balloon data from API..."):
            balloon_data = windborne.fetch_windborne_data()
        
        if len(balloon_data) > 0:
            if len(balloon_data) == 500:
                st.warning("Using mock balloon data for demonstration (Windborne API unavailable)")
        return balloon_data
    except Exception as e:
        st.error(f"Failed to load balloon data: {e}")
        logger.error(f"Balloon data loading error: {e}")
        return pd.DataFrame(columns=['latitude', 'longitude', 'altitude_km', 'hours_ago'])


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_station_data():
    """Load and cache weather station metadata."""
    try:
        with st.spinner("Loading weather station data..."):
            station_data = stations.fetch_station_metadata()
        return station_data
    except Exception as e:
        st.error(f"Failed to load station data: {e}")
        logger.error(f"Station data loading error: {e}")
        return pd.DataFrame(columns=['latitude', 'longitude', 'country_code'])


def main():
    """Main application function."""
    # Header
    st.title("Can the Sky Reveal Global Inequality?")
    
    st.markdown("""
    ### Exploring Weather Data Coverage Through Windborne Balloons
    
    This visualization compares **balloon-based** atmospheric observations from 
    [Windborne Systems](https://windbornesystems.com/) with traditional 
    **ground-based** weather stations. High-altitude balloons can reach remote 
    regions where weather stations are sparse or nonexistent, potentially revealing 
    global inequalities in meteorological infrastructure.
    
    **Balloons** float through the atmosphere collecting data in under-observed regions  
    **Stations** are fixed ground installations, often concentrated in wealthy nations  
    **Heatmap** shows where balloons fill critical gaps in station coverage
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        balloon_data = load_balloon_data()
        station_data = load_station_data()
    
    if balloon_data.empty and station_data.empty:
        st.error("No data available. Please check your internet connection and try again.")
        return
    
    # Compute coverage comparison
    with st.spinner("Analyzing coverage gaps..."):
        coverage_comparison = analysis.compare_coverage(
            balloon_data, 
            station_data, 
            grid_size=2.0
        )
    
    # Calculate global equality score
    equality_metrics = analysis.calculate_global_equality_score(coverage_comparison)
    
    # Display prominent equality score
    display_equality_score(equality_metrics)
    
    # Sidebar controls
    st.sidebar.header("Visualization Controls")
    
    # Educational narrative
    with st.sidebar.expander("Understanding the Data", expanded=False):
        st.markdown("""
        **What the Balloons Measure**
        
        Windborne's high-altitude balloons drift through the stratosphere 
        (15-20 km altitude), collecting atmospheric data including temperature, 
        pressure, humidity, and wind speed. They transmit observations via 
        satellite as they traverse the globe, providing measurements from 
        regions that traditional infrastructure cannot reach.
        
        **Why Ground Station Coverage is Unequal**
        
        Weather stations require substantial infrastructure: land access, 
        power, maintenance, and communication networks. This creates geographic 
        bias—stations cluster in wealthy nations, urban centers, and accessible 
        terrain. Remote regions, oceans, developing nations, and conflict zones 
        often lack coverage entirely.
        
        Historical factors compound this: colonial powers established networks 
        primarily in their territories and trade routes, creating persistent 
        gaps in equatorial and Southern Hemisphere coverage.
        
        **What Balloon-Dominated Regions Reveal**
        
        When balloons provide the only observations in a region, it indicates 
        a fundamental gap in permanent observational infrastructure. These 
        areas may be:
        - Geographically remote (oceans, deserts, polar regions)
        - Economically underinvested in meteorological infrastructure
        - Politically isolated or experiencing instability
        - Simply overlooked by historical network planning
        
        High observation gap scores highlight where atmospheric science depends 
        on mobile sensing to compensate for absent ground infrastructure.
        """)
    
    # Layer toggles
    st.sidebar.subheader("Layers")
    show_balloons = st.sidebar.checkbox("Show Balloons", value=True)
    show_stations = st.sidebar.checkbox("Show Stations", value=True)
    show_heatmap = st.sidebar.checkbox("Show Inequality Heatmap", value=True)
    
    # Time slider for balloons
    st.sidebar.subheader("Time Filter")
    if not balloon_data.empty:
        max_hours = int(balloon_data['hours_ago'].max())
        min_hours = int(balloon_data['hours_ago'].min())
        hours_range = st.sidebar.slider(
            "Hours ago (balloon data)",
            min_value=min_hours,
            max_value=max_hours,
            value=(min_hours, max_hours),
            help="Filter balloon observations by how many hours ago they were recorded"
        )
    else:
        hours_range = (0, 23)
    
    # View settings
    st.sidebar.subheader("View Settings")
    initial_zoom = st.sidebar.slider("Initial Zoom", min_value=0, max_value=10, value=1)
    
    # Filter balloon data by time
    filtered_balloons = balloon_data[
        (balloon_data['hours_ago'] >= hours_range[0]) &
        (balloon_data['hours_ago'] <= hours_range[1])
    ].copy() if not balloon_data.empty else balloon_data
    
    # Filter coverage data to land areas only (excludes oceans)
    if not coverage_comparison.empty:
        filtered_comparison = coverage_comparison[
            coverage_comparison['country_codes'] != 'Unknown/Ocean'
        ].copy()
    else:
        filtered_comparison = coverage_comparison.copy()
    
    # Display map
    st.subheader("Global Coverage Map - Land Areas")
    
    # Create the map
    map_layers = create_map_layers(
        filtered_balloons,
        station_data,
        filtered_comparison,
        show_balloons,
        show_stations,
        show_heatmap
    )
    
    # Render pydeck map
    view_state = pdk.ViewState(
        latitude=20,
        longitude=0,
        zoom=initial_zoom,
        pitch=0,
        bearing=0
    )
    
    deck = pdk.Deck(
        layers=map_layers,
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{name}</b><br/>{info}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
                "fontSize": "12px",
                "padding": "10px"
            }
        },
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    )
    
    st.pydeck_chart(deck, use_container_width=True)
    
    # Display statistics
    display_statistics(filtered_balloons, station_data, filtered_comparison)
    
    # Display population impact
    display_population_impact(filtered_comparison)
    
    # Display balloon-critical regions breakdown
    display_balloon_critical_regions(filtered_comparison)
    
    # Display high-gap regions
    display_high_gap_regions(balloon_data, station_data, filtered_comparison)
    
    # Display inequality metrics
    display_inequality_metrics(filtered_comparison)
    
    # Display country comparison tool
    display_country_comparison(filtered_comparison)
    
    # Concluding narrative
    display_conclusion()


def create_map_layers(balloon_data, station_data, coverage_comparison,
                     show_balloons, show_stations, show_heatmap):
    """Create pydeck layers for visualization."""
    layers = []
    
    # Inequality heatmap layer (render first, so it's below other layers)
    if show_heatmap and not coverage_comparison.empty:
        # Filter to cells with meaningful gap scores
        heatmap_data = coverage_comparison[
            coverage_comparison['observation_gap_score'] > 0
        ].copy()
        
        # Normalize gap scores for color mapping
        if not heatmap_data.empty:
            max_score = heatmap_data['observation_gap_score'].quantile(0.95)
            heatmap_data['normalized_score'] = (
                heatmap_data['observation_gap_score'].clip(0, max_score) / max_score
            )
            
            # Add tooltip info
            heatmap_data['name'] = 'Gap Score: ' + heatmap_data['observation_gap_score'].round(2).astype(str)
            
            # Build tooltip info
            tooltip_info = (
                'Lat: ' + heatmap_data['grid_lat'].round(3).astype(str) + '<br/>' +
                'Lon: ' + heatmap_data['grid_lon'].round(3).astype(str) + '<br/>' +
                'Balloons: ' + heatmap_data['balloon_count'].astype(str) + '<br/>' +
                'Stations: ' + heatmap_data['station_count'].astype(str) + '<br/>'
            )
            
            # Add country names if column exists
            if 'country_names' in heatmap_data.columns:
                tooltip_info += 'Countries: ' + heatmap_data['country_names'].astype(str) + '<br/>'
            
            tooltip_info += 'This region shows inequality in coverage'
            heatmap_data['info'] = tooltip_info
            
            # Use GridCellLayer for clear geographic representation
            # Only show high-gap cells to reduce clutter
            high_gap_heatmap = heatmap_data[heatmap_data['observation_gap_score'] >= 1.0].copy()
            
            if not high_gap_heatmap.empty:
                heatmap_layer = pdk.Layer(
                    "GridCellLayer",
                    data=high_gap_heatmap,
                    get_position=['grid_lon', 'grid_lat'],
                    cell_size=110000,  # ~1 degree at equator (for 2° cells)
                    elevation_scale=0,
                    extruded=False,
                    get_fill_color='[255 * normalized_score, 60, 255 * (1 - normalized_score), 120]',
                    pickable=True,
                    auto_highlight=True,
                )
                layers.append(heatmap_layer)
    
    # Station layer
    if show_stations and not station_data.empty:
        # Sample stations if too many (for performance)
        display_stations = station_data.copy()
        if len(display_stations) > 5000:
            display_stations = display_stations.sample(n=5000, random_state=42)
        
        # Add tooltip info
        display_stations['name'] = 'Weather Station'
        display_stations['info'] = (
            'Lat: ' + display_stations['latitude'].round(4).astype(str) + '<br/>' +
            'Lon: ' + display_stations['longitude'].round(4).astype(str) + '<br/>' +
            'Country: ' + display_stations['country_code'] + '<br/>' +
            'Traditional ground-based observation'
        )
        
        station_layer = pdk.Layer(
            "ScatterplotLayer",
            data=display_stations,
            get_position=['longitude', 'latitude'],  # [lon, lat] is correct for pydeck
            get_radius=20000,  # meters
            get_fill_color=[100, 255, 100, 200],
            pickable=True,
            auto_highlight=True,
        )
        layers.append(station_layer)
    
    # Balloon layer
    if show_balloons and not balloon_data.empty:
        # Add tooltip info
        balloon_display = balloon_data.copy()
        balloon_display['name'] = 'Windborne Balloon'
        balloon_display['info'] = (
            'Lat: ' + balloon_display['latitude'].round(4).astype(str) + '<br/>' +
            'Lon: ' + balloon_display['longitude'].round(4).astype(str) + '<br/>' +
            'Altitude: ' + balloon_display['altitude_km'].round(2).astype(str) + ' km<br/>' +
            'Hours ago: ' + balloon_display['hours_ago'].astype(str) + '<br/>' +
            'High-altitude atmospheric observation'
        )
        
        # Color by hours_ago (recent = brighter)
        if 'hours_ago' in balloon_display.columns:
            max_hours = balloon_display['hours_ago'].max()
            if max_hours > 0:
                balloon_display['recency'] = 255 - (balloon_display['hours_ago'] / max_hours * 200)
            else:
                balloon_display['recency'] = 255
        else:
            balloon_display['recency'] = 255
        
        balloon_layer = pdk.Layer(
            "ScatterplotLayer",
            data=balloon_display,
            get_position=['longitude', 'latitude'],  # [lon, lat] is correct for pydeck
            get_radius=50000,  # meters - larger for visibility
            get_fill_color=['recency', 100, 255, 220],
            pickable=True,
            auto_highlight=True,
        )
        layers.append(balloon_layer)
    
    return layers


def display_equality_score(equality_metrics):
    """Display the global coverage equality score prominently."""
    st.markdown("---")
    st.subheader("Global Coverage Equality Score")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        score = equality_metrics['score']
        grade = equality_metrics['grade']
        
        # Create a color based on score
        if score >= 70:
            color = "green"
        elif score >= 50:
            color = "orange"
        else:
            color = "red"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: rgba(0,0,0,0.05); border-radius: 10px;">
            <h1 style="color: {color}; font-size: 72px; margin: 0;">{score:.1f}</h1>
            <h2 style="margin: 5px 0; color: {color};">{grade}</h2>
            <p style="font-size: 18px; color: #666;">{equality_metrics['interpretation']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='margin-top: 40px;'>", unsafe_allow_html=True)
        st.markdown("**Score of 100** = Perfect equality")
        st.markdown("**Score of 0** = Complete inequality")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("**What this means:**")
        components = equality_metrics['components']
        pop_impact = equality_metrics['population_impact']
        
        st.metric("Distribution Equality", f"{components['gini_score']:.1f}/100")
        st.metric("Station Density", f"{components['density_score']:.1f}%")
        st.metric("People Underserved", f"{pop_impact['underserved_millions']:.0f}M")
    
    with st.expander("How is this calculated?"):
        st.markdown("""
        The Global Equality Score combines three factors:
        
        - **Distribution Equality (20%)**: Based on Gini coefficient - measures how evenly stations are distributed across regions
        - **Station Density (30%)**: Percentage of regions with good station density (2+ stations per grid cell)
        - **Coverage Adequacy (50%)**: Percentage of population with adequate coverage (3+ stations or 1+ station with balloon support)
        
        A high score means most people have access to quality weather observations. A low score indicates 
        severe inequality where many populated regions lack adequate meteorological infrastructure.
        
        **Current thresholds:**
        - Adequate coverage requires 3+ weather stations OR 1+ station with 3+ balloon observations
        - This is a realistic bar - a 2° × 2° grid cell (~500km × 500km) needs multiple stations for quality data
        """)
    
    st.markdown("---")


def display_population_impact(coverage_comparison):
    """Display population impact analysis."""
    st.subheader("Population Impact Analysis")
    
    st.markdown("""
    Understanding who is affected by data gaps is crucial. This analysis categorizes global population 
    by the quality of weather observation coverage in their region.
    
    **Coverage Levels:**
    - **Good**: 4+ weather stations (dense network)
    - **Adequate**: 2-3 stations or 1 station + balloon support
    - **Poor**: Only 1 station or only balloon observations (no permanent infrastructure)
    - **None**: No observations at all
    """)
    
    pop_impact = analysis.calculate_population_impact(coverage_comparison)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Population Analyzed",
            f"{pop_impact['total_population']:.0f}M",
            help="Population in grid cells with country data"
        )
    
    with col2:
        st.metric(
            "Underserved Population",
            f"{pop_impact['underserved_total']:.0f}M",
            delta=f"-{pop_impact['percentages']['underserved_pct']:.0f}%",
            delta_color="inverse",
            help="People with no coverage or poor coverage (inadequate for reliable forecasting)"
        )
    
    with col3:
        st.metric(
            "Balloon-Critical Regions",
            f"{pop_impact['balloon_critical']:.0f}M",
            help="People in regions critically dependent on balloon observations (see details below)"
        )
    
    with col4:
        st.metric(
            "Good Coverage",
            f"{pop_impact['good_coverage']:.0f}M",
            help="People with 4+ weather stations in their region"
        )
    
    # Create visualization of population distribution by coverage quality
    if pop_impact['total_population'] > 0:
        coverage_data = {
            'Coverage Quality': ['Good (4+ stations)', 'Adequate (2-3 stations)', 'Poor (1 station or balloons only)', 'None'],
            'Population (Millions)': [
                pop_impact['good_coverage'],
                pop_impact['adequate_coverage'],
                pop_impact['poor_coverage'],
                pop_impact['no_coverage']
            ],
            'Percentage': [
                pop_impact['percentages']['good_coverage_pct'],
                pop_impact['percentages']['adequate_coverage_pct'],
                pop_impact['percentages']['poor_coverage_pct'],
                pop_impact['percentages']['no_coverage_pct']
            ]
        }
        
        df = pd.DataFrame(coverage_data)
        df = df[df['Population (Millions)'] > 0]  # Only show non-zero categories
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Population (Millions)": st.column_config.NumberColumn(
                    "Population (Millions)",
                    format="%.1f"
                ),
                "Percentage": st.column_config.NumberColumn(
                    "Percentage",
                    format="%.1f%%"
                )
            }
        )
        
        # Key insights
        underserved_pct = pop_impact['percentages']['underserved_pct']
        balloon_critical_pct = pop_impact['percentages']['balloon_critical_pct']
        balloon_only_pct = pop_impact['percentages']['balloon_only_pct']
        
        if underserved_pct > 50:
            st.warning(
                f"**Critical Inequality**: {pop_impact['underserved_total']:.0f} million people "
                f"({underserved_pct:.0f}%) live in regions with inadequate weather observation infrastructure."
            )
        
        if balloon_critical_pct > 1:
            st.info(
                f"**Windborne's Critical Role**: {pop_impact['balloon_critical']:.0f} million people "
                f"({balloon_critical_pct:.1f}%) live in balloon-critical regions.\n\n"
                f"This includes:\n"
                f"- **{pop_impact['balloon_only']:.0f}M people ({balloon_only_pct:.1f}%)** with ONLY balloon observations (zero stations)\n"
                f"- Regions where balloons outnumber stations by 5:1 or more\n"
                f"- Areas with severe station gaps where balloons provide the majority of atmospheric data\n\n"
                f"Without balloon technology, these populations would have severely compromised or zero weather forecasting capability."
            )
        
        with st.expander("What makes a region 'balloon-critical'?"):
            st.markdown("""
            A region is considered **balloon-critical** if it meets any of these criteria:
            
            1. **Zero permanent infrastructure**: No weather stations, only balloon observations
            2. **Severe imbalance**: Balloons outnumber stations by 5:1 or more  
               (e.g., 1 station but 20 balloon observations)
            3. **High observation gap score**: Score > 3.0, indicating balloons are providing 
               the vast majority of atmospheric data despite some station presence
            
            These regions critically depend on mobile sensing technology. While they may technically 
            have "some" station coverage, it's so sparse that balloons are doing the heavy lifting 
            for weather observations. Losing balloon coverage would devastate forecast quality.
            
            **Example**: A 500km × 500km region with 1 aging weather station and 15 balloon 
            observations is balloon-critical. That single station cannot adequately cover such 
            a large area - the balloons are essential for spatial coverage and data density.
            """)


def display_country_comparison(coverage_comparison):
    """Display interactive country comparison tool."""
    st.subheader("Country Comparison Tool")
    
    st.markdown("""
    Compare weather observation coverage between any two countries to see disparities 
    in meteorological infrastructure.
    """)
    
    # Get list of countries from coverage data
    all_countries = set()
    for codes in coverage_comparison['country_codes'].dropna():
        if codes != 'Unknown/Ocean':
            for code in codes.split(','):
                code = code.strip()
                if code in analysis.COUNTRY_NAMES:
                    all_countries.add(code)
    
    country_options = {analysis.COUNTRY_NAMES[code]: code for code in sorted(all_countries)}
    
    if len(country_options) < 2:
        st.warning("Not enough countries in dataset for comparison.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        country1_name = st.selectbox(
            "Select first country",
            options=sorted(country_options.keys()),
            index=list(country_options.keys()).index('USA') if 'USA' in country_options else 0
        )
    
    with col2:
        country2_name = st.selectbox(
            "Select second country",
            options=sorted(country_options.keys()),
            index=list(country_options.keys()).index('Nigeria') if 'Nigeria' in country_options else 1
        )
    
    country1_code = country_options[country1_name]
    country2_code = country_options[country2_name]
    
    # Perform comparison
    comparison = analysis.compare_countries(coverage_comparison, country1_code, country2_code)
    
    # Display results
    col1, col2, col3 = st.columns([5, 1, 5])
    
    with col1:
        st.markdown(f"### {comparison['country1']['name']}")
        st.metric("Weather Stations", f"{comparison['country1']['stations']:,}")
        st.metric("Balloon Observations", f"{comparison['country1']['balloons']:,}")
        st.metric("Population", f"{comparison['country1']['population']:.0f}M")
        st.metric("Stations per Million", f"{comparison['country1']['stations_per_million']:.1f}")
        st.markdown(f"**Coverage Quality:** {comparison['country1']['coverage_quality']}")
    
    with col2:
        st.markdown("<div style='text-align: center; margin-top: 80px; font-size: 36px;'>vs</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"### {comparison['country2']['name']}")
        st.metric("Weather Stations", f"{comparison['country2']['stations']:,}")
        st.metric("Balloon Observations", f"{comparison['country2']['balloons']:,}")
        st.metric("Population", f"{comparison['country2']['population']:.0f}M")
        st.metric("Stations per Million", f"{comparison['country2']['stations_per_million']:.1f}")
        st.markdown(f"**Coverage Quality:** {comparison['country2']['coverage_quality']}")
    
    # Analysis
    st.markdown("### Analysis")
    better = comparison['comparison']['better_covered']
    st.markdown(f"**{better}** has better traditional weather station coverage.")
    
    if comparison['country1']['stations'] > 0 and comparison['country2']['stations'] > 0:
        ratio = comparison['comparison']['station_ratio']
        if ratio > 1:
            st.markdown(
                f"{comparison['country1']['name']} has {ratio:.1f}x more weather stations than "
                f"{comparison['country2']['name']}."
            )
        else:
            st.markdown(
                f"{comparison['country2']['name']} has {1/ratio:.1f}x more weather stations than "
                f"{comparison['country1']['name']}."
            )


def display_statistics(balloon_data, station_data, coverage_comparison):
    """Display summary statistics."""
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Balloon Observations",
            f"{len(balloon_data):,}",
            help="Total balloon observations in the last 24 hours"
        )
    
    with col2:
        st.metric(
            "Weather Stations",
            f"{len(station_data):,}",
            help="Total traditional weather stations globally"
        )
    
    with col3:
        cells_with_gap = (coverage_comparison['observation_gap_score'] > 2.0).sum()
        st.metric(
            "High-Gap Regions",
            f"{cells_with_gap:,}",
            help="Grid cells where balloons are filling significant coverage gaps (score > 2.0)"
        )
    
    with col4:
        improvement = analysis.calculate_coverage_improvement(
            balloon_data, station_data, grid_size=2.0
        )
        st.metric(
            "New Coverage",
            f"{improvement['new_cells_from_balloons']:,}",
            help="Grid cells covered only by balloons (no stations)"
        )


def display_balloon_critical_regions(coverage_comparison):
    """Display breakdown of balloon-critical regions."""
    st.subheader("Balloon-Critical Regions: Examples")
    
    st.markdown("""
    These are specific regions where balloon observations are critically important, 
    showing the ratio of balloons to stations and why these areas depend on mobile sensing.
    """)
    
    # Filter to balloon-critical regions
    land_cells = coverage_comparison[coverage_comparison['country_codes'] != 'Unknown/Ocean'].copy()
    
    # Mark balloon-critical regions (same logic as in analysis.py)
    land_cells['balloon_critical_flag'] = (
        ((land_cells['station_count'] == 0) & (land_cells['balloon_count'] > 0)) |
        ((land_cells['station_count'] > 0) & 
         (land_cells['balloon_count'] >= land_cells['station_count'] * 5)) |
        (land_cells['observation_gap_score'] > 3.0)
    )
    
    critical_regions = land_cells[land_cells['balloon_critical_flag']].copy()
    
    if critical_regions.empty:
        st.info("No balloon-critical regions identified in current data.")
        return
    
    # Add categorization
    def categorize_criticality(row):
        if row['station_count'] == 0:
            return "Zero stations"
        elif row['balloon_count'] >= row['station_count'] * 5:
            return "5:1+ balloon ratio"
        else:
            return "High gap score"
    
    critical_regions['criticality_type'] = critical_regions.apply(categorize_criticality, axis=1)
    
    # Sort by population impact
    critical_regions = critical_regions.nlargest(15, 'population_weighted_gap_score')
    
    # Prepare display
    display_df = critical_regions[[
        'grid_lat', 'grid_lon', 'balloon_count', 'station_count', 
        'observation_gap_score', 'estimated_population_millions', 
        'criticality_type', 'country_names'
    ]].copy()
    
    # Calculate balloon:station ratio
    display_df['ratio'] = display_df.apply(
        lambda row: f"{row['balloon_count']}:0" if row['station_count'] == 0 
        else f"{row['balloon_count']}:{row['station_count']}", 
        axis=1
    )
    
    display_df.columns = [
        'Latitude', 'Longitude', 'Balloons', 'Stations', 
        'Gap Score', 'Population (M)', 'Why Critical', 'Countries', 'Balloon:Station'
    ]
    
    st.dataframe(
        display_df[['Countries', 'Balloons', 'Stations', 'Balloon:Station', 
                    'Gap Score', 'Population (M)', 'Why Critical', 'Latitude', 'Longitude']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Gap Score": st.column_config.NumberColumn(
                "Gap Score",
                format="%.2f",
                help="Higher = more balloon-dependent"
            ),
            "Population (M)": st.column_config.NumberColumn(
                "Population (M)",
                format="%.1f"
            ),
            "Latitude": st.column_config.NumberColumn(
                "Latitude",
                format="%.2f°"
            ),
            "Longitude": st.column_config.NumberColumn(
                "Longitude",
                format="%.2f°"
            )
        }
    )
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        zero_station_count = (critical_regions['station_count'] == 0).sum()
        st.metric(
            "Regions with Zero Stations",
            zero_station_count,
            help="Completely dependent on balloons"
        )
    
    with col2:
        high_ratio_count = (
            (critical_regions['station_count'] > 0) & 
            (critical_regions['balloon_count'] >= critical_regions['station_count'] * 5)
        ).sum()
        st.metric(
            "Severe Imbalance (5:1+)",
            high_ratio_count,
            help="Balloons outnumber stations by 5x or more"
        )
    
    with col3:
        total_pop_critical = critical_regions['estimated_population_millions'].sum()
        st.metric(
            "People in These Regions",
            f"{total_pop_critical:.0f}M",
            help="Population in shown balloon-critical regions"
        )


def display_high_gap_regions(balloon_data, station_data, coverage_comparison):
    """Display top regions where balloons fill critical gaps."""
    st.subheader("Top Regions Where Balloons Fill Coverage Gaps")
    
    st.markdown("""
    These regions have high **population-weighted observation gap scores**, meaning balloons 
    are providing significant data coverage in **populated areas** with few traditional weather 
    stations. The weighting emphasizes regions where more people are affected by infrastructure 
    inequality. These are the places where Windborne is making the greatest impact on data 
    availability for human populations.
    """)
    
    # Get high-gap regions from already filtered comparison data
    # Sort by population-weighted gap score to prioritize populated areas
    high_gap = coverage_comparison[
        coverage_comparison['balloon_count'] > 0
    ].nlargest(25, 'population_weighted_gap_score').copy()
    
    # Add actual balloon coordinates (average of balloons in each grid cell)
    if not balloon_data.empty:
        # Assign balloons to grid cells (must match grid_size used in analysis)
        grid_size = 2.0
        balloon_data_copy = balloon_data.copy()
        balloon_data_copy['grid_lat'] = (balloon_data_copy['latitude'] // grid_size) * grid_size + (grid_size / 2)
        balloon_data_copy['grid_lon'] = (balloon_data_copy['longitude'] // grid_size) * grid_size + (grid_size / 2)
        
        # Calculate average coordinates per grid cell
        balloon_coords = balloon_data_copy.groupby(['grid_lat', 'grid_lon']).agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        # Merge with high_gap to get actual coordinates
        high_gap = high_gap.merge(
            balloon_coords[['grid_lat', 'grid_lon', 'latitude', 'longitude']], 
            on=['grid_lat', 'grid_lon'], 
            how='left',
            suffixes=('_grid', '_actual')
        )
        
        # Use actual coordinates if available, otherwise fall back to grid centers
        high_gap['display_lat'] = high_gap['latitude'].fillna(high_gap['grid_lat'])
        high_gap['display_lon'] = high_gap['longitude'].fillna(high_gap['grid_lon'])
    
    if not high_gap.empty:
        # Format for display - use actual balloon coordinates if available
        lat_col = 'display_lat' if 'display_lat' in high_gap.columns else 'grid_lat'
        lon_col = 'display_lon' if 'display_lon' in high_gap.columns else 'grid_lon'
        
        display_df = high_gap[
            [lat_col, lon_col, 'balloon_count', 'station_count', 
             'observation_gap_score', 'estimated_population_millions',
             'population_weighted_gap_score', 'country_names']
        ].copy()
        
        display_df.columns = [
            'Latitude', 'Longitude', 'Balloons', 'Stations', 
            'Gap Score', 'Population (M)', 'Weighted Score', 'Countries'
        ]
        
        # Don't round - let column_config handle formatting
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Latitude": st.column_config.NumberColumn(
                    "Latitude",
                    format="%.4f°"
                ),
                "Longitude": st.column_config.NumberColumn(
                    "Longitude", 
                    format="%.4f°"
                ),
                "Gap Score": st.column_config.NumberColumn(
                    "Gap Score",
                    format="%.3f"
                ),
                "Balloons": st.column_config.NumberColumn(
                    "Balloons",
                    format="%d"
                ),
                "Stations": st.column_config.NumberColumn(
                    "Stations",
                    format="%d"
                ),
                "Population (M)": st.column_config.NumberColumn(
                    "Population (M)",
                    format="%.2f",
                    help="Estimated population in millions"
                ),
                "Weighted Score": st.column_config.NumberColumn(
                    "Weighted Score",
                    format="%.2f",
                    help="Gap score weighted by population (higher = more people affected)"
                )
            }
        )
    else:
        st.info("No high-gap regions found with current data.")


def display_inequality_metrics(coverage_comparison):
    """Display inequality metrics and analysis."""
    st.subheader("Global Inequality Metrics")
    
    st.markdown("""
    The **Gini coefficient** measures inequality in distribution (0 = perfect equality, 1 = perfect inequality).
    High values indicate coverage is concentrated in certain regions while others are underserved.
    """)
    
    metrics = analysis.calculate_inequality_metrics(coverage_comparison)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Station Gini Coefficient",
            f"{metrics['gini_stations']:.3f}",
            help="Inequality in station distribution across the globe"
        )
    
    with col2:
        st.metric(
            "Balloon Gini Coefficient",
            f"{metrics['gini_balloons']:.3f}",
            help="Inequality in balloon observation distribution"
        )
    
    with col3:
        st.metric(
            "Average Gap Score",
            f"{metrics['avg_gap_score']:.2f}",
            help="Average observation gap score across regions with balloons"
        )
    
    # Additional metrics
    with st.expander("Detailed Coverage Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Coverage Distribution:**")
            st.write(f"- Total grid cells analyzed: {metrics['total_cells']:,}")
            st.write(f"- Cells with stations: {metrics['cells_with_stations']:,}")
            st.write(f"- Cells with balloons: {metrics['cells_with_balloons']:,}")
            st.write(f"- Cells with both: {metrics['cells_with_both']:,}")
        
        with col2:
            st.write("**Density Metrics:**")
            st.write(f"- Avg stations per cell: {metrics['avg_stations_per_cell']:.2f}")
            st.write(f"- Avg balloons per cell: {metrics['avg_balloons_per_cell']:.2f}")
            st.write(f"- Max gap score: {metrics['max_gap_score']:.2f}")


def display_conclusion():
    """Display concluding narrative about climate equity."""
    st.subheader("Why High-Altitude Sensing Matters for Climate Equity")
    
    st.markdown("""
    The unequal distribution of weather observations directly impacts global climate science, weather forecasting accuracy, and disaster 
    preparedness. When regions lack observational infrastructure, they become:
    
    - **Data Deserts**: Absent from global atmospheric models, reducing forecast accuracy 
      for the populations living there
    - **Climate Blind Spots**: Excluded from long-term climate monitoring, making it harder 
      to understand regional climate change impacts
    - **Forecast Gaps**: Unable to provide early warning for extreme weather events, 
      increasing vulnerability to disasters
    
    High-altitude balloon sensing technologies offer a path toward more equitable atmospheric 
    observation. By accessing previously under-sampled regions—remote areas, oceans, and 
    underinvested nations. These platforms can help democratize access to the atmospheric 
    data that underpins weather forecasts, climate models, and environmental policy.
    
    **The Path Forward**: Achieving truly global atmospheric monitoring requires both expanded 
    ground infrastructure in underserved regions and complementary mobile sensing platforms. 
    Balloons, drones, and satellite systems each play a role in filling observational gaps, 
    ensuring that all regions contribute to—and benefit from—global atmospheric science.
    
    Climate change is a global phenomenon and understanding it requires equitable global data.
    """)
    
    # Add a subtle call to action
    st.markdown("---")
    st.caption("""
    *This visualization analyzes publicly available data from Windborne Systems and NOAA. 
    The observation gap score is calculated as balloon_count / (station_count + 1) for 
    each 2° × 2° grid cell.*
    """)


if __name__ == "__main__":
    main()
