import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

# Add these lines for better deployment performance
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Page configuration
st.set_page_config(
    page_title="Bird Migration Dashboard",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Bird Migration Analysis Dashboard")
st.markdown("### Visualizing patterns and routes of migratory birds")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    try:
        df = pd.read_csv('Bird_Migration_Data_with_Origin.csv')
        if len(df) > 0:
            st.sidebar.success(f"✅ Loaded {len(df)} migration records")
            return df
        else:
            st.sidebar.warning("CSV file is empty. Using sample data.")
            return create_sample_data()
    except FileNotFoundError:
        st.sidebar.info("📋 CSV file not found. Using sample data for demonstration.")
        return create_sample_data()
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {str(e)}")
        st.sidebar.info("📋 Using sample data instead.")
        return create_sample_data()

def create_sample_data():
    return pd.DataFrame({
        'Species': np.random.choice(['Eagle', 'Falcon', 'Stork', 'Swallow'], 100),
        'Start_Latitude': np.random.uniform(20, 60, 100),
        'Start_Longitude': np.random.uniform(-120, -60, 100),
        'End_Latitude': np.random.uniform(0, 40, 100),
        'End_Longitude': np.random.uniform(-100, -40, 100),
        'Flight_Distance_km': np.random.uniform(500, 3000, 100),
        'Migration_Success': np.random.choice([True, False], 100, p=[0.8, 0.2]),
        'Migration_Start_Month': np.random.choice(['January', 'February', 'March', 'April', 'May', 'June', 
                                                 'July', 'August', 'September', 'October', 'November', 'December'], 100),
        'Region': np.random.choice(['North America', 'South America', 'Europe', 'Asia'], 100),
        'Migration_Reason': np.random.choice(['Breeding', 'Wintering', 'Feeding'], 100)
    })

# Helper functions
def convert_month_to_number(month_value):
    """Convert month name or number to numeric value"""
    if pd.isna(month_value):
        return None
    
    try:
        num_val = float(month_value)
        if 1 <= num_val <= 12:
            return int(num_val)
    except (ValueError, TypeError):
        pass
    
    # Month name mapping
    month_mapping = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9, 'sept': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }
    
    # Convert to lowercase string and look up
    month_str = str(month_value).lower().strip()
    return month_mapping.get(month_str, None)

def get_season_from_month(month_value):
    """Get season from month value (handles both numeric and text)"""
    month_num = convert_month_to_number(month_value)
    
    if month_num is None:
        return 'Unknown'
    
    if month_num in [12, 1, 2]:
        return 'Winter'
    elif month_num in [3, 4, 5]:
        return 'Spring'
    elif month_num in [6, 7, 8]:
        return 'Summer'
    elif month_num in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Unknown'

@st.cache_data
def prepare_migration_data_flexible(df):
    """Prepare migration data with flexible column handling"""
    columns_to_check = ['Start_Latitude', 'Start_Longitude', 'End_Latitude', 'End_Longitude', 'Species']
    
    # Check if required columns exist
    if not all(col in df.columns for col in columns_to_check):
        st.error("Missing required columns in dataset. Please check your data file.")
        return pd.DataFrame()
    
    # Filter out missing coordinates
    df_clean = df.dropna(subset=['Start_Latitude', 'Start_Longitude', 'End_Latitude', 'End_Longitude'])
    
    # Process month data if available
    if 'Migration_Start_Month' in df_clean.columns:
        df_clean['Migration_Start_Month_Numeric'] = df_clean['Migration_Start_Month'].apply(convert_month_to_number)
        df_clean['Start_Season'] = df_clean['Migration_Start_Month'].apply(get_season_from_month)
    
    return df_clean

@st.cache_data
def create_monthly_animation(df_clean, migration_reason_filter=None, show_routes=True, animation_speed=1000):
    """Create enhanced monthly animation with routes and migration reason filtering"""
    
    if len(df_clean) == 0:
        st.warning("No data available for monthly animation")
        return None
    
    # Apply migration reason filter
    if migration_reason_filter and 'Migration_Reason' in df_clean.columns:
        df_filtered = df_clean[df_clean['Migration_Reason'].isin(migration_reason_filter)]
        if len(df_filtered) == 0:
            st.warning("No data available for selected migration reasons")
            return None
    else:
        df_filtered = df_clean.copy()

    month_col = None
    if 'Migration_Start_Month_Numeric' in df_filtered.columns:
        month_col = 'Migration_Start_Month_Numeric'
    elif 'Migration_Start_Month' in df_filtered.columns:
        df_filtered['Migration_Start_Month_Numeric'] = df_filtered['Migration_Start_Month'].apply(convert_month_to_number)
        month_col = 'Migration_Start_Month_Numeric'
    
    if month_col is None or df_filtered[month_col].isna().all():
        st.warning("No valid month data available for animation")
        return None
    
    df_with_months = df_filtered.dropna(subset=[month_col])
    
    if len(df_with_months) == 0:
        st.warning("No records with valid month data")
        return None
    
    # Sample data for performance
    if len(df_with_months) > 500:
        df_sample = df_with_months.sample(n=500, random_state=42)
    else:
        df_sample = df_with_months
    
    # Create month labels for proper ordering
    month_labels = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August', 
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    # Add month labels to data
    df_sample['Month_Label'] = df_sample[month_col].map(month_labels)
    
    if show_routes and all(col in df_sample.columns for col in ['End_Latitude', 'End_Longitude']):
        # Create route-based visualization
        return create_route_animation_plotly(df_sample, month_col, animation_speed, month_labels)
    else:
        # Create point-based visualization (your original approach)
        return create_point_animation_plotly(df_sample, month_col, animation_speed, month_labels)

def create_route_animation_plotly(df_sample, month_col, animation_speed, month_labels):
    """Create animation showing migration routes by month"""
    
    # Determine color column (prefer Migration_Reason, fallback to Species)
    if 'Migration_Reason' in df_sample.columns and not df_sample['Migration_Reason'].isna().all():
        color_col = 'Migration_Reason'
    else:
        color_col = 'Species'
    
    fig = go.Figure()
    
    # Get color palette
    unique_categories = df_sample[color_col].unique()
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}
    
    frames = []
    available_months = sorted(df_sample[month_col].dropna().unique())
    
    for month_num in available_months:
        if month_num not in month_labels:
            continue
            
        month_data = df_sample[df_sample[month_col] == month_num]
        frame_traces = []
        
        legend_added = set()
        
        for _, row in month_data.iterrows():
            category = row[color_col]
            color = color_map[category]
            
            # Add migration route line
            frame_traces.append(
                go.Scattergeo(
                    lon=[row['Start_Longitude'], row['End_Longitude']],
                    lat=[row['Start_Latitude'], row['End_Latitude']],
                    mode='lines+markers',
                    line=dict(width=2, color=color),
                    marker=dict(
                        size=[10, 15], 
                        color=color,
                        symbol=['circle', 'triangle-up'],
                        line=dict(width=1, color='white')
                    ),
                    name=category,
                    legendgroup=category,
                    showlegend=(category not in legend_added),
                    hovertemplate=(
                        f'<b>{category}</b><br>' +
                        f'Species: {row.get("Species", "N/A")}<br>' +
                        f'Distance: {row.get("Flight_Distance_km", "N/A"):.0f} km<br>' +
                        f'Region: {row.get("Region", "N/A")}<br>' +
                        f'Month: {month_labels[month_num]}<br>' +
                        '<extra></extra>'
                    )
                )
            )
            legend_added.add(category)
        
        frames.append(go.Frame(
            data=frame_traces,
            name=str(month_num),
            layout=dict(title=f"Migration Routes - {month_labels[month_num]}")
        ))
    
    if frames:
        fig.add_traces(frames[0].data)

    slider_steps = []
    for month_num in available_months:
        if month_num in month_labels:
            slider_steps.append(dict(
                args=[[str(month_num)], 
                      dict(mode="immediate", 
                           frame=dict(duration=animation_speed, redraw=True),
                           transition=dict(duration=300))],
                label=month_labels[month_num][:3],  # Jan, Feb, etc.
                method="animate"
            ))

    fig.update_layout(
        title=dict(
            text='Monthly Migration Routes by Migration Reason<br><sub>Trace seasonal migration patterns month by month</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        geo=dict(
            projection_type='natural earth',
            showland=True,
            showcountries=True,
            countrywidth=0.5,
            showocean=True,
            oceancolor='lightblue',
            landcolor='lightgray',
            coastlinecolor='gray'
        ),
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Month: "},
            pad={"t": 50},
            steps=slider_steps
        )],
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=animation_speed, redraw=True),
                                    transition=dict(duration=300),
                                    fromcurrent=True,
                                    mode='immediate')]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                      mode='immediate',
                                      transition=dict(duration=0))])
            ],
            x=0.1, y=0, 
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray'
        )],
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)"
        )
    )

    fig.frames = frames
    
    return fig

def create_point_animation_plotly(df_sample, month_col, animation_speed, month_labels):
    """Create point-based animation (enhanced version of your original)"""
    
    color_column = 'Migration_Reason' if 'Migration_Reason' in df_sample.columns else 'Species'
    
    # Create the scatter plot
    fig = px.scatter_geo(
        df_sample,
        lat='Start_Latitude',
        lon='Start_Longitude',
        color=color_column,
        size='Flight_Distance_km' if 'Flight_Distance_km' in df_sample.columns else None,
        animation_frame=month_col,
        hover_data=['Species', 'Migration_Reason', 'Region', 'Flight_Distance_km'] if all(col in df_sample.columns for col in ['Species', 'Migration_Reason', 'Region', 'Flight_Distance_km']) else None,
        title='Bird Migration Starting Points by Month and Migration Reason',
        projection="natural earth",
        height=700
    )
    
    for frame in fig.frames:
        month_num = int(frame.name)
        if month_num in month_labels:
            frame.layout.update(title=f"Migration Patterns - {month_labels[month_num]}")
    
    for step in fig.layout.sliders[0].steps:
        month_num = int(step.args[0][0])
        if month_num in month_labels:
            step.label = month_labels[month_num][:3]

    if len(fig.frames) > 0:
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = animation_speed
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300

    fig.update_layout(
        title=dict(
            text='Monthly Migration Patterns by Migration Reason<br><sub>Starting points colored by migration purpose</sub>',
            x=0.5,
            font=dict(size=16)
        )
    )
    
    return fig

def generate_seasonal_insights(data):
    """Generate insights about seasonal migration patterns"""
    insights = []
    
    if 'Migration_Start_Month_Numeric' not in data.columns and 'Migration_Start_Month' in data.columns:
        data['Migration_Start_Month_Numeric'] = data['Migration_Start_Month'].apply(convert_month_to_number)
    
    if 'Migration_Start_Month_Numeric' in data.columns:
        # Peak migration months
        month_counts = data['Migration_Start_Month_Numeric'].value_counts().sort_index()
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        if len(month_counts) > 0:
            peak_month = month_counts.idxmax()
            peak_count = month_counts.max()
            insights.append(f"Peak migration occurs in **{month_names[peak_month]}** with {peak_count} recorded migrations")
            
            # Seasonal distribution
            spring_months = [3, 4, 5]
            summer_months = [6, 7, 8]
            fall_months = [9, 10, 11]
            winter_months = [12, 1, 2]
            
            spring_count = month_counts[month_counts.index.isin(spring_months)].sum()
            summer_count = month_counts[month_counts.index.isin(summer_months)].sum()
            fall_count = month_counts[month_counts.index.isin(fall_months)].sum()
            winter_count = month_counts[month_counts.index.isin(winter_months)].sum()
            
            seasonal_data = {'Spring': spring_count, 'Summer': summer_count, 
                           'Fall': fall_count, 'Winter': winter_count}
            dominant_season = max(seasonal_data, key=seasonal_data.get)
            insights.append(f"**{dominant_season}** is the dominant migration season ({seasonal_data[dominant_season]} migrations)")

    if 'Migration_Reason' in data.columns and 'Migration_Start_Month_Numeric' in data.columns:
        reason_timing = data.groupby(['Migration_Reason', 'Migration_Start_Month_Numeric']).size().reset_index(name='count')
        
        for reason in data['Migration_Reason'].unique():
            reason_data = reason_timing[reason_timing['Migration_Reason'] == reason]
            if len(reason_data) > 0:
                peak_month = reason_data.loc[reason_data['count'].idxmax(), 'Migration_Start_Month_Numeric']
                if peak_month in month_names:
                    insights.append(f"**{reason}** migrations peak in **{month_names[peak_month]}**")
    
    return insights

def generate_behavioral_insights(data):
    """Generate insights about migration behavior patterns"""
    insights = []
    
    # Distance analysis
    if 'Flight_Distance_km' in data.columns:
        avg_distance = data['Flight_Distance_km'].mean()
        max_distance = data['Flight_Distance_km'].max()
        min_distance = data['Flight_Distance_km'].min()
        
        insights.append(f"Average migration distance is **{avg_distance:.0f} km**")
        insights.append(f"Longest recorded journey: **{max_distance:.0f} km**")
        
        # Distance by migration reason
        if 'Migration_Reason' in data.columns:
            distance_by_reason = data.groupby('Migration_Reason')['Flight_Distance_km'].mean().sort_values(ascending=False)
            longest_reason = distance_by_reason.index[0]
            longest_distance = distance_by_reason.iloc[0]
            insights.append(f"**{longest_reason}** migrations cover the greatest distances (avg: {longest_distance:.0f} km)")
    
    # Speed analysis
    if 'Average_Speed_kmph' in data.columns:
        avg_speed = data['Average_Speed_kmph'].mean()
        insights.append(f"Average migration speed is **{avg_speed:.1f} km/h**")
        
        # Speed by species
        if 'Species' in data.columns:
            speed_by_species = data.groupby('Species')['Average_Speed_kmph'].mean().sort_values(ascending=False)
            if len(speed_by_species) > 0:
                fastest_species = speed_by_species.index[0]
                fastest_speed = speed_by_species.iloc[0]
                insights.append(f"**{fastest_species}** are the fastest migrants (avg: {fastest_speed:.1f} km/h)")
    
    # Flock behavior
    if 'Migrated_in_Flock' in data.columns:
        flock_percentage = (data['Migrated_in_Flock'] == 'Yes').mean() * 100
        insights.append(f"**{flock_percentage:.1f}%** of migrations occur in flocks")
        
        if 'Flock_Size' in data.columns:
            avg_flock_size = data[data['Migrated_in_Flock'] == 'Yes']['Flock_Size'].mean()
            if not pd.isna(avg_flock_size):
                insights.append(f"Average flock size is **{avg_flock_size:.0f} birds**")
    
    # Success rates - FIXED VERSION
    if 'Migration_Success' in data.columns:
        success_values = data['Migration_Success'].unique()
        if 'Success' in success_values:
            success_rate = (data['Migration_Success'] == 'Success').mean() * 100
        elif 'Successful' in success_values:
            success_rate = (data['Migration_Success'] == 'Successful').mean() * 100
        elif True in success_values:
            success_rate = (data['Migration_Success'] == True).mean() * 100
        else:
            success_rate = 0
        insights.append(f"Migration success rate is **{success_rate:.1f}%**")
    
    return insights

def generate_geographic_insights(data):
    """Generate insights about geographic patterns"""
    insights = []
    
    # Regional analysis
    if 'Region' in data.columns:
        region_counts = data['Region'].value_counts()
        most_active_region = region_counts.index[0]
        insights.append(f"**{most_active_region}** shows the highest migration activity ({region_counts.iloc[0]} records)")
    
    # Altitude patterns
    if 'Max_Altitude_m' in data.columns:
        avg_altitude = data['Max_Altitude_m'].mean()
        max_altitude = data['Max_Altitude_m'].max()
        insights.append(f"Birds fly at an average maximum altitude of **{avg_altitude:.0f} meters**")
        insights.append(f"Highest recorded altitude: **{max_altitude:.0f} meters**")
    
    # Weather impact - FIXED VERSION
    if 'Weather_Condition' in data.columns and 'Migration_Success' in data.columns:
        success_values = data['Migration_Success'].unique()
        if 'Success' in success_values:
            weather_success = data.groupby('Weather_Condition')['Migration_Success'].apply(
                lambda x: (x == 'Success').mean() * 100).sort_values(ascending=False)
        elif 'Successful' in success_values:
            weather_success = data.groupby('Weather_Condition')['Migration_Success'].apply(
                lambda x: (x == 'Successful').mean() * 100).sort_values(ascending=False)
        elif True in success_values:
            weather_success = data.groupby('Weather_Condition')['Migration_Success'].apply(
                lambda x: (x == True).mean() * 100).sort_values(ascending=False)
        else:
            weather_success = pd.Series()
        
        if len(weather_success) > 0:
            best_weather = weather_success.index[0]
            best_success_rate = weather_success.iloc[0]
            insights.append(f"**{best_weather}** weather conditions show highest success rates")
    
    # Rest stop patterns
    if 'Rest_Stops' in data.columns:
        avg_rest_stops = data['Rest_Stops'].mean()
        insights.append(f"Birds make an average of **{avg_rest_stops:.1f} rest stops** during migration")
        
        # Rest stops vs success - FIXED VERSION
        if 'Migration_Success' in data.columns:
            success_values = data['Migration_Success'].unique()
            if 'Success' in success_values:
                successful_rest_stops = data[data['Migration_Success'] == 'Success']['Rest_Stops'].mean()
                failed_rest_stops = data[data['Migration_Success'] == 'Failed']['Rest_Stops'].mean()
            elif 'Successful' in success_values:
                successful_rest_stops = data[data['Migration_Success'] == 'Successful']['Rest_Stops'].mean()
                failed_rest_stops = data[data['Migration_Success'] == 'Failed']['Rest_Stops'].mean()
            elif True in success_values:
                successful_rest_stops = data[data['Migration_Success'] == True]['Rest_Stops'].mean()
                failed_rest_stops = data[data['Migration_Success'] == False]['Rest_Stops'].mean()
            else:
                successful_rest_stops = failed_rest_stops = np.nan
            
            if not pd.isna(successful_rest_stops) and not pd.isna(failed_rest_stops):
                if successful_rest_stops > failed_rest_stops:
                    insights.append(f"Successful migrations involve more rest stops (avg: {successful_rest_stops:.1f} vs {failed_rest_stops:.1f})")
                else:
                    insights.append(f"Failed migrations tend to have more rest stops (avg: {failed_rest_stops:.1f} vs {successful_rest_stops:.1f})")
    
    return insights

def display_summary_stats(data):
    """Display summary statistics in a nice format"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Migrations", len(data))
        
    with col2:
        if 'Species' in data.columns:
            st.metric("Species Count", data['Species'].nunique())
        
    with col3:
        if 'Flight_Distance_km' in data.columns:
            avg_distance = data['Flight_Distance_km'].mean()
            st.metric("Avg Distance (km)", f"{avg_distance:.0f}")
    
    # Additional detailed statistics
    if st.checkbox("Show Detailed Statistics", key="detailed_stats"):
        st.subheader("Detailed Breakdown")
        
        # Species distribution
        if 'Species' in data.columns:
            st.write("**Species Distribution:**")
            species_dist = data['Species'].value_counts().head(10)
            st.bar_chart(species_dist)
        
        # Migration reasons
        if 'Migration_Reason' in data.columns:
            st.write("**Migration Reasons:**")
            reason_dist = data['Migration_Reason'].value_counts()
            st.bar_chart(reason_dist)

@st.cache_data
def create_animated_migration_globe(df, max_paths=60):
    """
    Create animated migration globe with lines and arrows
    """
    
    # Prepare data
    required_data = df[['Start_Latitude', 'Start_Longitude', 'End_Latitude', 'End_Longitude', 
                       'Species', 'Migration_Success']].copy()
    
    # Clean and convert Migration_Success to boolean
    required_data = required_data.dropna()
    
    # Handle different Migration_Success formats
    success_values = required_data['Migration_Success'].unique()
    if 'Success' in success_values:
        required_data['Migration_Success'] = required_data['Migration_Success'] == 'Success'
    elif 'Successful' in success_values:
        required_data['Migration_Success'] = required_data['Migration_Success'] == 'Successful'
    else:
        required_data['Migration_Success'] = required_data['Migration_Success'].astype(bool)
    
    # Filter successful migrations
    successful_data = required_data[required_data['Migration_Success'] == True]
    
    # Get unique species
    unique_species = sorted(successful_data['Species'].unique())
    
    # Create base figure
    fig = go.Figure()
    
    # Add static migration paths
    species_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    
    for i, species in enumerate(unique_species):
        species_data = successful_data[successful_data['Species'] == species]
        
        # Sample data
        if len(species_data) > max_paths // max(1, len(unique_species)):
            species_data = species_data.sample(n=max_paths // max(1, len(unique_species)), random_state=42)
        
        color = species_colors[i % len(species_colors)]
        
        for _, row in species_data.iterrows():
            # Create full path line
            fig.add_trace(
                go.Scattergeo(
                    lon=[row['Start_Longitude'], row['End_Longitude']],
                    lat=[row['Start_Latitude'], row['End_Latitude']],
                    mode='lines',
                    line=dict(width=1, color=color, dash='dot'),
                    opacity=0.5,
                    name=species,
                    legendgroup=species,
                    showlegend=(i == 0),  # Only show legend once per species
                    hoverinfo='skip'  # Don't show hover for static lines
                )
            )
    
    frames = []
    
    for frame_idx in range(1, 11):  # 10 frames for animation
        frame_data = []
        
        for i, species in enumerate(unique_species):
            species_data = successful_data[successful_data['Species'] == species]
            
            if len(species_data) > max_paths // max(1, len(unique_species)):
                species_data = species_data.sample(n=max_paths // max(1, len(unique_species)), random_state=42)
            
            color = species_colors[i % len(species_colors)]
            
            for _, row in species_data.iterrows():
                # Calculate current position
                start_lat = row['Start_Latitude']
                start_lon = row['Start_Longitude']
                end_lat = row['End_Latitude']
                end_lon = row['End_Longitude']
                
                # Handle longitude wrapping
                if abs(end_lon - start_lon) > 180:
                    if start_lon < end_lon:
                        start_lon += 360
                    else:
                        end_lon += 360
                
                current_lat = start_lat + (end_lat - start_lat) * (frame_idx / 10)
                current_lon = start_lon + (end_lon - start_lon) * (frame_idx / 10)
                current_lon = current_lon % 360
                
                # Create arrow marker
                arrow_trace = go.Scattergeo(
                    lon=[current_lon],
                    lat=[current_lat],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color,
                        symbol='arrow',
                        angleref="previous",
                        angle=(np.degrees(np.arctan2(end_lat - start_lat, end_lon - start_lon)) + 90) % 360
                    ),
                    name=species,
                    legendgroup=species,
                    showlegend=False,
                    hovertemplate=f'<b>{species}</b><br>Progress: {frame_idx*10}%<extra></extra>'
                )
                
                frame_data.append(arrow_trace)
        
        frames.append(go.Frame(data=frame_data, name=f'frame{frame_idx}'))
    
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Progress: "},
            pad={"t": 50},
            steps=[dict(
                args=[[f'frame{k}'], 
                       dict(mode="immediate",
                            frame=dict(duration=300, redraw=True),
                            transition=dict(duration=0))],
                label=f'{k*10}%',
                method="animate"
            ) for k in range(1, 11)]
        )],
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(args=[{'geo.projection.rotation': {'lon': 0, 'lat': 20, 'roll': 0}}], label='Default', method='relayout'),
                    dict(args=[{'geo.projection.rotation': {'lon': -100, 'lat': 40, 'roll': 0}}], label='N.America', method='relayout'),
                    dict(args=[{'geo.projection.rotation': {'lon': 10, 'lat': 50, 'roll': 0}}], label='Europe', method='relayout'),
                    dict(args=[{'geo.projection.rotation': {'lon': 100, 'lat': 30, 'roll': 0}}], label='Asia', method='relayout'),
                ],
                x=0.02, y=0.15,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgb(150, 150, 150)',
                font=dict(size=10)
            )
        ],
        title=dict(
            text='Animated Bird Migration Paths<br><sub>Use slider to see migration progress</sub>',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=18, color='rgb(70, 70, 70)')
        ),
        geo=dict(
            projection_type='orthographic',
            showland=True,
            showcountries=True,
            showocean=True,
            countrywidth=0.5,
            landcolor='rgb(230, 230, 230)',
            oceancolor='rgba(173, 216, 230, 0.6)',
            projection=dict(
                type='orthographic',
                rotation=dict(lon=0, lat=20, roll=0),
                scale=1.0
            ),
            showcoastlines=True,
            coastlinewidth=1,
            coastlinecolor='rgb(120, 120, 120)',
        ),
        width=1400,
        height=1000,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)"
        )
    )
    
    # Add frames to figure
    fig.frames = frames
    
    return fig

def create_weather_impact_dashboard(df):
    """Multi-panel dashboard showing weather effects on migration"""
    
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Success Rate by Weather', 'Speed vs Wind', 
                       'Temperature Impact', 'Visibility vs Success'),
        specs=[[{"secondary_y": True}, {"type": "scatter"}],
               [{"type": "violin"}, {"type": "box"}]]
    )
    
    # Success rate by weather condition - FIXED VERSION
    if 'Weather_Condition' in df.columns and 'Migration_Success' in df.columns:
        success_values = df['Migration_Success'].unique()
        if 'Success' in success_values:
            weather_success = df.groupby('Weather_Condition')['Migration_Success'].apply(
                lambda x: (x == 'Success').mean() * 100).reset_index()
        elif 'Successful' in success_values:
            weather_success = df.groupby('Weather_Condition')['Migration_Success'].apply(
                lambda x: (x == 'Successful').mean() * 100).reset_index()
        elif True in success_values:
            weather_success = df.groupby('Weather_Condition')['Migration_Success'].apply(
                lambda x: (x == True).mean() * 100).reset_index()
        else:
            weather_success = pd.DataFrame()
        
        if len(weather_success) > 0:
            fig.add_trace(
                go.Bar(x=weather_success['Weather_Condition'], 
                       y=weather_success['Migration_Success'],
                       name='Success Rate %'),
                row=1, col=1
            )
    
    # Speed vs Wind Speed
    if 'Wind_Speed_kmph' in df.columns and 'Average_Speed_kmph' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Wind_Speed_kmph'], y=df['Average_Speed_kmph'],
                      mode='markers', 
                      marker=dict(color=df['Temperature_C'] if 'Temperature_C' in df.columns else 'blue', 
                                colorscale='RdYlBu'),
                      name='Speed vs Wind'),
            row=1, col=2
        )
    
    # Temperature distribution by success
    if 'Temperature_C' in df.columns and 'Migration_Success' in df.columns:
        for success in df['Migration_Success'].unique():
            temp_data = df[df['Migration_Success'] == success]['Temperature_C']
            fig.add_trace(
                go.Violin(y=temp_data, name=f'Temp - {success}'),
                row=2, col=1
            )
    
    # Visibility vs Success
    if 'Visibility_km' in df.columns and 'Migration_Success' in df.columns:
        fig.add_trace(
            go.Box(x=df['Migration_Success'], y=df['Visibility_km'],
                   name='Visibility'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title='Weather Impact on Migration')
    return fig

def create_migration_flow_animation(bird_data):
    """Animated flow showing migration patterns over months"""
    
    # Prepare monthly data
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    bird_data['Month_Num'] = bird_data['Migration_Start_Month'].map(
        {month: i+1 for i, month in enumerate(month_order)})
    
    fig = px.scatter_geo(
        bird_data,
        lat='Start_Latitude',
        lon='Start_Longitude', 
        color='Species',
        size='Flock_Size' if 'Flock_Size' in bird_data.columns else None,
        animation_frame='Month_Num',
        hover_data=['Migration_Reason', 'Weather_Condition', 'Flight_Distance_km'] if all(col in bird_data.columns for col in ['Migration_Reason', 'Weather_Condition', 'Flight_Distance_km']) else None,
        title='Monthly Migration Flow Animation'
    )
    
    # Add destination points
    fig.add_trace(
        go.Scattergeo(
            lat=bird_data['End_Latitude'],
            lon=bird_data['End_Longitude'],
            mode='markers',
            marker=dict(size=8, symbol='triangle-up', color='red'),
            name='Destinations'
        )
    )
    
    return fig

def create_environmental_filters(data):
    """Create interactive filters for environmental analysis"""
    
    st.markdown("### 🎛️ Environmental Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Weather condition filter
        if 'Weather_Condition' in data.columns:
            weather_options = ['All'] + list(data['Weather_Condition'].dropna().unique())
            st.session_state.weather_filter = st.selectbox(
                "Weather Condition",
                options=weather_options,
                index=0,
                key="weather_select"
            )
    
    with col2:
        # Habitat filter
        if 'Habitat' in data.columns:
            habitat_options = ['All'] + list(data['Habitat'].dropna().unique())
            st.session_state.habitat_filter = st.selectbox(
                "Habitat Type",
                options=habitat_options,
                index=0,
                key="habitat_select"
            )
    
    with col3:
        # Food supply filter
        if 'Food_Supply_Level' in data.columns:
            food_options = ['All'] + list(data['Food_Supply_Level'].dropna().unique())
            st.session_state.food_filter = st.selectbox(
                "Food Supply Level",
                options=food_options,
                index=0,
                key="food_select"
            )
    
    with col4:
        # Temperature range
        if 'Temperature_C' in data.columns:
            temp_range = st.slider(
                "Temperature Range (°C)",
                min_value=int(data['Temperature_C'].min()),
                max_value=int(data['Temperature_C'].max()),
                value=(int(data['Temperature_C'].min()), int(data['Temperature_C'].max())),
                key="temp_range"
            )
            st.session_state.temp_filter = temp_range

def apply_environmental_filters(data):
    """Apply the selected filters to the data"""
    filtered_data = data.copy()

    if hasattr(st.session_state, 'weather_filter') and st.session_state.weather_filter != 'All':
        filtered_data = filtered_data[filtered_data['Weather_Condition'] == st.session_state.weather_filter]

    if hasattr(st.session_state, 'habitat_filter') and st.session_state.habitat_filter != 'All':
        filtered_data = filtered_data[filtered_data['Habitat'] == st.session_state.habitat_filter]

    if hasattr(st.session_state, 'food_filter') and st.session_state.food_filter != 'All':
        filtered_data = filtered_data[filtered_data['Food_Supply_Level'] == st.session_state.food_filter]

    if hasattr(st.session_state, 'temp_filter'):
        temp_min, temp_max = st.session_state.temp_filter
        filtered_data = filtered_data[
            (filtered_data['Temperature_C'] >= temp_min) & 
            (filtered_data['Temperature_C'] <= temp_max)
        ]
    
    return filtered_data

def create_environmental_insights_dashboard(data):
    """Create the main environmental insights dashboard"""

    create_environmental_metrics(data)
    
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        create_environmental_sweet_spot(data)
    
    with col2:
        # Risk Assessment Gauge
        create_risk_assessment_gauge(data)
    
    st.markdown("---")
    
    create_environmental_correlations(data)
    
    st.markdown("---")
    
    create_species_environmental_preferences(data)

    create_environmental_insights_panel(data)

def create_environmental_metrics(data):
    """Create key environmental metrics display"""
    
    st.markdown("### 📊 Environmental Impact Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if 'Migration_Success' in data.columns:
            # FIXED VERSION - handle different success formats
            success_values = data['Migration_Success'].unique()
            if 'Success' in success_values:
                success_rate = (data['Migration_Success'] == 'Success').mean() * 100
            elif 'Successful' in success_values:
                success_rate = (data['Migration_Success'] == 'Successful').mean() * 100
            elif True in success_values:
                success_rate = (data['Migration_Success'] == True).mean() * 100
            else:
                success_rate = 0
                
            st.metric(
                "Success Rate", 
                f"{success_rate:.1f}%",
                delta=f"{success_rate - 75:.1f}%" if success_rate != 75 else None
            )
    
    with col2:
        if 'Temperature_C' in data.columns:
            avg_temp = data['Temperature_C'].mean()
            st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    
    with col3:
        if 'Wind_Speed_kmph' in data.columns:
            avg_wind = data['Wind_Speed_kmph'].mean()
            st.metric("Avg Wind Speed", f"{avg_wind:.1f} km/h")
    
    with col4:
        if 'Food_Supply_Level' in data.columns:
            food_distribution = data['Food_Supply_Level'].value_counts()
            dominant_food_level = food_distribution.index[0] if len(food_distribution) > 0 else "N/A"
            st.metric("Dominant Food Level", dominant_food_level)
    
    with col5:
        risk_score = calculate_environmental_risk_score(data)
        st.metric(
            "Environmental Risk", 
            f"{risk_score:.1f}/10",
            delta=f"{5.0 - risk_score:.1f}" if risk_score != 5.0 else None,
            delta_color="inverse"
        )

def create_environmental_sweet_spot(data):
    """Create environmental sweet spot visualization"""
    
    st.subheader("🎯 Environmental Sweet Spot Analysis")
    
    if all(col in data.columns for col in ['Temperature_C', 'Wind_Speed_kmph', 'Migration_Success']):
        
        # FIXED VERSION - handle different success formats
        success_values = data['Migration_Success'].unique()
        if 'Success' in success_values:
            success_data = data[data['Migration_Success'] == 'Success']
            failed_data = data[data['Migration_Success'] == 'Failed']
        elif 'Successful' in success_values:
            success_data = data[data['Migration_Success'] == 'Successful']
            failed_data = data[data['Migration_Success'] == 'Failed']
        elif True in success_values:
            success_data = data[data['Migration_Success'] == True]
            failed_data = data[data['Migration_Success'] == False]
        else:
            success_data = failed_data = pd.DataFrame()
        
        if len(success_data) > 0 or len(failed_data) > 0:
            fig = go.Figure()
            
            # Add successful migrations
            if len(success_data) > 0:
                fig.add_trace(go.Scatter(
                    x=success_data['Temperature_C'],
                    y=success_data['Wind_Speed_kmph'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='green',
                        opacity=0.6,
                        line=dict(width=1, color='darkgreen')
                    ),
                    name='Successful',
                    hovertemplate='Temp: %{x}°C<br>Wind: %{y} km/h<br>Status: Successful<extra></extra>'
                ))
            
            # Add failed migrations
            if len(failed_data) > 0:
                fig.add_trace(go.Scatter(
                    x=failed_data['Temperature_C'],
                    y=failed_data['Wind_Speed_kmph'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        opacity=0.6,
                        line=dict(width=1, color='darkred')
                    ),
                    name='Failed',
                    hovertemplate='Temp: %{x}°C<br>Wind: %{y} km/h<br>Status: Failed<extra></extra>'
                ))
            
            # Add sweet spot zone
            if len(success_data) > 10:
                temp_sweet = success_data['Temperature_C'].quantile([0.25, 0.75])
                wind_sweet = success_data['Wind_Speed_kmph'].quantile([0.25, 0.75])
                
                fig.add_shape(
                    type="rect",
                    x0=temp_sweet.iloc[0], y0=wind_sweet.iloc[0],
                    x1=temp_sweet.iloc[1], y1=wind_sweet.iloc[1],
                    fillcolor="rgba(0,255,0,0.1)",
                    line=dict(color="green", width=2, dash="dash"),
                )
                
                fig.add_annotation(
                    x=temp_sweet.mean(),
                    y=wind_sweet.mean(),
                    text="Sweet Spot Zone",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="green"
                )
            
            fig.update_layout(
                title="Optimal Environmental Conditions for Migration Success",
                xaxis_title="Temperature (°C)",
                yaxis_title="Wind Speed (km/h)",
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Optimal conditions info
            if len(success_data) > 10:
                optimal_temp = success_data['Temperature_C'].median()
                optimal_wind = success_data['Wind_Speed_kmph'].median()
                
                st.info(f"🎯 **Optimal Conditions**: {optimal_temp:.1f}°C temperature, {optimal_wind:.1f} km/h wind speed")

def create_risk_assessment_gauge(data):
    """Create environmental risk assessment gauge"""
    
    st.subheader("⚠️ Risk Assessment")
    
    risk_score = calculate_environmental_risk_score(data)

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Environmental Risk Level"},
        delta = {'reference': 5.0},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 6], 'color': "yellow"},
                {'range': [6, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk interpretation
    if risk_score <= 3:
        st.success("🟢 **Low Risk**: Excellent environmental conditions for migration")
    elif risk_score <= 6:
        st.warning("🟡 **Moderate Risk**: Acceptable conditions with some challenges")
    else:
        st.error("🔴 **High Risk**: Challenging environmental conditions")

def create_environmental_correlations(data):
    """Create environmental factors correlation analysis"""
    
    st.subheader("🔗 Environmental Factor Relationships")
    
    # Select relevant environmental columns
    env_columns = []
    for col in ['Temperature_C', 'Wind_Speed_kmph', 'Humidity_%', 'Pressure_hPa', 'Visibility_km']:
        if col in data.columns:
            env_columns.append(col)
    
    if len(env_columns) > 2:
        correlation_matrix = data[env_columns].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Environmental Factors Correlation Matrix"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Find strongest correlations
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:  # Only show significant correlations
                    correlations.append({
                        'factors': f"{correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}",
                        'correlation': corr_value,
                        'strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                    })
        
        if correlations:
            st.markdown("**Key Relationships:**")
            for corr in sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)[:3]:
                direction = "positively" if corr['correlation'] > 0 else "negatively"
                st.write(f"• {corr['factors']} are {direction} correlated ({corr['strength'].lower()}: {corr['correlation']:.2f})")

def create_species_environmental_preferences(data):
    """Show species-specific environmental preferences"""
    
    st.subheader("🦅 Species Environmental Preferences")
    
    if 'Species' in data.columns and 'Temperature_C' in data.columns:
        
        # FIXED VERSION - handle different success formats
        def calculate_success_rate(x):
            success_values = x.unique()
            if 'Success' in success_values:
                return (x == 'Success').mean() * 100
            elif 'Successful' in success_values:
                return (x == 'Successful').mean() * 100
            elif True in success_values:
                return (x == True).mean() * 100
            else:
                return 0
        
        species_prefs = data.groupby('Species').agg({
            'Temperature_C': 'mean',
            'Wind_Speed_kmph': 'mean' if 'Wind_Speed_kmph' in data.columns else lambda x: np.nan,
            'Migration_Success': calculate_success_rate if 'Migration_Success' in data.columns else lambda x: np.nan
        }).round(1)
        
        species_prefs.columns = ['Preferred Temp (°C)', 'Preferred Wind (km/h)', 'Success Rate (%)']
        
        # Sort by success rate
        species_prefs = species_prefs.sort_values('Success Rate (%)', ascending=False)
        
        st.dataframe(
            species_prefs.head(10),
            use_container_width=True
        )
        
        # Highlight top performer
        if len(species_prefs) > 0:
            top_species = species_prefs.index[0]
            top_success = species_prefs.iloc[0]['Success Rate (%)']
            top_temp = species_prefs.iloc[0]['Preferred Temp (°C)']
            
            st.success(f"🏆 **{top_species}** shows the highest environmental adaptation with {top_success}% success rate in {top_temp}°C conditions")

def calculate_environmental_risk_score(data):
    """Calculate environmental risk score (0-10, where 10 is highest risk)"""
    
    risk_factors = []
    
    # Weather condition risk
    if 'Weather_Condition' in data.columns:
        weather_risk = {
            'Clear': 1, 'Cloudy': 3, 'Rainy': 6, 'Stormy': 9
        }
        avg_weather_risk = data['Weather_Condition'].map(weather_risk).mean()
        if not pd.isna(avg_weather_risk):
            risk_factors.append(avg_weather_risk)
    
    # Wind speed risk (too high or too low is risky)
    if 'Wind_Speed_kmph' in data.columns:
        wind_speeds = data['Wind_Speed_kmph'].dropna()
        if len(wind_speeds) > 0:
            # Risk increases for very low (<5) or very high (>25) wind speeds
            wind_risk = wind_speeds.apply(lambda x: 
                8 if x > 25 else 6 if x < 5 else 2 if 10 <= x <= 20 else 4).mean()
            risk_factors.append(wind_risk)
    
    # Food supply risk
    if 'Food_Supply_Level' in data.columns:
        food_risk = {'High': 1, 'Medium': 3, 'Low': 8}
        avg_food_risk = data['Food_Supply_Level'].map(food_risk).mean()
        if not pd.isna(avg_food_risk):
            risk_factors.append(avg_food_risk)
    
    # Temperature extremes risk
    if 'Temperature_C' in data.columns:
        temps = data['Temperature_C'].dropna()
        if len(temps) > 0:
            temp_risk = temps.apply(lambda x: 
                8 if x < -10 or x > 35 else 3 if -5 <= x <= 25 else 5).mean()
            risk_factors.append(temp_risk)
    
    return np.mean(risk_factors) if risk_factors else 5.0

def create_environmental_insights_panel(data):
    """Generate actionable environmental insights"""
    
    st.markdown("---")
    st.subheader("💡 Key Environmental Insights")
    
    insights = []
    
    # Weather impact insights - FIXED VERSION
    if 'Weather_Condition' in data.columns and 'Migration_Success' in data.columns:
        success_values = data['Migration_Success'].unique()
        if 'Success' in success_values:
            weather_success = data.groupby('Weather_Condition')['Migration_Success'].apply(
                lambda x: (x == 'Success').mean() * 100).sort_values(ascending=False)
        elif 'Successful' in success_values:
            weather_success = data.groupby('Weather_Condition')['Migration_Success'].apply(
                lambda x: (x == 'Successful').mean() * 100).sort_values(ascending=False)
        elif True in success_values:
            weather_success = data.groupby('Weather_Condition')['Migration_Success'].apply(
                lambda x: (x == True).mean() * 100).sort_values(ascending=False)
        else:
            weather_success = pd.Series()
        
        if len(weather_success) > 0:
            best_weather = weather_success.index[0]
            worst_weather = weather_success.index[-1]
            insights.append(f"🌤️ **{best_weather}** weather provides the best migration conditions ({weather_success.iloc[0]:.1f}% success)")
            insights.append(f"⛈️ **{worst_weather}** weather is most challenging ({weather_success.iloc[-1]:.1f}% success)")
    
    # Food supply insights
    if 'Food_Supply_Level' in data.columns and 'Flight_Distance_km' in data.columns:
        food_distance = data.groupby('Food_Supply_Level')['Flight_Distance_km'].mean().sort_values()
        if len(food_distance) > 1:
            insights.append(f"🍃 Birds with **{food_distance.index[-1]}** food supply travel furthest ({food_distance.iloc[-1]:.0f} km avg)")
    
    # Temperature tolerance insights - FIXED VERSION
    if 'Temperature_C' in data.columns and 'Migration_Success' in data.columns:
        success_values = data['Migration_Success'].unique()
        if 'Success' in success_values:
            successful_temps = data[data['Migration_Success'] == 'Success']['Temperature_C']
        elif 'Successful' in success_values:
            successful_temps = data[data['Migration_Success'] == 'Successful']['Temperature_C']
        elif True in success_values:
            successful_temps = data[data['Migration_Success'] == True]['Temperature_C']
        else:
            successful_temps = pd.Series()
            
        if len(successful_temps) > 10:
            optimal_range = (successful_temps.quantile(0.25), successful_temps.quantile(0.75))
            insights.append(f"🌡️ Optimal temperature range for success: **{optimal_range[0]:.1f}°C to {optimal_range[1]:.1f}°C**")
    
    # Habitat-specific insights
    if 'Habitat' in data.columns and 'Average_Speed_kmph' in data.columns:
        habitat_speed = data.groupby('Habitat')['Average_Speed_kmph'].mean().sort_values(ascending=False)
        if len(habitat_speed) > 0:
            fastest_habitat = habitat_speed.index[0]
            insights.append(f"🏞️ Birds from **{fastest_habitat}** habitats migrate fastest ({habitat_speed.iloc[0]:.1f} km/h avg)")
    
    # Habitat success rates - FIXED VERSION
    if 'Habitat' in data.columns and 'Migration_Success' in data.columns:
        success_values = data['Migration_Success'].unique()
        if 'Success' in success_values:
            habitat_success = data.groupby('Habitat')['Migration_Success'].apply(
                lambda x: (x == 'Success').mean() * 100).sort_values(ascending=False)
        elif 'Successful' in success_values:
            habitat_success = data.groupby('Habitat')['Migration_Success'].apply(
                lambda x: (x == 'Successful').mean() * 100).sort_values(ascending=False)
        elif True in success_values:
            habitat_success = data.groupby('Habitat')['Migration_Success'].apply(
                lambda x: (x == True).mean() * 100).sort_values(ascending=False)
        else:
            habitat_success = pd.Series()
            
        if len(habitat_success) > 0:
            best_habitat = habitat_success.index[0]
            insights.append(f"🏔️ **{best_habitat}** habitat shows highest success rate ({habitat_success.iloc[0]:.1f}%)")

    for insight in insights:
        st.write(f"• {insight}")
    
    if not insights:
        st.info("Insufficient data to generate detailed environmental insights with current filters.")

# Main app logic
bird_data = load_data()
df_clean = prepare_migration_data_flexible(bird_data)

st.sidebar.header("Filters")
species_filter = st.sidebar.multiselect(
    "Select Species",
    options=bird_data['Species'].unique(),
    default=bird_data['Species'].unique()[:min(2, len(bird_data['Species'].unique()))]
)

if species_filter:
    filtered_data = bird_data[bird_data['Species'].isin(species_filter)]
else:
    filtered_data = bird_data

tab1, tab2, tab3, tab4 = st.tabs(["Migration Map", "Migration Patterns", "Statistics", "Environmental Dashboard"])

# Tab 1: Migration Map
with tab1:
    st.header("Bird Migration Routes")
    
    try:
        with st.spinner("Creating animated migration globe..."):
            fig_animated = create_animated_migration_globe(filtered_data, max_paths=60)
            st.plotly_chart(fig_animated, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating migration globe: {str(e)}")
        st.info("Please check your data format and try again.")

# Tab 2: Migration Patterns
with tab2:
    st.header("Seasonal Migration Patterns")
    
    # Add filters and controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Migration reason filter
        if 'Migration_Reason' in df_clean.columns:
            available_reasons = df_clean['Migration_Reason'].dropna().unique()
            migration_reason_filter = st.multiselect(
                "Select Migration Reasons",
                options=available_reasons,
                default=available_reasons[:3] if len(available_reasons) > 3 else available_reasons,
                help="Filter birds by migration purpose"
            )
        else:
            migration_reason_filter = None
    
    with col2:
        # Show routes toggle
        show_routes = st.checkbox(
            "Show Migration Routes", 
            value=True,
            help="Display lines connecting start and end points"
        )
    
    with col3:
        # Animation speed control
        animation_speed = st.slider(
            "Animation Speed (ms)",
            min_value=500,
            max_value=3000,
            value=1000,
            step=250,
            help="Lower = faster animation"
        )
    
    # Create monthly animation with enhanced features
    try:
        with st.spinner("Creating enhanced monthly migration animation..."):
            monthly_fig = create_monthly_animation(df_clean, migration_reason_filter, show_routes, animation_speed)
            if monthly_fig:
                st.plotly_chart(monthly_fig, use_container_width=True)
                
                st.info("""
                🎮 **Animation Controls:**
                - **Play**: Start automatic month progression
                - **Slider**: Jump to specific months (Jan → Dec)
                - **Legend**: Click items to show/hide migration reasons
                - **Hover**: View detailed migration information
                """)
            else:
                st.info("Could not create monthly animation with the available data.")
    except Exception as e:
        st.error(f"Error creating monthly animation: {str(e)}")
        st.info("Trying with sample data...")

    st.markdown("---")
    st.header("📊 Key Migration Insights")
    
    if migration_reason_filter and 'Migration_Reason' in df_clean.columns:
        analysis_data = df_clean[df_clean['Migration_Reason'].isin(migration_reason_filter)]
    else:
        analysis_data = df_clean
    
    if len(analysis_data) > 0:
        # Create insights in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗓️ Seasonal Patterns")
            seasonal_insights = generate_seasonal_insights(analysis_data)
            for insight in seasonal_insights:
                st.write(f"• {insight}")
        
        with col2:
            st.subheader("🛤️ Migration Behavior")
            behavioral_insights = generate_behavioral_insights(analysis_data)
            for insight in behavioral_insights:
                st.write(f"• {insight}")
        
        # Full width insights
        st.subheader("🌍 Geographic & Performance Insights")
        geo_insights = generate_geographic_insights(analysis_data)
        for insight in geo_insights:
            st.write(f"• {insight}")
        
        # Add summary statistics
        st.subheader("📈 Summary Statistics")
        display_summary_stats(analysis_data)
    else:
        st.info("No data available for generating insights with current filters.")

# Tab 3: Statistics
with tab3:
    st.header("Migration Statistics")
    
    distance_col = None
    for possible_col in ['Flight_Distance_km', 'Distance', 'distance']:
        if possible_col in filtered_data.columns:
            distance_col = possible_col
            break
    
    if distance_col:
        st.subheader("Summary by Species")
        stats = filtered_data.groupby('Species')[distance_col].agg(['mean', 'min', 'max']).reset_index()
        stats.columns = ['Species', 'Average Distance (km)', 'Min Distance (km)', 'Max Distance (km)']
        st.dataframe(stats)
        
        st.header("Interactive Analysis")
        analysis_type = st.radio(
            "Choose analysis type:",
            ["Distance Distribution", "Time Analysis"]
        )
        
        if analysis_type == "Distance Distribution":
            hist_fig = px.histogram(filtered_data, x=distance_col, color='Species',
                                title="Distribution of Migration Distances")
            st.plotly_chart(hist_fig, use_container_width=True)
        elif 'Migration_Start_Month' in filtered_data.columns:
            # Time-based analysis
            time_fig = px.box(filtered_data, 
                              x='Migration_Start_Month', 
                              y=distance_col, 
                              color='Species',
                              title="Distance by Month")
            st.plotly_chart(time_fig, use_container_width=True)
        else:
            st.info("Time analysis requires Migration_Start_Month data.")
    else:
        st.warning("Distance data not found in the dataset. Cannot create statistical visualizations.")

with tab4:
    st.header("🌦️ Environmental Impact Dashboard")
    st.markdown("*Discover how weather, habitat, and food supply influence migration success*")
    
    # Interactive filters
    create_environmental_filters(bird_data)
    
    # Apply filters and create dashboard
    filtered_env_data = apply_environmental_filters(bird_data)
    
    if len(filtered_env_data) > 0:
        create_environmental_insights_dashboard(filtered_env_data)
    else:
        st.warning("No data available with current filter selection")

# End of file
if __name__ == "__main__":
    st.write("🦅 Bird Migration Dashboard is running successfully!")
