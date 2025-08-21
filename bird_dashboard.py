import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# Set pandas options
pd.options.mode.chained_assignment = None

# Page configuration with dark theme
st.set_page_config(
    page_title="Bird Migration",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme with custom CSS
# Dark theme with custom CSS - Professional sizing
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main .block-container {
        background-color: #0E1117;
        color: #FAFAFA;
        padding-top: 2rem;
        max-width: 1400px;
        margin: 0 auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Container width control */
    @media (min-width: 1400px) {
        .main .block-container {
            max-width: 1200px;
        }
    }
    
    @media (min-width: 1600px) {
        .main .block-container {
            max-width: 1400px;
        }
    }
    
    /* Professional spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    .css-1d391kg {
        background-color: #262730;
    }
    
    [data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
        color: #FAFAFA;
        margin-bottom: 1rem;
    }
    
    [data-testid="metric-container"] > div {
        color: #FAFAFA;
    }
    
    [data-testid="metric-container"] label {
        color: #CCCCCC !important;
    }
    
    .stMarkdown, .stText, .stWrite {
        color: #FAFAFA !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
        margin-bottom: 1rem;
    }
    
    h1 {
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2.5rem;
    }
    
    h2 {
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Input controls styling */
    .stSelectbox > div > div {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #333333;
    }
    
    .stMultiSelect > div > div {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #333333;
    }
    
    .stSlider > div > div > div {
        background-color: #262730;
    }
    
    .stCheckbox > label {
        color: #FAFAFA !important;
    }
    
    .stRadio > label {
        color: #FAFAFA !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #262730;
        gap: 8px;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FAFAFA;
        background-color: #1E1E1E;
        border-color: #333333;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #333333 !important;
        color: #FAFAFA !important;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: #262730;
        width: 280px !important;
    }
    
    .stSidebar > div > div {
        background-color: #262730;
        color: #FAFAFA;
        padding: 1rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* Alert styling */
    .stAlert > div {
        background-color: #1E1E1E;
        border-color: #333333;
        color: #FAFAFA;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stInfo {
        background-color: #1E3A5F;
        border-left-color: #3498DB;
        color: #FAFAFA;
    }
    
    .stSuccess {
        background-color: #1E4D2B;
        border-left-color: #27AE60;
        color: #FAFAFA;
    }
    
    .stWarning {
        background-color: #4D3A1E;
        border-left-color: #F39C12;
        color: #FAFAFA;
    }
    
    .stError {
        background-color: #4D1E1E;
        border-left-color: #E74C3C;
        color: #FAFAFA;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background-color: #1E1E1E;
        color: #FAFAFA;
        border-radius: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    /* Custom container for cards */
    .dashboard-card {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
    }
    
    /* Professional spacing for columns */
    [data-testid="column"] {
        padding: 0 0.75rem;
    }
    
    [data-testid="column"]:first-child {
        padding-left: 0;
    }
    
    [data-testid="column"]:last-child {
        padding-right: 0;
    }
    
    /* Plotly chart containers */
    .js-plotly-plot {
        margin: 1rem 0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Section dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #333333, transparent);
        margin: 2rem 0;
    }
    
    /* Fix for metric values */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #FAFAFA !important;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #CCCCCC !important;
    }
    
    /* Professional button styling */
    .stButton > button {
        background-color: #3498DB;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2980B9;
    }
    
    /* Compact spacing for insights */
    .insight-section {
        margin: 1.5rem 0;
    }
    
    .insight-section ul {
        margin: 0.5rem 0;
    }
    
    .insight-section li {
        margin: 0.25rem 0;
        padding-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib to dark theme
plt.style.use('dark_background')

st.title("Bird Migration Analysis")
st.markdown("### Visualizing patterns and routes of migratory birds")

@st.cache_data  
def load_data():
    try:
        df = pd.read_csv('Bird_Migration_Data_with_Origin.csv')
        return df
    except FileNotFoundError:
        st.warning("Data file not found. Using sample data instead.")
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
    
    # Get color palette (bright colors for dark background)
    unique_categories = df_sample[color_col].unique()
    bright_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#FF8A80', '#85C1E9']
    color_map = {cat: bright_colors[i % len(bright_colors)] for i, cat in enumerate(unique_categories)}
    
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
            font=dict(size=16, color='white')
        ),
        geo=dict(
            projection_type='natural earth',
            showland=True,
            showcountries=True,
            countrywidth=0.5,
            showocean=True,
            oceancolor='#1a1a2e',
            landcolor='#16213e',
            coastlinecolor='#4a90e2',
            bgcolor='#0E1117'
        ),
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Month: "},
            pad={"t": 50},
            steps=slider_steps,
            font=dict(color='white')
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
            bgcolor='rgba(30,30,30,0.8)',
            bordercolor='gray',
            font=dict(color='white')
        )],
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(30,30,30,0.9)",
            font=dict(color='white')
        ),
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='white')
    )

    fig.frames = frames
    
    return fig

def create_point_animation_plotly(df_sample, month_col, animation_speed, month_labels):
    """Create point-based animation (enhanced version of your original)"""
    
    color_column = 'Migration_Reason' if 'Migration_Reason' in df_sample.columns else 'Species'
    
    # Create the scatter plot with dark theme
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
        height=700,
        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    )
    
    # Update layout for dark theme
    fig.update_layout(
        geo=dict(
            bgcolor='#0E1117',
            showland=True,
            landcolor='#16213e',
            showocean=True,
            oceancolor='#1a1a2e',
            showcountries=True,
            countrycolor='#4a90e2'
        ),
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='white'),
        legend=dict(
            bgcolor="rgba(30,30,30,0.9)",
            font=dict(color='white')
        )
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
            font=dict(size=16, color='white')
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
    
    # Success rates - Fixed version
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
    
    # Weather impact
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
        
        # Rest stops vs success
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
            # Create bar chart with dark theme
            fig_species = px.bar(
                x=species_dist.index, 
                y=species_dist.values,
                color=species_dist.values,
                color_continuous_scale='viridis',
                labels={'x': 'Species', 'y': 'Count'}
            )
            fig_species.update_layout(
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117',
                font=dict(color='white'),
                showlegend=False
            )
            st.plotly_chart(fig_species, use_container_width=True)
        
        # Migration reasons
        if 'Migration_Reason' in data.columns:
            st.write("**Migration Reasons:**")
            reason_dist = data['Migration_Reason'].value_counts()
            fig_reason = px.bar(
                x=reason_dist.index, 
                y=reason_dist.values,
                color=reason_dist.values,
                color_continuous_scale='plasma',
                labels={'x': 'Migration Reason', 'y': 'Count'}
            )
            fig_reason.update_layout(
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117',
                font=dict(color='white'),
                showlegend=False
            )
            st.plotly_chart(fig_reason, use_container_width=True)

# Environmental functions (updated with dark theme)
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
        
        # Handle different success formats
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
                        color='#27AE60',
                        opacity=0.6,
                        line=dict(width=1, color='white')
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
                        color='#E74C3C',
                        opacity=0.6,
                        line=dict(width=1, color='white')
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
                    fillcolor="rgba(39,174,96,0.2)",
                    line=dict(color="#27AE60", width=2, dash="dash"),
                )
                
                fig.add_annotation(
                    x=temp_sweet.mean(),
                    y=wind_sweet.mean(),
                    text="Sweet Spot Zone",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#27AE60",
                    bgcolor="rgba(30,30,30,0.8)",
                    bordercolor="#27AE60",
                    font=dict(color='white')
                )
            
            fig.update_layout(
                title="Optimal Environmental Conditions for Migration Success",
                xaxis_title="Temperature (°C)",
                yaxis_title="Wind Speed (km/h)",
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, 
                           bgcolor="rgba(30,30,30,0.9)", font=dict(color='white')),
                paper_bgcolor='#0E1117',
                plot_bgcolor='#1E1E1E',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#333333'),
                yaxis=dict(gridcolor='#333333')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
        title = {'text': "Environmental Risk Level", 'font': {'color': 'white'}},
        delta = {'reference': 5.0},
        gauge = {
            'axis': {'range': [None, 10], 'tickcolor': 'white'},
            'bar': {'color': "#3498DB"},
            'steps': [
                {'range': [0, 3], 'color': "#27AE60"},
                {'range': [3, 6], 'color': "#F39C12"},
                {'range': [6, 10], 'color': "#E74C3C"}
            ],
            'threshold': {
                'line': {'color': "#E74C3C", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        },
        number={'font': {'color': 'white'}}
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='#0E1117',
        font=dict(color='white')
    )
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
        
        fig.update_layout(
            height=400,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='white')
        )
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
        
        # Fixed version - handle different success formats
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
    
    # Weather impact insights - Fixed version
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
    
    # Temperature tolerance insights - Fixed version
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
    
    # Habitat success rates - Fixed version
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

# MAIN EXECUTION CODE
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

tab1, tab2, tab3 = st.tabs(["Environmental Impacts", "Migration Patterns", "Statistics"])

# Tab 1: Environmental Dashboard
with tab1:
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
        st.error(f"Error creating animation: {str(e)}")

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
    st.header("📊 Migration Statistics Dashboard")
    st.markdown("*Comprehensive analysis of flight distances by species*")
    
    distance_col = None
    for possible_col in ['Flight_Distance_km', 'Distance', 'distance']:
        if possible_col in filtered_data.columns:
            distance_col = possible_col
            break
    
    if distance_col:
        # Calculate statistics
        stats = filtered_data.groupby('Species')[distance_col].agg(['mean', 'min', 'max', 'std', 'count']).reset_index()
        stats.columns = ['Species', 'Average Distance (km)', 'Min Distance (km)', 'Max Distance (km)', 'Std Dev', 'Sample Size']
        stats = stats.round(1)
        stats = stats.sort_values('Average Distance (km)', ascending=False)
        
        # Top-level metrics
        st.subheader("🎯 Key Insights")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            longest_species = stats.iloc[0]['Species']
            longest_distance = stats.iloc[0]['Average Distance (km)']
            st.metric(
                "🏆 Longest Distance Migrant", 
                longest_species,
                f"{longest_distance:.0f} km average"
            )
        
        with col2:
            shortest_species = stats.iloc[-1]['Species'] 
            shortest_distance = stats.iloc[-1]['Average Distance (km)']
            st.metric(
                "🏠 Shortest Distance Migrant",
                shortest_species, 
                f"{shortest_distance:.0f} km average"
            )
        
        with col3:
            overall_avg = filtered_data[distance_col].mean()
            st.metric(
                "📏 Overall Average",
                f"{overall_avg:.0f} km",
                "across all species"
            )
        
        with col4:
            max_distance = filtered_data[distance_col].max()
            max_species = filtered_data.loc[filtered_data[distance_col].idxmax(), 'Species']
            st.metric(
                "🚀 Record Distance",
                f"{max_distance:.0f} km", 
                f"by {max_species}"
            )
        
        st.markdown("---")
        
        # Pretty Species Cards using Streamlit native components
        st.subheader("🦅 Species Performance Cards")
        
        # Species icons
        species_icons = {
            'Eagle': '🦅',
            'Falcon': '🐦‍⬛', 
            'Stork': '🦢',
            'Swallow': '🐦',
            'Hawk': '🦅',
            'Robin': '🐦',
            'Crane': '🦢',
            'Goose': '🦢',
            'Duck': '🦆',
            'Heron': '🦢'
        }
        
        # Create cards in rows of 2 with professional styling
        for i in range(0, len(stats), 2):
            col_left, col_right = st.columns(2, gap="large")
            
            # Left card
            with col_left:
                if i < len(stats):
                    species_data = stats.iloc[i]
                    species = species_data['Species']
                    icon = species_icons.get(species, '🐦')
                    
                    # Determine performance level
                    avg_dist = species_data['Average Distance (km)']
                    if avg_dist > overall_avg * 1.2:
                        performance = "🟢 Long Distance"
                        emoji = "🚀"
                        card_color = "#1E4D2B"
                    elif avg_dist < overall_avg * 0.8:
                        performance = "🔵 Short Distance" 
                        emoji = "🏠"
                        card_color = "#1E3A5F"
                    else:
                        performance = "🟡 Medium Distance"
                        emoji = "✈️"
                        card_color = "#4D3A1E"
                    
                    # Create professional card
                    st.markdown(f"""
                    <div class="dashboard-card" style="background-color: {card_color}; min-height: 220px;">
                        <h3 style="color: #FAFAFA; margin: 0 0 0.5rem 0; font-size: 1.4rem;">{icon} {species} {emoji}</h3>
                        <p style="color: #FAFAFA; font-weight: bold; margin: 0 0 1rem 0;">{performance}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics in columns with better spacing
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            "Average", 
                            f"{species_data['Average Distance (km)']:.0f} km",
                            delta=f"{species_data['Average Distance (km)'] - overall_avg:.0f} km"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Minimum", 
                            f"{species_data['Min Distance (km)']:.0f} km"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Maximum", 
                            f"{species_data['Max Distance (km)']:.0f} km"
                        )
                    
                    # Additional info with better styling
                    st.markdown(f"""
                    <div style="background-color: #1E3A5F; padding: 0.75rem; border-radius: 6px; margin: 1rem 0; text-align: center;">
                        <span style="color: #FAFAFA;">📊 Sample Size: {species_data['Sample Size']:.0f} birds | 📐 Std Dev: {species_data['Std Dev']:.0f} km</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Right card  
            with col_right:
                if i + 1 < len(stats):
                    species_data = stats.iloc[i + 1]
                    species = species_data['Species']
                    icon = species_icons.get(species, '🐦')
                    
                    # Determine performance level
                    avg_dist = species_data['Average Distance (km)']
                    if avg_dist > overall_avg * 1.2:
                        performance = "🟢 Long Distance"
                        emoji = "🚀"
                        card_color = "#1E4D2B"
                    elif avg_dist < overall_avg * 0.8:
                        performance = "🔵 Short Distance"
                        emoji = "🏠"
                        card_color = "#1E3A5F"
                    else:
                        performance = "🟡 Medium Distance"
                        emoji = "✈️"
                        card_color = "#4D3A1E"
                    
                    # Create professional card
                    st.markdown(f"""
                    <div class="dashboard-card" style="background-color: {card_color}; min-height: 220px;">
                        <h3 style="color: #FAFAFA; margin: 0 0 0.5rem 0; font-size: 1.4rem;">{icon} {species} {emoji}</h3>
                        <p style="color: #FAFAFA; font-weight: bold; margin: 0 0 1rem 0;">{performance}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics in columns with better spacing
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            "Average", 
                            f"{species_data['Average Distance (km)']:.0f} km",
                            delta=f"{species_data['Average Distance (km)'] - overall_avg:.0f} km"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Minimum", 
                            f"{species_data['Min Distance (km)']:.0f} km"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Maximum", 
                            f"{species_data['Max Distance (km)']:.0f} km"
                        )
                    
                    # Additional info with better styling
                    st.markdown(f"""
                    <div style="background-color: #1E3A5F; padding: 0.75rem; border-radius: 6px; margin: 1rem 0; text-align: center;">
                        <span style="color: #FAFAFA;">📊 Sample Size: {species_data['Sample Size']:.0f} birds | 📐 Std Dev: {species_data['Std Dev']:.0f} km</span>
                    </div>
                    """, unsafe_allow_html=True)
        # Interactive Analysis Section
        st.subheader("📈 Interactive Analysis")
        
        analysis_col1, analysis_col2 = st.columns([2, 1])
        
        with analysis_col2:
            analysis_type = st.radio(
                "Choose Analysis:",
                ["Distance Distribution", "Species Comparison", "Performance Ranking"],
                help="Select different ways to analyze the data"
            )
        
        with analysis_col1:
            if analysis_type == "Distance Distribution":
                # Distance distribution histogram
                hist_fig = px.histogram(
                    filtered_data, 
                    x=distance_col, 
                    color='Species',
                    title="Distribution of Migration Distances by Species",
                    nbins=20,
                    opacity=0.7,
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
                )
                hist_fig.update_layout(
                    height=400,
                    paper_bgcolor='#0E1117',  
                    plot_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#333333'),
                    yaxis=dict(gridcolor='#333333'),
                    legend=dict(bgcolor="rgba(30,30,30,0.9)", font=dict(color='white'))
                )
                st.plotly_chart(hist_fig, use_container_width=True)
                
            elif analysis_type == "Species Comparison":
                # Box plot comparison
                box_fig = px.box(
                    filtered_data,
                    x='Species',
                    y=distance_col,
                    color='Species',
                    title="Distance Range Comparison by Species",
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
                )
                box_fig.update_xaxes(tickangle=45)
                box_fig.update_layout(
                    height=400, 
                    showlegend=False,
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#333333'),
                    yaxis=dict(gridcolor='#333333')
                )
                st.plotly_chart(box_fig, use_container_width=True)
                
            else:  # Performance Ranking
                # Ranking bar chart
                ranking_fig = px.bar(
                    stats,
                    x='Species',
                    y='Average Distance (km)',
                    color='Average Distance (km)',
                    title="Species Ranking by Average Migration Distance",
                    color_continuous_scale='viridis'
                )
                ranking_fig.update_xaxes(tickangle=45)
                ranking_fig.update_layout(
                    height=400,
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#333333'),
                    yaxis=dict(gridcolor='#333333')
                )
                st.plotly_chart(ranking_fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Distance data not found in the dataset. Cannot create statistical visualizations.")
        st.info("💡 Available columns: " + ", ".join(filtered_data.columns.tolist()))
# End of file
if __name__ == "__main__":
    st.write("🦅 Bird Migration Dashboard is running successfully!")
