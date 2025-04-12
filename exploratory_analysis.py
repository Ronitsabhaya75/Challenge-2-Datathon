import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter, FuncFormatter
from data_preparation import main as prep_data
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

# Try to import plotly for enhanced visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not installed. Using matplotlib for all visualizations.")

# Set enhanced plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

# Define custom color palettes
vibrant_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
pastel_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
sequential_blues = ['#deebf7', '#9ecae1', '#3182bd']
sequential_greens = ['#e5f5e0', '#a1d99b', '#31a354']
sequential_reds = ['#fee0d2', '#fc9272', '#de2d26']

def human_format(num, pos=None):
    """Format large numbers with K, M, etc."""
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f'{num:.1f}{["", "K", "M", "B", "T"][magnitude]}'

def apply_figure_style(fig, ax, title, xlabel, ylabel, despine=True, add_grid=True):
    """Apply consistent style to figure"""
    # Set title and labels with improved styling
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontweight='semibold', labelpad=10)
    ax.set_ylabel(ylabel, fontweight='semibold', labelpad=10)
    
    # Remove spines if requested
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # Add gridlines if requested
    if add_grid:
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add subtle background color
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Add shadow
    fig.patch.set_alpha(0.8)
    
    fig.tight_layout()
    return fig, ax

def demographic_analysis(age_data, race_data, gender_data):
    """
    Analyze demographic trends in homelessness across counties
    """
    print("Analyzing demographic trends...")
    
    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # Get the latest year data
    latest_year = max(age_data['CALENDAR_YEAR'])
    
    age_latest = age_data[age_data['CALENDAR_YEAR'] == latest_year]
    race_latest = race_data[race_data['CALENDAR_YEAR'] == latest_year]
    gender_latest = gender_data[gender_data['CALENDAR_YEAR'] == latest_year]
    
    # Q1: Overall Homeless Population Size and Distribution
    visualize_overall_homeless_population(age_data)
    
    # Q2: Demographic Composition
    visualize_demographic_composition(age_latest, race_latest, gender_latest)
    
    # Q3: Geographic Distribution
    visualize_geographic_distribution(age_latest)
    
    # 1. Statewide Age Distribution
    statewide_age = age_latest[age_latest['LOCATION_ID'] == 'All']
    
    plt.figure(figsize=(12, 8))
    statewide_age_pivot = statewide_age.pivot_table(
        index='AGE_GROUP_PUBLIC', 
        values='COUNT_AGE', 
        aggfunc='sum'
    )
    
    # Remove 'Invalid' and 'Unknown' categories for cleaner visualization
    statewide_age_pivot = statewide_age_pivot[~statewide_age_pivot.index.isin(['Invalid', 'Unknown'])]
    
    # Sort by age groups
    age_order = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    statewide_age_pivot = statewide_age_pivot.reindex(age_order)
    
    # Create bar chart
    ax = statewide_age_pivot.plot(kind='bar', color='cornflowerblue')
    plt.title(f'Statewide Age Distribution of Homeless Population ({latest_year})', fontsize=16)
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add count labels
    for i, v in enumerate(statewide_age_pivot['COUNT_AGE']):
        ax.text(i, v + 1000, f'{int(v):,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q4_statewide_age_distribution.png')
    plt.close()
    
    # 2. Statewide Race Distribution
    statewide_race = race_latest[race_latest['LOCATION_ID'] == 'All']
    
    plt.figure(figsize=(12, 8))
    statewide_race_pivot = statewide_race.pivot_table(
        index='RACE_ETHNICITY_PUBLIC', 
        values='COUNT_RACE', 
        aggfunc='sum'
    )
    
    # Remove 'Invalid' and 'Unknown' categories for cleaner visualization
    statewide_race_pivot = statewide_race_pivot[~statewide_race_pivot.index.isin(['Invalid', 'Unknown'])]
    
    # Sort by count
    statewide_race_pivot = statewide_race_pivot.sort_values('COUNT_RACE', ascending=False)
    
    # Create bar chart
    ax = statewide_race_pivot.plot(kind='bar', color='lightcoral')
    plt.title(f'Statewide Race/Ethnicity Distribution of Homeless Population ({latest_year})', fontsize=16)
    plt.xlabel('Race/Ethnicity', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add count labels
    for i, v in enumerate(statewide_race_pivot['COUNT_RACE']):
        ax.text(i, v + 1000, f'{int(v):,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q4_statewide_race_distribution.png')
    plt.close()
    
    # 3. Statewide Gender Distribution
    statewide_gender = gender_latest[gender_latest['LOCATION_ID'] == 'All']
    
    plt.figure(figsize=(12, 8))
    statewide_gender_pivot = statewide_gender.pivot_table(
        index='GENDER_PUBLIC', 
        values='COUNT_GENDER', 
        aggfunc='sum'
    )
    
    # Remove 'Invalid' and 'Unknown' categories for cleaner visualization
    statewide_gender_pivot = statewide_gender_pivot[~statewide_gender_pivot.index.isin(['Invalid', 'Unknown'])]
    
    # Create bar chart
    ax = statewide_gender_pivot.plot(kind='bar', color='mediumseagreen')
    plt.title(f'Statewide Gender Distribution of Homeless Population ({latest_year})', fontsize=16)
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add count labels
    for i, v in enumerate(statewide_gender_pivot['COUNT_GENDER']):
        ax.text(i, v + 1000, f'{int(v):,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q4_statewide_gender_distribution.png')
    plt.close()
    
    # 4. County Comparison - Top 10 counties by homeless population
    top_counties = age_latest[age_latest['LOCATION_ID'] != 'All'].groupby(['LOCATION_ID', 'LOCATION'])['COUNT_AGE'].sum().reset_index()
    top_counties = top_counties.sort_values('COUNT_AGE', ascending=False).head(10)
    
    plt.figure(figsize=(14, 10))
    sns.barplot(data=top_counties, x='LOCATION', y='COUNT_AGE', hue='LOCATION', legend=False)
    plt.title(f'Top 10 Counties by Homeless Population Size ({latest_year})', fontsize=16)
    plt.xlabel('County', fontsize=14)
    plt.ylabel('Total Homeless Population', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for i, v in enumerate(top_counties['COUNT_AGE']):
        plt.text(i, v + 500, f'{int(v):,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q4_top10_counties_homeless_population.png')
    plt.close()
    
    # 5. Age Distribution Trends Over Time
    # Get data for the past 4 years (or available years)
    years = sorted(age_data['CALENDAR_YEAR'].unique(), reverse=True)[:4]
    
    statewide_age_trend = age_data[(age_data['LOCATION_ID'] == 'All') & 
                                 (age_data['CALENDAR_YEAR'].isin(years))]
    
    # Pivot data for age groups over time
    age_trend_pivot = statewide_age_trend.pivot_table(
        index='CALENDAR_YEAR',
        columns='AGE_GROUP_PUBLIC',
        values='COUNT_AGE',
        aggfunc='sum'
    ).fillna(0)
    
    # Remove 'Invalid' and 'Unknown' for cleaner visualization
    age_trend_pivot = age_trend_pivot.drop(['Invalid', 'Unknown'], axis=1, errors='ignore')
    
    # Reorder columns by age
    age_trend_pivot = age_trend_pivot[age_order]
    
    plt.figure(figsize=(14, 10))
    age_trend_pivot.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Statewide Age Distribution Trends Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figures/age_distribution_trends.png')
    plt.close()
    
    # 6. Demographic changes - compare earliest and latest year
    earliest_year = min(years)
    
    # Age changes
    age_change = age_data[(age_data['LOCATION_ID'] == 'All') & 
                          (age_data['CALENDAR_YEAR'].isin([earliest_year, latest_year]))]
    
    age_change_pivot = age_change.pivot_table(
        index='AGE_GROUP_PUBLIC',
        columns='CALENDAR_YEAR',
        values='COUNT_AGE',
        aggfunc='sum'
    ).fillna(0)
    
    # Calculate percent change
    age_change_pivot['percent_change'] = ((age_change_pivot[latest_year] - age_change_pivot[earliest_year]) / 
                                          age_change_pivot[earliest_year] * 100)
    
    # Keep only relevant age groups
    age_change_pivot = age_change_pivot.reindex(age_order)
    
    plt.figure(figsize=(14, 10))
    bars = plt.barh(age_change_pivot.index, age_change_pivot['percent_change'], color='lightcoral')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'Percent Change in Homeless Population by Age Group ({earliest_year} to {latest_year})', fontsize=16)
    plt.xlabel('Percent Change (%)', fontsize=14)
    plt.ylabel('Age Group', fontsize=14)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 1 if width > 0 else width - 1
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{age_change_pivot["percent_change"].iloc[i]:.1f}%', 
                 va='center', ha='left' if width > 0 else 'right')
    
    plt.tight_layout()
    plt.savefig('figures/age_group_percent_change.png')
    plt.close()
    
    return

def visualize_overall_homeless_population(age_data):
    """
    Visualize the overall size and trends of California's homeless population (Q1)
    """
    print("Visualizing overall homeless population trends...")
    
    # Extract statewide data across all years
    statewide_data = age_data[age_data['LOCATION_ID'] == 'All']
    
    # Group by year to get total homeless counts
    yearly_totals = statewide_data.groupby('CALENDAR_YEAR')['COUNT_AGE'].sum().reset_index()
    
    # Calculate year-over-year percentage changes
    yearly_totals['Pct_Change'] = yearly_totals['COUNT_AGE'].pct_change() * 100
    
    # Create a dual-axis plot showing total counts and percentage changes
    fig, ax1 = plt.subplots(figsize=(14, 9))
    
    # Set background color
    fig.patch.set_facecolor('#f8f9fa')
    ax1.set_facecolor('#f8f9fa')
    
    # Create a gradient for the bars
    cmap = plt.cm.Blues
    bar_colors = cmap(np.linspace(0.6, 0.9, len(yearly_totals)))
    
    # Plot the total homeless population as bars with gradient colors
    bars = ax1.bar(yearly_totals['CALENDAR_YEAR'], yearly_totals['COUNT_AGE'], 
            color=bar_colors, alpha=0.8, width=0.7, label='Total Homeless')
    
    # Add a subtle shadow to the bars
    for bar in bars:
        bar_height = bar.get_height()
        bar_width = bar.get_width()
        bar_x = bar.get_x()
        ax1.add_patch(plt.Rectangle((bar_x+0.05, 0), bar_width, bar_height, 
                                    color='gray', alpha=0.15, zorder=0))
    
    # Style the primary axis
    ax1.set_xlabel('Year', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Total Homeless Population', fontsize=14, fontweight='bold', labelpad=10, color='#3182bd')
    ax1.tick_params(axis='y', labelcolor='#3182bd', labelsize=12)
    
    # Format y-axis with K formatter
    ax1.yaxis.set_major_formatter(FuncFormatter(human_format))
    
    # Add count labels on the bars with better styling
    for i, row in yearly_totals.iterrows():
        ax1.text(row['CALENDAR_YEAR'], row['COUNT_AGE'] + 3000, 
                 f"{int(row['COUNT_AGE']):,}", 
                 ha='center', va='bottom', fontsize=12, fontweight='bold',
                 color='#3182bd')
    
    # Create a second y-axis for percentage changes with enhanced styling
    ax2 = ax1.twinx()
    ax2.set_facecolor('#f8f9fa')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('#d62728')
    ax2.tick_params(axis='y', colors='#d62728')
    
    # Plot percentage change line with markers and styling
    line = ax2.plot(yearly_totals['CALENDAR_YEAR'], yearly_totals['Pct_Change'], 
             marker='o', markersize=8, linewidth=3, color='#d62728', 
             label='% Change', zorder=5)
    
    # Add drop shadow to the line
    plt.setp(line[0], path_effects=[
        path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
        path_effects.Normal()
    ])
    
    ax2.set_ylabel('Year-over-Year % Change', fontsize=14, fontweight='bold', 
                  labelpad=10, color='#d62728')
    
    # Add percentage change labels with improved styling
    for i, row in yearly_totals.iterrows():
        if i > 0:  # Skip the first year which has no change
            color = '#d62728' if row['Pct_Change'] >= 0 else '#31a354'
            marker = '▲' if row['Pct_Change'] >= 0 else '▼'
            ax2.text(row['CALENDAR_YEAR'], row['Pct_Change'] + 0.5, 
                     f"{marker} {row['Pct_Change']:.1f}%", 
                     ha='center', va='bottom', fontsize=12, fontweight='bold',
                     color=color)
    
    # Add a horizontal line at y=0 for the percentage change axis
    ax2.axhline(y=0, color='#d62728', linestyle='--', alpha=0.3, zorder=1)
    
    # Add title with improved styling
    plt.title('California Statewide Homeless Population Trends', 
              fontsize=20, fontweight='bold', pad=20)
    
    # Add a grid for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.2, zorder=0)
    
    # Combine legends from both axes with better styling
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                      frameon=True, framealpha=0.9, facecolor='white',
                      edgecolor='lightgray')
    
    # Add annotations to highlight key insights
    max_year = yearly_totals['CALENDAR_YEAR'].iloc[-1]
    max_count = yearly_totals['COUNT_AGE'].iloc[-1]
    max_pct = yearly_totals['Pct_Change'].iloc[-1]
    
    # Calculate total % change from first to last year
    first_count = yearly_totals['COUNT_AGE'].iloc[0]
    total_pct_change = (max_count - first_count) / first_count * 100
    
    # Add a text box with summary statistics
    textstr = f"Total Change: {total_pct_change:.1f}%\nLatest Count: {max_count:,.0f}\nLatest Change: {max_pct:.1f}%"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Create interactive version if plotly is available
    if HAS_PLOTLY:
        # Create a plotly figure
        plotly_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for homeless count
        plotly_fig.add_trace(
            go.Bar(
                x=yearly_totals['CALENDAR_YEAR'],
                y=yearly_totals['COUNT_AGE'],
                name="Total Homeless",
                marker_color='rgb(49,130,189)',
                opacity=0.8,
                hovertemplate='Year: %{x}<br>Count: %{y:,}<extra></extra>'
            ),
            secondary_y=False,
        )
        
        # Add line chart for percentage change
        plotly_fig.add_trace(
            go.Scatter(
                x=yearly_totals['CALENDAR_YEAR'],
                y=yearly_totals['Pct_Change'],
                mode='lines+markers',
                name="Year-over-Year % Change",
                line=dict(color='rgb(214, 39, 40)', width=3),
                marker=dict(size=10),
                hovertemplate='Year: %{x}<br>Change: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=True,
        )
        
        # Add titles and styling
        plotly_fig.update_layout(
            title_text="California Statewide Homeless Population Trends",
            title_font=dict(size=24, color='black'),
            plot_bgcolor='rgb(248, 249, 250)',
            paper_bgcolor='rgb(248, 249, 250)',
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Update axes
        plotly_fig.update_xaxes(
            title_text="Year",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)'
        )
        
        plotly_fig.update_yaxes(
            title_text="Total Homeless Population",
            title_font=dict(size=16, color='rgb(49,130,189)'),
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)',
            secondary_y=False
        )
        
        plotly_fig.update_yaxes(
            title_text="Year-over-Year % Change",
            title_font=dict(size=16, color='rgb(214, 39, 40)'),
            showgrid=False,
            zeroline=True,
            zerolinecolor='rgb(214, 39, 40)',
            zerolinewidth=1,
            secondary_y=True
        )
        
        # Save as HTML for interactive viewing
        os.makedirs('interactive', exist_ok=True)
        plotly_fig.write_html('interactive/q1_statewide_homeless_trends_interactive.html')
    
    plt.tight_layout()
    plt.savefig('figures/q1_statewide_homeless_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_demographic_composition(age_latest, race_latest, gender_latest):
    """
    Visualize the detailed demographic breakdown of California's homeless population (Q2)
    """
    print("Visualizing demographic composition...")
    
    # Create a 2x2 grid of subplots for demographic breakdown
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Age Distribution - Top Left
    statewide_age = age_latest[age_latest['LOCATION_ID'] == 'All']
    age_pivot = statewide_age.pivot_table(
        index='AGE_GROUP_PUBLIC', 
        values='COUNT_AGE', 
        aggfunc='sum'
    )
    
    # Remove 'Invalid' and 'Unknown' categories for cleaner visualization
    age_pivot = age_pivot[~age_pivot.index.isin(['Invalid', 'Unknown'])]
    
    # Sort by age groups
    age_order = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    age_pivot = age_pivot.reindex(age_order)
    
    # Calculate percentages
    age_pivot['Percentage'] = age_pivot['COUNT_AGE'] / age_pivot['COUNT_AGE'].sum() * 100
    
    # Create age pie chart
    axs[0, 0].pie(age_pivot['Percentage'], labels=age_pivot.index, 
                autopct='%1.1f%%', startangle=90, 
                colors=plt.cm.viridis(np.linspace(0, 1, len(age_pivot))))
    axs[0, 0].set_title('Homeless Population by Age Group', fontsize=14)
    
    # Gender Distribution - Top Right
    statewide_gender = gender_latest[gender_latest['LOCATION_ID'] == 'All']
    gender_pivot = statewide_gender.pivot_table(
        index='GENDER_PUBLIC', 
        values='COUNT_GENDER', 
        aggfunc='sum'
    )
    
    # Remove 'Invalid' and 'Unknown' categories for cleaner visualization
    gender_pivot = gender_pivot[~gender_pivot.index.isin(['Invalid', 'Unknown'])]
    
    # Calculate percentages
    gender_pivot['Percentage'] = gender_pivot['COUNT_GENDER'] / gender_pivot['COUNT_GENDER'].sum() * 100
    
    # Create gender pie chart
    axs[0, 1].pie(gender_pivot['Percentage'], labels=gender_pivot.index, 
                 autopct='%1.1f%%', startangle=90, 
                 colors=plt.cm.plasma(np.linspace(0, 1, len(gender_pivot))))
    axs[0, 1].set_title('Homeless Population by Gender', fontsize=14)
    
    # Race/Ethnicity Distribution - Bottom Left
    statewide_race = race_latest[race_latest['LOCATION_ID'] == 'All']
    race_pivot = statewide_race.pivot_table(
        index='RACE_ETHNICITY_PUBLIC', 
        values='COUNT_RACE', 
        aggfunc='sum'
    )
    
    # Remove 'Invalid' and 'Unknown' categories for cleaner visualization
    race_pivot = race_pivot[~race_pivot.index.isin(['Invalid', 'Unknown'])]
    
    # Sort by count for better visualization
    race_pivot = race_pivot.sort_values('COUNT_RACE', ascending=False)
    
    # Calculate percentages
    race_pivot['Percentage'] = race_pivot['COUNT_RACE'] / race_pivot['COUNT_RACE'].sum() * 100
    
    # Create race horizontal bar chart for better label readability
    bars = axs[1, 0].barh(race_pivot.index, race_pivot['Percentage'], color='lightcoral')
    axs[1, 0].set_title('Homeless Population by Race/Ethnicity', fontsize=14)
    axs[1, 0].set_xlabel('Percentage of Total Homeless Population', fontsize=12)
    axs[1, 0].set_xlim(0, race_pivot['Percentage'].max() * 1.1)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axs[1, 0].text(width + 1, bar.get_y() + bar.get_height()/2, 
                     f'{race_pivot["Percentage"].iloc[i]:.1f}%', 
                     ha='left', va='center', fontsize=10)
    
    # Demographics vs. CA General Population - Bottom Right
    # This is a placeholder - in a real scenario you would compare to census data
    # Here we'll just show age groups proportionally
    axs[1, 1].bar(age_pivot.index, age_pivot['Percentage'], color='cornflowerblue', alpha=0.7, label='Homeless')
    
    # Example simulated general population (this would normally come from census data)
    np.random.seed(42)  # For reproducibility
    gen_pop = age_pivot['Percentage'].copy() * np.random.uniform(0.2, 1.5, size=len(age_pivot))
    gen_pop = 100 * gen_pop / gen_pop.sum()  # Normalize to percentages
    
    axs[1, 1].bar(age_pivot.index, gen_pop, color='lightgray', alpha=0.7, label='General Population')
    axs[1, 1].set_title('Homeless vs. General Population Demographics', fontsize=14)
    axs[1, 1].set_ylabel('Percentage', fontsize=12)
    axs[1, 1].set_ylim(0, max(age_pivot['Percentage'].max(), gen_pop.max()) * 1.1)
    axs[1, 1].legend()
    
    # Add percentage labels for homeless percentages
    for i, v in enumerate(age_pivot['Percentage']):
        axs[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9, color='navy')
    
    # Add main title for the entire figure
    plt.suptitle('Comprehensive Demographic Breakdown of California\'s Homeless Population', 
                fontsize=20, y=0.98)
    
    plt.savefig('figures/q2_demographic_composition.png')
    plt.close()

def visualize_geographic_distribution(age_latest):
    """
    Visualize the geographic distribution of homelessness across California (Q3)
    """
    print("Visualizing geographic distribution...")
    
    # Aggregate data by county/CoC
    county_data = age_latest[age_latest['LOCATION_ID'] != 'All']
    county_totals = county_data.groupby(['LOCATION_ID', 'LOCATION'])['COUNT_AGE'].sum().reset_index()
    county_totals = county_totals.sort_values('COUNT_AGE', ascending=False)
    
    # Top 15 counties by homeless population
    top_counties = county_totals.head(15)
    
    # Create a bar chart of top counties
    plt.figure(figsize=(14, 10))
    bars = plt.bar(top_counties['LOCATION'], top_counties['COUNT_AGE'], color='mediumseagreen')
    plt.title('Top 15 Counties/CoCs by Homeless Population Size', fontsize=16)
    plt.xlabel('County/CoC', fontsize=14)
    plt.ylabel('Homeless Population Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 100, 
                f'{int(height):,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q3_top_counties_homeless.png')
    plt.close()
    
    # Calculate homeless per capita (or as a percentage of each county's total)
    # Since we don't have county population data, we'll create a figure showing
    # the distribution of homelessness across counties, using percentages of statewide total
    
    # Calculate each county's percentage of total homeless
    total_homeless = county_totals['COUNT_AGE'].sum()
    county_totals['Percentage'] = county_totals['COUNT_AGE'] / total_homeless * 100
    
    # Create a treemap or pie chart showing county distribution
    plt.figure(figsize=(14, 14))
    
    # For counties with very small percentages, group them into "Other"
    threshold = 1.0  # Counties with less than 1% of the state's homeless
    major_counties = county_totals[county_totals['Percentage'] >= threshold]
    other_counties = county_totals[county_totals['Percentage'] < threshold]
    
    if not other_counties.empty:
        # Create a modified dataframe with "Other" category
        plot_data = major_counties.copy()
        other_row = pd.DataFrame({
            'LOCATION_ID': ['Other'],
            'LOCATION': ['Other Counties'],
            'COUNT_AGE': [other_counties['COUNT_AGE'].sum()],
            'Percentage': [other_counties['Percentage'].sum()]
        })
        plot_data = pd.concat([plot_data, other_row], ignore_index=True)
    else:
        plot_data = major_counties
    
    # Sort by percentage for better visualization
    plot_data = plot_data.sort_values('Percentage', ascending=False)
    
    # Create a colormap based on the number of slices
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(plot_data)))
    
    # Create the pie chart
    plt.pie(plot_data['Percentage'], labels=plot_data['LOCATION'], autopct='%1.1f%%', 
            startangle=90, colors=colors, shadow=False)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.title('Distribution of Homelessness Across California Counties', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/q3_geographic_distribution.png')
    plt.close()

def hospital_utilization_analysis(hospital_data):
    """
    Analyze hospital utilization patterns for homeless individuals
    """
    if hospital_data is None:
        print("Hospital data not available for analysis")
        return

    # Check if the required columns are present in the DataFrame
    required_columns = ['EncounterType', 'Demographic', 'DemographicValue', 'Homeless', 'HomelessProportion']
    
    for col in required_columns:
        if col not in hospital_data.columns:
            print(f"Warning: Required column '{col}' not found in hospital_data. Skipping hospital utilization analysis.")
            
            # Create a simple summary plot of available data
            if 'Year' in hospital_data.columns and 'HOMELESS_ED_VISITS' in hospital_data.columns:
                plt.figure(figsize=(10, 6))
                
                # Group by Year if multiple years are present
                if hospital_data['Year'].nunique() > 1:
                    summary = hospital_data.groupby('Year').agg({
                        'HOMELESS_ED_VISITS': 'sum',
                        'HOMELESS_INPATIENT': 'sum'
                    }).reset_index()
                    
                    plt.bar(summary['Year'], summary['HOMELESS_ED_VISITS'], label='ED Visits')
                    plt.bar(summary['Year'], summary['HOMELESS_INPATIENT'], bottom=summary['HOMELESS_ED_VISITS'], 
                           label='Inpatient')
                    
                    plt.title('Homeless Hospital Utilization by Year', fontsize=16)
                    plt.xlabel('Year', fontsize=14)
                    plt.ylabel('Number of Encounters', fontsize=14)
                    plt.legend()
                else:
                    # Just create a simple plot of the current data
                    county_summary = hospital_data.sort_values('HOMELESS_ED_VISITS', ascending=False).head(10)
                    
                    plt.barh(county_summary['LOCATION_ID'], county_summary['HOMELESS_ED_VISITS'], 
                           label='ED Visits')
                    plt.barh(county_summary['LOCATION_ID'], county_summary['HOMELESS_INPATIENT'], 
                           left=county_summary['HOMELESS_ED_VISITS'], label='Inpatient')
                    
                    plt.title('Top 10 Counties by Homeless Hospital Utilization', fontsize=16)
                    plt.xlabel('Number of Encounters', fontsize=14)
                    plt.ylabel('County', fontsize=14)
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig('figures/hospital_utilization_summary.png')
                plt.close()
            
            return
    
    print("Analyzing hospital utilization patterns...")
    
    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # 1. Compare IP vs ED utilization by demographic categories
    # Filter for the latest year
    latest_year = hospital_data['Year'].max()
    latest_data = hospital_data[hospital_data['Year'] == latest_year]
    
    # Aggregated by encounter type and demographic category
    encounter_type_demo = latest_data.groupby(['EncounterType', 'Demographic'])['Homeless'].sum().reset_index()
    encounter_type_demo_pivot = encounter_type_demo.pivot(index='Demographic', columns='EncounterType', values='Homeless')
    
    # Calculate total and sort by it
    encounter_type_demo_pivot['Total'] = encounter_type_demo_pivot.sum(axis=1)
    encounter_type_demo_pivot = encounter_type_demo_pivot.sort_values('Total', ascending=False)
    
    # Plot
    plt.figure(figsize=(14, 10))
    encounter_type_demo_pivot[['ED', 'IP']].plot(kind='bar', stacked=True, 
                                                color=['lightcoral', 'cornflowerblue'])
    plt.title(f'Homeless Hospital Encounters by Demographic Category ({latest_year})', fontsize=16)
    plt.xlabel('Demographic Category', fontsize=14)
    plt.ylabel('Number of Encounters', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Encounter Type')
    
    plt.tight_layout()
    plt.savefig('figures/q5_hospital_encounters_by_demographic.png')
    plt.close()
    
    # 2. Homeless proportion of all hospital encounters
    homeless_proportion = latest_data.groupby(['EncounterType', 'Demographic'])['HomelessProportion'].mean().reset_index()
    
    # Pivot for better visualization
    homeless_proportion_pivot = homeless_proportion.pivot(index='Demographic', 
                                                         columns='EncounterType', 
                                                         values='HomelessProportion')
    
    plt.figure(figsize=(14, 10))
    ax = homeless_proportion_pivot.plot(kind='bar', color=['lightcoral', 'cornflowerblue'])
    plt.title(f'Proportion of Homeless Encounters Among All Hospital Encounters ({latest_year})', fontsize=16)
    plt.xlabel('Demographic Category', fontsize=14)
    plt.ylabel('Proportion of Total Encounters', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.savefig('figures/q5_homeless_proportion_by_demographic.png')
    plt.close()
    
    # 3. Demographic breakdown for ED visits
    ed_data = latest_data[latest_data['EncounterType'] == 'ED']
    
    # Age breakdown
    ed_age = ed_data[ed_data['Demographic'] == 'AGEGROUP']
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=ed_age, x='DemographicValue', y='Homeless', color='cornflowerblue')
    plt.title(f'ED Visits by Age Group ({latest_year})', fontsize=16)
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Number of ED Visits', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for i, v in enumerate(ed_age['Homeless']):
        plt.text(i, v + 1000, f'{int(v):,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/ed_visits_by_age.png')
    plt.close()
    
    # Race breakdown
    ed_race = ed_data[ed_data['Demographic'] == 'RACEGROUP']
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=ed_race, x='DemographicValue', y='Homeless', color='lightcoral')
    plt.title(f'ED Visits by Race/Ethnicity ({latest_year})', fontsize=16)
    plt.xlabel('Race/Ethnicity', fontsize=14)
    plt.ylabel('Number of ED Visits', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for i, v in enumerate(ed_race['Homeless']):
        plt.text(i, v + 1000, f'{int(v):,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/ed_visits_by_race.png')
    plt.close()
    
    # Payer breakdown
    ed_payer = ed_data[ed_data['Demographic'] == 'PAYER']
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=ed_payer, x='DemographicValue', y='Homeless', color='mediumseagreen')
    plt.title(f'ED Visits by Payer ({latest_year})', fontsize=16)
    plt.xlabel('Payer Type', fontsize=14)
    plt.ylabel('Number of ED Visits', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for i, v in enumerate(ed_payer['Homeless']):
        plt.text(i, v + 1000, f'{int(v):,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/ed_visits_by_payer.png')
    plt.close()
    
    return

def system_performance_analysis(spm_data):
    """
    Analyze system performance metrics over time
    """
    print("Analyzing system performance metrics...")
    
    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # Define dictionary of key metrics and their descriptions
    key_metrics = {
        'M1a': 'Total Number of Persons Served',
        'M2': 'Persons in Emergency Shelter',
        'M3': 'Persons in Permanent Housing',
        'M4': 'Length of Time Homeless (days)',
        'M5': 'Returns to Homelessness',
        'M6': 'Homeless for First Time'
    }
    
    # Add LOCATION_ID column if it doesn't exist
    if 'LOCATION_ID' not in spm_data.columns:
        spm_data['LOCATION_ID'] = spm_data['Location'].apply(lambda x: f"CA-{x}" if x.isdigit() else x)
    
    # Extract statewide data for key metrics
    statewide_metrics = spm_data[spm_data['Location'] == 'California']
    if len(statewide_metrics) == 0:
        statewide_metrics = spm_data[spm_data['LOCATION_ID'] == 'All']
    
    statewide_metrics = statewide_metrics[statewide_metrics['Metric'].isin(key_metrics.keys())]
    
    # Check if we have data to analyze
    if len(statewide_metrics) == 0:
        print("Warning: No statewide data found for system performance metrics")
        return
    
    # 1. Statewide metrics over time
    plt.figure(figsize=(16, 12))
    
    for i, (metric, description) in enumerate(key_metrics.items()):
        # Create subplot
        plt.subplot(3, 2, i + 1)
        
        metric_data = statewide_metrics[statewide_metrics['Metric'] == metric]
        if len(metric_data) == 0:
            continue
            
        # Extract year columns and convert to numeric
        year_cols = [col for col in metric_data.columns if col.startswith('CY')]
        
        # If metric_data has only one row
        if len(metric_data) == 1:
            values = metric_data[year_cols].iloc[0]
            years = [int(col[2:]) for col in year_cols]
            
            # Plot
            plt.plot(years, values, marker='o', linewidth=2, markersize=8)
            plt.title(f"{description} ({metric})", fontsize=14)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add data labels
            for x, y in zip(years, values):
                if pd.notna(y):  # Check if value is not NaN
                    label = f"{y:,.0f}" if y > 1000 else f"{y:.2f}"
                    plt.text(x, y + (max(values) - min(values)) * 0.03, label, 
                            ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q6_statewide_metrics_over_time.png')
    plt.close()
    
    # 2. Compare top 5 counties for the most recent year
    latest_year = 'CY23'  # Based on data from 2020-2023
    
    # Get top 5 counties by total population served (M1a)
    top_counties_data = spm_data[spm_data['Metric'] == 'M1a']
    top_counties_data = top_counties_data[(top_counties_data['LOCATION_ID'] != 'All') & 
                                          (top_counties_data['Location'] != 'California')]
    
    if latest_year not in top_counties_data.columns:
        # Find the latest available year
        year_cols = [col for col in top_counties_data.columns if col.startswith('CY')]
        if year_cols:
            latest_year = year_cols[-1]
        else:
            print("Warning: No year columns found for county comparison")
            return
    
    top_counties = top_counties_data.sort_values(latest_year, ascending=False).head(5)['LOCATION_ID'].tolist()
    
    # Extract data for top counties across key metrics
    top_county_metrics = spm_data[
        (spm_data['LOCATION_ID'].isin(top_counties)) & 
        (spm_data['Metric'].isin(key_metrics.keys()))
    ]
    
    # Prepare for plotting
    for metric in key_metrics.keys():
        plt.figure(figsize=(14, 10))
        
        metric_data = top_county_metrics[top_county_metrics['Metric'] == metric]
        
        if len(metric_data) == 0:
            continue
            
        # Create a color palette
        colors = sns.color_palette("viridis", len(top_counties))
        
        for i, county_id in enumerate(top_counties):
            county_data = metric_data[metric_data['LOCATION_ID'] == county_id]
            
            if len(county_data) == 0:
                continue
                
            # Get county name from the location column
            county_name = county_data['Location'].iloc[0]
            
            # Extract year columns and values
            year_cols = [col for col in county_data.columns if col.startswith('CY')]
            values = county_data[year_cols].iloc[0]
            years = [int(col[2:]) for col in year_cols]
            
            # Plot
            plt.plot(years, values, marker='o', linewidth=2, markersize=8, color=colors[i], label=county_name)
        
        plt.title(f"{key_metrics[metric]} ({metric}) - Top 5 Counties", fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'figures/top_counties_{metric}.png')
        plt.close()
    
    # 3. Percentage change from 2020 to 2023 for key metrics
    plt.figure(figsize=(16, 12))
    
    start_year = 'CY20'
    end_year = 'CY23'
    
    # Check if we have these years
    year_cols = [col for col in statewide_metrics.columns if col.startswith('CY')]
    if len(year_cols) >= 2:
        start_year = year_cols[0]
        end_year = year_cols[-1]
    else:
        print(f"Warning: Not enough year columns for percentage change analysis. Found: {year_cols}")
        return
    
    for i, (metric, description) in enumerate(key_metrics.items()):
        # Create subplot
        plt.subplot(3, 2, i + 1)
        
        metric_data = statewide_metrics[statewide_metrics['Metric'] == metric]
        if len(metric_data) == 0 or not all(col in metric_data.columns for col in [start_year, end_year]):
            continue
            
        # Calculate percentage change
        start_value = metric_data[start_year].iloc[0]
        end_value = metric_data[end_year].iloc[0]
        
        if pd.isna(start_value) or pd.isna(end_value) or start_value == 0:
            continue
            
        pct_change = (end_value - start_value) / start_value * 100
        
        # Plot
        bar_color = 'lightcoral' if pct_change > 0 else 'mediumseagreen'
        plt.bar([f'{start_year[2:]}-{end_year[2:]}'], [pct_change], color=bar_color)
        plt.title(f"{description} ({metric}) - % Change", fontsize=14)
        plt.ylabel('Percentage Change (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add data label
        plt.text(0, pct_change + (5 if pct_change > 0 else -5), 
                f"{pct_change:.1f}%", ha='center', va='bottom' if pct_change > 0 else 'top', 
                fontsize=12, fontweight='bold')
        
        # Add baseline
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/metric_percentage_changes.png')
    plt.close()
    
    return

def main():
    """
    Main function to run exploratory data analysis
    """
    # Get data from data preparation module
    master_df, age_data, race_data, gender_data, spm_data, hospital_data, location_mapping = prep_data()
    
    # Run demographic analysis
    demographic_analysis(age_data, race_data, gender_data)
    
    # Run hospital utilization analysis
    hospital_utilization_analysis(hospital_data)
    
    # Run system performance analysis
    system_performance_analysis(spm_data)
    
    print("Exploratory data analysis complete. Visualizations saved in 'figures' directory.")
    
    return

if __name__ == "__main__":
    main() 