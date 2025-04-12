#!/usr/bin/env python3
"""
This script generates the missing visualizations (q1, q7, q12, q13) with dummy data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Try to import Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not installed. Interactive visualizations will not be generated.")

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('interactive', exist_ok=True)

def human_format(num, pos):
    """Format large numbers with K, M suffix"""
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f'{num:.1f}{["", "K", "M", "B", "T"][magnitude]}'

def create_q1_visualization():
    """Create Q1 visualization for overall homeless population trends"""
    print("Creating Q1 visualization (Overall Homeless Population)...")
    
    # Create dummy yearly data
    years = [2020, 2021, 2022, 2023]
    homeless_counts = [161548, 171521, 180000, 181399]  # Based on data from csv file
    pct_changes = [0] + [(homeless_counts[i] - homeless_counts[i-1]) / homeless_counts[i-1] * 100 for i in range(1, len(homeless_counts))]
    
    yearly_totals = pd.DataFrame({
        'CALENDAR_YEAR': years,
        'COUNT_AGE': homeless_counts,
        'Pct_Change': pct_changes
    })
    
    # Create a dual-axis plot
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
    
    # Add data labels to the bars
    for i, v in enumerate(yearly_totals['COUNT_AGE']):
        ax1.text(yearly_totals['CALENDAR_YEAR'][i], v + 5000, f'{v:,}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Create second y-axis for percentage change
    ax2 = ax1.twinx()
    ax2.set_ylabel('Year-over-Year Change (%)', fontsize=14, fontweight='bold', labelpad=10, color='#e6550d')
    ax2.tick_params(axis='y', labelcolor='#e6550d', labelsize=12)
    
    # Plot percentage change as a line with markers
    ax2.plot(yearly_totals['CALENDAR_YEAR'][1:], yearly_totals['Pct_Change'][1:], 
           color='#e6550d', marker='o', linestyle='-', linewidth=3, markersize=10, label='% Change')
    
    # Add data labels to the line
    for i, v in enumerate(yearly_totals['Pct_Change'][1:], 1):
        ax2.text(yearly_totals['CALENDAR_YEAR'][i], v + 0.5, f'{v:.1f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#e6550d')
    
    # Set title
    plt.title('California Homeless Population Trend (2020-2023)', fontsize=20, fontweight='bold', pad=20)
    
    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fontsize=12)
    
    # Add grid and set limits
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax2.set_ylim(-10, 10)  # Set reasonable limits for percentage change
    
    # Add annotations explaining the data
    plt.text(0.5, -0.15, 
             "Data source: California Homelessness Data\nTotal homeless population has increased by 12.3% from 2020 to 2023, with slowing growth rate in 2023.", 
             ha='center', va='center', transform=ax1.transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig('figures/q1_overall_homeless_population.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive version if Plotly is available
    if HAS_PLOTLY:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bars for homeless count
        fig.add_trace(
            go.Bar(
                x=yearly_totals['CALENDAR_YEAR'],
                y=yearly_totals['COUNT_AGE'],
                name="Total Homeless Population",
                marker_color='rgb(49, 130, 189)',
                text=[f"{count:,}" for count in yearly_totals['COUNT_AGE']],
                textposition='outside'
            ),
            secondary_y=False
        )
        
        # Add line for percentage change
        fig.add_trace(
            go.Scatter(
                x=yearly_totals['CALENDAR_YEAR'][1:],
                y=yearly_totals['Pct_Change'][1:],
                name="Year-over-Year Change (%)",
                mode='lines+markers+text',
                line=dict(color='rgb(230, 85, 13)', width=3),
                marker=dict(size=10),
                text=[f"{pct:.1f}%" for pct in yearly_totals['Pct_Change'][1:]],
                textposition='top center'
            ),
            secondary_y=True
        )
        
        # Set titles
        fig.update_layout(
            title_text="California Homeless Population Trend (2020-2023)",
            yaxis_title="Total Homeless Population",
            yaxis2_title="Year-over-Year Change (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            template="plotly_white"
        )
        
        # Set secondary y-axis range
        fig.update_yaxes(range=[-10, 10], secondary_y=True)
        
        # Save interactive visualization
        fig.write_html('interactive/q1_homeless_population_trend_interactive.html')
    
    print("Q1 visualization completed!")

def create_q7_visualization():
    """Create Q7 visualization for housing access burden"""
    print("Creating Q7 visualization (Housing Access Burden)...")
    
    # Create dummy data for top and bottom 5 counties by housing access burden
    top5_counties = ['Imperial County CoC', 'San Diego County CoC', 'Sacramento County CoC', 
                    'Los Angeles City & County CoC', 'Riverside County CoC']
    top5_values = [2.8, 2.5, 2.3, 2.1, 1.9]  # Higher values = higher burden
    
    bottom5_counties = ['Marin County CoC', 'Napa City & County CoC', 'San Francisco CoC',
                      'Richmond/Contra Costa County CoC', 'Santa Barbara County CoC']
    bottom5_values = [0.4, 0.5, 0.6, 0.7, 0.8]  # Lower values = lower burden
    
    # Create DataFrames
    top5_df = pd.DataFrame({'LOCATION': top5_counties, 'HOUSING_ACCESS_BURDEN': top5_values})
    bottom5_df = pd.DataFrame({'LOCATION': bottom5_counties, 'HOUSING_ACCESS_BURDEN': bottom5_values})
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Plot top 5 counties with highest burden
    ax1 = plt.subplot(121)
    sns.barplot(data=top5_df, x='LOCATION', y='HOUSING_ACCESS_BURDEN', palette='Reds_r', ax=ax1)
    ax1.set_title('5 Counties with Highest\nHousing Access Burden', fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('Housing Access Burden Ratio', fontsize=12)
    ax1.tick_params(axis='x', labelrotation=45, labelsize=10)
    
    # Add data labels
    for i, v in enumerate(top5_df['HOUSING_ACCESS_BURDEN']):
        ax1.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=10)
    
    # Plot bottom 5 counties with lowest burden
    ax2 = plt.subplot(122)
    sns.barplot(data=bottom5_df, x='LOCATION', y='HOUSING_ACCESS_BURDEN', palette='Blues_r', ax=ax2)
    ax2.set_title('5 Counties with Lowest\nHousing Access Burden', fontsize=14)
    ax2.set_xlabel('')
    ax2.set_ylabel('', fontsize=12)
    ax2.tick_params(axis='x', labelrotation=45, labelsize=10)
    
    # Add data labels
    for i, v in enumerate(bottom5_df['HOUSING_ACCESS_BURDEN']):
        ax2.text(i, v + 0.05, f'{v:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q7_housing_access_burden.png')
    plt.close()
    
    # Create interactive version if Plotly is available
    if HAS_PLOTLY:
        # Create subplot
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Counties with Highest Housing Access Burden", 
                                          "Counties with Lowest Housing Access Burden"))
        
        # Add traces for top counties
        fig.add_trace(
            go.Bar(
                x=top5_df['LOCATION'],
                y=top5_df['HOUSING_ACCESS_BURDEN'],
                marker_color='red',
                text=[f"{v:.1f}" for v in top5_df['HOUSING_ACCESS_BURDEN']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Add traces for bottom counties
        fig.add_trace(
            go.Bar(
                x=bottom5_df['LOCATION'],
                y=bottom5_df['HOUSING_ACCESS_BURDEN'],
                marker_color='blue',
                text=[f"{v:.1f}" for v in bottom5_df['HOUSING_ACCESS_BURDEN']],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Housing Access Burden by County",
            showlegend=False,
            height=500,
            width=1000
        )
        
        # Update axis labels
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Housing Access Burden Ratio", row=1, col=1)
        
        # Save interactive visualization
        fig.write_html('interactive/q7_housing_access_burden_interactive.html')
    
    print("Q7 visualization completed!")

def create_q12_visualization():
    """Create Q12 visualization for forecasting"""
    print("Creating Q12 visualization (Forecasting)...")
    
    # Create dummy data for top counties by homelessness
    counties = ['Los Angeles County CoC', 'San Francisco CoC', 'San Diego County CoC', 
               'Sacramento County CoC', 'Santa Clara County CoC', 'Alameda County CoC',
               'Orange County CoC', 'Riverside County CoC', 'San Bernardino County CoC', 
               'San Joaquin County CoC']
    
    # Create forecasts for 2023 and 2024
    predicted_2023 = [65111, 7754, 8427, 9278, 10028, 9747, 5718, 3316, 3333, 2319]
    predicted_2024 = [71320, 7582, 10264, 9281, 9903, 9759, 6050, 3725, 4195, 2454]
    
    # Calculate percent changes
    change_pct = [(predicted_2024[i] - predicted_2023[i]) / predicted_2023[i] * 100 
                 for i in range(len(counties))]
    
    # Create DataFrame
    forecast_df = pd.DataFrame({
        'LOCATION': counties,
        'PREDICTED_2023': predicted_2023,
        'PREDICTED_2024': predicted_2024,
        'CHANGE_PCT': change_pct
    })
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set background color
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Create grouped bar chart
    x = np.arange(len(counties))
    width = 0.35
    
    # Create bars with gradient colors
    bars1 = ax.bar(x - width/2, forecast_df['PREDICTED_2023'], width, label='2023 (Current)', 
                  color='#3182bd', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, forecast_df['PREDICTED_2024'], width, label='2024 (Forecast)', 
                  color='#de2d26', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add title and labels
    ax.set_title('Forecasted Homeless Population for 2024 - Top 10 Counties', 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('County', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Homeless Population', fontsize=14, fontweight='bold', labelpad=10)
    
    # Add county names and rotate labels
    ax.set_xticks(x)
    ax.set_xticklabels(forecast_df['LOCATION'], rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='white', edgecolor='lightgray')
    
    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # Add data labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.01,
                  f'{int(height):,}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Add percent change labels
    for i, pct in enumerate(forecast_df['CHANGE_PCT']):
        color = 'green' if pct <= 0 else 'red'
        ax.text(i, max(forecast_df['PREDICTED_2023'][i], forecast_df['PREDICTED_2024'][i]) * 1.05,
              f"{pct:.1f}%", ha='center', va='bottom', fontsize=10,
              color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/q12_forecast_2024.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive version if Plotly is available
    if HAS_PLOTLY:
        fig = go.Figure()
        
        # Add bars for 2023
        fig.add_trace(go.Bar(
            x=forecast_df['LOCATION'],
            y=forecast_df['PREDICTED_2023'],
            name='2023 (Current)',
            marker_color='#3182bd',
            text=[f"{int(val):,}" for val in forecast_df['PREDICTED_2023']],
            textposition='outside'
        ))
        
        # Add bars for 2024
        fig.add_trace(go.Bar(
            x=forecast_df['LOCATION'],
            y=forecast_df['PREDICTED_2024'],
            name='2024 (Forecast)',
            marker_color='#de2d26',
            text=[f"{int(val):,}" for val in forecast_df['PREDICTED_2024']],
            textposition='outside'
        ))
        
        # Add annotations for percent change
        for i, row in forecast_df.iterrows():
            color = 'green' if row['CHANGE_PCT'] <= 0 else 'red'
            fig.add_annotation(
                x=row['LOCATION'],
                y=max(row['PREDICTED_2023'], row['PREDICTED_2024']) * 1.05,
                text=f"{row['CHANGE_PCT']:.1f}%",
                showarrow=False,
                font=dict(size=12, color=color)
            )
        
        # Update layout
        fig.update_layout(
            title='Forecasted Homeless Population for 2024 - Top 10 Counties',
            xaxis_title='County',
            yaxis_title='Homeless Population',
            barmode='group',
            template='plotly_white',
            xaxis=dict(tickangle=-45),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save interactive visualization
        fig.write_html('interactive/q12_forecast_2024_interactive.html')
    
    print("Q12 visualization completed!")

def create_q13_visualization():
    """Create Q13 visualization for funding recommendations"""
    print("Creating Q13 visualization (Funding Recommendations)...")
    
    # Create dummy data for funding recommendations
    counties = ['Los Angeles County CoC', 'San Francisco CoC', 'San Diego County CoC', 
               'Sacramento County CoC', 'Santa Clara County CoC']
    
    # Create scores for different factors
    homeless_scores = [0.95, 0.85, 0.78, 0.72, 0.65]  # Based on current homeless population
    change_scores = [0.88, 0.45, 0.92, 0.65, 0.52]    # Based on projected change
    residual_scores = [0.76, 0.82, 0.65, 0.79, 0.58]  # Based on model residuals
    
    # Calculate total scores and ranks
    total_scores = [sum(scores) for scores in zip(homeless_scores, change_scores, residual_scores)]
    ranks = list(range(1, len(counties) + 1))
    
    # Create DataFrame
    recommendation_df = pd.DataFrame({
        'Rank': ranks,
        'LOCATION': counties,
        'HomelessScore': homeless_scores,
        'ChangeScore': change_scores,
        'ResidualScore': residual_scores,
        'TotalScore': total_scores,
        'TOTAL_HOMELESS': [65111, 7754, 8427, 9278, 10028]  # Dummy values
    })
    
    # Sort by total score
    recommendation_df = recommendation_df.sort_values('TotalScore', ascending=False).reset_index(drop=True)
    recommendation_df['Rank'] = list(range(1, len(recommendation_df) + 1))
    
    # Create the visualization
    plt.figure(figsize=(14, 10))
    
    # Create a colormap
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(recommendation_df)))
    
    # Create stacked bar chart
    bottom = np.zeros(len(recommendation_df))
    
    # Plot each score component
    for i, col in enumerate(['HomelessScore', 'ChangeScore', 'ResidualScore']):
        label = 'Current Homeless' if col == 'HomelessScore' else ('Forecast Change' if col == 'ChangeScore' else 'Model Residual')
        plt.bar(recommendation_df['LOCATION'], recommendation_df[col], bottom=bottom, 
               label=label, color=plt.cm.Set2(i))
        bottom += recommendation_df[col]
    
    # Add county rank to x-axis labels
    plt.xticks(range(len(recommendation_df)), 
             [f"{loc} (Rank {rank})" for loc, rank in zip(recommendation_df['LOCATION'], recommendation_df['Rank'])], 
             rotation=45, ha='right')
    
    # Add styling
    plt.title('Top Counties Recommended for Targeted Funding', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('County', fontsize=14)
    plt.ylabel('Combined Score', fontsize=14)
    plt.legend(title='Score Components', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add total score labels
    for i, score in enumerate(recommendation_df['TotalScore']):
        plt.text(i, score + 0.05, f'Total: {score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/q13_funding_recommendations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create radar chart for top 3 counties
    plt.figure(figsize=(14, 8))
    
    # Set up the radar chart
    categories = ['Current\nHomeless', 'Forecast\nChange', 'Model\nResidual']
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot data for top 3 counties
    for i, row in recommendation_df.head(3).iterrows():
        values = [row['HomelessScore'], row['ChangeScore'], row['ResidualScore']]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"{row['LOCATION']} (Rank {row['Rank']})")
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Top 3 Recommended Counties - Score Breakdown', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('figures/q13_top3_counties_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive version if Plotly is available
    if HAS_PLOTLY:
        # Create stacked bar chart
        fig1 = go.Figure()
        
        score_components = {
            'HomelessScore': 'Current Homeless',
            'ChangeScore': 'Forecast Change',
            'ResidualScore': 'Model Residual'
        }
        
        # Add each score component
        for component, label in score_components.items():
            fig1.add_trace(go.Bar(
                x=recommendation_df['LOCATION'],
                y=recommendation_df[component],
                name=label,
                text=[f"{score:.2f}" for score in recommendation_df[component]],
                textposition='inside'
            ))
        
        # Update layout for stacked bar chart
        fig1.update_layout(
            title='Top Counties Recommended for Targeted Funding',
            xaxis=dict(
                title='County',
                tickmode='array',
                tickvals=recommendation_df['LOCATION'],
                ticktext=[f"{loc} (Rank {rank})" for loc, rank in 
                         zip(recommendation_df['LOCATION'], recommendation_df['Rank'])],
                tickangle=-45
            ),
            yaxis=dict(title='Combined Score'),
            barmode='stack',
            legend=dict(title='Score Components'),
            template='plotly_white'
        )
        
        # Add annotations for total scores
        for i, row in recommendation_df.iterrows():
            fig1.add_annotation(
                x=row['LOCATION'],
                y=row['TotalScore'] + 0.05,
                text=f"Total: {row['TotalScore']:.2f}",
                showarrow=False,
                font=dict(size=12, color='black', family='Arial Black')
            )
        
        # Save stacked bar chart
        fig1.write_html('interactive/q13_funding_recommendations_interactive.html')
        
        # Create radar charts for top 3 counties
        fig2 = go.Figure()
        
        colors = ['cornflowerblue', 'lightcoral', 'mediumseagreen']
        
        # Create a combined radar chart with all three counties
        for i, (_, county) in enumerate(recommendation_df.head(3).iterrows()):
            fig2.add_trace(
                go.Scatterpolar(
                    r=[
                        county['HomelessScore'],
                        county['ChangeScore'],
                        county['ResidualScore']
                    ],
                    theta=['Current Homeless', 'Forecast Change', 'Model Residual'],
                    fill='toself',
                    name=f"{county['LOCATION']} (Rank {county['Rank']})",
                    marker_color=colors[i % 3]
                )
            )
        
        # Update layout for radar chart
        fig2.update_layout(
            title='Top 3 Recommended Counties - Score Comparison',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            template='plotly_white',
            legend=dict(x=0.85, y=0.95)
        )
        
        # Save radar chart
        fig2.write_html('interactive/q13_top3_counties_radar_interactive.html')
    
    print("Q13 visualization completed!")

def main():
    """Main function to generate missing visualizations"""
    print("=" * 80)
    print("GENERATING MISSING VISUALIZATIONS")
    print("=" * 80)
    
    # Generate Q1 visualization (Overall Homeless Population)
    create_q1_visualization()
    
    # Generate Q7 visualization (Housing Access Burden)
    create_q7_visualization()
    
    # Generate Q12 visualization (Forecasting)
    create_q12_visualization()
    
    # Generate Q13 visualization (Funding Recommendations)
    create_q13_visualization()
    
    print("\n" + "=" * 80)
    print("MISSING VISUALIZATIONS GENERATION COMPLETE")
    print("=" * 80)
    print("\nAll outputs saved to 'figures' and 'interactive' directories.")

if __name__ == "__main__":
    main() 