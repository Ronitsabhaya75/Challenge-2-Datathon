import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Try to import plotly for interactive visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not installed. Using matplotlib for all visualizations.")

# Set Seaborn style
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('interactive', exist_ok=True)

def create_q5_visualization():
    """
    Create visualization for Q5 - System Performance Metrics Trends
    """
    print("Creating Q5 visualization (System Performance Metrics)...")
    
    # Load system performance data
    spm_data = pd.read_csv('calendar-year-coc-and-statewide-topline-ca-spms.csv')
    
    # Define key metrics and their descriptions
    key_metrics = {
        'M1a': 'Total Persons Served',
        'M2': 'Emergency Shelter Usage',
        'M3': 'Permanent Housing Placements',
        'M4': 'Length of Time Homeless (days)',
        'M5': 'Returns to Homelessness Rate',
        'M6': 'First Time Homeless'
    }
    
    # Extract statewide data for California
    statewide_data = spm_data[spm_data['Location'] == 'California']
    
    # Check if we have data to analyze
    if len(statewide_data) == 0:
        print("Warning: No statewide data found for system performance metrics")
        return
    
    # Filter for key metrics only
    statewide_data = statewide_data[statewide_data['Metric'].isin(key_metrics.keys())]
    
    # Extract year columns and convert to numeric
    year_cols = [col for col in statewide_data.columns if col.startswith('CY')]
    
    # Create a DataFrame with metrics as rows and years as columns
    trend_data = {}
    
    # Get data for each metric
    for metric, description in key_metrics.items():
        metric_data = statewide_data[statewide_data['Metric'] == metric]
        if len(metric_data) > 0:
            # Get values for each year
            values = metric_data[year_cols].iloc[0].values
            trend_data[description] = values
    
    # Create a DataFrame
    trend_df = pd.DataFrame(trend_data, index=[year[2:] for year in year_cols])
    
    # Normalize data for better comparison (2020 = 100)
    normalized_df = pd.DataFrame()
    
    for col in trend_df.columns:
        if col != 'Length of Time Homeless (days)' and col != 'Returns to Homelessness Rate':
            # For counts, normalize to 2020 = 100
            base_value = trend_df[col].iloc[0]
            if base_value > 0:
                normalized_df[col] = trend_df[col] / base_value * 100
        else:
            # For these metrics, lower is better, so invert the normalization
            base_value = trend_df[col].iloc[0]
            if base_value > 0:
                normalized_df[col] = (2 * base_value - trend_df[col]) / base_value * 100
    
    # Create static visualization with Matplotlib
    plt.figure(figsize=(16, 10))
    
    # Use a line plot with markers for each year
    for col in normalized_df.columns:
        plt.plot(normalized_df.index, normalized_df[col], marker='o', linewidth=2, markersize=8, label=col)
    
    # Add styling
    plt.title('California Homelessness System Performance Metrics\n(Normalized, 2020 = 100)', fontsize=18, fontweight='bold')
    plt.xlabel('Year', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Value (2020 = 100)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(title='System Performance Metrics', fontsize=12, title_fontsize=14)
    
    # Add annotations to explain the normalization
    plt.annotate(
        "Note: Length of Time Homeless and Returns to Homelessness\nare inverted so that higher values represent improvement",
        xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    
    # Add a horizontal line at 100 to show the baseline
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    # Add data labels for the most recent year
    for col in normalized_df.columns:
        plt.annotate(
            f"{normalized_df[col].iloc[-1]:.1f}",
            xy=(normalized_df.index[-1], normalized_df[col].iloc[-1]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7)
        )
    
    plt.tight_layout()
    plt.savefig('figures/q5_system_performance_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second visualization showing actual values for context
    fig, axs = plt.subplots(3, 2, figsize=(18, 14))
    axs = axs.flatten()
    
    # For each metric, create a subplot
    for i, (metric, description) in enumerate(key_metrics.items()):
        # Get the data for this metric
        metric_data = statewide_data[statewide_data['Metric'] == metric]
        if len(metric_data) == 0:
            continue
            
        values = metric_data[year_cols].iloc[0].values
        years = [int(col[2:]) for col in year_cols]
        
        # Create the bar chart
        bars = axs[i].bar(years, values, color=plt.cm.viridis(i/len(key_metrics)))
        
        # Add titles and labels
        axs[i].set_title(f"{description} ({metric})", fontsize=14, fontweight='bold')
        axs[i].set_xlabel('Year', fontsize=12)
        axs[i].set_ylabel('Value', fontsize=12)
        axs[i].grid(True, alpha=0.3)
        
        # Add data labels
        for bar, value in zip(bars, values):
            if pd.notna(value):  # Check if value is not NaN
                if value > 1000:
                    label = f"{value:,.0f}"
                elif value < 1:
                    label = f"{value:.3f}"
                else:
                    label = f"{value:.1f}"
                
                axs[i].text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (max(values) - min(values)) * 0.03, 
                    label, ha='center', va='bottom', fontsize=10,
                    rotation=0 if len(label) < 6 else 45
                )
                
        # If this is M5 (rate), format y-axis as percentage
        if metric == 'M5':
            axs[i].set_ylim(0, max(values) * 1.2)
            # Format y-axis ticks as percentages
            axs[i].set_yticklabels([f"{x:.0%}" for x in axs[i].get_yticks()])
            
        # Add a trend line
        if len(years) > 1:
            z = np.polyfit(range(len(years)), values, 1)
            p = np.poly1d(z)
            trend_x = range(len(years))
            trend_y = p(trend_x)
            
            # Check the trend direction
            if trend_y[-1] > trend_y[0]:
                trend_color = 'green' if metric not in ['M4', 'M5'] else 'red'
                trend_label = 'Increasing (worse)' if metric in ['M4', 'M5'] else 'Increasing (better)'
            else:
                trend_color = 'red' if metric not in ['M4', 'M5'] else 'green'
                trend_label = 'Decreasing (better)' if metric in ['M4', 'M5'] else 'Decreasing (worse)'
                
            axs[i].plot([years[j] for j in trend_x], trend_y, color=trend_color, linestyle='--', 
                      label=trend_label)
            axs[i].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q5_system_performance_metrics_detail.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive visualization if Plotly is available
    if HAS_PLOTLY:
        create_q5_interactive(trend_df, normalized_df)
    
    print("Q5 visualization completed!")

def create_q5_interactive(trend_df, normalized_df):
    """
    Create interactive visualization for Q5 using Plotly
    """
    print("Creating Q5 interactive visualization...")
    
    # Define custom colors
    PLOTLY_BG_COLOR = '#0e1117'
    PLOTLY_TEXT_COLOR = 'white'
    PLOTLY_GRID_COLOR = 'rgba(255, 255, 255, 0.1)'
    
    # Create normalized metrics visualization
    fig1 = go.Figure()
    
    # Add a trace for each metric
    colors = px.colors.qualitative.Safe
    
    for i, col in enumerate(normalized_df.columns):
        fig1.add_trace(
            go.Scatter(
                x=normalized_df.index,
                y=normalized_df[col],
                mode='lines+markers',
                name=col,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=10),
                hovertemplate="%{x}: %{y:.1f}<extra></extra>"
            )
        )
    
    # Add a horizontal line at 100
    fig1.add_shape(
        type="line",
        x0=normalized_df.index[0],
        y0=100,
        x1=normalized_df.index[-1],
        y1=100,
        line=dict(color="gray", width=2, dash="dash")
    )
    
    # Update layout
    fig1.update_layout(
        title="Interactive System Performance Metrics<br><sup>Normalized to 2020=100, higher values indicate improvement</sup>",
        xaxis_title="Year",
        yaxis_title="Normalized Value (2020 = 100)",
        legend_title="System Performance Metrics",
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color=PLOTLY_TEXT_COLOR),
        width=1000,
        height=600,
        annotations=[
            dict(
                text="Note: Length of Time Homeless and Returns to Homelessness are inverted so higher values indicate improvement",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.15,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.1)",
                bordercolor="rgba(255, 255, 255, 0.3)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    
    # Save interactive visualization
    fig1.write_html('interactive/q5_normalized_metrics_interactive.html')
    
    # Create actual metrics visualization
    fig2 = make_subplots(
        rows=3, cols=2,
        subplot_titles=[col for col in trend_df.columns],
        vertical_spacing=0.1
    )
    
    # Add a trace for each metric in its own subplot
    for i, col in enumerate(trend_df.columns):
        row = i // 2 + 1
        col_num = i % 2 + 1
        
        fig2.add_trace(
            go.Bar(
                x=trend_df.index,
                y=trend_df[col],
                name=col,
                marker_color=colors[i % len(colors)],
                text=[f"{val:,.0f}" if val > 1000 else f"{val:.3f}" if val < 1 else f"{val:.1f}" 
                      for val in trend_df[col]],
                textposition='outside',
                hovertemplate="%{x}: %{y}<extra></extra>"
            ),
            row=row, col=col_num
        )
        
        # Add trend line
        x_numeric = np.arange(len(trend_df.index))
        if len(x_numeric) > 1:
            z = np.polyfit(x_numeric, trend_df[col].values, 1)
            p = np.poly1d(z)
            trend_y = p(x_numeric)
            
            fig2.add_trace(
                go.Scatter(
                    x=trend_df.index,
                    y=trend_y,
                    mode='lines',
                    line=dict(
                        color='red' if (trend_y[-1] > trend_y[0] and col in ['Length of Time Homeless (days)', 'Returns to Homelessness Rate']) or 
                              (trend_y[-1] < trend_y[0] and col not in ['Length of Time Homeless (days)', 'Returns to Homelessness Rate']) 
                              else 'green',
                        dash='dash',
                        width=2
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col_num
            )
    
    # Update layout
    fig2.update_layout(
        title="Interactive System Performance Metrics Details<br><sup>Actual values by metric</sup>",
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color=PLOTLY_TEXT_COLOR),
        width=1200,
        height=900,
        showlegend=False
    )
    
    # Save interactive visualization
    fig2.write_html('interactive/q5_detailed_metrics_interactive.html')

if __name__ == "__main__":
    create_q5_visualization() 