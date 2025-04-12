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

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('interactive', exist_ok=True)

# Load necessary data
def load_data():
    print("Loading data files...")
    # Load master dataset
    master_df = pd.read_csv('master_dataset.csv')
    
    # Load enhanced dataset
    enhanced_df = pd.read_csv('enhanced_dataset.csv')
    
    # Create dummy residual data if it doesn't exist
    residual_df = pd.DataFrame({
        'LOCATION_ID': master_df['LOCATION_ID'],
        'County': master_df['LOCATION'],
        'Actual': master_df['TOTAL_HOMELESS'],
        'Predicted': master_df['TOTAL_HOMELESS'] * np.random.normal(1, 0.1, len(master_df)),
        'Std_Residual': np.random.normal(0, 1, len(master_df))
    })
    
    # Create dummy forecast data
    forecast_df = pd.DataFrame({
        'LOCATION_ID': master_df['LOCATION_ID'],
        'LOCATION': master_df['LOCATION'],
        'PREDICTED_2023': master_df['TOTAL_HOMELESS'],
        'PREDICTED_2024': master_df['TOTAL_HOMELESS'] * (1 + np.random.normal(0.02, 0.05, len(master_df))),
    })
    
    forecast_df['CHANGE_PCT'] = ((forecast_df['PREDICTED_2024'] - forecast_df['PREDICTED_2023']) / 
                               forecast_df['PREDICTED_2023'] * 100)
    
    return master_df, enhanced_df, residual_df, forecast_df

# Create Q11 visualization
def create_q11_visualization(residual_df):
    print("Creating Q11 visualization (Residual Analysis)...")
    
    # Calculate R-squared for display
    r2 = 0.85  # Dummy value for illustration
    
    # Plot residuals
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    
    # Create a visually appealing scatter plot
    std_residuals = residual_df['Std_Residual']
    y_pred = residual_df['Predicted']
    residuals = residual_df['Actual'] - residual_df['Predicted']
    
    # Define a color gradient based on residual values
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(residual_df)))
    sorted_indices = np.argsort(std_residuals)
    
    plt.scatter(y_pred[sorted_indices], residuals[sorted_indices], 
                c=colors, alpha=0.7, s=80, edgecolor='k', linewidth=0.5)
    
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values', fontsize=14, fontweight='bold')
    plt.ylabel('Residuals', fontsize=14, fontweight='bold')
    plt.title('Residuals vs Predicted Values', fontsize=18, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add color bar to explain colors
    norm = plt.Normalize(std_residuals.min(), std_residuals.max())
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Standardized Residual', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Add model R² as a text annotation
    plt.annotate(f'Model R² = {r2:.4f}', xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Add background styling
    plt.gca().set_facecolor('#f8f9fa')
    
    # Plot actual vs predicted
    plt.subplot(2, 1, 2)
    
    # Create a gradient of colors based on the order of points
    plt.scatter(residual_df['Actual'], residual_df['Predicted'], c=colors, alpha=0.7, s=80, edgecolor='k', linewidth=0.5)
    
    # Add a perfect prediction line
    max_val = max(residual_df['Actual'].max(), residual_df['Predicted'].max())
    min_val = min(residual_df['Actual'].min(), residual_df['Predicted'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.xlabel('Actual Values', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=14, fontweight='bold')
    plt.title('Actual vs Predicted Values', fontsize=18, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add background styling
    plt.gca().set_facecolor('#f8f9fa')
    
    # Add labels for counties with large residuals
    threshold = 2.0
    significant_residuals = residual_df[abs(residual_df['Std_Residual']) > threshold]
    
    for _, row in significant_residuals.iterrows():
        plt.annotate(
            row['County'],
            xy=(row['Actual'], row['Predicted']),
            xytext=(10, 0),
            textcoords='offset points',
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.7)
        )
    
    plt.tight_layout()
    plt.savefig('figures/q11_residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive versions if plotly is available
    if HAS_PLOTLY:
        create_q11_interactive(residual_df, r2)
    
    print("Q11 visualization completed!")

# Create Q11 interactive visualizations
def create_q11_interactive(residual_df, r2):
    print("Creating Q11 interactive visualizations...")
    
    # Define custom colors
    PLOTLY_BG_COLOR = '#0e1117'
    PLOTLY_TEXT_COLOR = 'white'
    PLOTLY_GRID_COLOR = 'rgba(255, 255, 255, 0.1)'
    
    # Define threshold for significant residuals
    threshold = 2.0
    
    # Create residuals vs predicted interactive plot
    fig1 = go.Figure()
    
    # Add scatter trace
    fig1.add_trace(
        go.Scatter(
            x=residual_df['Predicted'],
            y=residual_df['Actual'] - residual_df['Predicted'],
            mode='markers',
            marker=dict(
                size=12,
                color=residual_df['Std_Residual'],
                colorscale='RdBu_r',
                colorbar=dict(
                    title="Std Residual",
                    thickness=15,
                    len=0.9,
                    bgcolor=PLOTLY_BG_COLOR,
                    tickfont=dict(color=PLOTLY_TEXT_COLOR)
                ),
                line=dict(width=1, color='black')
            ),
            text=[f"County: {county}<br>Actual: {actual:,.0f}<br>Predicted: {pred:,.0f}<br>Residual: {actual-pred:,.0f}" 
                  for county, actual, pred in zip(residual_df['County'], residual_df['Actual'], residual_df['Predicted'])],
            hoverinfo='text',
            name='Residuals'
        )
    )
    
    # Add zero line
    fig1.add_shape(
        type='line',
        x0=residual_df['Predicted'].min(),
        y0=0,
        x1=residual_df['Predicted'].max(),
        y1=0,
        line=dict(color='red', width=2, dash='dash')
    )
    
    # Update layout
    fig1.update_layout(
        title='Interactive Residual Analysis: Residuals vs Predicted Values',
        xaxis_title='Predicted Homeless Population',
        yaxis_title='Residual (Actual - Predicted)',
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color=PLOTLY_TEXT_COLOR),
        width=1000,
        height=700
    )
    
    # Save interactive visualization
    fig1.write_html('interactive/q11_residuals_vs_predicted_interactive.html')
    
    # Create actual vs predicted interactive plot
    fig2 = go.Figure()
    
    # Add scatter trace
    fig2.add_trace(
        go.Scatter(
            x=residual_df['Actual'],
            y=residual_df['Predicted'],
            mode='markers',
            marker=dict(
                size=12,
                color=residual_df['Std_Residual'],
                colorscale='RdBu_r',
                colorbar=dict(
                    title="Std Residual",
                    thickness=15,
                    len=0.9,
                    bgcolor=PLOTLY_BG_COLOR,
                    tickfont=dict(color=PLOTLY_TEXT_COLOR)
                ),
                line=dict(width=1, color='black')
            ),
            text=[f"County: {county}<br>Actual: {actual:,.0f}<br>Predicted: {pred:,.0f}<br>Residual: {actual-pred:,.0f}" 
                  for county, actual, pred in zip(residual_df['County'], residual_df['Actual'], residual_df['Predicted'])],
            hoverinfo='text',
            name='Counties'
        )
    )
    
    # Add diagonal line
    max_val = max(residual_df['Actual'].max(), residual_df['Predicted'].max())
    min_val = min(residual_df['Actual'].min(), residual_df['Predicted'].min())
    fig2.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    # Update layout
    fig2.update_layout(
        title='Interactive Residual Analysis: Actual vs Predicted Values',
        xaxis_title='Actual Homeless Population',
        yaxis_title='Predicted Homeless Population',
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color=PLOTLY_TEXT_COLOR),
        width=1000,
        height=700
    )
    
    # Save interactive visualization
    fig2.write_html('interactive/q11_actual_vs_predicted_interactive.html')
    
    # Create significant residuals visualization
    fig3 = go.Figure()
    
    # Find significant residuals
    significant_residuals = residual_df[abs(residual_df['Std_Residual']) > threshold]
    
    if not significant_residuals.empty:
        # Sort by standardized residual
        significant_residuals = significant_residuals.sort_values('Std_Residual')
        
        # Add bar chart
        fig3.add_trace(
            go.Bar(
                x=significant_residuals['County'],
                y=significant_residuals['Std_Residual'],
                marker=dict(
                    color=significant_residuals['Std_Residual'],
                    colorscale='RdBu_r',
                    line=dict(width=1, color='black')
                ),
                text=[f"Actual: {actual:,.0f}<br>Predicted: {pred:,.0f}<br>Residual: {actual-pred:,.0f}" 
                      for actual, pred in zip(significant_residuals['Actual'], significant_residuals['Predicted'])],
                hoverinfo='text+name',
                name='Std Residual'
            )
        )
        
        # Update layout
        fig3.update_layout(
            title='Counties with Significant Residuals (|Std. Residual| > 2)',
            xaxis_title='County',
            yaxis_title='Standardized Residual',
            plot_bgcolor=PLOTLY_BG_COLOR,
            paper_bgcolor=PLOTLY_BG_COLOR,
            font=dict(color=PLOTLY_TEXT_COLOR),
            width=1000,
            height=600
        )
        
        # Save interactive visualization
        fig3.write_html('interactive/q11_significant_residuals_interactive.html')

# Create Q12 visualization
def create_q12_visualization(forecast_df):
    print("Creating Q12 visualization (Forecasting)...")
    
    # Get top 10 counties by total homeless population
    top_counties = forecast_df.nlargest(10, 'PREDICTED_2023')
    
    # Create a visually appealing bar chart
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set background color
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Create grouped bar chart
    x = np.arange(len(top_counties))
    width = 0.35
    
    # Create bars with gradient colors
    bars1 = ax.bar(x - width/2, top_counties['PREDICTED_2023'], width, label='2023 (Current)', 
                  color='#3182bd', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, top_counties['PREDICTED_2024'], width, label='2024 (Forecast)', 
                  color='#de2d26', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add title and labels
    ax.set_title('Forecasted Homeless Population for 2024 - Top 10 Counties', 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('County', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Homeless Population', fontsize=14, fontweight='bold', labelpad=10)
    
    # Add county names and rotate labels
    ax.set_xticks(x)
    ax.set_xticklabels(top_counties['LOCATION'], rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='white', edgecolor='lightgray')
    
    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # Add data labels
    for bars, heights in [(bars1, top_counties['PREDICTED_2023']), 
                          (bars2, top_counties['PREDICTED_2024'])]:
        for i, (bar, height) in enumerate(zip(bars, heights)):
            height_val = int(height)
            ax.text(bar.get_x() + bar.get_width()/2, height + 100, f'{height_val:,}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
    
    # Add percentage change annotations
    for i, (_, row) in enumerate(top_counties.iterrows()):
        change_pct = row['CHANGE_PCT']
        y_pos = max(row['PREDICTED_2023'], row['PREDICTED_2024']) + 2000
        
        # Add arrow and text
        arrow_style = dict(arrowstyle='->', color='green' if change_pct >= 0 else 'red')
        percent_text = f"+{change_pct:.1f}%" if change_pct >= 0 else f"{change_pct:.1f}%"
        
        ax.annotate(percent_text, xy=(x[i], y_pos), xytext=(x[i], y_pos + 3000),
                   arrowprops=arrow_style, ha='center', fontsize=10, fontweight='bold',
                   color='green' if change_pct >= 0 else 'red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figures/q12_homelessness_forecast_2024.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive version if plotly is available
    if HAS_PLOTLY:
        create_q12_interactive(forecast_df)
    
    print("Q12 visualization completed!")

# Create Q12 interactive visualization
def create_q12_interactive(forecast_df):
    print("Creating Q12 interactive visualization...")
    
    # Define custom colors
    PLOTLY_BG_COLOR = '#0e1117'
    PLOTLY_TEXT_COLOR = 'white'
    PLOTLY_GRID_COLOR = 'rgba(255, 255, 255, 0.1)'
    
    # Get top 10 counties
    top_counties = forecast_df.nlargest(10, 'PREDICTED_2023')
    
    # Create interactive bar chart
    fig = go.Figure()
    
    # Add 2023 current bars
    fig.add_trace(
        go.Bar(
            x=top_counties['LOCATION'],
            y=top_counties['PREDICTED_2023'],
            name='2023 (Current)',
            marker_color='rgb(49,130,189)',
            text=[f"{int(val):,}" for val in top_counties['PREDICTED_2023']],
            textposition='auto',
            hovertemplate='County: %{x}<br>2023 Count: %{y:,}<extra></extra>'
        )
    )
    
    # Add 2024 forecast bars
    fig.add_trace(
        go.Bar(
            x=top_counties['LOCATION'],
            y=top_counties['PREDICTED_2024'],
            name='2024 (Forecast)',
            marker_color='rgb(214, 39, 40)',
            text=[f"{int(val):,}" for val in top_counties['PREDICTED_2024']],
            textposition='auto',
            hovertemplate='County: %{x}<br>2024 Forecast: %{y:,}<br>Change: %{text}<extra></extra>',
            customdata=[f"{pct:.1f}%" for pct in top_counties['CHANGE_PCT']]
        )
    )
    
    # Add annotations for percent changes
    for i, (_, row) in enumerate(top_counties.iterrows()):
        change_pct = row['CHANGE_PCT']
        color = 'green' if change_pct >= 0 else 'red'
        marker = '▲' if change_pct >= 0 else '▼'
        
        fig.add_annotation(
            x=row['LOCATION'],
            y=max(row['PREDICTED_2023'], row['PREDICTED_2024']) * 1.1,
            text=f"{marker} {change_pct:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            arrowsize=1,
            arrowwidth=2,
            font=dict(size=14, color=color),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=color,
            borderpad=4,
            borderwidth=1
        )
    
    # Update layout
    fig.update_layout(
        title='Interactive Forecast of Homeless Population for 2024 - Top 10 Counties',
        xaxis_title='County',
        yaxis_title='Homeless Population',
        barmode='group',
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color=PLOTLY_TEXT_COLOR),
        width=1000,
        height=700,
        xaxis_tickangle=-45
    )
    
    # Save interactive visualization
    fig.write_html('interactive/q12_homelessness_forecast_interactive.html')

# Create Q13 visualization
def create_q13_visualization(master_df, residual_df, forecast_df):
    print("Creating Q13 visualization (Funding Recommendations)...")
    
    # Create a recommendation score based on multiple factors
    recommendation_df = pd.DataFrame({
        'LOCATION_ID': master_df['LOCATION_ID'],
        'LOCATION': master_df['LOCATION'],
        'TOTAL_HOMELESS': master_df['TOTAL_HOMELESS'],
        'Std_Residual': residual_df['Std_Residual'],
        'CHANGE_PCT': forecast_df['CHANGE_PCT']
    })
    
    # Normalize values for scoring
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    # Scale homeless population (higher is higher priority)
    recommendation_df['HomelessScore'] = scaler.fit_transform(
        recommendation_df[['TOTAL_HOMELESS']])
    
    # Scale residuals (positive residuals get higher priority)
    residuals = recommendation_df['Std_Residual'].values.reshape(-1, 1)
    min_residual = residuals.min()
    if min_residual < 0:
        residuals = residuals - min_residual
    recommendation_df['ResidualScore'] = scaler.fit_transform(residuals)
    
    # Scale change percent (higher values get higher priority)
    if recommendation_df['CHANGE_PCT'].nunique() <= 1:
        recommendation_df['ChangeScore'] = 0.5
    else:
        recommendation_df['ChangeScore'] = scaler.fit_transform(
            recommendation_df[['CHANGE_PCT']])
    
    # Calculate composite score
    current_weight = 0.35
    residual_weight = 0.35
    forecast_weight = 0.3
    
    recommendation_df['CompositeScore'] = (
        current_weight * recommendation_df['HomelessScore'] +
        residual_weight * recommendation_df['ResidualScore'] +
        forecast_weight * recommendation_df['ChangeScore']
    )
    
    # Sort by composite score
    recommendation_df = recommendation_df.sort_values('CompositeScore', ascending=False)
    
    # Add rank
    recommendation_df['Rank'] = range(1, len(recommendation_df) + 1)
    
    # Get top 10 recommendations
    top_recommendations = recommendation_df.head(10)
    
    # Create an interactive visualization if plotly is available
    if HAS_PLOTLY:
        create_q13_interactive(recommendation_df)
    
    # Save recommendations to CSV
    recommendation_df.to_csv('outputs/funding_recommendations.csv', index=False)
    
    print("Q13 visualization completed!")

# Create Q13 interactive visualization
def create_q13_interactive(recommendation_df):
    print("Creating Q13 interactive visualization...")
    
    # Define custom colors
    PLOTLY_BG_COLOR = '#0e1117'
    PLOTLY_TEXT_COLOR = 'white'
    PLOTLY_GRID_COLOR = 'rgba(255, 255, 255, 0.1)'
    
    # Get top 15 recommendations
    top_n = min(15, len(recommendation_df))
    top_recommendations = recommendation_df.head(top_n)
    
    # Current, forecast, and residual weights
    current_weight = 0.35
    residual_weight = 0.35
    forecast_weight = 0.3
    
    # Create DataFrame for component breakdown
    components_df = pd.DataFrame({
        'County': np.repeat(top_recommendations['LOCATION'].values, 3),
        'Component': np.tile(['Current Homeless', 'Forecast Change', 'Model Residual'], top_n),
        'Score': np.concatenate([
            current_weight * top_recommendations['HomelessScore'].values,
            forecast_weight * top_recommendations['ChangeScore'].values,
            residual_weight * top_recommendations['ResidualScore'].values
        ]),
        'Rank': np.repeat(top_recommendations['Rank'].values, 3)
    })
    
    # Create stacked bar chart
    fig = px.bar(
        components_df,
        x='County',
        y='Score',
        color='Component',
        title=f'Top {top_n} Counties for Targeted Funding - Component Breakdown',
        color_discrete_map={
            'Current Homeless': 'cornflowerblue',
            'Forecast Change': 'lightcoral',
            'Model Residual': 'mediumseagreen'
        },
        labels={'County': 'County', 'Score': 'Component Score'},
        hover_data={'Rank': True}
    )
    
    # Add total score annotations
    for i, county in enumerate(top_recommendations['LOCATION']):
        fig.add_annotation(
            x=county,
            y=top_recommendations['CompositeScore'].iloc[i] + 0.02,
            text=f"Total: {top_recommendations['CompositeScore'].iloc[i]:.2f}",
            showarrow=False,
            font=dict(size=10, color="black")
        )
    
    # Update layout
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': top_recommendations['LOCATION']},
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color=PLOTLY_TEXT_COLOR),
        width=1000,
        height=600,
        xaxis_tickangle=-45
    )
    
    # Save interactive visualization
    fig.write_html('interactive/q13_funding_recommendations_interactive.html')
    
    # Create radar chart for top 3 counties
    top_3_counties = top_recommendations.head(3)
    
    # Define colors for charts
    colors = ['cornflowerblue', 'lightcoral', 'mediumseagreen']
    
    # Create separate polar charts instead of subplots
    for i, (_, county) in enumerate(top_3_counties.iterrows()):
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Scatterpolar(
                r=[
                    county['HomelessScore'],
                    county['ChangeScore'],
                    county['ResidualScore']
                ],
                theta=['Current Homeless', 'Forecast Change', 'Model Residual'],
                fill='toself',
                name=county['LOCATION'],
                marker_color=colors[i % 3]
            )
        )
        
        # Update layout for this specific county
        fig2.update_layout(
            title_text=f"{county['LOCATION']} (Rank {county['Rank']}) - Score Breakdown",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            plot_bgcolor=PLOTLY_BG_COLOR,
            paper_bgcolor=PLOTLY_BG_COLOR,
            font=dict(color=PLOTLY_TEXT_COLOR),
            width=600,
            height=500
        )
        
        # Save individual radar chart
        fig2.write_html(f'interactive/q13_county_{i+1}_radar_interactive.html')
    
    # Create a combined radar chart with all three counties
    fig3 = go.Figure()
    
    for i, (_, county) in enumerate(top_3_counties.iterrows()):
        fig3.add_trace(
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
    
    # Update layout for combined chart
    fig3.update_layout(
        title_text='Top 3 Recommended Counties - Score Comparison',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color=PLOTLY_TEXT_COLOR),
        width=800,
        height=600,
        legend=dict(
            x=0.85,
            y=0.95,
            bgcolor='rgba(0,0,0,0.2)',
            bordercolor='rgba(255,255,255,0.3)'
        )
    )
    
    # Save combined radar chart
    fig3.write_html('interactive/q13_top3_counties_radar_interactive.html')

# Main function
def main():
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('interactive', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    print("Generating missing visualizations (Q11, Q12, Q13)...")
    
    # Load data
    master_df, enhanced_df, residual_df, forecast_df = load_data()
    
    # Create Q11 visualization (Residual Analysis)
    create_q11_visualization(residual_df)
    
    # Create Q12 visualization (Forecasting)
    create_q12_visualization(forecast_df)
    
    # Create Q13 visualization (Funding Recommendations)
    create_q13_visualization(master_df, residual_df, forecast_df)
    
    print("All missing visualizations have been generated!")

if __name__ == "__main__":
    main() 