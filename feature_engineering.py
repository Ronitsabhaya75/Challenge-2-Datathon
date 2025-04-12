import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_preparation import main as prep_data

# Try to import plotly for enhanced visualizations
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

def create_access_burden_indicators(master_df, age_data, spm_data):
    """
    Create indicators that reflect access burden or service capacity
    """
    print("Creating access burden indicators...")
    
    # Extract latest year
    latest_year = max(age_data['CALENDAR_YEAR'])
    
    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # 1. Create indicator: Ratio of homeless population to permanent housing placements
    # Get total homeless population by county
    homeless_by_county = master_df[['LOCATION_ID', 'LOCATION', 'TOTAL_HOMELESS']].copy()
    
    # Ensure LOCATION_ID exists in spm_data
    if spm_data is not None and 'LOCATION_ID' not in spm_data.columns:
        spm_data['LOCATION_ID'] = spm_data['Location'].apply(lambda x: f"CA-{x}" if x.isdigit() else x)
    
    # Get permanent housing placements (M3) for latest year
    if spm_data is None:
        print("Warning: System performance data not available for access burden indicators")
        return master_df
        
    m3_data = spm_data[spm_data['Metric'] == 'M3'] 
    m3_data = m3_data[m3_data['LOCATION_ID'].isin(master_df['LOCATION_ID'])]
    
    # Extract latest year column
    latest_col = f'CY{str(latest_year)[2:]}'
    
    if latest_col in m3_data.columns:
        m3_latest = m3_data[['LOCATION_ID', latest_col]].rename(columns={latest_col: 'PERMANENT_HOUSING'})
        
        # Merge with homeless population
        burden_df = homeless_by_county.merge(m3_latest, on='LOCATION_ID', how='left')
        
        # Calculate ratio (higher ratio means higher burden)
        burden_df['HOUSING_ACCESS_BURDEN'] = burden_df['TOTAL_HOMELESS'] / burden_df['PERMANENT_HOUSING']
        burden_df['HOUSING_ACCESS_BURDEN'] = burden_df['HOUSING_ACCESS_BURDEN'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Add to master dataset
        master_df = master_df.merge(
            burden_df[['LOCATION_ID', 'HOUSING_ACCESS_BURDEN', 'PERMANENT_HOUSING']], 
            on='LOCATION_ID', 
            how='left'
        )
        
        # Visualize top and bottom 5 counties by housing access burden
        burden_viz = burden_df.sort_values('HOUSING_ACCESS_BURDEN', ascending=False)
        
        # Filter out extreme outliers for better visualization
        q3 = burden_viz['HOUSING_ACCESS_BURDEN'].quantile(0.75)
        iqr = q3 - burden_viz['HOUSING_ACCESS_BURDEN'].quantile(0.25)
        upper_limit = q3 + 1.5 * iqr
        
        burden_viz = burden_viz[burden_viz['HOUSING_ACCESS_BURDEN'] <= upper_limit]
        
        top5 = burden_viz.head(5)
        bottom5 = burden_viz.tail(5)
        
        plt.figure(figsize=(12, 8))
        
        # Plot top 5 counties with highest burden
        ax1 = plt.subplot(121)
        sns.barplot(data=top5, x='LOCATION', y='HOUSING_ACCESS_BURDEN', palette='Reds_r', ax=ax1)
        ax1.set_title('5 Counties with Highest\nHousing Access Burden', fontsize=14)
        ax1.set_xlabel('')
        ax1.set_ylabel('Housing Access Burden Ratio', fontsize=12)
        ax1.tick_params(axis='x', labelrotation=45, labelsize=10)
        
        # Plot bottom 5 counties with lowest burden
        ax2 = plt.subplot(122)
        sns.barplot(data=bottom5, x='LOCATION', y='HOUSING_ACCESS_BURDEN', palette='Blues_r', ax=ax2)
        ax2.set_title('5 Counties with Lowest\nHousing Access Burden', fontsize=14)
        ax2.set_xlabel('')
        ax2.set_ylabel('', fontsize=12)
        ax2.tick_params(axis='x', labelrotation=45, labelsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/q7_housing_access_burden.png')
        plt.close()
    
    # 2. Create indicator: Emergency shelter utilization rate
    # Get emergency shelter usage (M2) for latest year
    m2_data = spm_data[spm_data['Metric'] == 'M2']
    m2_data = m2_data[m2_data['LOCATION_ID'].isin(master_df['LOCATION_ID'])]
    
    if latest_col in m2_data.columns:
        m2_latest = m2_data[['LOCATION_ID', latest_col]].rename(columns={latest_col: 'EMERGENCY_SHELTER'})
        
        # Merge with homeless population
        shelter_df = homeless_by_county.merge(m2_latest, on='LOCATION_ID', how='left')
        
        # Calculate ratio (this is utilization rate - higher means more sheltered)
        shelter_df['SHELTER_UTILIZATION'] = shelter_df['EMERGENCY_SHELTER'] / shelter_df['TOTAL_HOMELESS']
        shelter_df['SHELTER_UTILIZATION'] = shelter_df['SHELTER_UTILIZATION'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Add to master dataset
        master_df = master_df.merge(
            shelter_df[['LOCATION_ID', 'SHELTER_UTILIZATION', 'EMERGENCY_SHELTER']], 
            on='LOCATION_ID', 
            how='left'
        )
        
        # Visualize shelter utilization rates
        shelter_viz = shelter_df.sort_values('SHELTER_UTILIZATION', ascending=False)
        
        # Filter out extreme values
        shelter_viz = shelter_viz[shelter_viz['SHELTER_UTILIZATION'] <= 1.0]  # Cap at 100%
        
        top10 = shelter_viz.head(10)
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=top10, x='LOCATION', y='SHELTER_UTILIZATION', hue='LOCATION', legend=False)
        plt.title('Top 10 Counties by Emergency Shelter Utilization Rate', fontsize=16)
        plt.xlabel('County', fontsize=14)
        plt.ylabel('Shelter Utilization Rate', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Add percentage labels
        for i, v in enumerate(top10['SHELTER_UTILIZATION']):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/q7_shelter_utilization_rate.png')
        plt.close()
    
    # 3. Create indicator: Vulnerability Index based on demographics
    # Calculate vulnerability based on age groups (under 18 and over 65 are more vulnerable)
    if 'Under 18_PROP' in master_df.columns and '65+_PROP' in master_df.columns:
        master_df['VULNERABLE_AGE_PROP'] = master_df['Under 18_PROP'] + master_df['65+_PROP']
        
        # Create visualization of vulnerable age proportion
        vuln_df = master_df[['LOCATION_ID', 'LOCATION', 'VULNERABLE_AGE_PROP']].sort_values('VULNERABLE_AGE_PROP', ascending=False).head(10)
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=vuln_df, x='LOCATION', y='VULNERABLE_AGE_PROP', hue='LOCATION', legend=False)
        plt.title('Top 10 Counties by Proportion of Vulnerable Age Groups', fontsize=16)
        plt.xlabel('County', fontsize=14)
        plt.ylabel('Proportion of Vulnerable Age Groups', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Add percentage labels
        for i, v in enumerate(vuln_df['VULNERABLE_AGE_PROP']):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/q7_vulnerable_age_proportion.png')
        plt.close()
    
    # 4. Create a composite vulnerability score
    # Combine multiple factors: vulnerable age, shelter utilization (inverse), housing access burden
    if all(col in master_df.columns for col in ['VULNERABLE_AGE_PROP', 'SHELTER_UTILIZATION', 'HOUSING_ACCESS_BURDEN']):
        # Normalize all components to 0-1 scale
        scaler = StandardScaler()
        
        # Higher shelter utilization is good, so we invert it
        master_df['SHELTER_GAP'] = 1 - master_df['SHELTER_UTILIZATION']
        
        # Prepare data for scaling
        vulnerability_components = master_df[['VULNERABLE_AGE_PROP', 'SHELTER_GAP', 'HOUSING_ACCESS_BURDEN']].copy()
        
        # Replace inf and -inf with NaN and fill NaN with 0
        vulnerability_components = vulnerability_components.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale the components
        scaled_components = scaler.fit_transform(vulnerability_components)
        
        # Create a composite score (equal weights for simplicity)
        master_df['VULNERABILITY_SCORE'] = np.mean(scaled_components, axis=1)
        
        # Visualize vulnerability score for top counties
        vuln_score_df = master_df[['LOCATION_ID', 'LOCATION', 'VULNERABILITY_SCORE']].sort_values('VULNERABILITY_SCORE', ascending=False).head(10)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=vuln_score_df, x='LOCATION', y='VULNERABILITY_SCORE', hue='LOCATION', legend=False)
        plt.title('Top 10 Counties by Composite Vulnerability Score', fontsize=16)
        plt.xlabel('County', fontsize=14)
        plt.ylabel('Vulnerability Score', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('figures/q7_vulnerability_score.png')
        plt.close()
    
    return master_df

def create_trend_features(master_df, spm_data):
    """
    Generate trend-based features using year-over-year comparisons
    """
    print("Creating trend-based features...")
    
    # Define key metrics to calculate trends for
    key_metrics = ['M1a', 'M2', 'M3', 'M4', 'M5', 'M6']
    
    # Calculate trend strength for each metric
    for metric in key_metrics:
        # Get data for this metric
        metric_data = spm_data[spm_data['Metric'] == metric]
        
        # Only process if we have data for this metric
        if len(metric_data) == 0:
            continue
        
        # Extract year columns
        year_cols = [col for col in metric_data.columns if col.startswith('CY')]
        
        if len(year_cols) < 2:
            continue
            
        # Sort year columns
        year_cols.sort()
        
        # Calculate linear regression slope for each county
        slopes = {}
        
        for _, row in metric_data.iterrows():
            # Get location_id with fallback
            if 'LOCATION_ID' in row:
                location_id = row['LOCATION_ID']
            elif 'Location' in row:
                # Convert to location_id format
                location = row['Location']
                location_id = f"CA-{location}" if location.isdigit() else location
            else:
                # Skip if we can't identify the location
                continue
            
            # Get values for each year
            years = []
            values = []
            
            for col in year_cols:
                if pd.notna(row[col]):
                    years.append(int(col[2:]))
                    values.append(row[col])
            
            if len(years) < 2:
                slopes[location_id] = 0
                continue
            
            # Calculate linear regression slope
            from scipy import stats
            slope, _, _, _, _ = stats.linregress(years, values)
            
            # Normalize by dividing by the mean value to get percent change per year
            mean_value = np.mean(values)
            if mean_value != 0:
                normalized_slope = slope / mean_value
            else:
                normalized_slope = 0
                
            slopes[location_id] = normalized_slope
        
        # Convert to DataFrame
        slope_df = pd.DataFrame({
            'LOCATION_ID': list(slopes.keys()),
            f'{metric}_TREND': list(slopes.values())
        })
        
        # Merge with master dataset
        master_df = master_df.merge(slope_df, on='LOCATION_ID', how='left')
        
        # Skip visualization if Location column is missing
        if 'Location' not in spm_data.columns:
            continue
            
        # Visualize top positive and negative trends
        trend_viz = slope_df.copy()
        
        # Try to merge with location names
        locations_df = spm_data[['LOCATION_ID', 'Location']].drop_duplicates() if 'LOCATION_ID' in spm_data.columns else None
        
        if locations_df is not None and not locations_df.empty:
            trend_viz = trend_viz.merge(locations_df, on='LOCATION_ID', how='left')
        else:
            # Create a dummy Location column based on LOCATION_ID
            trend_viz['Location'] = trend_viz['LOCATION_ID'].apply(lambda x: x.replace('CA-', '') if 'CA-' in x else x)
        
        # Remove extreme values for better visualization
        q1 = trend_viz[f'{metric}_TREND'].quantile(0.25)
        q3 = trend_viz[f'{metric}_TREND'].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        
        trend_viz = trend_viz[(trend_viz[f'{metric}_TREND'] >= lower_limit) & 
                            (trend_viz[f'{metric}_TREND'] <= upper_limit)]
        
        # Get top 5 positive and negative trends
        top_positive = trend_viz.sort_values(f'{metric}_TREND', ascending=False).head(5)
        top_negative = trend_viz.sort_values(f'{metric}_TREND', ascending=True).head(5)
        
        plt.figure(figsize=(14, 10))
        
        # Plot top positive trends
        ax1 = plt.subplot(121)
        bars1 = sns.barplot(data=top_positive, x='Location', y=f'{metric}_TREND', hue='Location', legend=False, ax=ax1)
        ax1.set_title(f'Top 5 Counties with Positive\n{metric} Trend', fontsize=14)
        ax1.set_xlabel('')
        ax1.set_ylabel('Normalized Trend Slope', fontsize=12)
        ax1.tick_params(axis='x', labelrotation=45, labelsize=10)
        
        # Plot top negative trends
        ax2 = plt.subplot(122)
        bars2 = sns.barplot(data=top_negative, x='Location', y=f'{metric}_TREND', hue='Location', legend=False, ax=ax2)
        ax2.set_title(f'Top 5 Counties with Negative\n{metric} Trend', fontsize=14)
        ax2.set_xlabel('')
        ax2.set_ylabel('', fontsize=12)
        ax2.tick_params(axis='x', labelrotation=45, labelsize=10)
        
        plt.tight_layout()
        plt.savefig(f'figures/q8_{metric}_trend.png')
        plt.close()
    
    # Create a composite trend indicator
    trend_cols = [col for col in master_df.columns if col.endswith('_TREND')]
    
    if len(trend_cols) > 0:
        # For some metrics like M4 and M5, a negative trend is good
        # M4 = length of time homeless (lower is better)
        # M5 = returns to homelessness (lower is better)
        
        # Create a new column with the sign inverted for these metrics
        if 'M4_TREND' in master_df.columns:
            master_df['M4_TREND_INVERTED'] = -master_df['M4_TREND']
            trend_cols.remove('M4_TREND')
            trend_cols.append('M4_TREND_INVERTED')
        
        if 'M5_TREND' in master_df.columns:
            master_df['M5_TREND_INVERTED'] = -master_df['M5_TREND']
            trend_cols.remove('M5_TREND')
            trend_cols.append('M5_TREND_INVERTED')
        
        # Replace inf values and NaN with 0
        for col in trend_cols:
            master_df[col] = master_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale the trend features
        trend_features = master_df[trend_cols].copy()
        scaler = StandardScaler()
        scaled_trends = scaler.fit_transform(trend_features)
        
        # Calculate composite trend (positive means improvement)
        master_df['COMPOSITE_TREND'] = np.mean(scaled_trends, axis=1)
        
        # Visualize composite trend
        trend_composite = master_df[['LOCATION_ID', 'LOCATION', 'COMPOSITE_TREND']].sort_values('COMPOSITE_TREND', ascending=False)
        
        # Top 10 counties with best trends
        top10_trends = trend_composite.head(10)
        
        # Ensure non-zero data for visualization
        if top10_trends['COMPOSITE_TREND'].max() - top10_trends['COMPOSITE_TREND'].min() < 0.01:
            # If the range is too small, rescale for better visualization
            top10_trends['COMPOSITE_TREND'] = top10_trends['COMPOSITE_TREND'] * 100
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(top10_trends['LOCATION'], top10_trends['COMPOSITE_TREND'], color='cornflowerblue')
        plt.title('Top 10 Counties with Most Positive Composite Trend', fontsize=16)
        plt.xlabel('County', fontsize=14)
        plt.ylabel('Composite Trend Score', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels to the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('figures/q8_composite_trend.png')
        plt.close()
        
        # Create interactive version if plotly is available
        if HAS_PLOTLY:
            fig = px.bar(
                top10_trends,
                x='LOCATION',
                y='COMPOSITE_TREND',
                title='Top 10 Counties with Most Positive Composite Trend',
                labels={'LOCATION': 'County', 'COMPOSITE_TREND': 'Composite Trend Score'},
                text=[f"{score:.2f}" for score in top10_trends['COMPOSITE_TREND']],
                color='COMPOSITE_TREND',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(240,240,240,0.8)',
                width=1000,
                height=600
            )
            
            # Save interactive visualization
            os.makedirs('interactive', exist_ok=True)
            fig.write_html('interactive/q8_composite_trend_interactive.html')
    
    return master_df

def cluster_counties(master_df):
    """
    Group counties based on shared characteristics using unsupervised learning
    """
    print("Clustering counties based on shared characteristics...")
    
    # Select features for clustering
    # We'll use demographic proportions, vulnerability indicators, and trend features
    
    # Demographic features
    demo_cols = [col for col in master_df.columns if col.endswith('_PROP')]
    
    # Vulnerability indicators
    vulnerability_cols = ['HOUSING_ACCESS_BURDEN', 'SHELTER_UTILIZATION', 'VULNERABILITY_SCORE']
    vulnerability_cols = [col for col in vulnerability_cols if col in master_df.columns]
    
    # Trend features
    trend_cols = [col for col in master_df.columns if col.endswith('_TREND')]
    trend_cols = [col for col in trend_cols if not col.endswith('_INVERTED')]  # Skip inverted columns
    
    # Combine all features
    feature_cols = demo_cols + vulnerability_cols + trend_cols
    
    # Make sure all features exist in the dataset
    feature_cols = [col for col in feature_cols if col in master_df.columns]
    
    if len(feature_cols) == 0:
        print("No features available for clustering")
        return master_df
    
    # Handle missing values
    cluster_data = master_df[feature_cols].copy()
    cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Apply PCA for dimensionality reduction for visualization
    pca = PCA(n_components=3)  # Using 3 components for 3D visualization
    pca_result = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'PCA3': pca_result[:, 2]
    })
    
    # Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    K = range(2, 8)  # Try 2 to 7 clusters
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        
        # Calculate silhouette score
        score = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Find the best number of clusters
    best_k = K[np.argmax(silhouette_scores)]
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'o-', color='cornflowerblue')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.title('Silhouette Score Method to Determine Optimal k', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/q9_silhouette_scores.png')
    plt.close()
    
    # Apply KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to the master DataFrame - THIS IS THE KEY FIX
    master_df['CLUSTER'] = cluster_labels
    
    # Add cluster labels to the PCA DataFrame
    pca_df['Cluster'] = cluster_labels
    
    # Add county names for visualization
    pca_df['County'] = master_df['LOCATION'].values
    
    # Add total homeless population for sizing the markers
    pca_df['Total Homeless'] = master_df['TOTAL_HOMELESS'].values
    
    # Visualize the clusters
    plt.figure(figsize=(14, 10))
    
    # Create a color palette for clusters
    colors = sns.color_palette("viridis", best_k)
    
    # Plot each cluster with different colors
    for i in range(best_k):
        cluster_points = pca_df[pca_df['Cluster'] == i]
        
        # Scale point sizes by homeless population (with min and max sizes)
        sizes = 100 * (cluster_points['Total Homeless'] / master_df['TOTAL_HOMELESS'].max())
        sizes = sizes.clip(30, 500)  # Min and max sizes
        
        plt.scatter(
            cluster_points['PCA1'], 
            cluster_points['PCA2'], 
            s=sizes,
            c=[colors[i]],
            alpha=0.7,
            label=f'Cluster {i+1}'
        )
        
        # Add county labels for the top counties by homeless population
        top_counties = cluster_points.nlargest(3, 'Total Homeless')
        for _, row in top_counties.iterrows():
            plt.annotate(
                row['County'],
                (row['PCA1'], row['PCA2']),
                fontsize=10,
                ha='center',
                va='bottom'
            )
    
    plt.title('County Clusters Based on Homelessness Indicators', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.legend(title='Clusters')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/q9_county_clusters.png')
    plt.close()
    
    # Create 3D visualization using Plotly if available
    if HAS_PLOTLY:
        # Create a 3D scatter plot with improved aesthetics
        fig = px.scatter_3d(
            pca_df, 
            x='PCA1', 
            y='PCA2', 
            z='PCA3',
            color='Cluster',
            size='Total Homeless',
            hover_name='County',
            hover_data={
                'Total Homeless': True,
                'PCA1': False,
                'PCA2': False,
                'PCA3': False,
                'Cluster': False
            },
            opacity=0.85,
            color_continuous_scale='plasma',
            size_max=60,  # Larger maximum marker size
            title='3D Visualization of California County Homelessness Clusters',
            labels={'Cluster': 'Cluster Group'}
        )
        
        # Customize layout for more appeal
        fig.update_layout(
            scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Principal Component 3',
                xaxis=dict(showbackground=False, showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showbackground=False, showgrid=True, gridcolor='lightgray'),
                zaxis=dict(showbackground=False, showgrid=True, gridcolor='lightgray'),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.2)  # Adjust camera position for better view
                ),
                aspectmode='cube'  # Ensure equal scaling
            ),
            width=1200,
            height=900,
            template='plotly_dark',  # Use dark theme for better contrast
            margin=dict(l=0, r=0, t=50, b=10),
            title=dict(
                text='3D Visualization of California County Homelessness Clusters',
                font=dict(size=24, color='white')
            ),
            legend=dict(
                title=dict(text='Cluster Groups', font=dict(size=16)),
                font=dict(size=14)
            )
        )
        
        # Add text labels for major counties instead of annotations
        # Get top 5 counties by homeless population
        top_counties = pca_df.nlargest(5, 'Total Homeless')
        
        # Add text markers for these counties
        for i, row in top_counties.iterrows():
            fig.add_trace(
                go.Scatter3d(
                    x=[row['PCA1']],
                    y=[row['PCA2']],
                    z=[row['PCA3']],
                    mode='text',
                    text=[row['County']],
                    textposition='top center',
                    textfont=dict(size=14, color='white'),
                    showlegend=False
                )
            )
        
        # Add custom buttons for different views
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='Reset View',
                            method='relayout',
                            args=['scene.camera', dict(
                                up=dict(x=0, y=0, z=1),
                                center=dict(x=0, y=0, z=0),
                                eye=dict(x=1.5, y=1.5, z=1.2)
                            )]
                        ),
                        dict(
                            label='Top View',
                            method='relayout',
                            args=['scene.camera', dict(
                                up=dict(x=0, y=1, z=0),
                                center=dict(x=0, y=0, z=0),
                                eye=dict(x=0, y=0, z=2.5)
                            )]
                        ),
                        dict(
                            label='Side View',
                            method='relayout',
                            args=['scene.camera', dict(
                                up=dict(x=0, y=0, z=1),
                                center=dict(x=0, y=0, z=0),
                                eye=dict(x=2.5, y=0, z=0)
                            )]
                        )
                    ],
                    direction='down',
                    pad={'r': 10, 't': 10},
                    x=0.9,
                    y=0.1,
                    xanchor='right',
                    yanchor='bottom'
                )
            ]
        )
        
        # Save as HTML for interactive viewing
        os.makedirs('interactive', exist_ok=True)
        fig.write_html('interactive/q9_county_clusters_3d.html')
    
    # Analyze cluster characteristics
    cluster_profiles = master_df.groupby('CLUSTER')[feature_cols].mean()
    
    # Determine distinguishing features for each cluster
    top_features = {}
    
    for cluster in range(best_k):
        # Calculate how much each feature deviates from the overall mean
        cluster_profile = cluster_profiles.loc[cluster]
        overall_mean = master_df[feature_cols].mean()
        
        # Calculate z-scores
        z_scores = (cluster_profile - overall_mean) / master_df[feature_cols].std()
        
        # Get top 3 most distinguishing features (both positive and negative)
        distinguishing_features = z_scores.abs().nlargest(3)
        
        top_features[cluster] = {
            'features': distinguishing_features.index.tolist(),
            'z_scores': distinguishing_features.values.tolist(),
            'directions': ['+' if z_scores[feat] > 0 else '-' for feat in distinguishing_features.index]
        }
    
    # Visualize cluster characteristics
    plt.figure(figsize=(15, 10))
    
    # Create a heatmap of feature values by cluster
    sns.heatmap(
        cluster_profiles.apply(lambda x: (x - x.mean()) / x.std(), axis=0),  # Standardize for better visualization
        cmap='RdBu_r',
        center=0,
        annot=False
    )
    
    plt.title('Cluster Characteristics (Standardized Feature Values)', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Cluster', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/q9_cluster_characteristics.png')
    plt.close()
    
    # Create interactive cluster characteristics visualization with Plotly if available
    if HAS_PLOTLY:
        # Prepare data for visualization
        std_cluster_profiles = cluster_profiles.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        
        # Create a heatmap
        fig = px.imshow(
            std_cluster_profiles,
            labels=dict(x="Features", y="Cluster", color="Z-Score"),
            x=std_cluster_profiles.columns,
            y=[f"Cluster {i+1}" for i in range(best_k)],
            color_continuous_scale="RdBu_r",
            title="Interactive Cluster Characteristics Heatmap"
        )
        
        # Customize layout
        fig.update_layout(
            width=1200,
            height=600,
            xaxis=dict(tickangle=45)
        )
        
        # Save as HTML for interactive viewing
        fig.write_html('interactive/q9_cluster_characteristics_interactive.html')
    
    # Create a more detailed profile of each cluster
    cluster_details = {}
    
    for cluster in range(best_k):
        cluster_counties = master_df[master_df['CLUSTER'] == cluster]
        
        cluster_details[cluster] = {
            'size': len(cluster_counties),
            'total_homeless': cluster_counties['TOTAL_HOMELESS'].sum(),
            'avg_homeless': cluster_counties['TOTAL_HOMELESS'].mean(),
            'top_counties': cluster_counties.nlargest(3, 'TOTAL_HOMELESS')['LOCATION'].tolist(),
            'distinguishing_features': [
                f"{feat} ({top_features[cluster]['directions'][i]})" 
                for i, feat in enumerate(top_features[cluster]['features'])
            ]
        }
    
    # Print cluster details for reference
    print("\nCluster Profiles:")
    for cluster, details in cluster_details.items():
        print(f"\nCluster {cluster+1} ({details['size']} counties):")
        print(f"  Total homeless population: {details['total_homeless']:,.0f}")
        print(f"  Average homeless per county: {details['avg_homeless']:,.0f}")
        print(f"  Representative counties: {', '.join(details['top_counties'])}")
        print(f"  Distinguishing features: {', '.join(details['distinguishing_features'])}")
    
    return master_df

def main():
    """
    Main function to run feature engineering pipeline
    """
    # Get data from data preparation module
    master_df, age_data, race_data, gender_data, spm_data, hospital_data, location_mapping = prep_data()
    
    # Add LOCATION_ID column to spm_data if it doesn't exist
    if spm_data is not None and 'LOCATION_ID' not in spm_data.columns:
        print("Adding LOCATION_ID column to system performance data")
        spm_data['LOCATION_ID'] = spm_data['Location'].apply(lambda x: f"CA-{x}" if x.isdigit() else x)
    
    # Create access burden indicators
    print("\nFeature Engineering Process:")
    master_df = create_access_burden_indicators(master_df, age_data, spm_data)
    
    # Create trend-based features
    master_df = create_trend_features(master_df, spm_data)
    
    # Cluster counties
    master_df = cluster_counties(master_df)
    
    # Save enhanced dataset
    master_df.to_csv('enhanced_dataset.csv', index=False)
    print("\nEnhanced dataset saved to 'enhanced_dataset.csv'")
    
    return master_df

if __name__ == "__main__":
    main() 