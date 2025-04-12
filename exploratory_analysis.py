import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter
from data_preparation import main as prep_data

# Set plot style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100

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
    plt.savefig('figures/statewide_age_distribution.png')
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
    plt.savefig('figures/statewide_race_distribution.png')
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
    plt.savefig('figures/statewide_gender_distribution.png')
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
    plt.savefig('figures/top10_counties_homeless_population.png')
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

def hospital_utilization_analysis(hospital_data):
    """
    Analyze hospital utilization patterns for homeless individuals
    """
    if hospital_data is None:
        print("Hospital data not available for analysis")
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
    plt.savefig('figures/hospital_encounters_by_demographic.png')
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
    plt.savefig('figures/homeless_proportion_by_demographic.png')
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
    
    # Extract statewide data for key metrics
    statewide_metrics = spm_data[spm_data['LOCATION_ID'] == 'All']
    statewide_metrics = statewide_metrics[statewide_metrics['Metric'].isin(key_metrics.keys())]
    
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
    plt.savefig('figures/statewide_metrics_over_time.png')
    plt.close()
    
    # 2. Compare top 5 counties for the most recent year
    latest_year = 'CY23'  # Based on data from 2020-2023
    
    # Get top 5 counties by total population served (M1a)
    top_counties_data = spm_data[spm_data['Metric'] == 'M1a']
    top_counties_data = top_counties_data[top_counties_data['LOCATION_ID'] != 'All']
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
    
    for i, (metric, description) in enumerate(key_metrics.items()):
        # Create subplot
        plt.subplot(3, 2, i + 1)
        
        metric_data = statewide_metrics[statewide_metrics['Metric'] == metric]
        if len(metric_data) == 0 or not all(col in metric_data.columns for col in ['CY20', 'CY23']):
            continue
            
        # Calculate percentage change
        start_value = metric_data['CY20'].iloc[0]
        end_value = metric_data['CY23'].iloc[0]
        
        if pd.isna(start_value) or pd.isna(end_value) or start_value == 0:
            continue
            
        pct_change = (end_value - start_value) / start_value * 100
        
        # Plot
        bar_color = 'lightcoral' if pct_change > 0 else 'mediumseagreen'
        plt.bar(['2020-2023'], [pct_change], color=bar_color)
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