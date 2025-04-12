import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

def load_demographic_data():
    """
    Load and merge demographic data (age, race, gender)
    """
    def load_and_rename(file_path, new_col_name):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        print(f"[DEBUG] Columns in {file_path}: {df.columns.tolist()}")

        # Try known names first
        if 'EXPERIENCING_HOMELESSNESS_CNT' in df.columns:
            df = df.rename(columns={'EXPERIENCING_HOMELESSNESS_CNT': new_col_name})
        elif 'CNT' in df.columns:
            df = df.rename(columns={'CNT': new_col_name})
        else:
            # Fallback
            match = [col for col in df.columns if 'HOMELESS' in col.upper() or 'CNT' in col.upper()]
            if match:
                df = df.rename(columns={match[0]: new_col_name})
            else:
                raise KeyError(f"{new_col_name} column not found in {file_path}")
        return df

    age_data = load_and_rename('cy_age.csv', 'COUNT_AGE')
    race_data = load_and_rename('cy_race.csv', 'COUNT_RACE')
    gender_data = load_and_rename('cy_gender.csv', 'COUNT_GENDER')

    # Fill LOCATION if missing
    if 'LOCATION' not in race_data.columns:
        race_data['LOCATION'] = race_data['LOCATION_ID']
    if 'LOCATION' not in gender_data.columns:
        gender_data['LOCATION'] = gender_data['LOCATION_ID']

    # Handle race data specifics
    if 'RACE_ETHNICITY' in race_data.columns and 'RACE_ETHNICITY_PUBLIC' not in race_data.columns:
        race_data = race_data.rename(columns={'RACE_ETHNICITY': 'RACE_ETHNICITY_PUBLIC'})

    # Handle gender data specifics
    if 'GENDER' in gender_data.columns and 'GENDER_PUBLIC' not in gender_data.columns:
        gender_data = gender_data.rename(columns={'GENDER': 'GENDER_PUBLIC'})

    # Convert values
    age_data['COUNT_AGE'] = pd.to_numeric(age_data['COUNT_AGE'].replace('*', np.nan), errors='coerce')
    race_data['COUNT_RACE'] = pd.to_numeric(race_data['COUNT_RACE'].replace('*', np.nan), errors='coerce')
    gender_data['COUNT_GENDER'] = pd.to_numeric(gender_data['COUNT_GENDER'].replace('*', np.nan), errors='coerce')

    age_data['LOCATION_ID'] = age_data['LOCATION_ID'].astype(str)
    race_data['LOCATION_ID'] = race_data['LOCATION_ID'].astype(str)
    gender_data['LOCATION_ID'] = gender_data['LOCATION_ID'].astype(str)

    return age_data, race_data, gender_data

def load_hospital_data():
    try:
        statewide_data = pd.read_csv('homeless-hospital-encounters-age-race-sex-expected-payer-statewide.csv')
        for col in ['Homeless', 'All']:
            statewide_data[col] = pd.to_numeric(statewide_data[col].str.replace(',', '').str.strip(), errors='coerce')
        statewide_data['HomelessProportion'] = statewide_data['Homeless'] / statewide_data['All']
        return statewide_data
    except Exception as e:
        print(f"Error loading hospital data: {e}")
        return None

def load_system_performance_data():
    calendar_spm = pd.read_csv('calendar-year-coc-and-statewide-topline-ca-spms.csv')
    try:
        cy_dict = pd.read_csv('cy-ca-spms-data-dictionary.csv')
    except:
        print("Data dictionary not found, proceeding without it")

    for col in ['CY20', 'CY21', 'CY22', 'CY23']:
        calendar_spm[col] = pd.to_numeric(calendar_spm[col], errors='coerce')

    calendar_spm['LOCATION_ID'] = calendar_spm['Location'].str.extract(r'(CA-\d+)')
    calendar_spm['LOCATION_ID'] = calendar_spm['LOCATION_ID'].fillna('All')

    return calendar_spm

def create_unified_identifier(age_data, race_data, gender_data, spm_data):
    locations_age = age_data[['LOCATION_ID', 'LOCATION']].drop_duplicates()
    locations_race = race_data[['LOCATION_ID', 'LOCATION']].drop_duplicates()
    locations_gender = gender_data[['LOCATION_ID', 'LOCATION']].drop_duplicates()
    locations_spm = spm_data[['LOCATION_ID', 'Location']].drop_duplicates().rename(columns={'Location': 'LOCATION'})

    all_locations = pd.concat([locations_age, locations_race, locations_gender, locations_spm]).drop_duplicates()
    location_mapping = all_locations.dropna(subset=['LOCATION_ID']).drop_duplicates()
    location_id_map = location_mapping.set_index('LOCATION')['LOCATION_ID'].to_dict()

    return location_mapping, location_id_map

def clean_and_standardize(age_data, race_data, gender_data, spm_data):
    imputer = SimpleImputer(strategy='median')
    age_data['COUNT_AGE'] = imputer.fit_transform(age_data[['COUNT_AGE']])
    race_data['COUNT_RACE'] = imputer.fit_transform(race_data[['COUNT_RACE']])
    gender_data['COUNT_GENDER'] = imputer.fit_transform(gender_data[['COUNT_GENDER']])

    for col in ['CY20', 'CY21', 'CY22', 'CY23']:
        if spm_data[col].dtype == 'object':
            spm_data[col] = spm_data[col].replace('N/A', np.nan)
        spm_data[col] = pd.to_numeric(spm_data[col], errors='coerce')
        if spm_data[col].isna().any():
            spm_data[col] = imputer.fit_transform(spm_data[[col]])

    return age_data, race_data, gender_data, spm_data

def calculate_derived_metrics(age_data, race_data, gender_data, spm_data):
    derived_metrics = {}
    latest_year = max(age_data['CALENDAR_YEAR'])

    # Create age pivot table
    age_latest = age_data[age_data['CALENDAR_YEAR'] == latest_year]
    age_pivot = age_latest.pivot_table(index='LOCATION_ID', columns='AGE_GROUP_PUBLIC', values='COUNT_AGE', aggfunc='sum').fillna(0)
    age_total = age_latest.groupby('LOCATION_ID')['COUNT_AGE'].sum().to_frame('TOTAL_HOMELESS')
    
    # Calculate proportions
    for col in age_pivot.columns:
        age_pivot[f'{col}_PROP'] = age_pivot[col] / age_pivot.sum(axis=1)

    # Create race pivot table
    race_latest = race_data[race_data['CALENDAR_YEAR'] == latest_year]
    race_col = 'RACE_ETHNICITY_PUBLIC' if 'RACE_ETHNICITY_PUBLIC' in race_latest.columns else 'RACE_ETHNICITY'
    race_pivot = race_latest.pivot_table(index='LOCATION_ID', columns=race_col, values='COUNT_RACE', aggfunc='sum').fillna(0)
    
    # Calculate proportions
    for col in race_pivot.columns:
        race_pivot[f'{col}_PROP'] = race_pivot[col] / race_pivot.sum(axis=1)

    # Create gender pivot table
    gender_latest = gender_data[gender_data['CALENDAR_YEAR'] == latest_year]
    gender_col = 'GENDER_PUBLIC' if 'GENDER_PUBLIC' in gender_latest.columns else 'GENDER'
    gender_pivot = gender_latest.pivot_table(index='LOCATION_ID', columns=gender_col, values='COUNT_GENDER', aggfunc='sum').fillna(0)
    
    # Calculate proportions
    for col in gender_pivot.columns:
        gender_pivot[f'{col}_PROP'] = gender_pivot[col] / gender_pivot.sum(axis=1)

    # Calculate system performance metrics trends
    # Create a dataframe with the unique LOCATION_ID values from spm_data
    unique_locations = spm_data['LOCATION_ID'].unique()
    spm_change = pd.DataFrame({'LOCATION_ID': unique_locations})
    spm_change = spm_change.set_index('LOCATION_ID')
    
    metrics = spm_data['Metric'].unique()
    
    for metric in metrics:
        # Get data for this metric, handle duplicates by taking the mean
        metric_data = (spm_data[spm_data['Metric'] == metric]
                      .groupby('LOCATION_ID')[['CY20', 'CY21', 'CY22', 'CY23']]
                      .mean())
        
        # Calculate year-over-year changes
        if 'CY20' in metric_data.columns and 'CY21' in metric_data.columns:
            # Calculate change safely, avoiding division by zero
            change_20_21 = (metric_data['CY21'] - metric_data['CY20']) / metric_data['CY20'].replace(0, np.nan)
            spm_change[f'{metric}_change_20_21'] = change_20_21
            
        if 'CY21' in metric_data.columns and 'CY22' in metric_data.columns:
            change_21_22 = (metric_data['CY22'] - metric_data['CY21']) / metric_data['CY21'].replace(0, np.nan)
            spm_change[f'{metric}_change_21_22'] = change_21_22
            
        if 'CY22' in metric_data.columns and 'CY23' in metric_data.columns:
            change_22_23 = (metric_data['CY23'] - metric_data['CY22']) / metric_data['CY22'].replace(0, np.nan)
            spm_change[f'{metric}_change_22_23'] = change_22_23
            
        # Store latest value
        if 'CY23' in metric_data.columns:
            spm_change[f'{metric}_latest'] = metric_data['CY23']
    
    # Reset index to make LOCATION_ID a column again
    spm_change = spm_change.reset_index().fillna(0)

    # Store all derived metrics
    derived_metrics['age_pivot'] = age_pivot
    derived_metrics['race_pivot'] = race_pivot
    derived_metrics['gender_pivot'] = gender_pivot
    derived_metrics['age_total'] = age_total
    derived_metrics['spm_change'] = spm_change

    return derived_metrics

def create_master_dataset(derived_metrics, location_mapping):
    master_df = derived_metrics['age_total'].reset_index()
    master_df = master_df.merge(location_mapping[['LOCATION_ID', 'LOCATION']], on='LOCATION_ID', how='left')
    
    # Add demographic proportions
    age_props = derived_metrics['age_pivot'].filter(regex='_PROP$').reset_index()
    master_df = master_df.merge(age_props, on='LOCATION_ID', how='left')
    
    race_props = derived_metrics['race_pivot'].filter(regex='_PROP$').reset_index()
    master_df = master_df.merge(race_props, on='LOCATION_ID', how='left', suffixes=('', '_race'))
    
    gender_props = derived_metrics['gender_pivot'].filter(regex='_PROP$').reset_index()
    master_df = master_df.merge(gender_props, on='LOCATION_ID', how='left', suffixes=('', '_gender'))
    
    # Add performance metrics
    spm_change = derived_metrics['spm_change']
    master_df = master_df.merge(spm_change, on='LOCATION_ID', how='left')
    
    # Exclude statewide data for county-level analysis
    master_df = master_df[master_df['LOCATION_ID'] != 'All']
    
    # Fill NaN values with 0 for analysis
    master_df = master_df.fillna(0)

    return master_df

def main():
    print("Loading demographic data...")
    age_data, race_data, gender_data = load_demographic_data()

    print("Loading system performance data...")
    spm_data = load_system_performance_data()

    print("Loading hospital data...")
    hospital_data = load_hospital_data()

    print("Creating unified identifier...")
    location_mapping, location_id_map = create_unified_identifier(age_data, race_data, gender_data, spm_data)

    print("Cleaning and standardizing data...")
    age_data, race_data, gender_data, spm_data = clean_and_standardize(age_data, race_data, gender_data, spm_data)

    print("Calculating derived metrics...")
    derived_metrics = calculate_derived_metrics(age_data, race_data, gender_data, spm_data)

    print("Creating master dataset...")
    master_df = create_master_dataset(derived_metrics, location_mapping)

    master_df.to_csv('master_dataset.csv', index=False)
    print("✅ Master dataset saved to 'master_dataset.csv'")

    return master_df, age_data, race_data, gender_data, spm_data, hospital_data, location_mapping

if __name__ == "__main__":
    main()
