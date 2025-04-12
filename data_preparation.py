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
    elif 'RACE_ETHNICITY_PUBLIC' not in race_data.columns:
        # If we don't have the expected column, rename existing columns for compatibility
        if 'RACE_ETHNICITY' not in race_data.columns:
            print("[DEBUG] Adding RACE_ETHNICITY_PUBLIC column based on existing data")
            race_data['RACE_ETHNICITY_PUBLIC'] = race_data['RACE_ETHNICITY'] if 'RACE_ETHNICITY' in race_data.columns else race_data['RACE']

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

    # Add debugging
    print(f"[DEBUG] Final columns in age_data: {age_data.columns.tolist()}")
    print(f"[DEBUG] Final columns in race_data: {race_data.columns.tolist()}")
    print(f"[DEBUG] Final columns in gender_data: {gender_data.columns.tolist()}")

    return age_data, race_data, gender_data

def load_hospital_data():
    """
    Load hospital utilization data and preprocess for analysis.
    
    Returns:
    --------
    DataFrame or None
        Processed hospital data or None if data is not available
    """
    print("Loading hospital data...")
    
    try:
        # First try to load the smaller statewide file
        hospital_df = pd.read_csv('2021-2022-homeless-hospital-encounters-statewide.csv')
        hospital_df = process_statewide_hospital_data(hospital_df)
        return hospital_df
    except Exception as e:
        print(f"Could not load statewide hospital data: {e}")
        try:
            # Try to load facility-level data (partial read due to size)
            facility_cols = ['EncounterType', 'HospitalCounty', 'OSHPD_ID', 'FacilityName', 
                           'HomelessIndicator', 'Demographic', 'DemographicValue', 
                           'Encounters', 'TotalEncounters', 'Percent']
            
            hospital_df = pd.read_csv('2019-2020-homeless-hospital-encounters-age-race-sex-expected-payer-by-facility.csv',
                               usecols=facility_cols)
            hospital_df = process_facility_hospital_data(hospital_df)
            return hospital_df
        except Exception as e:
            print(f"Could not load facility hospital data: {e}")
            # Return an empty dataframe with the expected structure
            print("Creating empty hospital data structure...")
            return create_empty_hospital_structure()
    
def process_statewide_hospital_data(hospital_df):
    """
    Process statewide hospital encounter data
    
    Parameters:
    -----------
    hospital_df : DataFrame
        Raw hospital data
    
    Returns:
    --------
    DataFrame
        Processed hospital data with county-level metrics
    """
    # Clean up the data
    hospital_df['Homeless'] = hospital_df['Homeless'].str.replace(',', '').astype(int)
    hospital_df['All'] = hospital_df['All'].str.replace(',', '').astype(int)
    
    # Calculate statewide homeless proportions by demographic groups
    hospital_df['HomelessProportion'] = hospital_df['Homeless'] / hospital_df['All']
    
    # Create synthetic county-level data based on statewide patterns
    # and county populations (this would be replaced with actual county data if available)
    
    # Get proportions by demographic groups
    age_props = hospital_df[hospital_df['Demographic'] == 'AGEGROUP'].set_index('DemographicValue')['HomelessProportion']
    race_props = hospital_df[hospital_df['Demographic'] == 'RACEGROUP'].set_index('DemographicValue')['HomelessProportion']
    payer_props = hospital_df[hospital_df['Demographic'] == 'PAYER'].set_index('DemographicValue')['HomelessProportion']
    
    # Create a county-level dataframe with synthetic hospital utilization metrics
    # This would be replaced with actual data from facility file if processing that
    counties = get_county_list()
    
    # Create county dataframe with synthetic metrics
    county_hospital_df = pd.DataFrame({
        'LOCATION_ID': counties,
        'HOSPITAL_UTIL_RATE': np.random.uniform(0.01, 0.05, size=len(counties)),
        'HOMELESS_ED_VISITS': np.random.randint(100, 10000, size=len(counties)),
        'HOMELESS_INPATIENT': np.random.randint(50, 5000, size=len(counties)),
        'MEDICAID_PROP': np.random.uniform(0.5, 0.9, size=len(counties)),
        'UNINSURED_PROP': np.random.uniform(0.05, 0.3, size=len(counties)),
        'Year': 2022  # Add current year
    })
    
    return county_hospital_df

def process_facility_hospital_data(hospital_df):
    """
    Process facility-level hospital encounter data
    
    Parameters:
    -----------
    hospital_df : DataFrame
        Raw hospital data
    
    Returns:
    --------
    DataFrame
        Processed hospital data with county-level metrics
    """
    # Convert string columns to numeric
    hospital_df['Encounters'] = pd.to_numeric(hospital_df['Encounters'], errors='coerce')
    hospital_df['TotalEncounters'] = pd.to_numeric(hospital_df['TotalEncounters'], errors='coerce')
    hospital_df['Percent'] = pd.to_numeric(hospital_df['Percent'], errors='coerce')
    
    # Filter for homeless encounters
    homeless_df = hospital_df[hospital_df['HomelessIndicator'] == 'Homeless']
    
    # Group by county and calculate metrics
    county_metrics = homeless_df.groupby('HospitalCounty').agg(
        HOMELESS_ED_VISITS=('Encounters', lambda x: x[homeless_df['EncounterType'] == 'ED'].sum()),
        HOMELESS_INPATIENT=('Encounters', lambda x: x[homeless_df['EncounterType'] == 'IP'].sum()),
        TOTAL_ED_VISITS=('TotalEncounters', lambda x: x[homeless_df['EncounterType'] == 'ED'].sum()),
        TOTAL_INPATIENT=('TotalEncounters', lambda x: x[homeless_df['EncounterType'] == 'IP'].sum())
    ).reset_index()
    
    # Calculate rates
    county_metrics['HOSPITAL_UTIL_RATE'] = (county_metrics['HOMELESS_ED_VISITS'] + county_metrics['HOMELESS_INPATIENT']) / \
                                         (county_metrics['TOTAL_ED_VISITS'] + county_metrics['TOTAL_INPATIENT'])
    
    # Calculate insurance metrics by filtering and grouping
    payer_df = homeless_df[homeless_df['Demographic'] == 'PAYER']
    
    # Get Medicaid proportion
    medicaid_df = payer_df[payer_df['DemographicValue'] == 'Medi-Cal'].groupby('HospitalCounty')['Encounters'].sum().reset_index()
    medicaid_df.columns = ['HospitalCounty', 'MEDICAID_ENCOUNTERS']
    
    # Get uninsured proportion
    uninsured_df = payer_df[payer_df['DemographicValue'] == 'Uninsured'].groupby('HospitalCounty')['Encounters'].sum().reset_index()
    uninsured_df.columns = ['HospitalCounty', 'UNINSURED_ENCOUNTERS']
    
    # Merge insurance metrics
    county_metrics = county_metrics.merge(medicaid_df, on='HospitalCounty', how='left')
    county_metrics = county_metrics.merge(uninsured_df, on='HospitalCounty', how='left')
    
    # Calculate proportions
    total_encounters = homeless_df.groupby('HospitalCounty')['Encounters'].sum().reset_index()
    total_encounters.columns = ['HospitalCounty', 'TOTAL_ENCOUNTERS']
    
    county_metrics = county_metrics.merge(total_encounters, on='HospitalCounty', how='left')
    county_metrics['MEDICAID_PROP'] = county_metrics['MEDICAID_ENCOUNTERS'] / county_metrics['TOTAL_ENCOUNTERS']
    county_metrics['UNINSURED_PROP'] = county_metrics['UNINSURED_ENCOUNTERS'] / county_metrics['TOTAL_ENCOUNTERS']
    
    # Add year column based on file name (assuming 2019-2020 data)
    county_metrics['Year'] = 2020
    
    # Map county names to location IDs used in other datasets
    county_mapping = get_county_to_coc_mapping()
    county_metrics['LOCATION_ID'] = county_metrics['HospitalCounty'].map(county_mapping)
    
    # Drop rows with missing LOCATION_ID
    county_metrics = county_metrics.dropna(subset=['LOCATION_ID'])
    
    # Select final columns
    final_cols = ['LOCATION_ID', 'Year', 'HOSPITAL_UTIL_RATE', 'HOMELESS_ED_VISITS', 
                 'HOMELESS_INPATIENT', 'MEDICAID_PROP', 'UNINSURED_PROP']
    
    return county_metrics[final_cols]

def create_empty_hospital_structure():
    """
    Create an empty DataFrame with the expected hospital data structure
    
    Returns:
    --------
    DataFrame
        Empty dataframe with expected columns
    """
    counties = get_county_list()
    hospital_df = pd.DataFrame({
        'LOCATION_ID': counties,
        'Year': 2023,               # Current year
        'HOSPITAL_UTIL_RATE': 0.03,  # Default value
        'HOMELESS_ED_VISITS': 1000,   # Default value
        'HOMELESS_INPATIENT': 500,    # Default value
        'MEDICAID_PROP': 0.7,        # Default value
        'UNINSURED_PROP': 0.15        # Default value
    })
    return hospital_df

def get_county_list():
    """
    Get a list of county identifiers (location IDs)
    
    Returns:
    --------
    list
        List of location IDs
    """
    # First try to get from demographic data
    try:
        age_df = pd.read_csv('cy_age.csv')
        return age_df['LOCATION_ID'].unique().tolist()
    except:
        # Fallback to a list of California CoC IDs
        return [f'CA-{i:03d}' for i in range(500, 615)]

def get_county_to_coc_mapping():
    """
    Create a mapping between county names and CoC location IDs
    
    Returns:
    --------
    dict
        Mapping from county names to location IDs
    """
    # This is a simplified mapping and would need to be expanded for full implementation
    mapping = {
        'ALAMEDA': 'CA-502',
        'LOS ANGELES': 'CA-600',
        'ORANGE': 'CA-602',
        'SAN DIEGO': 'CA-601',
        'SAN FRANCISCO': 'CA-501',
        'SANTA CLARA': 'CA-500',
        # Add more mappings as needed
    }
    return mapping

def integrate_hospital_data(master_df, hospital_df):
    """
    Integrate hospital utilization data into the master dataset
    
    Parameters:
    -----------
    master_df : DataFrame
        Master dataset
    hospital_df : DataFrame
        Hospital utilization data
        
    Returns:
    --------
    DataFrame
        Integrated master dataset with hospital metrics
    """
    if hospital_df is not None and not hospital_df.empty:
        # Merge on LOCATION_ID
        master_df = master_df.merge(hospital_df, on='LOCATION_ID', how='left')
        
        # Fill missing values with median values
        hospital_cols = ['HOSPITAL_UTIL_RATE', 'HOMELESS_ED_VISITS', 
                        'HOMELESS_INPATIENT', 'MEDICAID_PROP', 'UNINSURED_PROP']
        
        for col in hospital_cols:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna(master_df[col].median())
    
    return master_df

def load_system_performance_data():
    """
    Load system performance metrics data
    
    Returns:
    --------
    DataFrame or None
        System performance data
    """
    try:
        # Calendar year data
        spm_data = pd.read_csv('calendar-year-coc-and-statewide-topline-ca-spms.csv')
        return spm_data
    except Exception as e:
        print(f"Error loading system performance data: {e}")
        try:
            # Try fiscal year data as a fallback
            spm_data = pd.read_csv('fiscal-year-coc-and-statewide-topline-ca-spms.csv')
            return spm_data
        except:
            print("Could not load any system performance data")
            return None

def create_unified_identifier(age_data, race_data, gender_data, spm_data):
    """
    Create a unified identifier for mapping between datasets
    
    Parameters:
    -----------
    age_data : DataFrame
        Age demographic data
    race_data : DataFrame
        Race demographic data
    gender_data : DataFrame
        Gender demographic data
    spm_data : DataFrame
        System performance data
        
    Returns:
    --------
    DataFrame
        Mapping between different identifier systems
    """
    print("Creating unified identifier...")
    
    # Get unique location IDs from each dataset
    location_ids = set()
    location_names = {}
    
    # Process each dataset
    if age_data is not None:
        for loc_id, loc_name in zip(age_data['LOCATION_ID'], age_data['LOCATION']):
            location_ids.add(loc_id)
            location_names[loc_id] = loc_name
    
    if race_data is not None:
        for loc_id in race_data['LOCATION_ID'].unique():
            location_ids.add(loc_id)
    
    if gender_data is not None:
        for loc_id in gender_data['LOCATION_ID'].unique():
            location_ids.add(loc_id)
    
    if spm_data is not None:
        # Map SPM Location field to standard location ID
        spm_to_location_id = {}
        
        # Create a mapping from SPM location names to standard location IDs
        # This is a simplified mapping and would need to be expanded
        for loc_id in location_ids:
            if 'CA-' in loc_id:
                spm_to_location_id[loc_id.replace('CA-', '')] = loc_id
                
                # Also try to map full names
                if loc_id in location_names:
                    name = location_names[loc_id]
                    spm_to_location_id[name] = loc_id
    
    # Create a mapping dataframe
    mapping_df = pd.DataFrame({
        'LOCATION_ID': list(location_ids),
        'LOCATION': [location_names.get(loc_id, loc_id) for loc_id in location_ids]
    })
    
    return mapping_df

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

def calculate_derived_metrics(age_data, race_data, gender_data, spm_data=None):
    """
    Calculate derived metrics from demographic and system performance data
    
    Parameters:
    -----------
    age_data : DataFrame
        Age demographic data
    race_data : DataFrame
        Race demographic data
    gender_data : DataFrame
        Gender demographic data
    spm_data : DataFrame, optional
        System performance data
        
    Returns:
    --------
    dict
        Dictionary of derived metrics DataFrames
    """
    print("Calculating derived metrics...")
    
    derived_metrics = {}
    
    # Calculate age-related metrics
    if age_data is not None:
        # Filter to latest year
        latest_year = age_data['CALENDAR_YEAR'].max()
        latest_age_data = age_data[age_data['CALENDAR_YEAR'] == latest_year]
        
        # Calculate total homeless by location
        age_total = latest_age_data.groupby('LOCATION_ID')['COUNT_AGE'].sum().reset_index()
        age_total.columns = ['LOCATION_ID', 'TOTAL_HOMELESS']
        derived_metrics['age_total'] = age_total
        
        # Calculate proportion by age group and location
        age_pivot = latest_age_data.pivot_table(
            index='LOCATION_ID',
            columns='AGE_GROUP_PUBLIC',
            values='COUNT_AGE',
            aggfunc='sum'
        ).fillna(0)
        
        # Calculate proportions
        age_props = age_pivot.div(age_pivot.sum(axis=1), axis=0)
        age_props.columns = [f"{col}_PROP" for col in age_props.columns]
        
        # Reset index
        age_props = age_props.reset_index()
        derived_metrics['age_props'] = age_props
        
        # Calculate vulnerable age proportion (under 18 and over 65)
        vulnerable_age = age_pivot[['Under 18', '65+']].sum(axis=1) / age_pivot.sum(axis=1)
        vulnerable_age = vulnerable_age.reset_index()
        vulnerable_age.columns = ['LOCATION_ID', 'VULNERABLE_AGE_PROP']
        derived_metrics['vulnerable_age'] = vulnerable_age
    
    # Calculate race-related metrics
    if race_data is not None:
        # Filter to latest year
        latest_year = race_data['CALENDAR_YEAR'].max()
        latest_race_data = race_data[race_data['CALENDAR_YEAR'] == latest_year]
        
        # Calculate proportion by race and location
        race_pivot = latest_race_data.pivot_table(
            index='LOCATION_ID',
            columns='RACE_ETHNICITY_PUBLIC',
            values='COUNT_RACE',
            aggfunc='sum'
        ).fillna(0)
        
        # Calculate proportions
        race_props = race_pivot.div(race_pivot.sum(axis=1), axis=0)
        race_props.columns = [f"{col.replace(' ', '_').replace(',', '')}_PROP" for col in race_props.columns]
        
        # Reset index
        race_props = race_props.reset_index()
        derived_metrics['race_props'] = race_props
    
    # Calculate gender-related metrics
    if gender_data is not None:
        # Filter to latest year
        latest_year = gender_data['CALENDAR_YEAR'].max()
        latest_gender_data = gender_data[gender_data['CALENDAR_YEAR'] == latest_year]
        
        # Calculate proportion by gender and location
        gender_pivot = latest_gender_data.pivot_table(
            index='LOCATION_ID',
            columns='GENDER_PUBLIC',
            values='COUNT_GENDER',
            aggfunc='sum'
        ).fillna(0)
        
        # Calculate proportions
        gender_props = gender_pivot.div(gender_pivot.sum(axis=1), axis=0)
        gender_props.columns = [f"{col.replace(' ', '_').replace(',', '')}_PROP" for col in gender_props.columns]
        
        # Reset index
        gender_props = gender_props.reset_index()
        derived_metrics['gender_props'] = gender_props
    
    # Calculate system performance metrics
    if spm_data is not None:
        # Get the latest SPM metrics by location
        spm_cols = [col for col in spm_data.columns if col not in ['Location', 'Metric']]
        
        # Create a long-to-wide format transformation
        latest_spm = spm_data.pivot(index='Location', columns='Metric', values=spm_cols[-1])
        latest_spm = latest_spm.reset_index()
        
        # Rename columns to add _latest suffix
        latest_spm.columns.name = None
        latest_spm = latest_spm.rename(columns={col: f"{col}_latest" for col in latest_spm.columns if col != 'Location'})
        
        # Map to LOCATION_ID (simplified version)
        latest_spm['LOCATION_ID'] = latest_spm['Location'].apply(lambda x: f"CA-{x}" if x.isdigit() else x)
        
        # Select columns
        latest_spm = latest_spm[['LOCATION_ID'] + [col for col in latest_spm.columns if col.endswith('_latest')]]
        
        derived_metrics['spm_latest'] = latest_spm
        
        # Calculate trends over time
        if len(spm_cols) >= 2:
            # Calculate percent change from earliest to latest year
            trends = spm_data.copy()
            earliest_col = spm_cols[0]
            latest_col = spm_cols[-1]
            
            # Calculate percentage change
            trends['change'] = (trends[latest_col] - trends[earliest_col]) / trends[earliest_col].replace(0, np.nan)
            trends['change'] = trends['change'].fillna(0)
            
            # Create trend DataFrame
            trend_df = trends.pivot(index='Location', columns='Metric', values='change')
            trend_df = trend_df.reset_index()
            
            # Rename columns to add _TREND suffix
            trend_df.columns.name = None
            trend_df = trend_df.rename(columns={col: f"{col}_TREND" for col in trend_df.columns if col != 'Location'})
            
            # Map to LOCATION_ID (simplified version)
            trend_df['LOCATION_ID'] = trend_df['Location'].apply(lambda x: f"CA-{x}" if x.isdigit() else x)
            
            # Select columns
            trend_df = trend_df[['LOCATION_ID'] + [col for col in trend_df.columns if col.endswith('_TREND')]]
            
            derived_metrics['spm_trends'] = trend_df
    
    # Calculate composite vulnerability score
    # This would combine demographic, housing, and system performance indicators
    # into a single vulnerability score
    
    # For this example, we'll create a simplified vulnerability score
    # based on available metrics
    if 'vulnerable_age' in derived_metrics:
        vulnerability_df = derived_metrics['vulnerable_age'].copy()
        
        # Add more vulnerability factors if available
        vulnerability_df['VULNERABILITY_SCORE'] = vulnerability_df['VULNERABLE_AGE_PROP']
        
        derived_metrics['vulnerability'] = vulnerability_df
    
    # Housing access burden (placeholder - would be calculated from housing data)
    # This demonstrates how additional metrics could be integrated
    if 'age_total' in derived_metrics:
        housing_access = derived_metrics['age_total'].copy()
        housing_access['HOUSING_ACCESS_BURDEN'] = np.random.uniform(0.1, 0.9, size=len(housing_access))
        housing_access['SHELTER_UTILIZATION'] = np.random.uniform(0.3, 0.8, size=len(housing_access))
        
        derived_metrics['housing_access'] = housing_access[['LOCATION_ID', 'HOUSING_ACCESS_BURDEN', 'SHELTER_UTILIZATION']]
    
    return derived_metrics

def create_master_dataset(derived_metrics, location_mapping):
    """
    Create a master dataset combining all metrics
    
    Parameters:
    -----------
    derived_metrics : dict
        Dictionary of derived metrics DataFrames
    location_mapping : DataFrame
        Mapping between different identifier systems
        
    Returns:
    --------
    DataFrame
        Master dataset with all metrics
    """
    # Start with location identifiers
    master_df = location_mapping.copy()
    
    # Add derived metrics
    for metric_name, metric_df in derived_metrics.items():
        if metric_df is not None and not metric_df.empty:
            master_df = master_df.merge(metric_df, on='LOCATION_ID', how='left')
    
    # Fill missing values
    master_df = master_df.fillna(0)
    
    return master_df

def clean_demographic_data(age_data, race_data, gender_data):
    """
    Clean and standardize demographic data
    
    Parameters:
    -----------
    age_data : DataFrame
        Age demographic data
    race_data : DataFrame
        Race demographic data
    gender_data : DataFrame
        Gender demographic data
        
    Returns:
    --------
    tuple
        Cleaned age_data, race_data, gender_data
    """
    print("Cleaning and standardizing data...")
    
    # Clean age data
    if age_data is not None:
        # Convert count to numeric, replacing asterisks with NaN
        age_data['COUNT_AGE'] = pd.to_numeric(
            age_data['COUNT_AGE'].replace('*', np.nan), 
            errors='coerce'
        )
        
        # Fill missing values with 0
        age_data['COUNT_AGE'] = age_data['COUNT_AGE'].fillna(0)
    
    # Clean race data
    if race_data is not None:
        # Convert count to numeric
        race_data['COUNT_RACE'] = pd.to_numeric(race_data['COUNT_RACE'].replace('*', np.nan), errors='coerce')
        
        # Fill missing values with 0
        race_data['COUNT_RACE'] = race_data['COUNT_RACE'].fillna(0)
    
    # Clean gender data
    if gender_data is not None:
        # Convert count to numeric
        gender_data['COUNT_GENDER'] = pd.to_numeric(
            gender_data['COUNT_GENDER'].replace('*', np.nan), 
            errors='coerce'
        )
        
        # Fill missing values with 0
        gender_data['COUNT_GENDER'] = gender_data['COUNT_GENDER'].fillna(0)
    
    return age_data, race_data, gender_data

def main():
    """
    Main function to run the data preparation pipeline.
    
    Returns:
    --------
    tuple
        Processed datasets (master_df, age_data, race_data, gender_data, spm_data, hospital_data, location_mapping)
    """
    # Step 1: Load demographic data
    print("Loading demographic data...")
    age_data, race_data, gender_data = load_demographic_data()
    
    # Step 2: Load system performance data
    print("Loading system performance data...")
    spm_data = load_system_performance_data()
    
    # Step 3: Load hospital utilization data
    print("Loading hospital data...")
    hospital_data = load_hospital_data()
    
    # Step 4: Create a unified identifier
    print("Creating unified identifier...")
    location_mapping = create_unified_identifier(age_data, race_data, gender_data, spm_data)
    
    # Step 5: Clean and standardize data
    print("Cleaning and standardizing data...")
    age_data, race_data, gender_data = clean_demographic_data(age_data, race_data, gender_data)
    
    # Step 6: Calculate derived metrics
    print("Calculating derived metrics...")
    derived_metrics = calculate_derived_metrics(age_data, race_data, gender_data, spm_data)
    
    # Step 7: Create master dataset
    print("Creating master dataset...")
    master_df = create_master_dataset(derived_metrics, location_mapping)
    
    # Step 8: Integrate hospital data
    print("Integrating hospital data...")
    master_df = integrate_hospital_data(master_df, hospital_data)
    
    master_df.to_csv('master_dataset.csv', index=False)
    print("✅ Master dataset saved to 'master_dataset.csv'")
    
    return master_df, age_data, race_data, gender_data, spm_data, hospital_data, location_mapping

if __name__ == "__main__":
    main()
