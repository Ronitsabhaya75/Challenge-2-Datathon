import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from feature_engineering import main as get_enhanced_data

# Try to import plotly for enhanced visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
    
    # Define custom color schemes and templates for more appealing visuals
    PLOTLY_CUSTOM_DARK = {
        'bg_color': '#0e1117',
        'grid_color': 'rgba(255, 255, 255, 0.1)',
        'text_color': 'white',
        'colorscales': {
            'sequential': 'Plasma',
            'diverging': 'RdBu', 
            'categorical': 'Turbo'
        }
    }
    
    PLOTLY_CUSTOM_LIGHT = {
        'bg_color': '#f8f9fa',
        'grid_color': 'rgba(0, 0, 0, 0.1)',
        'text_color': '#333333',
        'colorscales': {
            'sequential': 'Viridis',
            'diverging': 'RdYlBu',
            'categorical': 'Turbo'
        }
    }
    
    # Choose the theme (dark or light)
    PLOTLY_THEME = PLOTLY_CUSTOM_DARK
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not installed. Using matplotlib for all visualizations.")

# Set Seaborn style
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

def prepare_model_data(master_df):
    """
    Prepare data for modeling with more targeted feature sets
    """
    print("Preparing data for modeling...")
    
    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # Our target variable is the total homeless population
    y = master_df['TOTAL_HOMELESS']
    
    # Create feature groups for targeted analysis
    feature_groups = {}
    
    # 1. Demographic groups - analyze age-specific, gender-specific, etc. relationships
    age_cols = [col for col in master_df.columns if col.endswith('_PROP') and any(age in col for age in ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'])]
    gender_cols = [col for col in master_df.columns if col.endswith('_PROP') and any(gender in col for gender in ['Man', 'Woman', 'Non-Binary', 'Transgender'])]
    race_cols = [col for col in master_df.columns if col.endswith('_PROP') and any(race in col for race in ['American Indian', 'Asian', 'Black', 'Hispanic', 'White', 'Pacific Islander'])]
    
    feature_groups['age_demographics'] = age_cols
    feature_groups['gender_demographics'] = gender_cols
    feature_groups['race_demographics'] = race_cols
    
    # 2. System performance metrics
    spm_cols = [col for col in master_df.columns if col.endswith('_latest')]
    feature_groups['system_performance'] = spm_cols
    
    # 3. Trend features - how changes over time affect homelessness
    trend_cols = [col for col in master_df.columns if col.endswith('_TREND')]
    feature_groups['trends'] = trend_cols
    
    # 4. Hospital utilization metrics
    hospital_cols = ['HOSPITAL_UTIL_RATE', 'HOMELESS_ED_VISITS', 'HOMELESS_INPATIENT', 
                    'MEDICAID_PROP', 'UNINSURED_PROP']
    hospital_cols = [col for col in hospital_cols if col in master_df.columns]
    feature_groups['hospital_utilization'] = hospital_cols
    
    # 5. Vulnerability and access indicators
    indicator_cols = [
        'HOUSING_ACCESS_BURDEN', 
        'SHELTER_UTILIZATION', 
        'VULNERABLE_AGE_PROP', 
        'VULNERABILITY_SCORE'
    ]
    indicator_cols = [col for col in indicator_cols if col in master_df.columns]
    feature_groups['vulnerability_indicators'] = indicator_cols
    
    # 6. All features combined
    all_features = []
    for group in feature_groups.values():
        all_features.extend(group)
    
    # Remove duplicates while preserving order
    all_features = list(dict.fromkeys(all_features))
    feature_groups['all_features'] = all_features
    
    # Create feature matrices for each group
    X_groups = {}
    for group_name, features in feature_groups.items():
        if features:  # Skip empty feature groups
            # Filter to features that exist in the dataset
            features = [col for col in features if col in master_df.columns]
            
            if features:  # Proceed only if we have valid features
                # Create feature matrix
                X_group = master_df[features].copy()
                
                # Handle missing values and infinities
                X_group = X_group.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Scale features
                scaler = StandardScaler()
                X_group_scaled = scaler.fit_transform(X_group)
                X_group_scaled = pd.DataFrame(X_group_scaled, columns=features)
                
                X_groups[group_name] = X_group_scaled
    
    # For backward compatibility, return the full feature set as X
    X = X_groups['all_features'] if 'all_features' in X_groups else pd.DataFrame()
    
    return X, y, feature_groups, X_groups

def build_targeted_models(X_groups, y):
    """
    Build and evaluate models for targeted feature groups
    to identify which specific factors lead to homelessness
    """
    print("Building targeted models for different feature groups...")
    
    # Results dictionary to store models and their performance for each feature group
    group_results = {}
    
    # Use the same test split across all models for fair comparison
    random_state = 123
    
    # For each feature group, build a gradient boosting model
    for group_name, X_group in X_groups.items():
        print(f"\nEvaluating {group_name} as predictors:")
        
        # Skip if no features
        if X_group.empty:
            print(f"  No features available for {group_name}")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_group, y, test_size=0.2, random_state=random_state
        )
        
        # Train gradient boosting model (best performer from earlier)
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"  R² = {r2:.4f}, RMSE = {rmse:.2f}")
        
        # Get feature importance for this group
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X_group.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Print top features
            top_n = min(5, len(feature_importance))
            print(f"  Top {top_n} most important features:")
            for i, row in feature_importance.head(top_n).iterrows():
                print(f"    - {row['Feature']}: {row['Importance']:.4f}")
            
            # Store results
            group_results[group_name] = {
                'model': model,
                'r2': r2,
                'rmse': rmse,
                'feature_importance': feature_importance
            }
    
    # Compare performance across feature groups with a bar chart
    if group_results:
        group_names = list(group_results.keys())
        r2_scores = [group_results[name]['r2'] for name in group_names]
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(group_names, r2_scores, color='cornflowerblue')
        plt.title('Model Performance by Feature Group (R² Score)', fontsize=16)
        plt.xlabel('Feature Group', fontsize=14)
        plt.ylabel('R² Score', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha='center',
                fontsize=10
            )
        
        plt.tight_layout()
        plt.savefig('figures/q10_feature_group_performance.png')
        plt.close()
    
    return group_results

def analyze_age_homelessness_relationships(master_df, X_groups, group_results):
    """
    Perform detailed analysis of how specific age demographics 
    relate to homelessness rates
    """
    print("\nAnalyzing specific age-to-homelessness relationships...")
    
    if 'age_demographics' not in group_results:
        print("No age demographic features available for analysis")
        return
    
    # Get age-related features and their importance
    age_importance = group_results['age_demographics']['feature_importance']
    
    # Create a figure showing age group relationships to homelessness
    plt.figure(figsize=(14, 10))
    
    # Plot importance of different age groups
    plt.subplot(211)
    sns.barplot(data=age_importance, x='Feature', y='Importance', palette='viridis')
    plt.title('Impact of Different Age Groups on Homelessness Predictions', fontsize=16)
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Create scatterplots for top 2 age groups vs homelessness
    plt.subplot(212)
    top_age_features = age_importance.head(2)['Feature'].tolist()
    
    # Only proceed if we have age features
    if top_age_features:
        # Get the original (unscaled) data for these features
        age_data = master_df[top_age_features + ['TOTAL_HOMELESS']]
        
        # Create a 1x2 grid of scatterplots
        for i, feature in enumerate(top_age_features):
            plt.subplot(2, 2, i+3)  # Start from position 3 in the 2x2 grid
            plt.scatter(age_data[feature], age_data['TOTAL_HOMELESS'], alpha=0.6)
            
            # Add trendline
            z = np.polyfit(age_data[feature], age_data['TOTAL_HOMELESS'], 1)
            p = np.poly1d(z)
            plt.plot(age_data[feature], p(age_data[feature]), "r--")
            
            plt.title(f'{feature} vs Homelessness', fontsize=12)
            plt.xlabel(feature, fontsize=10)
            plt.ylabel('Total Homeless Population', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q10_age_homelessness_relationships.png')
    plt.close()
    
    # Print key findings
    print("\nKey findings about age-homelessness relationships:")
    for i, row in age_importance.head(3).iterrows():
        print(f"  - {row['Feature']} has a {row['Importance']:.4f} importance score in predicting homelessness")

def analyze_hospital_homelessness_relationship(master_df, X_groups, group_results):
    """
    Analyze the relationship between hospital utilization metrics and homelessness
    """
    print("\nAnalyzing hospital utilization and homelessness relationship...")
    
    if 'hospital_utilization' not in group_results:
        print("No hospital utilization features available for analysis")
        return
    
    # Get hospital-related features and their importance
    hospital_importance = group_results['hospital_utilization']['feature_importance']
    
    # Create a figure showing hospital metric relationships to homelessness
    plt.figure(figsize=(14, 10))
    
    # Plot importance of different hospital metrics
    plt.subplot(211)
    sns.barplot(x='Feature', y='Importance', data=hospital_importance, palette='viridis')
    plt.title('Impact of Hospital Utilization Metrics on Homelessness Predictions', fontsize=16)
    plt.xlabel('Hospital Metric', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Create scatterplots for top 2 hospital metrics vs homelessness
    plt.subplot(212)
    top_hospital_features = hospital_importance.head(2)['Feature'].tolist()
    
    # Only proceed if we have hospital features
    if top_hospital_features:
        # Get the original (unscaled) data for these features
        hospital_data = master_df[top_hospital_features + ['TOTAL_HOMELESS']]
        
        # Create a 1x2 grid of scatterplots
        for i, feature in enumerate(top_hospital_features):
            plt.subplot(2, 2, i+3)  # Start from position 3 in the 2x2 grid
            plt.scatter(hospital_data[feature], hospital_data['TOTAL_HOMELESS'], alpha=0.6)
            
            # Add trendline
            z = np.polyfit(hospital_data[feature], hospital_data['TOTAL_HOMELESS'], 1)
            p = np.poly1d(z)
            plt.plot(hospital_data[feature], p(hospital_data[feature]), "r--")
            
            plt.title(f'{feature} vs Homelessness', fontsize=12)
            plt.xlabel(feature, fontsize=10)
            plt.ylabel('Total Homeless Population', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/q10_hospital_homelessness_relationships.png')
    plt.close()
    
    # Print key findings
    print("\nKey findings about hospital utilization and homelessness relationship:")
    for i, row in hospital_importance.head(3).iterrows():
        print(f"  - {row['Feature']} has a {row['Importance']:.4f} importance score in predicting homelessness")

def analyze_cross_category_relationships(master_df, X_groups, group_results):
    """
    Analyze how factors across different categories relate to each other
    """
    print("\nAnalyzing cross-category relationships with homelessness...")
    
    # Get top features from each category
    top_factors = {}
    for category, results in group_results.items():
        if 'feature_importance' in results and not results['feature_importance'].empty:
            # Get the most important feature from each category
            top_factors[category] = results['feature_importance'].iloc[0]['Feature']
    
    # If we have at least 2 categories, analyze correlations
    if len(top_factors) >= 2:
        # Get correlation matrix for all features
        all_features = []
        for group_name, features in X_groups.items():
            if group_name in top_factors:
                all_features.append(features)
        
        if all_features:
            try:
                X_combined = pd.concat(all_features, axis=1)
                corr_matrix = X_combined.corr()
                
                # Find strongest correlations between top factors from different categories
                print("\nKey cross-category relationships:")
                for cat1 in top_factors:
                    for cat2 in top_factors:
                        if cat1 != cat2:  # Only compare different categories
                            feature1 = top_factors[cat1]
                            feature2 = top_factors[cat2]
                            
                            # Ensure features are in the correlation matrix
                            if feature1 in corr_matrix.index and feature2 in corr_matrix.columns:
                                # Get the correlation value - ensure it's a scalar
                                corr_value = corr_matrix.loc[feature1, feature2]
                                
                                # Check if it's a DataFrame or Series
                                if isinstance(corr_value, (pd.DataFrame, pd.Series)):
                                    continue
                                
                                # Now it's safe to compare
                                if abs(corr_value) > 0.3:
                                    direction = "positive" if corr_value > 0 else "negative"
                                    print(f"  - {feature1} ({cat1}) has a {direction} correlation ({corr_value:.2f}) with {feature2} ({cat2})")
            except Exception as e:
                print(f"Error in cross-category analysis: {e}")

def identify_key_homelessness_drivers(master_df, group_results):
    """
    Identify key drivers of homelessness across all datasets
    """
    print("\nIdentifying key drivers of homelessness across all datasets...")
    
    # Create the figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Combine feature importance from all group models
    key_drivers = []
    categories = []
    
    for category, result in group_results.items():
        if 'feature_importance' in result:
            # Add category information to each feature
            category_importance = result['feature_importance'].copy()
            category_importance['Category'] = category
            key_drivers.append(category_importance)
            categories.append(category)
    
    # Combine all feature importances
    if key_drivers:
        combined_importance = pd.concat(key_drivers)
        
        # Get top 20 features overall
        top_features = combined_importance.sort_values('Importance', ascending=False).head(20)
        
        # Visualize top features by category
        plt.figure(figsize=(14, 10))
        sns.barplot(x='Importance', y='Feature', hue='Category', data=top_features, palette='viridis')
        plt.title('Top 20 Drivers of Homelessness Across All Datasets', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.legend(title='Category', loc='lower right')
        plt.tight_layout()
        plt.savefig('figures/q10_key_homelessness_drivers.png')
        plt.close()
        
        # Create a summary of key drivers by category
        category_summary = {}
        for category in top_features['Category'].unique():
            category_features = top_features[top_features['Category'] == category]
            category_summary[category] = category_features.head(3)['Feature'].tolist()
        
        # Print key drivers by category
        print("\nKey drivers of homelessness by category:")
        for category, features in category_summary.items():
            print(f"  {category.upper()}:")
            for feature in features:
                # Get importance value
                importance = float(top_features[(top_features['Category'] == category) & 
                                              (top_features['Feature'] == feature)]['Importance'])
                print(f"    - {feature} ({importance:.4f})")
        
        return top_features
    
    return None

def analyze_targeted_predictors(master_df, X_groups, group_results):
    """
    Analyze how different categories of factors predict homelessness
    """
    print("\nAnalyzing targeted predictors across categories...")
    
    # Store top predictors from each category
    top_predictors = {}
    
    for group_name, results in group_results.items():
        if 'feature_importance' in results:
            # Get top 3 predictors from each category
            top_n = min(3, len(results['feature_importance']))
            top_predictors[group_name] = results['feature_importance'].head(top_n)
    
    # Combine all top predictors across categories
    all_top_predictors = pd.concat(top_predictors.values())
    all_top_predictors = all_top_predictors.sort_values('Importance', ascending=False).head(15)
    
    # Visualize top 15 predictors across all categories
    plt.figure(figsize=(14, 10))
    sns.barplot(data=all_top_predictors, x='Importance', y='Feature', palette='viridis')
    plt.title('Top 15 Predictors of Homelessness Across Categories', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/q10_top_predictors_all_categories.png')
    plt.close()
    
    print("\nTop predictors across all categories:")
    for i, row in all_top_predictors.head(7).iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f}")
    
    return all_top_predictors

def build_models(X, y):
    """
    Build and evaluate multiple regression models
    """
    print("Building and evaluating models...")
    
    # Split data into train and test sets with a different random_state
    # Change from 42 to 123 to get a different train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.2f}")
    
    # Visualize model performance comparison
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(model_names, r2_scores, color='cornflowerblue')
    plt.title('Model Performance Comparison (R² Score)', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.4f}',
            ha='center',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig('figures/q10_model_performance_comparison.png')
    plt.close()
    
    # Find the best model
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")
    
    return best_model, results

def tune_best_model(X, y, best_model_name, best_model):
    """
    Tune hyperparameters for the best model
    """
    print(f"Tuning hyperparameters for {best_model_name}...")
    
    # Define parameter grid based on model type
    param_grid = {}
    
    if best_model_name == 'Linear Regression':
        # Linear regression doesn't have hyperparameters to tune
        return best_model
    
    elif best_model_name == 'Ridge Regression':
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    
    elif best_model_name == 'Lasso Regression':
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    
    elif best_model_name == 'Elastic Net':
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    
    elif best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    
    # If no parameters to tune, return the original model
    if not param_grid:
        return best_model
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X, y)
    
    # Get the best model
    tuned_model = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Tuned model R²: {grid_search.best_score_:.4f}")
    
    return tuned_model

def analyze_feature_importance(X, y, best_model, feature_cols):
    """
    Analyze feature importance from the best model
    """
    print("Analyzing feature importance...")
    
    # Different models have different ways to get feature importance
    importances = []
    
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        # Linear models
        importances = np.abs(best_model.coef_)
    else:
        # Use permutation importance as a backup
        perm_importance = permutation_importance(best_model, X, y, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
    
    # Create dataframe of feature importances
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Visualize top 15 most important features
    top_features = feature_importance.head(15)
    
    plt.figure(figsize=(14, 10))
    sns.barplot(data=top_features, x='Importance', y='Feature', hue='Feature', legend=False)
    plt.title('Top 15 Most Important Features', fontsize=16)
    plt.xlabel('Feature Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/q10_feature_importance.png')
    plt.close()
    
    return feature_importance

def analyze_residuals(X, y, best_model, master_df):
    """
    Analyze the residuals from the model predictions
    and identify counties with unusual patterns
    """
    print("\nAnalyzing model residuals...")
    
    # Make predictions
    y_pred = best_model.predict(X)
    
    # Calculate residuals and standardized residuals
    residuals = y - y_pred
    std_residuals = (residuals - residuals.mean()) / residuals.std()
    
    # Create a residual dataframe
    residual_df = pd.DataFrame({
        'LOCATION_ID': master_df['LOCATION_ID'],
        'County': master_df['LOCATION'],
        'Actual': y,
        'Predicted': y_pred,
        'Residual': residuals,
        'Std_Residual': std_residuals
    })
    
    # Add a residual factor column (positive = under-predicted, negative = over-predicted)
    residual_df['RESIDUAL_FACTOR'] = residual_df['Std_Residual'] / residual_df['Std_Residual'].abs().max()
    
    # Sort by absolute standardized residual
    residual_df = residual_df.sort_values('Std_Residual', key=lambda x: x.abs(), ascending=False)
    
    # Calculate R-squared of the model
    r2 = r2_score(y, y_pred)
    
    # Plot residuals
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    
    # Create a more visually appealing scatter plot with a color gradient based on residual value
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(y_pred)))
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
    
    # Add a bit of background styling
    plt.gca().set_facecolor('#f8f9fa')
    
    # Plot actual vs predicted
    plt.subplot(2, 1, 2)
    
    # Create a gradient of colors based on the order of points
    plt.scatter(y, y_pred, c=colors, alpha=0.7, s=80, edgecolor='k', linewidth=0.5)
    
    # Add a perfect prediction line
    max_val = max(y.max(), y_pred.max())
    min_val = min(y.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.xlabel('Actual Values', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=14, fontweight='bold')
    plt.title('Actual vs Predicted Values', fontsize=18, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add a bit of background styling
    plt.gca().set_facecolor('#f8f9fa')
    
    # Add labels for counties with large residuals
    threshold = 2.0  # Label counties with standardized residuals > 2 or < -2
    significant_residuals = residual_df[residual_df['Std_Residual'].abs() > threshold]
    
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
    
    # Highlight significant residuals
    print("\nCounties with significant residuals (|std residual| > 2):")
    for _, row in significant_residuals.iterrows():
        direction = "under-predicted" if row['Residual'] > 0 else "over-predicted"
        print(f"  - {row['County']}: {direction} by {abs(row['Residual']):.0f} " + 
              f"(std. residual = {row['Std_Residual']:.2f})")
    
    # If we have plotly available, create interactive visualizations
    if HAS_PLOTLY:
        # Create an enhanced interactive residual vs predicted plot
        fig1 = go.Figure()
        
        # Add color based on standardized residual
        colorscale = PLOTLY_THEME['colorscales']['diverging']
        
        # Add hover text
        hover_text = [
            f"County: {county}<br>" +
            f"Actual: {actual:,.0f}<br>" +
            f"Predicted: {pred:,.0f}<br>" +
            f"Residual: {resid:,.0f}<br>" +
            f"Std Residual: {std_resid:.2f}"
            for county, actual, pred, resid, std_resid in zip(
                residual_df['County'], residual_df['Actual'], 
                residual_df['Predicted'], residual_df['Residual'], 
                residual_df['Std_Residual']
            )
        ]
        
        # Add scatter plot with colored points
        fig1.add_trace(
            go.Scatter(
                x=residual_df['Predicted'],
                y=residual_df['Residual'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=residual_df['Std_Residual'],
                    colorscale=colorscale,
                    colorbar=dict(
                        title="Std Residual",
                        thickness=15,
                        len=0.9,
                        bgcolor=PLOTLY_THEME['bg_color'],
                        titlefont=dict(color=PLOTLY_THEME['text_color']),
                        tickfont=dict(color=PLOTLY_THEME['text_color'])
                    ),
                    line=dict(width=1, color='black')
                ),
                text=hover_text,
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
            line=dict(
                color='red',
                width=2,
                dash='dash',
            )
        )
        
        # Add lines for +/-2 std residuals
        std_dev = residual_df['Residual'].std()
        fig1.add_shape(
            type='line',
            x0=residual_df['Predicted'].min(),
            y0=2*std_dev,
            x1=residual_df['Predicted'].max(),
            y1=2*std_dev,
            line=dict(
                color='rgba(255, 0, 0, 0.5)',
                width=1.5,
                dash='dot',
            )
        )
        
        fig1.add_shape(
            type='line',
            x0=residual_df['Predicted'].min(),
            y0=-2*std_dev,
            x1=residual_df['Predicted'].max(),
            y1=-2*std_dev,
            line=dict(
                color='rgba(255, 0, 0, 0.5)',
                width=1.5,
                dash='dot',
            )
        )
        
        # Update layout with theme
        fig1.update_layout(
            title={
                'text': 'Interactive Residual Analysis: Residuals vs Predicted Values',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color=PLOTLY_THEME['text_color'])
            },
            xaxis_title={
                'text': 'Predicted Homeless Population',
                'font': dict(size=16, color=PLOTLY_THEME['text_color'])
            },
            yaxis_title={
                'text': 'Residual (Actual - Predicted)',
                'font': dict(size=16, color=PLOTLY_THEME['text_color'])
            },
            plot_bgcolor=PLOTLY_THEME['bg_color'],
            paper_bgcolor=PLOTLY_THEME['bg_color'],
            font=dict(color=PLOTLY_THEME['text_color']),
            width=1000,
            height=700,
            hovermode='closest',
            xaxis=dict(
                gridcolor=PLOTLY_THEME['grid_color'],
                zerolinecolor=PLOTLY_THEME['grid_color'],
            ),
            yaxis=dict(
                gridcolor=PLOTLY_THEME['grid_color'],
                zerolinecolor=PLOTLY_THEME['grid_color'],
            ),
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Model R² = {r2:.4f}",
                    showarrow=False,
                    font=dict(size=14, color=PLOTLY_THEME['text_color']),
                    bgcolor="rgba(255, 255, 255, 0.1)",
                    bordercolor="rgba(255, 255, 255, 0.3)",
                    borderwidth=1,
                    borderpad=4,
                )
            ]
        )
        
        # Save interactive visualization
        os.makedirs('interactive', exist_ok=True)
        fig1.write_html('interactive/q11_residuals_vs_predicted_interactive.html')
        
        # Create an enhanced interactive actual vs predicted plot
        fig2 = go.Figure()
        
        # Add actual vs predicted scatter plot
        fig2.add_trace(
            go.Scatter(
                x=residual_df['Actual'],
                y=residual_df['Predicted'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=residual_df['Std_Residual'],
                    colorscale=colorscale,
                    colorbar=dict(
                        title="Std Residual",
                        thickness=15,
                        len=0.9,
                        bgcolor=PLOTLY_THEME['bg_color'],
                        titlefont=dict(color=PLOTLY_THEME['text_color']),
                        tickfont=dict(color=PLOTLY_THEME['text_color'])
                    ),
                    line=dict(width=1, color='black')
                ),
                text=hover_text,
                hoverinfo='text',
                name='Counties'
            )
        )
        
        # Add diagonal line (perfect prediction)
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
        
        # Add labels for counties with significant residuals
        for _, row in significant_residuals.iterrows():
            fig2.add_annotation(
                x=row['Actual'],
                y=row['Predicted'],
                text=row['County'],
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowcolor='white',
                arrowwidth=1.5,
                font=dict(color=PLOTLY_THEME['text_color'])
            )
        
        # Update layout with theme
        fig2.update_layout(
            title={
                'text': 'Interactive Residual Analysis: Actual vs Predicted Values',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color=PLOTLY_THEME['text_color'])
            },
            xaxis_title={
                'text': 'Actual Homeless Population',
                'font': dict(size=16, color=PLOTLY_THEME['text_color'])
            },
            yaxis_title={
                'text': 'Predicted Homeless Population',
                'font': dict(size=16, color=PLOTLY_THEME['text_color'])
            },
            plot_bgcolor=PLOTLY_THEME['bg_color'],
            paper_bgcolor=PLOTLY_THEME['bg_color'],
            font=dict(color=PLOTLY_THEME['text_color']),
            width=1000,
            height=700,
            hovermode='closest',
            xaxis=dict(
                gridcolor=PLOTLY_THEME['grid_color'],
                zerolinecolor=PLOTLY_THEME['grid_color'],
            ),
            yaxis=dict(
                gridcolor=PLOTLY_THEME['grid_color'],
                zerolinecolor=PLOTLY_THEME['grid_color'],
            ),
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Model R² = {r2:.4f}",
                    showarrow=False,
                    font=dict(size=14, color=PLOTLY_THEME['text_color']),
                    bgcolor="rgba(255, 255, 255, 0.1)",
                    bordercolor="rgba(255, 255, 255, 0.3)",
                    borderwidth=1,
                    borderpad=4,
                )
            ]
        )
        
        # Save interactive visualization
        fig2.write_html('interactive/q11_actual_vs_predicted_interactive.html')
        
        # Create a dedicated visualization for significant residuals
        if not significant_residuals.empty:
            fig3 = go.Figure()
            
            # Sort significant residuals by standardized residual value
            significant_residuals = significant_residuals.sort_values('Std_Residual')
            
            # Create a more informative bar chart
            fig3.add_trace(
                go.Bar(
                    x=significant_residuals['County'],
                    y=significant_residuals['Std_Residual'],
                    marker=dict(
                        color=significant_residuals['Std_Residual'],
                        colorscale=colorscale,
                        line=dict(width=1, color='black')
                    ),
                    text=[
                        f"Actual: {actual:,.0f}<br>" +
                        f"Predicted: {pred:,.0f}<br>" +
                        f"Residual: {resid:,.0f}"
                        for actual, pred, resid in zip(
                            significant_residuals['Actual'], 
                            significant_residuals['Predicted'], 
                            significant_residuals['Residual']
                        )
                    ],
                    hoverinfo='text+name',
                    name='Std Residual'
                )
            )
            
            # Add horizontal lines at +/-2
            fig3.add_shape(
                type='line',
                x0=-0.5,
                y0=2,
                x1=len(significant_residuals) - 0.5,
                y1=2,
                line=dict(
                    color='rgba(255, 0, 0, 0.5)',
                    width=1.5,
                    dash='dot',
                )
            )
            
            fig3.add_shape(
                type='line',
                x0=-0.5,
                y0=-2,
                x1=len(significant_residuals) - 0.5,
                y1=-2,
                line=dict(
                    color='rgba(255, 0, 0, 0.5)',
                    width=1.5,
                    dash='dot',
                )
            )
            
            # Update layout with theme
            fig3.update_layout(
                title={
                    'text': 'Counties with Significant Residuals (|Std. Residual| > 2)',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24, color=PLOTLY_THEME['text_color'])
                },
                xaxis_title={
                    'text': 'County',
                    'font': dict(size=16, color=PLOTLY_THEME['text_color'])
                },
                yaxis_title={
                    'text': 'Standardized Residual',
                    'font': dict(size=16, color=PLOTLY_THEME['text_color'])
                },
                plot_bgcolor=PLOTLY_THEME['bg_color'],
                paper_bgcolor=PLOTLY_THEME['bg_color'],
                font=dict(color=PLOTLY_THEME['text_color']),
                width=1000,
                height=600,
                hovermode='closest',
                xaxis=dict(
                    gridcolor=PLOTLY_THEME['grid_color'],
                    zerolinecolor=PLOTLY_THEME['grid_color'],
                ),
                yaxis=dict(
                    gridcolor=PLOTLY_THEME['grid_color'],
                    zerolinecolor=PLOTLY_THEME['grid_color'],
                ),
                annotations=[
                    dict(
                        x=0.02,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text="Counties potentially needing targeted intervention",
                        showarrow=False,
                        font=dict(size=14, color=PLOTLY_THEME['text_color']),
                        bgcolor="rgba(255, 255, 255, 0.1)",
                        bordercolor="rgba(255, 255, 255, 0.3)",
                        borderwidth=1,
                        borderpad=4,
                    )
                ]
            )
            
            # Save interactive visualization
            fig3.write_html('interactive/q11_significant_residuals_interactive.html')
    
    return residual_df

def forecast_future_trends(master_df, best_model, feature_cols):
    """
    Forecast key homelessness indicators for 2024
    """
    print("Forecasting homelessness trends for 2024...")
    
    # Check if trend columns are present for forecasting
    trend_cols = [col for col in master_df.columns if col.endswith('_TREND')]
    
    if len(trend_cols) == 0:
        print("Not enough trend data available for forecasting")
        return master_df
    
    # Create a forecast dataframe
    forecast_df = master_df[['LOCATION_ID', 'LOCATION', 'TOTAL_HOMELESS']].copy()
    
    # Create a simple linear extrapolation for key metrics
    # For demo purposes, we'll focus on M1a (Total Persons Served)
    # and M3 (Permanent Housing Placements)
    # These would normally come from time series forecasting
    
    key_metrics = ['M1a', 'M3']
    
    for metric in key_metrics:
        col = f'{metric}_latest'
        trend_col = f'{metric}_TREND'
        
        if col in master_df.columns and trend_col in master_df.columns:
            # Simple linear extrapolation: 2024 = 2023 * (1 + trend)
            forecast_df[f'{metric}_2024'] = master_df[col] * (1 + master_df[trend_col])
    
    # For homeless population forecast, use the trained model
    # Get features used in the model
    forecast_features = master_df[feature_cols].copy()
    
    # Handle missing values
    forecast_features = forecast_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Get predictions for 2023 (current)
    y_pred_2023 = best_model.predict(forecast_features)
    
    # Modify features based on trends for 2024 prediction
    forecast_features_2024 = forecast_features.copy()
    
    # Apply trends to relevant features
    for col in forecast_features.columns:
        # Find matching trend column if exists
        trend_col = None
        for tc in trend_cols:
            feature_name = tc.replace('_TREND', '')
            if feature_name in col or col.startswith(feature_name):
                trend_col = tc
                break
        
        if trend_col and trend_col in master_df.columns:
            # Update feature with trend
            forecast_features_2024[col] = forecast_features[col] * (1 + master_df[trend_col])
    
    # Get predictions for 2024
    y_pred_2024 = best_model.predict(forecast_features_2024)
    
    # Add predictions to forecast dataframe
    forecast_df['PREDICTED_2023'] = y_pred_2023
    forecast_df['PREDICTED_2024'] = y_pred_2024
    forecast_df['CHANGE_PCT'] = (forecast_df['PREDICTED_2024'] - forecast_df['PREDICTED_2023']) / forecast_df['PREDICTED_2023'] * 100
    
    # Visualize forecasted changes for top counties
    top_counties = forecast_df.nlargest(10, 'TOTAL_HOMELESS')
    
    # Create a more visually appealing plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set background color
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Create grouped bar chart with custom styling
    x = np.arange(len(top_counties))
    width = 0.35
    
    # Create bars with gradient colors and shadows
    bars1 = ax.bar(x - width/2, top_counties['PREDICTED_2023'], width, label='2023 (Current)', color='#3182bd', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, top_counties['PREDICTED_2024'], width, label='2024 (Forecasted)', color='#de2d26', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add shadows to bars
    for bar_set in [bars1, bars2]:
        for bar in bar_set:
            x_pos = bar.get_x()
            width = bar.get_width()
            height = bar.get_height()
            ax.add_patch(plt.Rectangle((x_pos+0.03, 0), width, height, color='gray', alpha=0.15))
    
    # Add title and labels with improved styling
    ax.set_title('Forecasted Homeless Population for 2024 - Top 10 Counties', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('County', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Homeless Population', fontsize=14, fontweight='bold', labelpad=10)
    
    # Add county names and rotate labels
    ax.set_xticks(x)
    ax.set_xticklabels(top_counties['LOCATION'], rotation=45, ha='right')
    
    # Add a legend with better styling
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='white', edgecolor='lightgray')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # Add data labels for both bars
    for bars, heights in [(bars1, top_counties['PREDICTED_2023']), (bars2, top_counties['PREDICTED_2024'])]:
        for i, (bar, height) in enumerate(zip(bars, heights)):
            height_val = int(height)
            ax.text(bar.get_x() + bar.get_width()/2, height + 100, f'{height_val:,}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   color='black')
            
    # Add percentage change annotations
    for i, county in enumerate(top_counties.iterrows()):
        _, row = county
        change_pct = row['CHANGE_PCT']
        y_pos = max(row['PREDICTED_2023'], row['PREDICTED_2024']) + 2000
        
        # Add arrow and text with change percentage
        arrow_style = dict(arrowstyle='->', color='green' if change_pct >= 0 else 'red')
        percent_text = f"+{change_pct:.1f}%" if change_pct >= 0 else f"{change_pct:.1f}%"
        
        ax.annotate(percent_text, xy=(x[i], y_pos), xytext=(x[i], y_pos + 3000),
                  arrowprops=arrow_style, ha='center', fontsize=10, fontweight='bold',
                  color='green' if change_pct >= 0 else 'red',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figures/q12_homelessness_forecast_2024.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive forecast visualization using Plotly if available
    if HAS_PLOTLY:
        # Create an interactive bar chart comparing 2023 and 2024 forecasts
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
        
        # Add 2024 forecasted bars
        fig.add_trace(
            go.Bar(
                x=top_counties['LOCATION'],
                y=top_counties['PREDICTED_2024'],
                name='2024 (Forecasted)',
                marker_color='rgb(214, 39, 40)',
                text=[f"{int(val):,}" for val in top_counties['PREDICTED_2024']],
                textposition='auto',
                hovertemplate='County: %{x}<br>2024 Forecast: %{y:,}<br>Change: %{text}<extra></extra>',
                customdata=[f"{pct:.1f}%" for pct in top_counties['CHANGE_PCT']]
            )
        )
        
        # Update layout with the custom theme
        fig.update_layout(
            title={
                'text': 'Interactive Forecast of Homeless Population for 2024 - Top 10 Counties',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color=PLOTLY_THEME['text_color'])
            },
            xaxis={
                'title': 'County',
                'titlefont': dict(size=16, color=PLOTLY_THEME['text_color']),
                'tickangle': -45
            },
            yaxis={
                'title': 'Homeless Population',
                'titlefont': dict(size=16, color=PLOTLY_THEME['text_color']),
                'gridcolor': PLOTLY_THEME['grid_color']
            },
            barmode='group',
            plot_bgcolor=PLOTLY_THEME['bg_color'],
            paper_bgcolor=PLOTLY_THEME['bg_color'],
            font=dict(color=PLOTLY_THEME['text_color']),
            width=1000,
            height=700,
            legend={
                'title': {'text': 'Year', 'font': dict(size=14, color=PLOTLY_THEME['text_color'])},
                'bgcolor': 'rgba(255, 255, 255, 0.1)',
                'bordercolor': 'rgba(255, 255, 255, 0.2)',
                'borderwidth': 1
            }
        )
        
        # Add annotations for percent changes
        for i, row in enumerate(top_counties.iterrows()):
            _, data = row
            change_pct = data['CHANGE_PCT']
            color = 'green' if change_pct >= 0 else 'red'
            marker = '▲' if change_pct >= 0 else '▼'
            
            y_pos = max(data['PREDICTED_2023'], data['PREDICTED_2024']) * 1.1
            
            fig.add_annotation(
                x=data['LOCATION'],
                y=y_pos,
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
        
        # Save as HTML for interactive viewing
        os.makedirs('interactive', exist_ok=True)
        fig.write_html('interactive/q12_homelessness_forecast_interactive.html')
        
        # Create an interactive map visualization of forecasted changes across counties
        # This requires county codes to be in the format CA-601, etc.
        if 'LOCATION_ID' in forecast_df.columns and forecast_df['LOCATION_ID'].str.contains('CA-').any():
            # Create a choropleth map using California counties
            # (This is a simplified version; in practice you would use GeoJSON data for California counties)
            fig_map = px.choropleth(
                forecast_df,
                locations='LOCATION_ID',
                locationmode='USA-states',  # Use USA-states mode with California county codes
                color='CHANGE_PCT',
                hover_name='LOCATION',
                color_continuous_scale=PLOTLY_THEME['colorscales']['diverging'],
                range_color=[-5, 5],  # Adjust this range based on your data
            )
            
            # Update layout
            fig_map.update_layout(
                title={
                    'text': 'Forecasted Change in Homelessness by County (2023-2024)',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24, color=PLOTLY_THEME['text_color'])
                },
                geo=dict(scope='usa', projection=dict(type='albers usa')),
                plot_bgcolor=PLOTLY_THEME['bg_color'],
                paper_bgcolor=PLOTLY_THEME['bg_color'],
                font=dict(color=PLOTLY_THEME['text_color']),
                width=1000,
                height=700
            )
            
            # Save as HTML
            fig_map.write_html('interactive/q12_forecast_map_interactive.html')
    
    # Print counties with significant forecasted changes
    print("\nCounties with largest forecasted increases in homelessness:")
    increasing_counties = forecast_df.sort_values('CHANGE_PCT', ascending=False).head(5)
    for _, row in increasing_counties.iterrows():
        print(f"  - {row['LOCATION']}: {row['CHANGE_PCT']:.1f}% increase " +
              f"(from {row['PREDICTED_2023']:.0f} to {row['PREDICTED_2024']:.0f})")
    
    print("\nCounties with largest forecasted decreases in homelessness:")
    decreasing_counties = forecast_df.sort_values('CHANGE_PCT', ascending=True).head(5)
    for _, row in decreasing_counties.iterrows():
        print(f"  - {row['LOCATION']}: {abs(row['CHANGE_PCT']):.1f}% decrease " +
              f"(from {row['PREDICTED_2023']:.0f} to {row['PREDICTED_2024']:.0f})")
    
    return forecast_df

def recommend_targeted_funding(master_df, residual_df, forecast_df, vulnerability_threshold=0.5, 
                              residual_weight=0.35, current_weight=0.35, forecast_weight=0.3):
    """
    Recommend counties for targeted funding based on:
    1. Current homelessness (TOTAL_HOMELESS)
    2. Vulnerability (high vulnerability counties are prioritized)
    3. Projected change (prioritize counties projected to worsen)
    4. Model residuals (prioritize counties where model under-predicts, 
       indicating factors not captured in model may be contributing)
    
    Returns a DataFrame sorted by composite score.
    """
    print("\nGenerating county funding recommendations...")
    
    # Create copy of master_df with relevant columns
    recommendation_df = master_df[['LOCATION_ID', 'LOCATION', 'TOTAL_HOMELESS']].copy()
    
    # Merge with residual_df using LOCATION_ID
    recommendation_df = recommendation_df.merge(
        residual_df[['LOCATION_ID', 'Std_Residual']], 
        on='LOCATION_ID', 
        how='left'
    )
    
    # Add forecasted change
    if 'CHANGE_PCT' in forecast_df.columns:
        recommendation_df = recommendation_df.merge(
            forecast_df[['LOCATION_ID', 'CHANGE_PCT']], 
            on='LOCATION_ID', 
            how='left'
        )
    else:
        recommendation_df['CHANGE_PCT'] = 0
    
    # Fill NaN values in CHANGE_PCT
    recommendation_df['CHANGE_PCT'] = recommendation_df['CHANGE_PCT'].fillna(0)
    
    # Normalize values (min-max scaling)
    scaler = MinMaxScaler()
    
    # For total homeless, higher values should result in higher scores
    recommendation_df['HomelessScore'] = scaler.fit_transform(
        recommendation_df[['TOTAL_HOMELESS']])
    
    # For residuals, positive values indicate underprediction (should be prioritized)
    # Adjust residuals to prioritize positive values
    adjusted_residuals = recommendation_df['Std_Residual'].values.reshape(-1, 1)
    min_residual = adjusted_residuals.min()
    
    # Scale residuals while preserving sign
    if min_residual < 0:
        # Shift all values to be positive, then scale
        adjusted_residuals = adjusted_residuals - min_residual
    
    recommendation_df['ResidualScore'] = scaler.fit_transform(adjusted_residuals)
    
    # For change percent, higher values (worsening) should be prioritized
    # If all values are identical, set uniform scores
    if recommendation_df['CHANGE_PCT'].nunique() <= 1:
        recommendation_df['ChangeScore'] = 0.5
    else:
        recommendation_df['ChangeScore'] = scaler.fit_transform(
            recommendation_df[['CHANGE_PCT']])
    
    # Calculate composite score based on weighting
    recommendation_df['CompositeScore'] = (
        current_weight * recommendation_df['HomelessScore'] +
        residual_weight * recommendation_df['ResidualScore'] +
        forecast_weight * recommendation_df['ChangeScore']
    )
    
    # Sort by composite score
    recommendation_df = recommendation_df.sort_values('CompositeScore', ascending=False)
    
    # Add rank column
    recommendation_df['Rank'] = range(1, len(recommendation_df) + 1)
    
    # Create output with formatted string
    recommendation_df['Rationale'] = recommendation_df.apply(
        lambda row: f"Current: {row['TOTAL_HOMELESS']:.0f} homeless, " +
                   f"Forecast change: {row['CHANGE_PCT']*100:.1f}%, " +
                   (f"Residual: {row['Std_Residual']:.2f} std" if pd.notna(row['Std_Residual']) else "Residual: N/A"),
        axis=1
    )
    
    # Select columns for final output
    final_output = recommendation_df[[
        'Rank', 'LOCATION', 'CompositeScore', 'Rationale', 
        'TOTAL_HOMELESS', 'CHANGE_PCT', 'Std_Residual'
    ]].copy()
    
    # Save recommendations to CSV
    final_output.to_csv("outputs/funding_recommendations.csv", index=False)
    
    # Create an interactive visualization of the top recommendations using Plotly if available
    if HAS_PLOTLY:
        # Create directory for interactive visualizations if it doesn't exist
        os.makedirs('interactive', exist_ok=True)
        
        # Get top 15 counties for visualization
        top_n = min(15, len(final_output))
        top_recommendations = final_output.head(top_n).copy()
        
        # Create a DataFrame for the component breakdown
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
        
        # Create a stacked bar chart
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
            hover_data={'Rank': True},
        )
        
        # Add total score as text annotations
        for i, county in enumerate(top_recommendations['LOCATION']):
            fig.add_annotation(
                x=county,
                y=top_recommendations['CompositeScore'].iloc[i] + 0.02,
                text=f"Total: {top_recommendations['CompositeScore'].iloc[i]:.2f}",
                showarrow=False,
                font=dict(
                    size=10,
                    color="black"
                )
            )
        
        # Customize layout
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': top_recommendations['LOCATION']},
            width=1000,
            height=600,
            xaxis_tickangle=-45
        )
        
        # Save as HTML for interactive viewing
        fig.write_html('interactive/q13_funding_recommendations_interactive.html')
        
        # Create a radar chart for the top 3 counties to visualize their scores across dimensions
        top_3_counties = top_recommendations.head(3).copy()
        
        # Create a figure with subplots (1 row, 3 columns)
        fig2 = make_subplots(
            rows=1, 
            cols=3,
            subplot_titles=[f"{county} (Rank {rank})" for county, rank in zip(top_3_counties['LOCATION'], top_3_counties['Rank'])]
        )
        
        # Add radar charts for each county
        for i, (_, county) in enumerate(top_3_counties.iterrows()):
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
                    marker_color=['cornflowerblue', 'lightcoral', 'mediumseagreen'][i % 3]
                ),
                row=1, 
                col=i+1
            )
        
        # Update layout for all subplots
        fig2.update_layout(
            title_text='Top 3 Recommended Counties - Score Breakdown',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            width=1200,
            height=400
        )
        
        # Save as HTML for interactive viewing
        fig2.write_html('interactive/q13_top3_counties_radar_interactive.html')
    
    # Print top recommendations
    print("\nTop counties for targeted funding:")
    top_n = min(10, len(final_output))
    for idx, row in final_output.head(top_n).iterrows():
        print(f"{row['Rank']}. {row['LOCATION']}: {row['Rationale']}")
    
    return final_output

def main():
    """
    Main function to run the modeling pipeline
    """
    # Get enhanced data from feature engineering module
    master_df = get_enhanced_data()
    
    # Prepare data for modeling with targeted feature groups
    X, y, feature_groups, X_groups = prepare_model_data(master_df)
    
    # Build standard models with all features
    print("\n--- Standard Models with All Features ---")
    best_model, results = build_models(X, y)
    
    # Build targeted models for each feature group
    print("\n--- Targeted Models by Feature Group ---")
    group_results = build_targeted_models(X_groups, y)
    
    # Perform targeted analysis on age-homelessness relationships
    analyze_age_homelessness_relationships(master_df, X_groups, group_results)
    
    # Analyze hospital utilization relationship with homelessness
    analyze_hospital_homelessness_relationship(master_df, X_groups, group_results)
    
    # Analyze cross-category relationships
    analyze_cross_category_relationships(master_df, X_groups, group_results)
    
    # Identify key drivers of homelessness across all datasets
    key_drivers = identify_key_homelessness_drivers(master_df, group_results)
    
    # Analyze targeted predictors across categories
    top_predictors = analyze_targeted_predictors(master_df, X_groups, group_results)
    
    # Find the best model name from standard evaluation
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    
    # Tune the best model
    tuned_model = tune_best_model(X, y, best_model_name, best_model)
    
    # Analyze feature importance for the overall best model
    feature_importance = analyze_feature_importance(X, y, tuned_model, X.columns)
    
    # Analyze residuals
    residual_df = analyze_residuals(X, y, tuned_model, master_df)
    
    # Forecast future trends
    forecast_df = forecast_future_trends(master_df, tuned_model, X.columns)
    
    # Recommend counties for targeted funding
    recommendations = recommend_targeted_funding(master_df, residual_df, forecast_df)
    
    print("\nModeling and analysis complete. Results saved in 'figures' directory.")
    
    # Compile targeted results to return
    targeted_results = {
        'group_results': group_results,
        'top_predictors': top_predictors,
        'key_drivers': key_drivers,
        'age_demographics': group_results.get('age_demographics', {}),
        'gender_demographics': group_results.get('gender_demographics', {}),
        'race_demographics': group_results.get('race_demographics', {}),
        'hospital_utilization': group_results.get('hospital_utilization', {})
    }
    
    return tuned_model, feature_importance, residual_df, forecast_df, recommendations, targeted_results

if __name__ == "__main__":
    main() 