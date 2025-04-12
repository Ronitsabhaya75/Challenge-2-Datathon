import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from feature_engineering import main as get_enhanced_data

# Set Seaborn style
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

def prepare_model_data(master_df):
    """
    Prepare data for modeling
    """
    print("Preparing data for modeling...")
    
    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # Our target variable is the total homeless population
    y = master_df['TOTAL_HOMELESS']
    
    # Select features for the model
    # Demographic proportions
    demo_cols = [col for col in master_df.columns if col.endswith('_PROP')]
    
    # Vulnerability and access indicators
    indicator_cols = [
        'HOUSING_ACCESS_BURDEN', 
        'SHELTER_UTILIZATION', 
        'VULNERABLE_AGE_PROP', 
        'VULNERABILITY_SCORE'
    ]
    indicator_cols = [col for col in indicator_cols if col in master_df.columns]
    
    # Trend features
    trend_cols = [col for col in master_df.columns if col.endswith('_TREND')]
    
    # System performance metrics (latest values)
    spm_cols = [col for col in master_df.columns if col.endswith('_latest')]
    
    # Combine all features
    feature_cols = demo_cols + indicator_cols + trend_cols + spm_cols
    
    # Filter to features that exist in the dataset
    feature_cols = [col for col in feature_cols if col in master_df.columns]
    
    # Create feature matrix
    X = master_df[feature_cols].copy()
    
    # Handle missing values and infinities
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y, feature_cols

def build_models(X, y):
    """
    Build and evaluate multiple regression models
    """
    print("Building and evaluating models...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    plt.savefig('figures/model_performance_comparison.png')
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
    plt.savefig('figures/feature_importance.png')
    plt.close()
    
    return feature_importance

def analyze_residuals(X, y, best_model, master_df):
    """
    Analyze counties where model predictions deviate significantly from actual values
    """
    print("Analyzing model residuals...")
    
    # Make predictions on the full dataset
    y_pred = best_model.predict(X)
    
    # Calculate residuals (actual - predicted)
    residuals = y - y_pred
    
    # Calculate standardized residuals
    std_residuals = (residuals - residuals.mean()) / residuals.std()
    
    # Add to dataframe
    residual_df = pd.DataFrame({
        'County': master_df['LOCATION'],
        'Actual': y,
        'Predicted': y_pred,
        'Residual': residuals,
        'Std_Residual': std_residuals
    })
    
    # Flag significant outliers (|std_residual| > 2)
    residual_df['Is_Outlier'] = np.abs(residual_df['Std_Residual']) > 2
    
    # Visualize residuals
    plt.figure(figsize=(14, 10))
    
    plt.subplot(221)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted Values', fontsize=14)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    
    plt.subplot(222)
    plt.scatter(y_pred, y, alpha=0.6)
    # Add a line representing perfect prediction
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Actual vs Predicted Values', fontsize=14)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Actual Values', fontsize=12)
    
    plt.subplot(223)
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals', fontsize=14)
    plt.xlabel('Residual', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    plt.subplot(224)
    # Create a bar plot of counties with largest absolute standardized residuals
    top_outliers = residual_df.loc[residual_df['Is_Outlier']].sort_values('Std_Residual', key=abs, ascending=False)
    
    if len(top_outliers) > 0:
        colors = ['red' if val > 0 else 'blue' for val in top_outliers['Std_Residual']]
        sns.barplot(data=top_outliers, x='County', y='Std_Residual', hue='County', palette=colors, legend=False)
        plt.title('Counties with Significant Residuals', fontsize=14)
        plt.xlabel('County', fontsize=12)
        plt.ylabel('Standardized Residual', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    else:
        plt.text(0.5, 0.5, 'No significant outliers found', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Counties with Significant Residuals', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figures/residual_analysis.png')
    plt.close()
    
    # Print counties with largest residuals
    if len(top_outliers) > 0:
        print("\nCounties with significant prediction deviations:")
        for i, row in top_outliers.iterrows():
            direction = "under-predicted" if row['Residual'] > 0 else "over-predicted"
            print(f"  {row['County']}: Actual = {row['Actual']:.0f}, Predicted = {row['Predicted']:.0f}, "
                  f"{direction} by {abs(row['Residual']):.0f} ({row['Std_Residual']:.2f} std)")
    
    return residual_df

def forecast_future_trends(master_df, best_model, feature_cols):
    """
    Forecast key homelessness indicators for 2024
    """
    print("Forecasting homelessness trends for 2024...")
    
    # We need historical data to make projections
    # For simplicity, we'll assume a linear trend continuation from 2020-2023
    
    # Check if we have trend columns
    trend_cols = [col for col in master_df.columns if col.endswith('_TREND')]
    
    if len(trend_cols) == 0:
        print("Not enough trend data available for forecasting")
        return pd.DataFrame()
    
    # For each county, project 2024 values based on latest data and trends
    forecast_df = master_df[['LOCATION_ID', 'LOCATION', 'TOTAL_HOMELESS']].copy()
    
    # Assuming an average annual change based on trends
    # We'll create a simple projection by applying the trend factors
    
    # Get the latest year SPM metrics ending with '_latest'
    latest_cols = [col for col in master_df.columns if col.endswith('_latest')]
    
    if len(latest_cols) > 0:
        # Calculate projected 2024 values
        for col in latest_cols:
            # Get base metric name
            metric = col.replace('_latest', '')
            
            # Check if we have a trend for this metric
            trend_col = f'{metric}_TREND'
            
            if trend_col in master_df.columns:
                # Project 2024 value using the trend
                forecast_df[f'{metric}_2024'] = master_df[col] * (1 + master_df[trend_col])
    
    # Predict homeless population for 2024
    # First, prepare the feature data
    forecast_features = master_df[feature_cols].copy()
    
    # Handle missing values and infinities
    forecast_features = forecast_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Make predictions for 2023 (current)
    y_pred_2023 = best_model.predict(forecast_features)
    
    # For 2024 prediction, adjust the feature values based on trends
    forecast_features_2024 = forecast_features.copy()
    
    # Apply trend factors to each feature that has a trend
    for col in feature_cols:
        # Check if this feature has a corresponding trend
        base_name = col.split('_')[0] if '_' in col else col
        trend_col = f'{base_name}_TREND'
        
        if trend_col in master_df.columns:
            # Apply the trend to project 2024 value
            forecast_features_2024[col] = forecast_features[col] * (1 + master_df[trend_col])
    
    # Make predictions for 2024
    y_pred_2024 = best_model.predict(forecast_features_2024)
    
    # Add predictions to forecast dataframe
    forecast_df['PREDICTED_2023'] = y_pred_2023
    forecast_df['PREDICTED_2024'] = y_pred_2024
    forecast_df['CHANGE_PCT'] = (forecast_df['PREDICTED_2024'] - forecast_df['PREDICTED_2023']) / forecast_df['PREDICTED_2023'] * 100
    
    # Visualize forecasted changes for top counties
    top_counties = forecast_df.nlargest(10, 'TOTAL_HOMELESS')
    
    plt.figure(figsize=(16, 10))
    
    # Create a bar chart showing 2023 vs 2024
    x = np.arange(len(top_counties))
    width = 0.35
    
    plt.bar(x - width/2, top_counties['PREDICTED_2023'], width, label='2023 (Predicted)', color='cornflowerblue')
    plt.bar(x + width/2, top_counties['PREDICTED_2024'], width, label='2024 (Forecasted)', color='lightcoral')
    
    plt.title('Forecasted Homeless Population for 2024 - Top 10 Counties', fontsize=16)
    plt.xlabel('County', fontsize=14)
    plt.ylabel('Homeless Population', fontsize=14)
    plt.xticks(x, top_counties['LOCATION'], rotation=45, ha='right')
    plt.legend()
    
    # Add percentage change labels
    for i, county in enumerate(top_counties['LOCATION']):
        change_pct = top_counties['CHANGE_PCT'].iloc[i]
        y_pos = max(top_counties['PREDICTED_2023'].iloc[i], top_counties['PREDICTED_2024'].iloc[i])
        plt.text(i, y_pos + 100, f'{change_pct:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/homelessness_forecast_2024.png')
    plt.close()
    
    # Identify counties with highest projected increases
    increasing_counties = forecast_df.sort_values('CHANGE_PCT', ascending=False).head(5)
    
    print("\nCounties with highest projected increases in homelessness for 2024:")
    for i, row in increasing_counties.iterrows():
        print(f"  {row['LOCATION']}: {row['PREDICTED_2023']:.0f} → {row['PREDICTED_2024']:.0f} ({row['CHANGE_PCT']:.1f}%)")
    
    # Identify counties with decreases or lowest increases
    decreasing_counties = forecast_df.sort_values('CHANGE_PCT', ascending=True).head(5)
    
    print("\nCounties with decreases or lowest increases in homelessness for 2024:")
    for i, row in decreasing_counties.iterrows():
        print(f"  {row['LOCATION']}: {row['PREDICTED_2023']:.0f} → {row['PREDICTED_2024']:.0f} ({row['CHANGE_PCT']:.1f}%)")
    
    return forecast_df

def recommend_targeted_funding(master_df, residual_df, forecast_df):
    """
    Recommend counties for targeted funding based on current homelessness,
    vulnerability, projected change, and model residuals.
    
    Parameters:
    -----------
    master_df : DataFrame
        Master dataset with county information
    residual_df : DataFrame
        DataFrame with model residuals
    forecast_df : DataFrame
        DataFrame with forecasted homelessness values
        
    Returns:
    --------
    DataFrame
        Top recommended counties for funding with scores
    """
    # Merge relevant data
    recommendation_df = master_df[['LOCATION_ID', 'LOCATION', 'TOTAL_HOMELESS']].copy()
    recommendation_df = recommendation_df.merge(residual_df[['County', 'Std_Residual']], left_on='LOCATION', right_on='County', how='left')
    
    # Add forecasted change
    if 'CHANGE_PCT' in forecast_df.columns:
        recommendation_df = recommendation_df.merge(
            forecast_df[['LOCATION_ID', 'CHANGE_PCT']], 
            on='LOCATION_ID', 
            how='left'
        )
    else:
        recommendation_df['CHANGE_PCT'] = 0
    
    # Create a composite score for prioritization
    # Standardize each component first
    recommendation_df['TOTAL_HOMELESS_SCALED'] = (recommendation_df['TOTAL_HOMELESS'] - recommendation_df['TOTAL_HOMELESS'].mean()) / recommendation_df['TOTAL_HOMELESS'].std()
    
    # Fix clip usage by using numpy directly
    change_std = max(recommendation_df['CHANGE_PCT'].std(), 0.0001) # Avoid division by zero
    recommendation_df['CHANGE_PCT_SCALED'] = (recommendation_df['CHANGE_PCT'] - recommendation_df['CHANGE_PCT'].mean()) / change_std
    
    if 'VULNERABILITY_SCORE' in recommendation_df.columns:
        vulnerability_std = max(recommendation_df['VULNERABILITY_SCORE'].std(), 0.0001) # Avoid division by zero
        recommendation_df['VULNERABILITY_SCORE_SCALED'] = (recommendation_df['VULNERABILITY_SCORE'] - recommendation_df['VULNERABILITY_SCORE'].mean()) / vulnerability_std
    else:
        # If vulnerability score is not available, create a dummy column
        recommendation_df['VULNERABILITY_SCORE'] = 0
        recommendation_df['VULNERABILITY_SCORE_SCALED'] = 0
    
    # For residuals, we only want to prioritize under-predicted counties
    recommendation_df['RESIDUAL_FACTOR'] = np.maximum(recommendation_df['Std_Residual'], 0)
    if recommendation_df['RESIDUAL_FACTOR'].max() > 0:
        recommendation_df['RESIDUAL_FACTOR'] = recommendation_df['RESIDUAL_FACTOR'] / recommendation_df['RESIDUAL_FACTOR'].max()
    
    # Calculate composite score with weights
    recommendation_df['PRIORITY_SCORE'] = (
        0.4 * recommendation_df['TOTAL_HOMELESS_SCALED'] +
        0.3 * recommendation_df['CHANGE_PCT_SCALED'] +
        0.2 * recommendation_df['VULNERABILITY_SCORE_SCALED'] +
        0.1 * recommendation_df['RESIDUAL_FACTOR']
    )
    
    # Sort by priority score
    recommendation_df = recommendation_df.sort_values('PRIORITY_SCORE', ascending=False)
    
    # Get top 3 recommended counties
    top_recommendations = recommendation_df.head(3)
    
    print("\nTop 3 Recommended Counties for Targeted Funding:")
    for i, row in top_recommendations.iterrows():
        print(f"\n{i+1}. {row['LOCATION']}")
        print(f"   - Current homeless population: {row['TOTAL_HOMELESS']:.0f}")
        print(f"   - Projected change in 2024: {row['CHANGE_PCT']:.1f}%")
        print(f"   - Vulnerability score: {row['VULNERABILITY_SCORE']:.2f}")
        
        # Add specific reasons based on highest factors
        factors = []
        if row['TOTAL_HOMELESS_SCALED'] > 0.5:
            factors.append("Large current homeless population")
        if row['CHANGE_PCT_SCALED'] > 0.5:
            factors.append("High projected increase in homelessness")
        if 'VULNERABILITY_SCORE' in master_df.columns and row['VULNERABILITY_SCORE_SCALED'] > 0.5:
            factors.append("High vulnerability score")
        if row['RESIDUAL_FACTOR'] > 0.2:
            factors.append("Under-predicted by model (potential unmet needs)")
        
        if not factors:
            factors.append("Overall priority based on combined factors")
            
        print(f"   - Key factors: {', '.join(factors)}")
    
    # Visualize recommended counties
    plt.figure(figsize=(14, 10))
    
    # Create a horizontal bar chart showing priority scores
    bars = plt.barh(top_recommendations['LOCATION'], top_recommendations['PRIORITY_SCORE'], color='mediumseagreen')
    plt.title('Top 3 Counties Recommended for Targeted Funding', fontsize=16)
    plt.xlabel('Priority Score', fontsize=14)
    plt.ylabel('County', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    
    # Add score labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                 va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/funding_recommendations.png')
    plt.close()
    
    return top_recommendations

def main():
    """
    Main function to run the modeling pipeline
    """
    # Get enhanced data from feature engineering module
    master_df = get_enhanced_data()
    
    # Prepare data for modeling
    X, y, feature_cols = prepare_model_data(master_df)
    
    # Build and evaluate models
    best_model, results = build_models(X, y)
    
    # Find the best model name
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    
    # Tune the best model
    tuned_model = tune_best_model(X, y, best_model_name, best_model)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(X, y, tuned_model, feature_cols)
    
    # Analyze residuals
    residual_df = analyze_residuals(X, y, tuned_model, master_df)
    
    # Forecast future trends
    forecast_df = forecast_future_trends(master_df, tuned_model, feature_cols)
    
    # Recommend counties for targeted funding
    recommendations = recommend_targeted_funding(master_df, residual_df, forecast_df)
    
    print("\nModeling and analysis complete. Results saved in 'figures' directory.")
    
    return tuned_model, feature_importance, residual_df, forecast_df, recommendations

if __name__ == "__main__":
    main() 