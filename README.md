# Safe Haven _ CrisesCopilot

This project analyzes homelessness data across California counties using demographic information, system performance metrics, hospital utilization data, and other sources to build predictive models and develop county-level insights to support targeted interventions.

## Project Overview

This analysis addresses California's homelessness crisis by providing data-driven insights and recommendations. The pipeline follows a structured approach:

1. **Data Understanding & Preparation**: Loading, cleaning, and integrating data from multiple sources
2. **Exploratory Data Analysis**: Visualizing trends, patterns, and relationships in the data
3. **Feature Engineering**: Creating enhanced indicators and county clustering
4. **Baseline & Advanced Modeling**: Building predictive models and analyzing key drivers
5. **Policy Framing**: Generating targeted funding recommendations
6. **Forecasting & Strategic Insights**: Projecting future homelessness trends

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Ensure all data files are in the root directory:
   - `cy_age.csv`, `cy_race.csv`, `cy_gender.csv` - Demographic data
   - `calendar-year-coc-and-statewide-topline-ca-spms.csv` - System performance data
   - `2021-2022-homeless-hospital-encounters-statewide.csv` - Hospital utilization data (if available)

## Running the Analysis

Execute the main script to run the complete analysis pipeline:
```
python main.py
```

The script will automatically:
- Create all necessary output directories
- Process and clean the data
- Generate all visualizations (Q1-Q13)
- Create interactive HTML visualizations
- Generate a summary report
- Handle errors gracefully if any data sources are missing

## Output Files

### Visualizations

Static visualizations are saved in the `figures/` directory with filenames that correspond to specific analysis questions:

**Data Understanding & Preparation (Q1-Q3)**
- `q1_statewide_homeless_trends.png` - Overall homeless population trends
- `q1_homeless_population_trend.png` - Detailed population trends by county
- `q2_demographic_composition.png` - Demographic breakdown
- `q3_geographic_distribution.png` - Map of homelessness by county

**Exploratory Data Analysis (Q4-Q6)**
- `q4_statewide_age_distribution.png`, `q4_statewide_race_distribution.png`, `q4_statewide_gender_distribution.png`
- `q4_top10_counties_homeless_population.png` - Top counties by homeless population
- `q5_detailed_metrics.png` - System performance metrics analysis
- `q5_normalized_metrics.png` - Normalized metrics for comparison
- `q6_hospital_utilization.png` - Hospital utilization patterns (if data available)

**Feature Engineering (Q7-Q9)**
- `q7_housing_access_burden.png` - Housing access indicators
- `q7_shelter_utilization_rate.png` - Shelter capacity and utilization
- `q8_composite_trend.png` - Composite trend analysis
- `q9_county_clusters.png` - County clustering results
- `q9_cluster_characteristics.png` - Characteristics of each cluster
- `q9_silhouette_scores.png` - Cluster validation

**Modeling & Prediction (Q10-Q11)**
- `q10_model_performance_comparison.png` - Comparison of model performance
- `q10_feature_importance.png` - Feature importance analysis
- `q10_key_homelessness_drivers.png` - Key drivers of homelessness
- `q11_residuals_vs_predicted.png` - Residual analysis
- `q11_actual_vs_predicted.png` - Actual vs predicted values
- `q11_significant_residuals.png` - Counties with significant residuals

**Forecasting & Policy Insight (Q12-Q13)**
- `q12_forecast_2024.png` - Forecasted homelessness for 2024
- `q12_homelessness_forecast.png` - Time series forecast visualization
- `q13_funding_recommendations.png` - Top counties for targeted funding
- `q13_county_1_radar.png`, `q13_county_2_radar.png`, `q13_county_3_radar.png` - Radar charts for top counties

### Interactive Visualizations

Interactive HTML visualizations are saved in the `interactive/` directory:

- `q1_statewide_homeless_trends_interactive.html` - Interactive population trends
- `q1_homeless_population_trend_interactive.html` - Interactive county trends
- `q5_detailed_metrics_interactive.html` - Interactive system metrics
- `q5_normalized_metrics_interactive.html` - Interactive normalized metrics
- `q7_housing_access_burden_interactive.html` - Interactive housing access indicators
- `q8_composite_trend_interactive.html` - Interactive composite trends
- `q9_county_clusters_3d.html` - 3D visualization of county clusters
- `q9_cluster_characteristics_interactive.html` - Interactive cluster characteristics
- `q11_residuals_vs_predicted_interactive.html` - Interactive residual analysis
- `q11_actual_vs_predicted_interactive.html` - Interactive actual vs predicted values
- `q11_significant_residuals_interactive.html` - Interactive significant residuals
- `q12_homelessness_forecast_interactive.html` - Interactive forecast visualization
- `q12_forecast_2024_interactive.html` - Interactive 2024 projections
- `q13_funding_recommendations_interactive.html` - Interactive funding recommendations
- `q13_top3_counties_radar_interactive.html` - Interactive radar chart of top counties
- `q13_county_1_radar_interactive.html`, `q13_county_2_radar_interactive.html`, `q13_county_3_radar_interactive.html` - Individual county radar charts

### Summary Report

A comprehensive summary report is generated in Markdown format:
- `outputs/summary_report.md` - Contains key findings, model performance, and recommendations

### Data Files

Processed and enhanced datasets:
- `master_dataset.csv` - Cleaned and integrated dataset
- `enhanced_dataset.csv` - Dataset with engineered features
- `outputs/funding_recommendations.csv` - County funding recommendations with component scores

## Evaluation Criteria

This project addresses all required evaluation criteria:

1. **Data Understanding & Preparation**
   - Comprehensive data loading, cleaning, and integration
   - Handling of missing data and standardization

2. **Exploratory Data Analysis**
   - Thorough visualization of demographic trends
   - Analysis of system performance metrics
   - Geographic and temporal pattern identification

3. **Feature Engineering**
   - Creation of novel composite indicators
   - County clustering based on shared characteristics
   - Trend analysis and pattern extraction

4. **Baseline & Advanced Modeling**
   - Implementation of multiple regression algorithms
   - Model comparison and evaluation
   - Feature importance analysis
   - Residual analysis to identify outliers

5. **Policy Framing**
   - Data-driven funding recommendations
   - Component-based scoring system
   - Targeted intervention strategies

6. **Forecasting & Strategic Insights (Bonus)**
   - Time series forecasting for 2024
   - County-level projections
   - Change percentage analysis

7. **Timeliness & Clarity of Submission**
   - End-to-end pipeline with single command execution
   - Clear documentation and visualization naming
   - Comprehensive README with all project details

## Technical Implementation

- **Python Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly
- **Modeling Techniques**: Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting
- **Visualization**: Static (matplotlib/seaborn) and interactive (plotly) visualizations
- **Error Handling**: Robust pipeline with fallback mechanisms for missing data

## Acknowledgments

This analysis uses data from the California Department of Housing and Community Development and other public sources.

## License

This project is licensed under the MIT License - see the LICENSE file for details.