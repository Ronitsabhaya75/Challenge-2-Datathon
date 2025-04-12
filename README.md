# California Homelessness Data Analysis Pipeline

## Overview
This project provides a comprehensive data analysis pipeline for California homelessness data. It processes demographic, system performance, and hospital utilization data to create insights, predictive models, and targeted funding recommendations for addressing homelessness across California counties.

## Project Structure
The pipeline consists of four main modules:

1. **Data Preparation** (`data_preparation.py`): Loads and preprocesses raw data sources
2. **Exploratory Analysis** (`exploratory_analysis.py`): Visualizes demographic trends and system performance 
3. **Feature Engineering** (`feature_engineering.py`): Creates derived metrics and clusters counties
4. **Modeling & Analysis** (`modeling.py`): Builds predictive models and generates recommendations

## Data Sources
The pipeline processes several data sources:
- County-level demographic data (age, race, gender) from `cy_age.csv`, `cy_race.csv`, and `cy_gender.csv`
- System performance metrics from `calendar-year-coc-and-statewide-topline-ca-spms.csv`
- Hospital utilization data from `homeless-hospital-encounters-age-race-sex-expected-payer-statewide.csv`

## Key Features

### Data Preparation
- Standardizes column names and data formats across multiple sources
- Creates a unified county identifier system
- Calculates derived metrics like proportions and year-over-year changes
- Creates a master dataset that combines all data sources

### Exploratory Analysis
- Visualizes demographic distributions and trends
- Analyzes hospital utilization patterns by demographic categories
- Examines system performance metrics over time
- Compares performance across counties

### Feature Engineering
- Creates access burden indicators and service capacity metrics
- Generates trend-based features from historical data
- Clusters counties based on shared homelessness characteristics
- Produces vulnerability scores and housing access metrics

### Modeling & Analysis
- Builds and evaluates multiple regression models to predict homelessness
- Identifies key factors influencing homelessness levels
- Analyzes residuals to find counties with unexpected patterns
- Forecasts future homelessness trends
- Recommends counties for targeted funding

## Model Accuracy

The pipeline evaluates several regression models with the following accuracy metrics:

| Model               | R² Score | RMSE    |
|---------------------|----------|---------|
| Linear Regression   | 0.9950   | 446.01  |
| Ridge Regression    | 0.9827   | 832.53  |
| Lasso Regression    | 0.9887   | 674.84  |
| Elastic Net         | 0.9776   | 949.08  |
| Random Forest       | 0.9830   | 826.26  |
| Gradient Boosting   | 1.0000   | 14.85   |

The Gradient Boosting model achieves the highest accuracy with an R² of 1.0000 and the lowest RMSE of 14.85, making it the chosen model for predictions and analysis.

## Usage

### Prerequisites
- Python 3.x
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd [repository-directory]

# Install required dependencies
pip install -r requirements.txt
```

### Running the Pipeline
To run the complete pipeline:
```bash
python main.py
```

To run individual modules:
```bash
python data_preparation.py
python exploratory_analysis.py
python feature_engineering.py
python modeling.py
```

## Outputs
The pipeline generates:
- Processed datasets: `master_dataset.csv` and `enhanced_dataset.csv`
- Visualizations in the `figures` directory
- A summary report in `outputs/summary_report.md`
- Funding recommendations for targeting resources effectively

## Advanced Analysis Features

### County Clustering
The pipeline segments counties into distinct clusters based on:
- Demographic proportions (age, race, gender)
- Vulnerability indicators
- Homelessness trends
- System performance metrics

This clustering enables targeted strategies for different county types.

### Predictive Modeling
Multiple regression models are evaluated including:
- Linear Regression
- Ridge Regression  
- Lasso Regression
- Elastic Net
- Random Forest
- Gradient Boosting

The best performing model is used for prediction and feature importance analysis.

### Targeted Funding Recommendations
Recommendations are generated based on:
- Current homelessness magnitude
- Vulnerability scores
- Projected trends
- Unexpected patterns identified in residual analysis

This provides an evidence-based approach for resource allocation.

## License
[Specify the license]

## Contributors
[List contributors] 