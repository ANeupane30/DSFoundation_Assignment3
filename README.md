# HIT140 Assessment 3 - Bat Seasonal Behavior Analysis

## 🦇 Professional Data Analytics Project

This repository contains a comprehensive data analytics pipeline for analyzing bat seasonal behavior patterns and their interactions with rat populations, developed for HIT140 Data Science Foundations Assessment 3.

## 📊 Project Overview

**Objective**: Implement Linear Regression modeling to analyze bat landing behavior across different seasons, with focus on rat interaction patterns and environmental factors.

**Key Features**:
- ✅ Professional modular code architecture
- ✅ Comprehensive feature engineering pipeline  
- ✅ Seasonal Linear Regression modeling
- ✅ Advanced data visualizations
- ✅ Automated analysis workflow
- ✅ Reproducible research methodology

## 🏗️ Project Structure

```
DSFoundation_Assignment3/
│
├── 📁 config/                     # Configuration files
│   └── settings.yaml              # Main configuration settings
│
├── 📁 data/                       # Data storage
│   ├── raw/                       # Original datasets
│   └── processed/                 # Cleaned and engineered data
│
├── 📁 src/                        # Source code modules
│   ├── data/                      # Data processing modules
│   ├── features/                  # Feature engineering
│   ├── models/                    # Model training and evaluation
│   └── visualization/             # Plotting and visualization
│
├── 📁 notebooks/                  # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_analysis.ipynb
│
├── 📁 results/                    # Analysis outputs
│   ├── models/                    # Trained models
│   ├── plots/                     # Generated visualizations  
│   └── reports/                   # Analysis reports
│
├── 📁 tests/                      # Unit tests
│
├── requirements.txt               # Python dependencies
├── main.py                       # Main execution script
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
venv\\Scripts\\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Place your cleaned datasets in data/raw/
# - Dataset1_Cleaner.csv
# - Dataset2_Cleaner.csv
```

### 3. Run Complete Analysis
```bash
python main.py
```

### 4. View Results
- **Models**: `results/models/`
- **Visualizations**: `results/plots/`
- **Report**: `results/reports/analysis_report.html`

## 📈 Analysis Pipeline

### Phase 1: Data Processing
- ✅ Load and validate cleaned datasets
- ✅ Data quality assessment and missing value handling
- ✅ Outlier detection and treatment

### Phase 2: Feature Engineering
- ✅ **Temporal Features**: Rolling averages, seasonal indicators
- ✅ **Seasonal Analysis**: Australian seasonality mapping
- ✅ **Encounter Features**: Bat-rat interaction patterns
- ✅ **Environmental Features**: Food availability indices

### Phase 3: Model Development
- ✅ **Overall Linear Regression**: General bat behavior prediction
- ✅ **Seasonal Models**: Summer vs Autumn comparison
- ✅ **Model Evaluation**: R², MSE, MAE metrics
- ✅ **Feature Importance**: Coefficient analysis

### Phase 4: Results & Visualization
- ✅ **Performance Metrics**: Model comparison charts
- ✅ **Seasonal Patterns**: Distribution and behavior plots
- ✅ **Correlation Analysis**: Feature relationship heatmaps
- ✅ **Residual Analysis**: Model assumption validation

## 🔬 Key Findings

### Model Performance
- **Overall Model**: R² = 0.114 (11.4% variance explained)
- **Summer Model**: R² = 0.127 (Best performing season)
- **Autumn Model**: R² = 0.059 (More variable behavior)

### Behavioral Insights
- **Summer**: Higher bat activity (36.1 avg landings), better predictability
- **Autumn**: Lower activity (27.3 avg landings), more variable patterns
- **Rat Interactions**: Complex seasonal dynamics with opposite effects
- **Food Availability**: Strong correlation with seasonal bat activity

## 📊 Technical Implementation

### Feature Engineering (52 variables created)
- **Temporal**: `month_sin`, `month_cos`, `days_since_season_start`
- **Seasonal**: `is_winter`, `is_spring`, `is_summer`, `is_autumn`
- **Rolling Statistics**: 7-day, 14-day, 30-day moving averages
- **Interactions**: Rat-environment interaction terms

### Model Architecture
- **Algorithm**: Linear Regression with standardized features
- **Validation**: Train-test split (80%-20%)
- **Optimization**: Feature selection and multicollinearity handling

## 🛠️ Module Documentation

### `src/features/feature_engineer.py`
Comprehensive feature engineering pipeline with temporal, seasonal, and interaction features.

### `src/models/regression_models.py`  
Linear regression model training, evaluation, and seasonal comparison functionality.

### `src/visualization/plots.py`
Advanced visualization suite for model results and data exploration.

### `src/data/data_processor.py`
Data loading, cleaning, and preprocessing utilities.

## 📋 Dependencies

- **Core**: `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Configuration**: `pyyaml`, `python-dotenv`
- **Development**: `jupyter`, `pytest`, `black`

## 🎯 Assessment Criteria Fulfillment

✅ **Linear Regression Implementation**: Complete with evaluation metrics  
✅ **Seasonal Analysis**: Summer vs Autumn model comparison  
✅ **Feature Engineering**: 52 engineered variables including rat interactions  
✅ **Data Visualization**: Comprehensive plots and analysis charts  
✅ **Professional Code**: Modular, documented, and reproducible  
✅ **Statistical Validation**: R², MSE, MAE metrics with proper interpretation  

## 📝 Usage Examples

### Basic Analysis
```python
from src.main import BatAnalysisPipeline

pipeline = BatAnalysisPipeline()
results = pipeline.run_complete_analysis()
```

### Custom Feature Engineering
```python
from src.features.feature_engineer import BatSeasonalFeatureEngineer

engineer = BatSeasonalFeatureEngineer(data)
features = engineer.create_all_features()
```

### Seasonal Model Comparison
```python
from src.models.regression_models import SeasonalModelComparator

comparator = SeasonalModelComparator()
results = comparator.compare_seasons(['summer', 'autumn'])
```

## 🤝 Contributing

This is an academic project for HIT140 Assessment 3. Code follows professional data science best practices for educational purposes.

## 📄 License

Academic use only - HIT140 Data Science Foundations, Charles Darwin University

---

**Author**: S396689 - HIT140 Data Science Foundations  
**Date**: October 2025  
**Institution**: Charles Darwin University