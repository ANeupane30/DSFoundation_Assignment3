# HIT140 Assessment 3 - Usage Guide

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- Required packages (install with: `pip install -r requirements.txt`)

### Running the Complete Analysis

1. **Clone/Download the project**
2. **Navigate to the project directory**
   ```bash
   cd DSFoundation_Assignment3
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the complete analysis**
   ```bash
   python main.py
   ```

This will automatically:
- Load and process your datasets
- Engineer 52+ features
- Train Linear Regression models
- Create seasonal comparisons
- Generate visualizations and reports

### Expected Outputs

After running `main.py`, you'll find:

📁 **results/plots/** - All visualization files
- `data_overview.png` - Dataset overview plots
- `correlation_matrix.png` - Feature correlation heatmap
- `seasonal_analysis.png` - Seasonal behavior patterns
- `model_performance.png` - Model evaluation plots
- `seasonal_comparison.png` - Season-to-season comparison

📁 **results/models/** - Trained model files
- `overall_model.joblib` - Main Linear Regression model
- `summer_model.joblib` - Summer-specific model
- `autumn_model.joblib` - Autumn-specific model
- Associated scaler files

📁 **results/reports/**
- `analysis_report.html` - Comprehensive analysis report
- `interactive_dashboard.html` - Interactive dashboard

📁 **data/processed/**
- `engineered_features.csv` - Complete feature set
- `combined_processed_data.csv` - Cleaned and processed data

## 🛠️ Manual Usage (Advanced)

### Step-by-Step Execution

```python
from main import BatAnalysisPipeline

# Initialize pipeline
pipeline = BatAnalysisPipeline()

# Run individual steps
pipeline.load_data()
pipeline.process_data()
pipeline.engineer_features()
pipeline.train_models()
pipeline.create_visualizations()
```

### Using Individual Modules

#### Feature Engineering
```python
from src.features.feature_engineer import BatSeasonalFeatureEngineer

engineer = BatSeasonalFeatureEngineer(df=your_data)
features = engineer.create_all_features()
```

#### Model Training
```python
from src.models.regression_models import BatBehaviorModeler

modeler = BatBehaviorModeler()
X, y = modeler.prepare_modeling_data(df, 'bat_landing_number')
model = modeler.train_linear_regression(X_train, y_train)
```

#### Visualization
```python
from src.visualization.plots import BatBehaviorVisualizer

visualizer = BatBehaviorVisualizer()
visualizer.plot_seasonal_analysis(df, 'bat_landing_number', encounter_cols)
```

## 📊 Data Requirements

The pipeline expects cleaned datasets with columns like:
- `date` - Date column
- `bat_landing_number` - Target variable (bat behavior)
- `rat_arrival_number` - Rat encounter data
- Additional environmental variables

If your data files are not found, the pipeline will create sample data for demonstration.

## ⚙️ Configuration

Modify `config/settings.yaml` to customize:
- Data file paths
- Model parameters
- Visualization settings
- Feature engineering options

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're running from the project root directory
   - Install all requirements: `pip install -r requirements.txt`

2. **Data Not Found**
   - Place your CSV files in `data/raw/` folder
   - Or update paths in `config/settings.yaml`
   - Pipeline creates sample data if files not found

3. **Memory Issues**
   - Reduce `max_features` in config
   - Use smaller dataset for testing

### Getting Help

- Check the console output for detailed progress logs
- Review `results/analysis.log` for technical details
- Ensure all dependencies are installed correctly

## 📈 Expected Results

### Model Performance
- **Overall Model**: R² ≈ 0.11-0.14
- **Seasonal Models**: Varying performance by season
- **Feature Importance**: Top predictors identified

### Key Findings
- Seasonal differences in bat behavior
- Rat interaction patterns vary by season
- Food availability strongly correlates with activity
- Summer typically shows better model predictability

## 🎯 Assessment Requirements Fulfilled

✅ **Linear Regression Implementation**: Complete with evaluation metrics  
✅ **Seasonal Analysis**: Summer vs Autumn model comparison  
✅ **Feature Engineering**: 52 engineered variables  
✅ **Data Visualization**: Comprehensive plots and dashboards  
✅ **Professional Code**: Modular, documented, reproducible  
✅ **Statistical Validation**: R², MSE, MAE with proper interpretation  

---

**Need help?** Check the README.md for more detailed information!