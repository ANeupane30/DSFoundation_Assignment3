# API Documentation - HIT140 Assessment 3

## 🔧 Module Documentation

### BatSeasonalFeatureEngineer

**Location**: `src/features/feature_engineer.py`

```python
class BatSeasonalFeatureEngineer:
    """Comprehensive feature engineering for bat behavior analysis."""
    
    def __init__(self, data_path=None, df=None):
        """Initialize with data path or DataFrame."""
    
    def analyze_data_structure(self) -> pd.DataFrame:
        """Analyze dataset structure and identify column types."""
    
    def prepare_temporal_features(self, date_col=None, date_format=None):
        """Create temporal features from date column."""
    
    def create_seasonal_features(self, location="australia"):
        """Create seasonal features based on geographic location."""
    
    def create_encounter_features(self, encounter_cols=None):
        """Create encounter-based features and rolling statistics."""
    
    def create_seasonal_interaction_features(self):
        """Create interaction features between seasonal and encounter variables."""
    
    def get_final_feature_set(self, save_path=None) -> pd.DataFrame:
        """Get final engineered feature set."""
    
    def create_all_features(self) -> pd.DataFrame:
        """Run complete feature engineering pipeline."""
```

### BatBehaviorModeler

**Location**: `src/models/regression_models.py`

```python
class BatBehaviorModeler:
    """Linear regression modeling for bat behavior analysis."""
    
    def __init__(self, random_state=42):
        """Initialize modeler with random state."""
    
    def prepare_modeling_data(self, df, target_column, feature_columns=None):
        """Prepare data for modeling by selecting features and target."""
    
    def train_linear_regression(self, X_train, y_train, model_name="overall"):
        """Train a Linear Regression model."""
    
    def evaluate_model(self, model, X_test, y_test, scaler=None):
        """Evaluate trained model on test data."""
    
    def train_seasonal_models(self, df, target_column, seasons=['summer', 'autumn']):
        """Train separate models for different seasons."""
    
    def compare_seasonal_models(self, seasonal_results):
        """Compare performance across seasonal models."""
    
    def save_models(self, output_dir):
        """Save trained models to disk."""
```

### BatBehaviorVisualizer

**Location**: `src/visualization/plots.py`

```python
class BatBehaviorVisualizer:
    """Comprehensive visualization for bat behavior analysis."""
    
    def __init__(self, style='seaborn-v0_8', color_palette='husl'):
        """Initialize visualizer with styling preferences."""
    
    def plot_data_overview(self, df, target_column, save_plots=True):
        """Create comprehensive data overview plots."""
    
    def plot_correlation_matrix(self, df, target_column, max_features=20):
        """Create correlation matrix heatmap."""
    
    def plot_seasonal_analysis(self, df, target_column, encounter_columns):
        """Create comprehensive seasonal analysis plots."""
    
    def plot_model_performance(self, model_results, save_plots=True):
        """Create model performance comparison plots."""
    
    def create_interactive_dashboard(self, df, target_column, model_results):
        """Create interactive dashboard using Plotly."""
    
    def generate_analysis_report(self, df, model_results):
        """Generate comprehensive HTML analysis report."""
```

### DataProcessor

**Location**: `src/data/data_processor.py`

```python
class DataProcessor:
    """Data processing for bat behavior analysis."""
    
    def __init__(self, config_path=None):
        """Initialize data processor with configuration."""
    
    def load_dataset(self, file_path, dataset_name, encoding='utf-8'):
        """Load a single dataset from file."""
    
    def validate_data_quality(self, df, dataset_name):
        """Perform comprehensive data quality assessment."""
    
    def handle_missing_values(self, df, strategy='auto', threshold=0.5):
        """Handle missing values using specified strategy."""
    
    def handle_outliers(self, df, method='iqr', action='cap'):
        """Handle outliers in numeric columns."""
    
    def combine_datasets(self, datasets=None, join_method='outer'):
        """Combine multiple datasets into single DataFrame."""
```

## 📊 Data Structures

### ModelMetrics
```python
@dataclass
class ModelMetrics:
    r2_score: float
    mse: float
    mae: float
    rmse: float
    explained_variance: float
    max_error: float
    n_samples: int
    n_features: int
```

### ModelResults
```python
@dataclass
class ModelResults:
    model: LinearRegression
    metrics: ModelMetrics
    feature_importance: pd.DataFrame
    predictions: np.ndarray
    residuals: np.ndarray
    scaler: StandardScaler
    feature_names: List[str]
```

## 🔧 Configuration Schema

### settings.yaml Structure
```yaml
project:
  name: "HIT140_Assessment3_BatSeasonalBehavior"
  version: "1.0.0"

data:
  raw_data_path: "data/raw/"
  processed_data_path: "data/processed/"
  datasets:
    dataset1_cleaned: "Dataset1_Cleaner.csv"
    dataset2_cleaned: "Dataset2_Cleaner.csv"

feature_engineering:
  temporal:
    rolling_windows: [7, 14, 30]
  seasonal:
    summer_months: [12, 1, 2]
    # ... etc

modeling:
  linear_regression:
    test_size: 0.2
    random_state: 42
    normalize_features: true
```

## 📈 Feature Categories

### Temporal Features
- `year`, `month`, `day`, `day_of_year`, `weekday`, `week`
- `month_sin`, `month_cos` (cyclical encoding)
- `days_since_season_start`

### Seasonal Features
- `season` (categorical)
- `is_winter`, `is_spring`, `is_summer`, `is_autumn` (binary indicators)
- `food_scarcity_index`, `food_abundance_index`
- `is_transition_period`

### Encounter Features
- Rolling averages: `*_rolling_7d`, `*_rolling_14d`, `*_rolling_30d`
- Rolling standard deviations: `*_std_7d`, `*_std_14d`, `*_std_30d`
- Binary indicators: `has_*`
- Temporal: `days_since_last_*`
- Categories: `*_frequency`

### Interaction Features
- `winter_encounters`, `spring_encounters`
- `scarcity_encounter_ratio`, `abundance_encounter_product`
- `prev_month_encounters`, `prev_season_effect`
- `winter_rolling_encounters`

## 🎯 Usage Examples

### Basic Pipeline
```python
from main import BatAnalysisPipeline

pipeline = BatAnalysisPipeline()
results = pipeline.run_complete_analysis()
print(f"Best model: {results['best_model']}")
```

### Custom Feature Engineering
```python
from src.features.feature_engineer import BatSeasonalFeatureEngineer

engineer = BatSeasonalFeatureEngineer(df=data)
features = engineer.create_all_features(
    date_col='date',
    location='australia'
)
```

### Custom Modeling
```python
from src.models.regression_models import BatBehaviorModeler

modeler = BatBehaviorModeler(random_state=42)
X, y = modeler.prepare_modeling_data(df, 'bat_landing_number')
X_selected = modeler.feature_selection(X, y, max_features=12)
```

### Custom Visualization
```python
from src.visualization.plots import BatBehaviorVisualizer

viz = BatBehaviorVisualizer(style='seaborn-v0_8')
fig = viz.plot_seasonal_analysis(df, 'bat_landing_number', encounter_cols)
```

## 🔍 Error Handling

All modules include comprehensive error handling:
- **FileNotFoundError**: When data files are missing
- **ValueError**: For invalid parameters or data structure issues  
- **RuntimeError**: For processing failures
- **KeyError**: For missing required columns

## 📝 Logging

All modules use Python's logging framework:
- **INFO**: Progress updates and key findings
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Critical failures requiring attention

Log output saved to: `results/analysis.log`

---

**For more examples, see the `examples/` directory and `USAGE.md`**