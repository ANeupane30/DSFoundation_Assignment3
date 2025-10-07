#!/usr/bin/env python3
"""
Debug version of the main pipeline to identify execution issues
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

print(f"[{datetime.now()}] Starting debug pipeline...")
print(f"Current working directory: {os.getcwd()}")

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    print("Importing modules...")
    from src.data.data_processor import DataProcessor
    from src.features.feature_engineer import BatSeasonalFeatureEngineer
    from src.models.regression_models import BatBehaviorModeler
    from src.visualization.plots import BatBehaviorVisualizer
    print("✅ All modules imported successfully")
    
    # Create a simple test pipeline
    print("\nCreating test pipeline...")
    
    # Step 1: Initialize components
    print("1. Initializing components...")
    data_processor = DataProcessor()
    print("✅ DataProcessor initialized")
    
    # Step 2: Load existing engineered features data
    print("2. Loading existing engineered features data...")
    data_file = "bat_rat_features_engineered.csv"
    if os.path.exists(data_file):
        df_features = pd.read_csv(data_file)
        print(f"✅ Data loaded from {data_file} with shape: {df_features.shape}")
        print(f"Columns: {list(df_features.columns)[:10]}...")  # Show first 10 columns
    else:
        print(f"❌ File {data_file} not found")
        # Fallback: create simple sample data
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n_samples = 1000
        df_features = pd.DataFrame({
            'month': np.random.randint(1, 13, n_samples),
            'day_of_year': np.random.randint(1, 366, n_samples),
            'season_encoded': np.random.randint(0, 4, n_samples),
            'temperature': np.random.normal(20, 5, n_samples),
            'humidity': np.random.uniform(40, 90, n_samples),
            'wind_speed': np.random.exponential(2, n_samples),
            'encounters': np.random.poisson(3, n_samples),
            'duration': np.random.exponential(5, n_samples),
            'activity_level': np.random.uniform(0, 10, n_samples)
        })
        print(f"✅ Sample data created with shape: {df_features.shape}")
    
    # Step 3: Initialize other components
    print("3. Initializing modeler and visualizer...")
    modeler = BatBehaviorModeler()
    visualizer = BatBehaviorVisualizer()
    print("✅ All components initialized")
    
    # Step 4: Train models
    print("4. Training models...")
    results = modeler.train_models(df_features)
    print(f"✅ Models trained. Results keys: {list(results.keys())}")
    
    # Display key results
    print("\n=== FINAL RESULTS ===")
    if 'overall_model' in results:
        overall_r2 = results['overall_model'].get('r2_score', 'N/A')
        print(f"Overall Model R² Score: {overall_r2}")
    
    if 'seasonal_models' in results:
        seasonal = results['seasonal_models']
        for season, model_data in seasonal.items():
            r2 = model_data.get('r2_score', 'N/A')
            print(f"{season} Model R² Score: {r2}")
    
    # Step 5: Create basic visualization
    print("\n5. Creating visualizations...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    try:
        visualizer.plot_data_overview(df_features, save_path=str(results_dir / "data_overview.png"))
        print("✅ Data overview plot created")
    except Exception as viz_error:
        print(f"Warning: Visualization failed: {viz_error}")
    
    print(f"\n🎉 Pipeline completed successfully at {datetime.now()}")
    print(f"Check the 'results' directory for outputs!")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    print(f"Traceback:\n{traceback.format_exc()}")
    sys.exit(1)