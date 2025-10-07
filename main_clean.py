#!/usr/bin/env python3
"""
Clean Main Analysis Pipeline for Bat Seasonal Behavior Analysis
HIT140 Assessment 3 - Professional Data Analytics Pipeline

This script runs the complete analysis using the existing engineered features.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
from src.features.feature_engineer import BatSeasonalFeatureEngineer
from src.models.regression_models import BatBehaviorModeler
from src.visualization.plots import BatBehaviorVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("🚀 STARTING BAT SEASONAL BEHAVIOR ANALYSIS")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Create results directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
        os.makedirs('results/models', exist_ok=True)
        
        # Step 1: Load the existing engineered features
        logger.info("STEP 1: Loading engineered features data")
        
        # Try to find the engineered features file
        feature_files = [
            'bat_rat_features_engineered.csv',
            'data/processed/engineered_features.csv',
            'results/engineered_features.csv'
        ]
        
        df = None
        for file_path in feature_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                logger.info(f"✅ Loaded features from: {file_path}")
                logger.info(f"   Data shape: {df.shape}")
                break
        
        if df is None:
            raise FileNotFoundError("No engineered features file found!")
        
        # Step 2: Prepare data for modeling
        logger.info("STEP 2: Preparing data for modeling")
        
        # Initialize modeler
        modeler = BatBehaviorModeler(random_state=42)
        
        # Determine target column
        target_candidates = ['bat_landing_number', 'bat_landings', 'landings']
        target_column = None
        for candidate in target_candidates:
            if candidate in df.columns:
                target_column = candidate
                break
        
        if target_column is None:
            raise ValueError(f"No target column found. Available columns: {list(df.columns)}")
        
        logger.info(f"Using target column: {target_column}")
        
        # Prepare features and target
        exclude_cols = ['date', 'season_start'] + [col for col in df.columns if 'frequency' in col.lower()]
        
        X, y = modeler.prepare_modeling_data(
            df, 
            target_column,
            exclude_columns=exclude_cols
        )
        
        logger.info(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        
        # Step 3: Train models
        logger.info("STEP 3: Training Linear Regression models")
        
        # Feature selection if too many features
        if len(X.columns) > 15:
            X = modeler.feature_selection(X, y, max_features=15)
            logger.info(f"Selected top {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = modeler.split_data(X, y, test_size=0.2)
        
        # Train overall model
        logger.info("Training overall Linear Regression model")
        overall_model = modeler.train_linear_regression(X_train, y_train, model_name="overall")
        overall_metrics = modeler.evaluate_model(
            overall_model, X_test, y_test, 
            modeler.scalers["overall"], "overall"
        )
        overall_importance = modeler.get_feature_importance(overall_model, X.columns.tolist())
        
        # Store results (create custom result objects)
        model_results = {
            "overall": {
                'model': overall_model,
                'metrics': overall_metrics,
                'feature_importance': overall_importance
            }
        }
        
        # Train seasonal models if season column exists
        if 'season' in df.columns:
            logger.info("Training seasonal Linear Regression models")
            seasonal_results = modeler.train_seasonal_models(
                df,
                target_column,
                feature_columns=X.columns.tolist(),
                seasons=['summer', 'autumn']
            )
            model_results.update(seasonal_results)
        
        # Step 4: Display results
        logger.info("STEP 4: Displaying model results")
        logger.info("=" * 60)
        
        # Display model performance
        for model_name, results in model_results.items():
            # Handle both dictionary format and ModelResults objects
            if hasattr(results, 'metrics'):  # ModelResults object
                metrics = results.metrics
            else:  # Dictionary format
                metrics = results['metrics']
                
            logger.info(f"📊 {model_name.upper()} MODEL RESULTS:")
            logger.info(f"   R² Score: {metrics.r2_score:.4f}")
            logger.info(f"   Mean Absolute Error: {metrics.mae:.4f}")
            logger.info(f"   Root Mean Square Error: {metrics.rmse:.4f}")
            logger.info("")
        
        # Display feature importance for overall model
        if 'overall' in model_results:
            results = model_results['overall']
            if hasattr(results, 'feature_importance'):  # ModelResults object
                importance_df = results.feature_importance
            else:  # Dictionary format
                importance_df = results['feature_importance']
                
            logger.info("🔍 TOP 10 MOST IMPORTANT FEATURES (Overall Model):")
            for idx, row in importance_df.head(10).iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.4f}")
            logger.info("")
        
        # Step 5: Create basic visualizations
        logger.info("STEP 5: Creating visualizations")
        
        visualizer = BatBehaviorVisualizer()
        
        # Create data overview
        visualizer.plot_data_overview(df, target_column)
        logger.info("✅ Data overview plot created")
        
        # Create model performance plots
        visualizer.plot_model_performance(model_results)
        logger.info("✅ Model performance plots created")
        
        # Create correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]  # Limit to avoid crowding
        visualizer.plot_correlation_matrix(df[numeric_cols], target_column)
        logger.info("✅ Correlation matrix created")
        
        # Save models
        modeler.save_models('results/models')
        logger.info("✅ Models saved")
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # Final summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total execution time: {execution_time}")
        logger.info(f"📊 Dataset analyzed: {df.shape[0]} samples, {df.shape[1]} features")
        logger.info(f"🤖 Models trained: {len(model_results)}")
        
        # Find best model
        def get_r2_score(x):
            results = model_results[x]
            if hasattr(results, 'metrics'):  # ModelResults object
                return results.metrics.r2_score
            else:  # Dictionary format
                return results['metrics'].r2_score
                
        best_model = max(model_results.keys(), key=get_r2_score)
        best_r2 = get_r2_score(best_model)
        logger.info(f"🏆 Best performing model: {best_model} (R² = {best_r2:.4f})")
        
        logger.info("")
        logger.info("📁 Output files generated:")
        logger.info("   📊 results/plots/ - Visualization plots")
        logger.info("   🤖 results/models/ - Trained model files")
        logger.info("")
        
        print("\\n🎉 SUCCESS! Analysis pipeline completed successfully.")
        print("📊 Check the 'results' folder for all outputs.")
        
        return model_results
        
    except Exception as e:
        logger.error(f"❌ Analysis pipeline failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()