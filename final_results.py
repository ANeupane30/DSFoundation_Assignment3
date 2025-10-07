#!/usr/bin/env python3
"""
Final Results Script - Bat Seasonal Behavior Analysis
HIT140 Assessment 3 - Get Final Results Now
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🚀 FINAL BAT SEASONAL BEHAVIOR ANALYSIS RESULTS")
    print("=" * 60)
    
    try:
        # Load the engineered features data
        print("📊 Loading data...")
        df = pd.read_csv('bat_rat_features_engineered.csv')
        print(f"✅ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Import modules
        from src.models.regression_models import BatBehaviorModeler
        
        # Initialize modeler
        modeler = BatBehaviorModeler(random_state=42)
        
        # Prepare data
        print("\n🔧 Preparing data for modeling...")
        target_column = 'bat_landing_number'
        exclude_cols = ['date', 'season_start'] + [col for col in df.columns if 'frequency' in col.lower()]
        
        X, y = modeler.prepare_modeling_data(df, target_column, exclude_columns=exclude_cols)
        print(f"✅ Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        # Feature selection
        if len(X.columns) > 15:
            X = modeler.feature_selection(X, y, max_features=15)
        
        # Split data
        X_train, X_test, y_train, y_test = modeler.split_data(X, y, test_size=0.2)
        
        print("\n🤖 Training models...")
        
        # Train overall model
        overall_model = modeler.train_linear_regression(X_train, y_train, model_name="overall")
        overall_metrics = modeler.evaluate_model(overall_model, X_test, y_test, modeler.scalers["overall"], "overall")
        overall_importance = modeler.get_feature_importance(overall_model, X.columns.tolist())
        
        # Train seasonal models
        seasonal_results = modeler.train_seasonal_models(
            df, target_column, feature_columns=X.columns.tolist(), seasons=['summer', 'autumn']
        )
        
        print("\n" + "=" * 60)
        print("🎉 FINAL RESULTS - HIT140 ASSESSMENT 3 OBJECTIVE 2")
        print("=" * 60)
        
        # Overall Model Results
        print("📊 OVERALL LINEAR REGRESSION MODEL:")
        print(f"   R² Score: {overall_metrics.r2_score:.4f}")
        print(f"   Mean Absolute Error: {overall_metrics.mae:.4f}")
        print(f"   Root Mean Square Error: {overall_metrics.rmse:.4f}")
        print(f"   Explained Variance: {overall_metrics.explained_variance:.4f}")
        print()
        
        # Seasonal Model Results
        for season_name, season_result in seasonal_results.items():
            if season_name.endswith('_model'):
                season = season_name.replace('_model', '').upper()
                metrics = season_result.metrics
                print(f"📊 {season} SEASONAL MODEL:")
                print(f"   R² Score: {metrics.r2_score:.4f}")
                print(f"   Mean Absolute Error: {metrics.mae:.4f}")
                print(f"   Root Mean Square Error: {metrics.rmse:.4f}")
                print(f"   Explained Variance: {metrics.explained_variance:.4f}")
                print()
        
        # Feature Importance
        print("🔍 TOP 10 MOST IMPORTANT FEATURES:")
        print(f"Feature Importance DataFrame columns: {list(overall_importance.columns)}")
        
        # Check what columns are available and use the right one
        if 'importance' in overall_importance.columns:
            imp_col = 'importance'
            feat_col = 'feature'
        elif 'coefficient' in overall_importance.columns:
            imp_col = 'coefficient'
            feat_col = 'feature'
        else:
            # Use the first numeric column
            numeric_cols = overall_importance.select_dtypes(include=[np.number]).columns
            imp_col = numeric_cols[0] if len(numeric_cols) > 0 else overall_importance.columns[1]
            feat_col = overall_importance.columns[0]
        
        for idx, row in overall_importance.head(10).iterrows():
            print(f"   {idx+1:2d}. {row[feat_col]:<35} {abs(row[imp_col]):.4f}")
        print()
        
        # Model Comparison Summary
        all_r2_scores = [overall_metrics.r2_score]
        all_model_names = ['Overall']
        
        for season_name, season_result in seasonal_results.items():
            if season_name.endswith('_model'):
                season = season_name.replace('_model', '').title()
                all_r2_scores.append(season_result.metrics.r2_score)
                all_model_names.append(season)
        
        print("📈 MODEL PERFORMANCE COMPARISON:")
        for name, r2 in zip(all_model_names, all_r2_scores):
            print(f"   {name:<10}: R² = {r2:.4f}")
        
        best_idx = np.argmax(all_r2_scores)
        print(f"\n🏆 BEST MODEL: {all_model_names[best_idx]} (R² = {all_r2_scores[best_idx]:.4f})")
        
        print("\n" + "=" * 60)
        print("✅ OBJECTIVE 2 COMPLETED SUCCESSFULLY!")
        print("Linear Regression analysis for bat seasonal behavior complete.")
        print("=" * 60)
        
        # Create basic visualizations
        print("\n📊 Creating visualizations...")
        try:
            from src.visualization.plots import BatBehaviorVisualizer
            os.makedirs('results/plots', exist_ok=True)
            
            visualizer = BatBehaviorVisualizer()
            
            # Create combined results for visualization
            viz_results = {'overall': {'metrics': overall_metrics, 'feature_importance': overall_importance}}
            viz_results.update(seasonal_results)
            
            visualizer.plot_data_overview(df, target_column)
            visualizer.plot_model_performance(viz_results)
            
            print("✅ Basic plots created in results/plots/")
            
        except Exception as e:
            print(f"⚠️  Visualization creation had issues: {e}")
            print("   Models trained successfully - main results above are valid")
        
        print(f"\n🎯 ANALYSIS SUMMARY:")
        print(f"   Dataset: {df.shape[0]} bat behavior observations")
        print(f"   Features: {X.shape[1]} engineered features")
        print(f"   Models: {len(all_model_names)} Linear Regression models")
        print(f"   Best Performance: R² = {max(all_r2_scores):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Details: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ SUCCESS! All results generated.")
    else:
        print("\n❌ FAILED! Check errors above.")