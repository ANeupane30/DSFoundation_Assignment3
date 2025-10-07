#!/usr/bin/env python3
"""
Simplified Pipeline to Execute Complete Analysis Using Existing Data
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from src.models.regression_models import BatBehaviorModeler
from src.visualization.plots import BatBehaviorVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/analysis.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_analysis():
    """Run complete analysis using existing engineered features."""
    
    logger.info("="*60)
    logger.info("STARTING BAT SEASONAL BEHAVIOR ANALYSIS")
    logger.info("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    
    # Load existing engineered features
    logger.info("STEP 1: Loading existing engineered features data")
    
    if os.path.exists('bat_rat_features_engineered.csv'):
        df = pd.read_csv('bat_rat_features_engineered.csv')
        logger.info(f"✅ Loaded engineered features: {df.shape}")
    else:
        logger.error("❌ No engineered features file found!")
        return
    
    # Display data info
    logger.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Identify target column
    target_candidates = ['bat_landing_number', 'bat_landings', 'landings']
    target_column = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target_column = candidate
            break
    
    if target_column is None:
        logger.error("❌ No target column found!")
        return
    
    logger.info(f"Using target column: {target_column}")
    
    # STEP 2: Model Training
    logger.info("STEP 2: Training Linear Regression Models")
    
    modeler = BatBehaviorModeler(random_state=42)
    
    # Prepare features for modeling
    exclude_cols = ['date', 'season_start'] + [col for col in df.columns if 'frequency' in col.lower()]
    
    # Remove target column from features
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target_column]]
    
    # Prepare data
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_column]
    
    # Remove any remaining non-numeric or problematic columns
    X = X.dropna(axis=1, how='all')  # Drop columns that are all NaN
    
    logger.info(f"Features for modeling: {X.shape[1]} columns")
    logger.info(f"Target variable: {target_column} ({len(y)} samples)")
    
    # Feature selection if too many features
    if X.shape[1] > 20:
        logger.info("Performing feature selection...")
        X = modeler.feature_selection(X, y, max_features=20)
    
    # Split data
    X_train, X_test, y_train, y_test = modeler.split_data(X, y, test_size=0.2)
    
    # Train overall model
    logger.info("Training overall Linear Regression model...")
    overall_model = modeler.train_linear_regression(X_train, y_train, model_name="overall")
    overall_metrics = modeler.evaluate_model(
        overall_model, X_test, y_test, 
        modeler.scalers["overall"], "overall"
    )
    
    logger.info(f"✅ Overall Model R² Score: {overall_metrics.r2_score:.4f}")
    logger.info(f"✅ Overall Model RMSE: {overall_metrics.rmse:.4f}")
    
    # Get feature importance
    feature_importance = modeler.get_feature_importance(overall_model, X.columns.tolist())
    
    # Train seasonal models if season data available
    seasonal_results = {}
    if 'season' in df.columns:
        logger.info("Training seasonal models...")
        seasons = df['season'].unique()
        logger.info(f"Available seasons: {list(seasons)}")
        
        for season in ['summer', 'autumn']:  # Focus on main seasons
            if season in seasons:
                season_mask = df['season'] == season
                if season_mask.sum() > 50:  # Enough data for modeling
                    X_season = X[season_mask]
                    y_season = y[season_mask]
                    
                    if len(X_season) > 10:  # Minimum samples for training
                        X_train_s, X_test_s, y_train_s, y_test_s = modeler.split_data(
                            X_season, y_season, test_size=0.2
                        )
                        
                        model = modeler.train_linear_regression(
                            X_train_s, y_train_s, model_name=season
                        )
                        metrics = modeler.evaluate_model(
                            model, X_test_s, y_test_s,
                            modeler.scalers[season], season
                        )
                        
                        seasonal_results[season] = {
                            'model': model,
                            'metrics': metrics,
                            'feature_importance': modeler.get_feature_importance(model, X.columns.tolist())
                        }
                        
                        logger.info(f"✅ {season.title()} Model R² Score: {metrics.r2_score:.4f}")
    
    # STEP 3: Create Visualizations
    logger.info("STEP 3: Creating Visualizations")
    
    visualizer = BatBehaviorVisualizer()
    
    # Store all results
    model_results = {
        'overall': {
            'model': overall_model,
            'metrics': overall_metrics,
            'feature_importance': feature_importance
        }
    }
    model_results.update(seasonal_results)
    
    try:
        # Data overview
        logger.info("Creating data overview plots...")
        visualizer.plot_data_overview(df, target_column)
        
        # Model performance
        logger.info("Creating model performance plots...")
        visualizer.plot_model_performance(model_results)
        
        # Feature importance
        logger.info("Creating feature importance plot...")
        top_features = feature_importance.head(10)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.savefig('results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Visualizations created successfully")
        
    except Exception as e:
        logger.warning(f"Some visualizations failed: {e}")
    
    # STEP 4: Generate Summary Report
    logger.info("STEP 4: Generating Summary Report")
    
    # Create a simple HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bat Seasonal Behavior Analysis - HIT140 Assessment 3</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #e74c3c; }}
        </style>
    </head>
    <body>
        <h1>🦇 Bat Seasonal Behavior Analysis Results</h1>
        <h2>HIT140 Assessment 3 - Objective 2: Linear Regression Analysis</h2>
        
        <h2>📊 Dataset Summary</h2>
        <ul>
            <li><strong>Total Samples:</strong> {df.shape[0]:,}</li>
            <li><strong>Total Features:</strong> {df.shape[1]:,}</li>
            <li><strong>Target Variable:</strong> {target_column}</li>
            <li><strong>Features Used for Modeling:</strong> {len(X.columns)}</li>
        </ul>
        
        <h2>🤖 Model Performance Results</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>R² Score</th>
                <th>RMSE</th>
                <th>Sample Size</th>
            </tr>
            <tr>
                <td>Overall Linear Regression</td>
                <td class="metric">{overall_metrics.r2_score:.4f}</td>
                <td>{overall_metrics.rmse:.4f}</td>
                <td>{len(y):,}</td>
            </tr>
    """
    
    # Add seasonal results to report
    for season, results in seasonal_results.items():
        html_report += f"""
            <tr>
                <td>{season.title()} Linear Regression</td>
                <td class="metric">{results['metrics'].r2_score:.4f}</td>
                <td>{results['metrics'].rmse:.4f}</td>
                <td>{len(df[df['season'] == season]):,}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>🏆 Key Findings</h2>
        <ul>
    """
    
    # Add key findings
    if seasonal_results:
        best_season = max(seasonal_results.keys(), 
                         key=lambda x: seasonal_results[x]['metrics'].r2_score)
        best_r2 = seasonal_results[best_season]['metrics'].r2_score
        html_report += f"<li><strong>Best Performing Season:</strong> {best_season.title()} (R² = {best_r2:.4f})</li>"
    
    html_report += f"""
            <li><strong>Overall Model Performance:</strong> R² = {overall_metrics.r2_score:.4f}</li>
            <li><strong>Top Feature:</strong> {feature_importance.index[0]} (Importance: {feature_importance.values[0]:.4f})</li>
        </ul>
        
        <h2>📈 Model Interpretation</h2>
        <p>The Linear Regression analysis reveals insights into bat seasonal behavior patterns:</p>
        <ul>
            <li>The overall model explains <strong>{overall_metrics.r2_score*100:.1f}%</strong> of the variance in bat landing numbers</li>
    """
    
    if seasonal_results:
        html_report += "<li>Seasonal differences in model performance suggest varying behavioral patterns throughout the year</li>"
    
    html_report += f"""
            <li>Feature engineering created {df.shape[1]} variables from the original dataset</li>
        </ul>
        
        <p><em>Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </body>
    </html>
    """
    
    # Save report
    with open('results/analysis_report.html', 'w') as f:
        f.write(html_report)
    
    # Save model results as CSV
    results_df = pd.DataFrame({
        'Model': ['Overall'] + list(seasonal_results.keys()),
        'R2_Score': [overall_metrics.r2_score] + [seasonal_results[s]['metrics'].r2_score for s in seasonal_results.keys()],
        'RMSE': [overall_metrics.rmse] + [seasonal_results[s]['metrics'].rmse for s in seasonal_results.keys()],
        'Sample_Size': [len(y)] + [len(df[df['season'] == s]) for s in seasonal_results.keys()]
    })
    results_df.to_csv('results/model_results.csv', index=False)
    
    # Save feature importance
    feature_importance.to_csv('results/feature_importance.csv')
    
    logger.info("="*60)
    logger.info("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"📊 Overall Model R² Score: {overall_metrics.r2_score:.4f}")
    logger.info(f"📈 Results saved to 'results/' directory")
    logger.info(f"📄 Open 'results/analysis_report.html' to view the complete report")
    logger.info(f"📊 Model results saved to 'results/model_results.csv'")
    
    return {
        'overall_r2': overall_metrics.r2_score,
        'seasonal_results': seasonal_results,
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    try:
        results = run_analysis()
        print("\n🎉 SUCCESS! Check the 'results/' folder for all outputs.")
        print("📄 Open 'results/analysis_report.html' for the comprehensive report.")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        logger.error(f"Analysis failed: {e}", exc_info=True)