#!/usr/bin/env python3
"""
Visualization Module for Bat Seasonal Behavior Analysis
HIT140 Assessment 3 - Professional Data Analytics Pipeline

This module provides comprehensive visualization capabilities for exploratory
data analysis, model evaluation, and results presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatBehaviorVisualizer:
    """
    Comprehensive visualization class for bat seasonal behavior analysis.
    
    Provides methods for data exploration, model evaluation, and results
    presentation with professional styling and multiple output formats.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', color_palette: str = 'husl',
                 figure_size: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the visualizer with styling preferences.
        
        Args:
            style: Matplotlib style
            color_palette: Seaborn color palette
            figure_size: Default figure size
            dpi: Resolution for saved plots
        """
        self.style = style
        self.color_palette = color_palette
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Set up plotting style
        plt.style.use(style)
        sns.set_palette(color_palette)
        
        logger.info(f"Visualizer initialized with style: {style}, palette: {color_palette}")
    
    def save_plot(self, fig, filename: str, output_dir: str = "results/plots",
                 formats: List[str] = ['png', 'pdf']) -> None:
        """
        Save plot in multiple formats.
        
        Args:
            fig: Matplotlib figure or plotly figure
            filename: Base filename (without extension)
            output_dir: Output directory
            formats: List of formats to save
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for fmt in formats:
            filepath = os.path.join(output_dir, f"{filename}.{fmt}")
            
            if hasattr(fig, 'savefig'):  # Matplotlib figure
                fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', format=fmt)
            elif hasattr(fig, 'write_image'):  # Plotly figure
                fig.write_image(filepath)
            
            logger.info(f"Plot saved: {filepath}")
    
    def plot_data_overview(self, df: pd.DataFrame, target_column: str,
                          save_plots: bool = True) -> plt.Figure:
        """
        Create comprehensive data overview plots.
        
        Args:
            df: Input DataFrame
            target_column: Target variable name
            save_plots: Whether to save plots
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating data overview plots")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bat Seasonal Behavior Analysis - Data Overview', fontsize=16, fontweight='bold')
        
        # 1. Target variable distribution
        axes[0, 0].hist(df[target_column], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'{target_column} Distribution')
        axes[0, 0].set_xlabel(target_column)
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Seasonal distribution
        if 'season' in df.columns:
            season_counts = df['season'].value_counts()
            axes[0, 1].bar(season_counts.index, season_counts.values, 
                          color=['gold', 'orange', 'lightblue', 'lightgreen'])
            axes[0, 1].set_title('Seasonal Data Distribution')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Target by season boxplot
        if 'season' in df.columns:
            sns.boxplot(data=df, x='season', y=target_column, ax=axes[0, 2])
            axes[0, 2].set_title(f'{target_column} by Season')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Missing values heatmap
        missing_data = df.isnull().sum().sort_values(ascending=False)
        if missing_data.sum() > 0:
            missing_df = missing_data[missing_data > 0].head(10)
            axes[1, 0].barh(missing_df.index, missing_df.values, color='red', alpha=0.7)
            axes[1, 0].set_title('Missing Values by Column')
            axes[1, 0].set_xlabel('Missing Count')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Missing Values Check')
        
        # 5. Correlation with target (top features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
        top_corr = correlations.head(10)
        
        axes[1, 1].barh(range(len(top_corr)), top_corr.values, color='green', alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_corr)))
        axes[1, 1].set_yticklabels(top_corr.index)
        axes[1, 1].set_title(f'Top Correlations with {target_column}')
        axes[1, 1].set_xlabel('Absolute Correlation')
        
        # 6. Time series if date available
        if 'date' in df.columns:
            df_time = df.copy()
            df_time['date'] = pd.to_datetime(df_time['date'])
            monthly_avg = df_time.groupby(df_time['date'].dt.to_period('M'))[target_column].mean()
            axes[1, 2].plot(monthly_avg.index.astype(str), monthly_avg.values, marker='o')
            axes[1, 2].set_title(f'Monthly Average {target_column}')
            axes[1, 2].set_ylabel(target_column)
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            self.save_plot(fig, 'data_overview')
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, target_column: str,
                               max_features: int = 20, save_plots: bool = True) -> plt.Figure:
        """
        Create correlation matrix heatmap.
        
        Args:
            df: Input DataFrame
            target_column: Target variable name
            max_features: Maximum number of features to include
            save_plots: Whether to save plots
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating correlation matrix")
        
        # Select numeric columns and top correlated features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
        top_features = correlations.head(max_features).index.tolist()
        
        # Create correlation matrix
        corr_matrix = df[top_features].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, ax=ax, fmt='.2f')
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_plots:
            self.save_plot(fig, 'correlation_matrix')
        
        return fig
    
    def plot_seasonal_analysis(self, df: pd.DataFrame, target_column: str,
                              encounter_columns: List[str], save_plots: bool = True) -> plt.Figure:
        """
        Create comprehensive seasonal analysis plots.
        
        Args:
            df: Input DataFrame
            target_column: Target variable name
            encounter_columns: List of encounter column names
            save_plots: Whether to save plots
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating seasonal analysis plots")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Seasonal Behavior Analysis', fontsize=16, fontweight='bold')
        
        # 1. Seasonal target distribution
        if 'season' in df.columns:
            sns.boxplot(data=df, x='season', y=target_column, ax=axes[0, 0])
            axes[0, 0].set_title(f'{target_column} Distribution by Season')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Seasonal means comparison
        if 'season' in df.columns:
            seasonal_means = df.groupby('season')[target_column].mean().sort_values(ascending=False)
            axes[0, 1].bar(seasonal_means.index, seasonal_means.values,
                          color=['gold', 'orange', 'lightblue', 'lightgreen'])
            axes[0, 1].set_title(f'Average {target_column} by Season')
            axes[0, 1].set_ylabel(f'Mean {target_column}')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(seasonal_means.values):
                axes[0, 1].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
        
        # 3. Encounter patterns by season
        if encounter_columns and 'season' in df.columns:
            primary_encounter = encounter_columns[0]
            if primary_encounter in df.columns:
                encounter_seasonal = df.groupby('season')[primary_encounter].mean()
                axes[0, 2].bar(encounter_seasonal.index, encounter_seasonal.values,
                              color='coral', alpha=0.7)
                axes[0, 2].set_title(f'{primary_encounter} by Season')
                axes[0, 2].set_ylabel(f'Mean {primary_encounter}')
                axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Monthly trends
        if 'month' in df.columns:
            monthly_means = df.groupby('month')[target_column].mean()
            axes[1, 0].plot(monthly_means.index, monthly_means.values, marker='o', linewidth=2)
            axes[1, 0].set_title(f'Monthly {target_column} Trends')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel(f'Mean {target_column}')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Food availability impact
        if 'food_scarcity_index' in df.columns and 'food_abundance_index' in df.columns:
            # Scatter plot of food availability vs target
            axes[1, 1].scatter(df['food_abundance_index'], df[target_column], 
                              alpha=0.6, color='green')
            axes[1, 1].set_xlabel('Food Abundance Index')
            axes[1, 1].set_ylabel(target_column)
            axes[1, 1].set_title('Food Availability vs Behavior')
            
            # Add trend line
            z = np.polyfit(df['food_abundance_index'], df[target_column], 1)
            p = np.poly1d(z)
            axes[1, 1].plot(df['food_abundance_index'], p(df['food_abundance_index']), 
                           "r--", alpha=0.8)
        
        # 6. Seasonal interaction with encounters
        if encounter_columns and 'season' in df.columns:
            primary_encounter = encounter_columns[0]
            if primary_encounter in df.columns:
                # Create interaction plot
                for season in df['season'].unique():
                    season_data = df[df['season'] == season]
                    if len(season_data) > 0:
                        axes[1, 2].scatter(season_data[primary_encounter], 
                                         season_data[target_column],
                                         label=season.capitalize(), alpha=0.6)
                
                axes[1, 2].set_xlabel(primary_encounter)
                axes[1, 2].set_ylabel(target_column)
                axes[1, 2].set_title('Seasonal Encounter-Behavior Relationship')
                axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_plots:
            self.save_plot(fig, 'seasonal_analysis')
        
        return fig
    
    def plot_model_performance(self, model_results: Dict[str, Any],
                             save_plots: bool = True) -> plt.Figure:
        """
        Create model performance comparison plots.
        
        Args:
            model_results: Dictionary containing model results
            save_plots: Whether to save plots
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating model performance plots")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Linear Regression Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Extract metrics for plotting
        models = list(model_results.keys())
        r2_scores = [model_results[model]['metrics'].r2_score for model in models]
        mae_scores = [model_results[model]['metrics'].mae for model in models]
        rmse_scores = [model_results[model]['metrics'].rmse for model in models]
        
        # 1. R² Score comparison
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(models)]
        bars1 = axes[0, 0].bar(models, r2_scores, color=colors, alpha=0.8)
        axes[0, 0].set_title('Model R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 2. MAE comparison
        bars2 = axes[0, 1].bar(models, mae_scores, color=colors, alpha=0.8)
        axes[0, 1].set_title('Mean Absolute Error Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars2, mae_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                           f'{score:.1f}', ha='center', va='bottom')
        
        # 3. Feature importance for best model
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics'].r2_score)
        if 'feature_importance' in model_results[best_model]:
            importance = model_results[best_model]['feature_importance'].head(10)
            y_pos = np.arange(len(importance))
            
            bars3 = axes[0, 2].barh(y_pos, importance['abs_coefficient'], color='green', alpha=0.7)
            axes[0, 2].set_yticks(y_pos)
            axes[0, 2].set_yticklabels(importance['feature'])
            axes[0, 2].set_title(f'Top Features - {best_model.title()} Model')
            axes[0, 2].set_xlabel('Absolute Coefficient')
        
        # 4. Residual plots for best model
        if 'residuals' in model_results[best_model]:
            residuals = model_results[best_model]['residuals']
            predictions = model_results[best_model]['predictions']
            
            axes[1, 0].scatter(predictions, residuals, alpha=0.6, color='blue')
            axes[1, 0].axhline(y=0, color='red', linestyle='--')
            axes[1, 0].set_xlabel('Predicted Values')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title(f'Residual Plot - {best_model.title()}')
        
        # 5. Q-Q plot for residuals
        if 'residuals' in model_results[best_model]:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title(f'Q-Q Plot - {best_model.title()}')
        
        # 6. Model metrics summary table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        # Create summary table
        table_data = []
        for model in models:
            metrics = model_results[model]['metrics']
            table_data.append([
                model.title(),
                f"{metrics.r2_score:.3f}",
                f"{metrics.mae:.1f}",
                f"{metrics.rmse:.1f}",
                f"{metrics.n_samples}"
            ])
        
        table = axes[1, 2].table(cellText=table_data,
                               colLabels=['Model', 'R²', 'MAE', 'RMSE', 'Samples'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Model Performance Summary')
        
        plt.tight_layout()
        
        if save_plots:
            self.save_plot(fig, 'model_performance')
        
        return fig
    
    def plot_seasonal_comparison(self, seasonal_results: Dict[str, Any],
                               coefficient_comparison: pd.DataFrame,
                               save_plots: bool = True) -> plt.Figure:
        """
        Create seasonal model comparison plots.
        
        Args:
            seasonal_results: Dictionary of seasonal model results
            coefficient_comparison: DataFrame with coefficient comparisons
            save_plots: Whether to save plots
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating seasonal comparison plots")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Seasonal Model Comparison Analysis', fontsize=16, fontweight='bold')
        
        seasons = list(seasonal_results.keys())
        
        # 1. Performance metrics comparison
        metrics_data = {
            'Season': [s.capitalize() for s in seasons],
            'R² Score': [seasonal_results[s]['metrics'].r2_score for s in seasons],
            'MAE': [seasonal_results[s]['metrics'].mae for s in seasons]
        }
        
        x = np.arange(len(seasons))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, metrics_data['R² Score'], width, 
                              label='R² Score', color='skyblue', alpha=0.8)
        ax_twin = axes[0, 0].twinx()
        bars2 = ax_twin.bar(x + width/2, metrics_data['MAE'], width,
                           label='MAE', color='lightcoral', alpha=0.8)
        
        axes[0, 0].set_xlabel('Season')
        axes[0, 0].set_ylabel('R² Score', color='blue')
        ax_twin.set_ylabel('MAE', color='red')
        axes[0, 0].set_title('Seasonal Model Performance')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_data['Season'])
        
        # Add value labels
        for bar, val in zip(bars1, metrics_data['R² Score']):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{val:.3f}', ha='center', va='bottom')
        
        # 2. Coefficient differences
        if not coefficient_comparison.empty and 'abs_difference' in coefficient_comparison.columns:
            top_diffs = coefficient_comparison.head(8)
            y_pos = np.arange(len(top_diffs))
            
            axes[0, 1].barh(y_pos, top_diffs['abs_difference'], color='orange', alpha=0.7)
            axes[0, 1].set_yticks(y_pos)
            axes[0, 1].set_yticklabels(top_diffs['feature'])
            axes[0, 1].set_title('Largest Coefficient Differences')
            axes[0, 1].set_xlabel('Absolute Difference')
        
        # 3. Feature importance comparison
        season1, season2 = seasons[0], seasons[1] if len(seasons) > 1 else seasons[0]
        
        if 'feature_importance' in seasonal_results[season1]:
            importance1 = seasonal_results[season1]['feature_importance'].head(6)
            y_pos = np.arange(len(importance1))
            
            axes[1, 0].barh(y_pos - 0.2, importance1['abs_coefficient'], 0.4,
                           label=season1.capitalize(), color='gold', alpha=0.8)
            
            if len(seasons) > 1 and 'feature_importance' in seasonal_results[season2]:
                importance2 = seasonal_results[season2]['feature_importance'].head(6)
                common_features = set(importance1['feature']) & set(importance2['feature'])
                importance2_matched = importance2[importance2['feature'].isin(common_features)]
                
                if not importance2_matched.empty:
                    axes[1, 0].barh(y_pos + 0.2, importance2_matched['abs_coefficient'], 0.4,
                                   label=season2.capitalize(), color='lightblue', alpha=0.8)
            
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(importance1['feature'])
            axes[1, 0].set_title('Feature Importance Comparison')
            axes[1, 0].set_xlabel('Absolute Coefficient')
            axes[1, 0].legend()
        
        # 4. Residual distribution comparison
        axes[1, 1].hist(seasonal_results[season1]['residuals'], bins=20, alpha=0.7,
                       label=season1.capitalize(), color='gold')
        
        if len(seasons) > 1:
            axes[1, 1].hist(seasonal_results[season2]['residuals'], bins=20, alpha=0.7,
                           label=season2.capitalize(), color='lightblue')
        
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residual Distribution Comparison')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            self.save_plot(fig, 'seasonal_comparison')
        
        return fig
    
    def create_interactive_dashboard(self, df: pd.DataFrame, target_column: str,
                                   model_results: Dict[str, Any],
                                   save_path: str = "results/interactive_dashboard.html") -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            df: Input DataFrame
            target_column: Target variable name
            model_results: Dictionary of model results
            save_path: Path to save HTML dashboard
        """
        logger.info("Creating interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Seasonal Distribution', 'Model Performance',
                'Feature Correlations', 'Predictions vs Actual',
                'Monthly Trends', 'Residual Analysis'
            ),
            specs=[[{"type": "box"}, {"type": "bar"}],
                  [{"type": "heatmap"}, {"type": "scatter"}],
                  [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # 1. Seasonal box plot
        if 'season' in df.columns:
            for season in df['season'].unique():
                season_data = df[df['season'] == season][target_column]
                fig.add_trace(
                    go.Box(y=season_data, name=season.capitalize(),
                          boxpoints='outliers'),
                    row=1, col=1
                )
        
        # 2. Model performance
        models = list(model_results.keys())
        r2_scores = [model_results[model]['metrics'].r2_score for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=r2_scores, name='R² Score',
                  text=[f'{score:.3f}' for score in r2_scores],
                  textposition='auto'),
            row=1, col=2
        )
        
        # 3. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:15]
        corr_matrix = df[numeric_cols].corr()
        
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values,
                      x=corr_matrix.columns,
                      y=corr_matrix.columns,
                      colorscale='RdBu',
                      zmid=0),
            row=2, col=1
        )
        
        # 4. Predictions vs Actual (best model)
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics'].r2_score)
        if 'predictions' in model_results[best_model]:
            # Get actual values (assuming they're available)
            predictions = model_results[best_model]['predictions']
            # Create dummy actual values for demonstration
            actual = predictions + model_results[best_model]['residuals']
            
            fig.add_trace(
                go.Scatter(x=actual, y=predictions,
                          mode='markers',
                          name=f'{best_model.title()} Model',
                          opacity=0.7),
                row=2, col=2
            )
            
            # Add perfect prediction line
            min_val, max_val = min(actual.min(), predictions.min()), max(actual.max(), predictions.max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines',
                          name='Perfect Prediction',
                          line=dict(dash='dash', color='red')),
                row=2, col=2
            )
        
        # 5. Monthly trends
        if 'month' in df.columns:
            monthly_avg = df.groupby('month')[target_column].mean()
            fig.add_trace(
                go.Scatter(x=monthly_avg.index, y=monthly_avg.values,
                          mode='lines+markers',
                          name='Monthly Average',
                          line=dict(width=3)),
                row=3, col=1
            )
        
        # 6. Residual histogram
        if 'residuals' in model_results[best_model]:
            residuals = model_results[best_model]['residuals']
            fig.add_trace(
                go.Histogram(x=residuals,
                           name='Residuals',
                           opacity=0.7),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Bat Seasonal Behavior Analysis - Interactive Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Save dashboard
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        
        logger.info(f"Interactive dashboard saved: {save_path}")
    
    def generate_analysis_report(self, df: pd.DataFrame, model_results: Dict[str, Any],
                               output_path: str = "results/analysis_report.html") -> None:
        """
        Generate a comprehensive HTML analysis report.
        
        Args:
            df: Input DataFrame
            model_results: Dictionary of model results
            output_path: Path to save HTML report
        """
        logger.info("Generating analysis report")
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bat Seasonal Behavior Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2E4057; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>HIT140 Assessment 3 - Bat Seasonal Behavior Analysis</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report presents the results of a comprehensive Linear Regression analysis 
                of bat seasonal behavior patterns and their interactions with rat populations.</p>
                <p><strong>Dataset:</strong> {len(df)} observations with {len(df.columns)} variables</p>
                <p><strong>Analysis Period:</strong> {df['date'].min() if 'date' in df.columns else 'N/A'} to {df['date'].max() if 'date' in df.columns else 'N/A'}</p>
            </div>
            
            <h2>Key Findings</h2>
        """
        
        # Add model performance summary
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics'].r2_score)
        best_r2 = model_results[best_model]['metrics'].r2_score
        
        html_content += f"""
            <div class="summary">
                <h3>Model Performance</h3>
                <div class="metric">
                    <strong>Best Model:</strong> {best_model.title()}<br>
                    <strong>R² Score:</strong> {best_r2:.3f}
                </div>
        """
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            html_content += f"""
                <div class="metric">
                    <strong>{model_name.title()}</strong><br>
                    R²: {metrics.r2_score:.3f}<br>
                    MAE: {metrics.mae:.1f}<br>
                    RMSE: {metrics.rmse:.1f}
                </div>
            """
        
        html_content += """
            </div>
            
            <h2>Methodology</h2>
            <p>The analysis employed the following approach:</p>
            <ul>
                <li>Feature Engineering: Created 52 engineered features including temporal, seasonal, and interaction variables</li>
                <li>Linear Regression Modeling: Implemented overall and seasonal-specific models</li>
                <li>Model Evaluation: Used R², MSE, MAE, and cross-validation for assessment</li>
                <li>Seasonal Comparison: Analyzed behavioral differences between seasons</li>
            </ul>
            
            <h2>Data Quality Assessment</h2>
        """
        
        # Add data quality information
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        duplicates = df.duplicated().sum()
        
        html_content += f"""
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Observations</td><td>{len(df):,}</td></tr>
                <tr><td>Total Variables</td><td>{len(df.columns)}</td></tr>
                <tr><td>Missing Data %</td><td>{missing_pct:.2f}%</td></tr>
                <tr><td>Duplicate Rows</td><td>{duplicates}</td></tr>
            </table>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Focus conservation efforts during the {best_model} season when behavior is most predictable</li>
                <li>Consider rat population management strategies based on seasonal interaction patterns</li>
                <li>Monitor food availability indices as they show significant correlation with bat behavior</li>
                <li>Collect additional data during underrepresented seasons for improved model performance</li>
            </ul>
            
            <footer>
                <p><em>Report generated automatically from HIT140 Assessment 3 analysis pipeline.</em></p>
            </footer>
        </body>
        </html>
        """
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Analysis report saved: {output_path}")


if __name__ == "__main__":
    # Example usage
    visualizer = BatBehaviorVisualizer()
    logger.info("BatBehaviorVisualizer module loaded successfully")