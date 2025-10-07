#!/usr/bin/env python3
"""
Data Processing Module for Bat Seasonal Behavior Analysis
HIT140 Assessment 3 - Professional Data Analytics Pipeline

This module handles data loading, cleaning, validation, and preprocessing
for the bat seasonal behavior analysis project.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import logging
import yaml
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Comprehensive data processing class for bat behavior analysis.
    
    Handles loading, cleaning, validation, and preprocessing of datasets
    from the HIT140 Assessment 3 bat-rat interaction study.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data processor.
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.datasets = {}
        self.combined_data = None
        self.processed_data = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def load_dataset(self, file_path: str, dataset_name: str, 
                    encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load a single dataset from file.
        
        Args:
            file_path: Path to the data file
            dataset_name: Name identifier for the dataset
            encoding: File encoding (default: utf-8)
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading {dataset_name} from {file_path}")
        
        # Determine file type and load accordingly
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, encoding=encoding)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                # Try CSV as default
                df = pd.read_csv(file_path, encoding=encoding)
                
        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {e}")
            raise ValueError(f"Could not load {dataset_name}: {e}")
        
        # Store the dataset
        self.datasets[dataset_name] = df
        
        # Log basic info
        logger.info(f"✅ {dataset_name} loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"   Columns: {list(df.columns)}")
        
        return df
    
    def load_multiple_datasets(self, dataset_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple datasets from a dictionary of paths.
        
        Args:
            dataset_paths: Dictionary mapping dataset names to file paths
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of loaded datasets
        """
        logger.info(f"Loading {len(dataset_paths)} datasets")
        
        for name, path in dataset_paths.items():
            try:
                self.load_dataset(path, name)
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                continue
                
        logger.info(f"Successfully loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, any]:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dict: Data quality metrics and issues
        """
        logger.info(f"Validating data quality for {dataset_name}")
        
        quality_report = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_rows': 0,
            'data_types': {},
            'outliers': {},
            'date_columns': [],
            'numeric_columns': [],
            'categorical_columns': []
        }
        
        # Missing values analysis
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        quality_report['missing_values'] = {
            col: {'count': missing[col], 'percentage': missing_pct[col]}
            for col in missing[missing > 0].index
        }
        
        # Duplicate rows
        quality_report['duplicate_rows'] = df.duplicated().sum()
        
        # Data types analysis
        quality_report['data_types'] = df.dtypes.to_dict()
        
        # Categorize columns
        for col in df.columns:
            dtype = str(df[col].dtype)
            if df[col].dtype in ['int64', 'float64']:
                quality_report['numeric_columns'].append(col)
            elif 'datetime' in dtype:
                quality_report['date_columns'].append(col)
            else:
                quality_report['categorical_columns'].append(col)
        
        # Outlier detection for numeric columns
        for col in quality_report['numeric_columns']:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                quality_report['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df) * 100).round(2),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        # Log key findings
        logger.info(f"Data Quality Summary for {dataset_name}:")
        logger.info(f"  - Shape: {quality_report['total_rows']} rows × {quality_report['total_columns']} columns")
        logger.info(f"  - Missing values: {len(quality_report['missing_values'])} columns affected")
        logger.info(f"  - Duplicate rows: {quality_report['duplicate_rows']}")
        logger.info(f"  - Numeric columns: {len(quality_report['numeric_columns'])}")
        logger.info(f"  - Date columns: {len(quality_report['date_columns'])}")
        logger.info(f"  - Categorical columns: {len(quality_report['categorical_columns'])}")
        
        return quality_report
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto',
                            threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values using specified strategy.
        
        Args:
            df: DataFrame to process
            strategy: Missing value strategy ('drop', 'fill_mean', 'fill_median', 'fill_mode', 'auto')
            threshold: Drop columns/rows if missing percentage exceeds this
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        df_clean = df.copy()
        initial_shape = df_clean.shape
        
        # Drop columns with too many missing values
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_pct[missing_pct > threshold].index
        if len(cols_to_drop) > 0:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'fill_mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'fill_median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == 'fill_mode':
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    mode_value = df_clean[col].mode()
                    if len(mode_value) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_value[0])
        elif strategy == 'auto':
            # Intelligent filling based on data type and distribution
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if df_clean[col].dtype in ['int64', 'float64']:
                        # Use median for numeric columns
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    else:
                        # Use mode for categorical columns
                        mode_value = df_clean[col].mode()
                        if len(mode_value) > 0:
                            df_clean[col] = df_clean[col].fillna(mode_value[0])
        
        final_shape = df_clean.shape
        logger.info(f"Missing value handling complete: {initial_shape} → {final_shape}")
        
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr',
                       action: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers in numeric columns.
        
        Args:
            df: DataFrame to process
            method: Outlier detection method ('iqr', 'zscore')
            action: Action to take ('remove', 'cap', 'transform')
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        logger.info(f"Handling outliers using {method} method with {action} action")
        
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        outliers_handled = 0
        
        for col in numeric_cols:
            if df_clean[col].notna().sum() == 0:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                # Map z-scores back to original indices
                non_null_indices = df_clean[col].dropna().index
                outlier_mask = pd.Series(False, index=df_clean.index)
                outlier_mask.loc[non_null_indices] = z_scores > 3
            else:
                continue
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                outliers_handled += outlier_count
                
                if action == 'remove':
                    df_clean = df_clean[~outlier_mask]
                elif action == 'cap':
                    if method == 'iqr':
                        df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                        df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                    elif method == 'zscore':
                        # Cap at 3 standard deviations
                        mean_val = df_clean[col].mean()
                        std_val = df_clean[col].std()
                        df_clean.loc[outlier_mask, col] = np.clip(
                            df_clean.loc[outlier_mask, col],
                            mean_val - 3 * std_val,
                            mean_val + 3 * std_val
                        )
                elif action == 'transform':
                    # Log transform if all values are positive
                    if (df_clean[col] > 0).all():
                        df_clean[col] = np.log1p(df_clean[col])
        
        logger.info(f"Outlier handling complete: {outliers_handled} outliers processed")
        return df_clean
    
    def combine_datasets(self, datasets: Optional[Dict[str, pd.DataFrame]] = None,
                        join_method: str = 'outer') -> pd.DataFrame:
        """
        Combine multiple datasets into a single DataFrame.
        
        Args:
            datasets: Dictionary of datasets to combine (uses self.datasets if None)
            join_method: How to join datasets ('inner', 'outer', 'left', 'right')
            
        Returns:
            pd.DataFrame: Combined dataset
        """
        if datasets is None:
            datasets = self.datasets
            
        if len(datasets) == 0:
            raise ValueError("No datasets to combine")
        
        logger.info(f"Combining {len(datasets)} datasets using {join_method} join")
        
        # Start with the first dataset
        dataset_names = list(datasets.keys())
        combined = datasets[dataset_names[0]].copy()
        logger.info(f"Starting with {dataset_names[0]}: {combined.shape}")
        
        # Combine with remaining datasets
        for name in dataset_names[1:]:
            df = datasets[name]
            
            # Find common columns for joining
            common_cols = list(set(combined.columns) & set(df.columns))
            
            if len(common_cols) == 0:
                # No common columns - concatenate with different column names
                logger.warning(f"No common columns with {name}, concatenating with suffixes")
                combined = pd.concat([combined, df], axis=1, sort=False)
            else:
                # Try to merge on common columns
                logger.info(f"Merging with {name} on columns: {common_cols}")
                try:
                    # Use date column for merging if available
                    date_cols = [col for col in common_cols if 'date' in col.lower()]
                    if date_cols:
                        merge_on = date_cols[0]
                    else:
                        merge_on = common_cols[0]
                    
                    combined = pd.merge(combined, df, on=merge_on, how=join_method, suffixes=('', f'_{name}'))
                except Exception as e:
                    logger.warning(f"Could not merge with {name}: {e}. Concatenating instead.")
                    combined = pd.concat([combined, df], axis=1, sort=False)
            
            logger.info(f"After adding {name}: {combined.shape}")
        
        self.combined_data = combined
        logger.info(f"Dataset combination complete: Final shape {combined.shape}")
        
        return combined
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names (lowercase, underscores, no spaces).
        
        Args:
            df: DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        df_clean = df.copy()
        
        # Store original column mapping
        original_columns = df_clean.columns.tolist()
        
        # Standardize column names
        new_columns = []
        for col in original_columns:
            # Convert to lowercase and replace spaces/special chars with underscores
            new_col = col.lower().replace(' ', '_').replace('-', '_')
            # Remove special characters except underscores
            new_col = ''.join(c if c.isalnum() or c == '_' else '' for c in new_col)
            # Remove multiple consecutive underscores
            while '__' in new_col:
                new_col = new_col.replace('__', '_')
            # Remove leading/trailing underscores
            new_col = new_col.strip('_')
            new_columns.append(new_col)
        
        df_clean.columns = new_columns
        
        # Log changes
        changes = [(orig, new) for orig, new in zip(original_columns, new_columns) if orig != new]
        if changes:
            logger.info(f"Column name changes made: {len(changes)} columns")
            for orig, new in changes[:5]:  # Show first 5 changes
                logger.info(f"  '{orig}' → '{new}'")
        
        return df_clean
    
    def create_data_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Create a comprehensive summary of the dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dict: Comprehensive data summary
        """
        summary = {
            'basic_info': {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'dtypes': df.dtypes.value_counts().to_dict()
            },
            'missing_data': df.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {},
            'date_range': {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'frequency': df[col].value_counts().head().to_dict()
            }
        
        # Date columns summary
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            if df[col].notna().any():
                summary['date_range'][col] = {
                    'min_date': df[col].min(),
                    'max_date': df[col].max(),
                    'date_range_days': (df[col].max() - df[col].min()).days
                }
        
        return summary
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save based on file extension
        file_extension = Path(filepath).suffix.lower()
        
        if file_extension == '.csv':
            df.to_csv(filepath, index=False)
        elif file_extension in ['.xlsx', '.xls']:
            df.to_excel(filepath, index=False)
        elif file_extension == '.parquet':
            df.to_parquet(filepath, index=False)
        else:
            # Default to CSV
            df.to_csv(filepath, index=False)
        
        logger.info(f"Processed data saved to: {filepath}")
        logger.info(f"File size: {os.path.getsize(filepath)} bytes")
    
    def process_complete_pipeline(self, dataset_paths: Dict[str, str],
                                output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.
        
        Args:
            dataset_paths: Dictionary mapping dataset names to file paths
            output_path: Optional path to save processed data
            
        Returns:
            pd.DataFrame: Fully processed dataset
        """
        logger.info("STARTING COMPLETE DATA PROCESSING PIPELINE")
        
        # Load datasets
        self.load_multiple_datasets(dataset_paths)
        
        # Validate data quality for each dataset
        quality_reports = {}
        for name, df in self.datasets.items():
            quality_reports[name] = self.validate_data_quality(df, name)
        
        # Process each dataset individually
        processed_datasets = {}
        for name, df in self.datasets.items():
            logger.info(f"Processing {name}")
            
            # Standardize column names
            df_processed = self.standardize_column_names(df)
            
            # Handle missing values
            df_processed = self.handle_missing_values(df_processed, strategy='auto')
            
            # Handle outliers
            df_processed = self.handle_outliers(df_processed, method='iqr', action='cap')
            
            processed_datasets[name] = df_processed
            logger.info(f"✅ {name} processing complete")
        
        # Combine processed datasets
        combined_data = self.combine_datasets(processed_datasets)
        
        # Final standardization
        final_data = self.standardize_column_names(combined_data)
        
        # Store processed data
        self.processed_data = final_data
        
        # Save if output path provided
        if output_path:
            self.save_processed_data(final_data, output_path)
        
        # Create final summary
        final_summary = self.create_data_summary(final_data)
        logger.info("COMPLETE DATA PROCESSING PIPELINE FINISHED")
        logger.info(f"Final dataset shape: {final_data.shape}")
        
        return final_data


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    logger.info("DataProcessor module loaded successfully")