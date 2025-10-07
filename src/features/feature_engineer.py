#!/usr/bin/env python3
"""
Feature Engineering Module for Bat Seasonal Behavior Analysis
HIT140 Assessment 3 - Professional Data Analytics Pipeline

This module contains the BatSeasonalFeatureEngineer class for comprehensive
feature engineering including temporal, seasonal, encounter, and interaction features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
from typing import Optional, List, Union
import logging

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatSeasonalFeatureEngineer:
    """
    Comprehensive feature engineering class for bat seasonal behavior analysis.
    
    This class provides methods to create temporal, seasonal, encounter, and 
    interaction features from bat behavior and rat encounter data.
    
    Attributes:
        df: Main DataFrame containing the data
        original_shape: Original data dimensions
        date_column: Identified date column name
        encounter_columns: List of encounter-related columns
        behavior_columns: List of behavior-related columns
        id_columns: List of ID-related columns
    """
    
    def __init__(self, data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """
        Initialize the feature engineer with data.
        
        Args:
            data_path: Path to data file (CSV/Excel)
            df: Pandas DataFrame with data
            
        Raises:
            ValueError: If neither data_path nor df is provided
            RuntimeError: If data loading fails
        """
        if data_path and os.path.exists(data_path):
            logger.info(f"Loading data from: {data_path}")
            try:
                if data_path.lower().endswith(".csv"):
                    self.df = pd.read_csv(data_path)
                elif data_path.lower().endswith((".xls", ".xlsx")):
                    self.df = pd.read_excel(data_path)
                else:
                    self.df = pd.read_csv(data_path)
            except Exception as e:
                raise RuntimeError(f"Error loading data: {e}")
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Provide either data_path or df parameter")

        self.original_shape = self.df.shape
        logger.info(f"Data loaded: {self.original_shape[0]} rows, {self.original_shape[1]} cols")

        # Initialize tracking variables
        self.date_column = None
        self.encounter_columns = []
        self.behavior_columns = []
        self.id_columns = []

    def analyze_data_structure(self) -> pd.DataFrame:
        """
        Analyze the structure of the dataset and identify column types.
        
        Returns:
            pd.DataFrame: Descriptive statistics of the dataset
        """
        logger.info("ANALYZING DATA STRUCTURE")
        logger.info(f"Dataset shape: {self.df.shape}")
        
        # Log column information
        logger.info("Columns and dtypes:")
        for i, (col, dtype) in enumerate(self.df.dtypes.items()):
            logger.info(f"  {i+1:2d}. {col:<30} ({dtype})")

        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            logger.info("Missing values detected:")
            missing_pct = (missing / len(self.df) * 100).round(2)
            for col in missing[missing > 0].index:
                logger.info(f"  {col:<30} {missing[col]:>6} ({missing_pct[col]:>6.2f}%)")
        else:
            logger.info("No missing values found!")

        self._detect_column_types()

        # Log identified columns
        logger.info("Key columns identified:")
        if self.date_column:
            logger.info(f"  Date column: {self.date_column}")
        if self.encounter_columns:
            logger.info(f"  Encounter columns: {self.encounter_columns}")
        if self.behavior_columns:
            logger.info(f"  Behavior columns: {self.behavior_columns}")
        if self.id_columns:
            logger.info(f"  ID columns: {self.id_columns}")

        return self.df.describe(include="all")

    def _detect_column_types(self) -> None:
        """Automatically detect column types based on naming patterns."""
        cols = list(self.df.columns)
        lower = [c.lower() for c in cols]

        # Date detection
        date_keywords = ["date", "time", "timestamp", "day", "month", "year"]
        for i, name in enumerate(lower):
            if any(k in name for k in date_keywords):
                self.date_column = cols[i]
                break

        # Encounter detection
        encounter_keywords = ["encounter", "rat", "contact", "interaction", "prey", "encounters", "arrival"]
        for i, name in enumerate(lower):
            if any(k in name for k in encounter_keywords):
                self.encounter_columns.append(cols[i])

        # Behavior detection
        behavior_keywords = ["activity", "behavior", "behaviour", "movement", "flight", "foraging", "landing"]
        for i, name in enumerate(lower):
            if any(k in name for k in behavior_keywords):
                self.behavior_columns.append(cols[i])

        # ID detection
        id_keywords = ["id", "bat", "individual", "animal", "subject"]
        for i, name in enumerate(lower):
            if any(k in name for k in id_keywords) and "date" not in name:
                self.id_columns.append(cols[i])

    def prepare_temporal_features(self, date_col: Optional[str] = None, 
                                date_format: Optional[str] = None) -> None:
        """
        Create temporal features from date column.
        
        Args:
            date_col: Name of date column (if not auto-detected)
            date_format: Specific date format for parsing
            
        Raises:
            ValueError: If no date column is specified or found
        """
        logger.info("PREPARING TEMPORAL FEATURES")

        if date_col:
            self.date_column = date_col

        if not self.date_column:
            raise ValueError("No date column specified. Provide date_col or ensure auto-detection found a date column.")

        # Parse dates
        if date_format:
            self.df["date"] = pd.to_datetime(self.df[self.date_column], format=date_format, errors="coerce")
        else:
            self.df["date"] = pd.to_datetime(self.df[self.date_column], infer_datetime_format=True, errors="coerce")

        null_dates = self.df["date"].isnull().sum()
        if null_dates > 0:
            logger.warning(f"{null_dates} date(s) could not be parsed and are NaT")

        # Sort by date
        self.df = self.df.sort_values("date").reset_index(drop=True)

        # Create basic temporal features
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day
        self.df["day_of_year"] = self.df["date"].dt.dayofyear
        self.df["weekday"] = self.df["date"].dt.dayofweek
        self.df["week"] = self.df["date"].dt.isocalendar().week.astype(int)

        # Log date range
        if self.df["date"].notna().any():
            date_range = f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}"
            total_days = (self.df['date'].max() - self.df['date'].min()).days
            logger.info(f"Date range: {date_range}")
            logger.info(f"Total days: {total_days}")
        else:
            logger.warning("No valid dates to report range")

        logger.info("Basic temporal features created successfully")

    def create_seasonal_features(self, location: str = "australia") -> None:
        """
        Create seasonal features based on geographic location.
        
        Args:
            location: Geographic location for seasonal mapping ("australia" or "northern")
            
        Raises:
            ValueError: If temporal features haven't been created first
        """
        logger.info("CREATING SEASONAL FEATURES")

        if "month" not in self.df.columns:
            raise ValueError("Call prepare_temporal_features first to create month/day columns.")

        loc = location.lower()
        
        # Define seasonal classification and food availability
        if loc == "australia":
            def classify_season(month):
                if month in [12, 1, 2]:
                    return "summer"
                elif month in [3, 4, 5]:
                    return "autumn"
                elif month in [6, 7, 8]:
                    return "winter"
                else:
                    return "spring"
            
            season_food_map = {"winter": 0.9, "spring": 0.1, "summer": 0.4, "autumn": 0.6}
        else:
            def classify_season(month):
                if month in [12, 1, 2]:
                    return "winter"
                elif month in [3, 4, 5]:
                    return "spring"
                elif month in [6, 7, 8]:
                    return "summer"
                else:
                    return "autumn"
            
            season_food_map = {"winter": 0.9, "spring": 0.1, "summer": 0.3, "autumn": 0.7}

        # Create seasonal features
        self.df["season"] = self.df["month"].apply(classify_season)
        self.df["is_winter"] = (self.df["season"] == "winter").astype(int)
        self.df["is_spring"] = (self.df["season"] == "spring").astype(int)
        self.df["is_summer"] = (self.df["season"] == "summer").astype(int)
        self.df["is_autumn"] = (self.df["season"] == "autumn").astype(int)

        # Cyclical encoding
        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)

        # Food availability indices
        self.df["food_scarcity_index"] = self.df["season"].map(season_food_map).fillna(0.5)
        self.df["food_abundance_index"] = 1 - self.df["food_scarcity_index"]

        # Days since season start
        self._calculate_days_since_season_start(loc)
        
        # Transition periods
        self._identify_transition_periods(loc)

        # Log seasonal distribution
        seasonal_counts = self.df["season"].value_counts()
        logger.info("Seasonal distribution:")
        for season, count in seasonal_counts.items():
            pct = count / len(self.df) * 100
            scarcity = season_food_map.get(season, np.nan)
            logger.info(f"  {season.capitalize():<8} {count:>6} ({pct:5.1f}%) - Food scarcity: {scarcity}")

        logger.info("Seasonal features created successfully")

    def _calculate_days_since_season_start(self, location: str) -> None:
        """Calculate days since the start of current season."""
        def season_start_date(row):
            year = int(row["year"])
            season = row["season"]
            month = row["month"]
            
            if location == "australia":
                if season == "summer":
                    if month == 12:
                        start = pd.Timestamp(year=year, month=12, day=1)
                    else:
                        start = pd.Timestamp(year=year - 1, month=12, day=1)
                elif season == "autumn":
                    start = pd.Timestamp(year=year, month=3, day=1)
                elif season == "winter":
                    start = pd.Timestamp(year=year, month=6, day=1)
                else:  # spring
                    start = pd.Timestamp(year=year, month=9, day=1)
            else:
                if season == "winter":
                    if month == 12:
                        start = pd.Timestamp(year=year, month=12, day=1)
                    else:
                        start = pd.Timestamp(year=year - 1, month=12, day=1)
                elif season == "spring":
                    start = pd.Timestamp(year=year, month=3, day=1)
                elif season == "summer":
                    start = pd.Timestamp(year=year, month=6, day=1)
                else:  # autumn
                    start = pd.Timestamp(year=year, month=9, day=1)
            return start

        self.df["season_start"] = self.df.apply(season_start_date, axis=1)
        self.df["days_since_season_start"] = (self.df["date"] - self.df["season_start"]).dt.days.clip(lower=0)

    def _identify_transition_periods(self, location: str) -> None:
        """Identify seasonal transition periods (±14 days around boundaries)."""
        def is_seasonal_transition(row):
            date = row["date"]
            year = row["year"]
            
            if location == "australia":
                boundaries = [
                    pd.Timestamp(year=year, month=3, day=1),
                    pd.Timestamp(year=year, month=6, day=1),
                    pd.Timestamp(year=year, month=9, day=1),
                    pd.Timestamp(year=year, month=12, day=1)
                ]
            else:
                boundaries = [
                    pd.Timestamp(year=year, month=3, day=1),
                    pd.Timestamp(year=year, month=6, day=1),
                    pd.Timestamp(year=year, month=9, day=1),
                    pd.Timestamp(year=year, month=12, day=1)
                ]
            
            # Check current year boundaries
            for boundary in boundaries:
                if abs((date - boundary).days) <= 14:
                    return 1
            
            # Check adjacent year boundaries for year transitions
            for boundary in boundaries:
                for offset in [1, -1]:
                    adj_boundary = boundary + pd.DateOffset(years=offset)
                    if abs((date - adj_boundary).days) <= 14:
                        return 1
            return 0

        self.df["is_transition_period"] = self.df.apply(is_seasonal_transition, axis=1)

    def create_encounter_features(self, encounter_cols: Optional[List[str]] = None) -> None:
        """
        Create encounter-based features including rolling statistics and temporal patterns.
        
        Args:
            encounter_cols: List of encounter column names (if not auto-detected)
        """
        logger.info("CREATING ENCOUNTER FEATURES")

        if encounter_cols:
            if isinstance(encounter_cols, str):
                encounter_cols = [encounter_cols]
            self.encounter_columns = encounter_cols

        # Create simulated encounter data if none found
        if not self.encounter_columns:
            np.random.seed(42)
            base = np.random.poisson(1.5, len(self.df))
            seasonal_multiplier = self.df["food_abundance_index"] * 1.8 + 0.2
            self.df["encounter_count"] = np.maximum(0, (base * seasonal_multiplier).astype(int))
            self.encounter_columns = ["encounter_count"]
            logger.info("Created simulated encounter_count")

        # Sort data for temporal calculations
        if self.id_columns:
            id_col = self.id_columns[0]
            self.df = self.df.sort_values([id_col, "date"]).reset_index(drop=True)
            group_col = id_col
        else:
            self.df = self.df.sort_values("date").reset_index(drop=True)
            group_col = None

        # Process each encounter column
        for enc_col in self.encounter_columns:
            logger.info(f"Processing encounter column: {enc_col}")
            total = self.df[enc_col].sum()
            mean_val = self.df[enc_col].mean()
            logger.info(f"  Total: {total}, Mean: {mean_val:.3f}")

            # Rolling statistics
            self._create_rolling_features(enc_col, group_col)
            
            # Binary indicators
            self._create_encounter_indicators(enc_col, group_col)
            
            # Frequency categories
            self._create_frequency_categories(enc_col)

            logger.info("  Rolling windows and temporal features created")

    def _create_rolling_features(self, enc_col: str, group_col: Optional[str]) -> None:
        """Create rolling mean and standard deviation features."""
        windows = [7, 14, 30]
        for window in windows:
            if group_col:
                self.df[f"{enc_col}_rolling_{window}d"] = (
                    self.df.groupby(group_col)[enc_col]
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                self.df[f"{enc_col}_std_{window}d"] = (
                    self.df.groupby(group_col)[enc_col]
                    .rolling(window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                ).fillna(0)
            else:
                self.df[f"{enc_col}_rolling_{window}d"] = self.df[enc_col].rolling(window, min_periods=1).mean()
                self.df[f"{enc_col}_std_{window}d"] = self.df[enc_col].rolling(window, min_periods=1).std().fillna(0)

    def _create_encounter_indicators(self, enc_col: str, group_col: Optional[str]) -> None:
        """Create binary indicators and days-since-last-encounter features."""
        has_col = f"has_{enc_col}"
        self.df[has_col] = (self.df[enc_col] > 0).astype(int)

        if group_col:
            def compute_days_since(group):
                last_encounter_date = pd.Series(pd.NaT, index=group.index)
                last_seen = pd.NaT
                for idx, row in group.iterrows():
                    if row[has_col] == 1:
                        last_seen = row["date"]
                        last_encounter_date.loc[idx] = 0
                    else:
                        if pd.isna(last_seen):
                            last_encounter_date.loc[idx] = np.nan
                        else:
                            last_encounter_date.loc[idx] = (row["date"] - last_seen).days
                return last_encounter_date

            self.df[f"days_since_last_{enc_col}"] = (
                self.df.groupby(group_col).apply(compute_days_since).reset_index(level=0, drop=True)
            ).fillna(-1)
        else:
            last_seen = pd.NaT
            days_since = []
            for _, row in self.df.iterrows():
                if row[has_col] == 1:
                    last_seen = row["date"]
                    days_since.append(0)
                else:
                    if pd.isna(last_seen):
                        days_since.append(np.nan)
                    else:
                        days_since.append((row["date"] - last_seen).days)
            self.df[f"days_since_last_{enc_col}"] = pd.Series(days_since).fillna(-1)

    def _create_frequency_categories(self, enc_col: str) -> None:
        """Create frequency categories based on quantiles."""
        if self.df[enc_col].max() > 0:
            try:
                q33, q67 = self.df[enc_col].quantile([0.33, 0.67]).values
                if q33 == q67:  # Handle case where values are the same
                    # Create simple binary categories
                    self.df[f"{enc_col}_frequency"] = np.where(
                        self.df[enc_col] > q33, "high", "low"
                    )
                else:
                    self.df[f"{enc_col}_frequency"] = pd.cut(
                        self.df[enc_col],
                        bins=[-0.1, q33, q67, np.inf],
                        labels=["low", "medium", "high"]
                    ).astype(object)
            except Exception as e:
                logger.warning(f"Could not create frequency categories for {enc_col}: {e}")
                # Create simple binary high/low categories
                median_val = self.df[enc_col].median()
                self.df[f"{enc_col}_frequency"] = np.where(
                    self.df[enc_col] > median_val, "high", "low"
                )

    def create_seasonal_interaction_features(self) -> None:
        """Create interaction features between seasonal and encounter variables."""
        logger.info("CREATING SEASONAL-ENCOUNTER INTERACTIONS")

        if not self.encounter_columns:
            logger.error("No encounter columns; run create_encounter_features first")
            return

        primary = self.encounter_columns[0]
        logger.info(f"Primary encounter: {primary}")

        # Season-encounter interactions
        self.df["winter_encounters"] = self.df["is_winter"] * self.df[primary]
        self.df["spring_encounters"] = self.df["is_spring"] * self.df[primary]

        # Food availability interactions
        eps = 1e-6
        self.df["scarcity_encounter_ratio"] = self.df[primary] / (self.df["food_scarcity_index"] + eps)
        self.df["abundance_encounter_product"] = self.df[primary] * self.df["food_abundance_index"]

        # Rolling encounter interactions
        if f"{primary}_rolling_7d" in self.df.columns:
            self.df["winter_rolling_encounters"] = self.df["is_winter"] * self.df[f"{primary}_rolling_7d"]

        # Lagged effects
        if self.id_columns:
            id_col = self.id_columns[0]
            self.df["prev_month_encounters"] = self.df.groupby(id_col)[primary].shift(30).fillna(0)
            self.df["prev_season_effect"] = self.df.groupby(id_col)["food_scarcity_index"].shift(90).fillna(0.5)
        else:
            self.df["prev_month_encounters"] = self.df[primary].shift(30).fillna(0)
            self.df["prev_season_effect"] = self.df["food_scarcity_index"].shift(90).fillna(0.5)

        logger.info("Interaction features created successfully")

    def analyze_seasonal_patterns(self) -> pd.DataFrame:
        """
        Analyze seasonal patterns in encounter data.
        
        Returns:
            pd.DataFrame: Seasonal summary statistics
        """
        logger.info("ANALYZING SEASONAL PATTERNS")

        if not self.encounter_columns:
            logger.error("No encounter columns; run create_encounter_features first")
            return pd.DataFrame()

        primary = self.encounter_columns[0]
        seasonal_summary = self.df.groupby("season")[primary].agg(["count", "mean", "std", "min", "max"]).round(3)
        
        logger.info("Seasonal Encounter Summary:")
        logger.info(f"\n{seasonal_summary}")

        # Winter vs Spring comparison
        winter = self.df[self.df["is_winter"] == 1][primary].dropna()
        spring = self.df[self.df["is_spring"] == 1][primary].dropna()
        
        if len(winter) > 0 and len(spring) > 0:
            w_mean = winter.mean()
            s_mean = spring.mean()
            logger.info(f"Winter mean: {w_mean:.3f} | Spring mean: {s_mean:.3f}")
            
            ratio = (s_mean / w_mean) if w_mean != 0 else np.nan
            logger.info(f"Spring/Winter ratio: {ratio:.3f}")

            # Statistical test
            try:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(winter, spring, equal_var=False, nan_policy="omit")
                logger.info(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
                significance = "Significant difference" if p_value < 0.05 else "No significant difference"
                logger.info(significance)
            except Exception as e:
                logger.warning(f"Could not run t-test: {e}")

        # Food scarcity correlation
        if "food_scarcity_index" in self.df.columns:
            corr = self.df["food_scarcity_index"].corr(self.df[primary])
            logger.info(f"Food scarcity vs encounters correlation: {corr:.3f}")

        return seasonal_summary

    def get_final_feature_set(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Get the final engineered feature set.
        
        Args:
            save_path: Optional path to save the final feature set
            
        Returns:
            pd.DataFrame: Final feature set with all engineered features
        """
        logger.info("CREATING FINAL FEATURE SET")

        # Core temporal & seasonal features
        core_features = [
            "date", "year", "month", "day", "day_of_year", "weekday", "week",
            "season", "is_winter", "is_spring", "is_summer", "is_autumn",
            "month_sin", "month_cos", "days_since_season_start", "is_transition_period",
            "food_scarcity_index", "food_abundance_index"
        ]

        # Encounter features
        encounter_features = []
        for enc in self.encounter_columns:
            encounter_features.extend([
                enc,
                f"{enc}_rolling_7d", f"{enc}_rolling_14d", f"{enc}_rolling_30d",
                f"{enc}_std_7d", f"{enc}_std_14d", f"{enc}_std_30d",
                f"has_{enc}", f"days_since_last_{enc}", f"{enc}_frequency"
            ])

        # Interaction features
        interaction_features = [
            "winter_encounters", "spring_encounters",
            "scarcity_encounter_ratio", "abundance_encounter_product",
            "prev_month_encounters", "prev_season_effect", "winter_rolling_encounters"
        ]

        # Behavior features
        behavior_features = []
        for behavior in self.behavior_columns:
            behavior_features.extend([
                behavior, f"{behavior}_zscore",
                f"{behavior}_rolling_7d", f"{behavior}_rolling_14d",
                f"winter_{behavior}", f"spring_{behavior}"
            ])

        # Combine all features and filter to existing columns
        all_features = core_features + encounter_features + interaction_features + behavior_features
        final_columns = [col for col in all_features if col in self.df.columns]

        final_df = self.df[final_columns].copy()

        if save_path:
            final_df.to_csv(save_path, index=False)
            logger.info(f"Final feature set saved to: {save_path}")

        logger.info(f"Final feature set prepared: {final_df.shape[0]} rows, {final_df.shape[1]} cols")
        return final_df

    def create_all_features(self, date_col: Optional[str] = None, 
                          encounter_cols: Optional[List[str]] = None,
                          location: str = "australia") -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            date_col: Date column name
            encounter_cols: Encounter column names
            location: Geographic location for seasonal mapping
            
        Returns:
            pd.DataFrame: Complete engineered feature set
        """
        logger.info("RUNNING COMPLETE FEATURE ENGINEERING PIPELINE")
        
        # Analyze data structure
        self.analyze_data_structure()
        
        # Create temporal features
        self.prepare_temporal_features(date_col)
        
        # Create seasonal features
        self.create_seasonal_features(location)
        
        # Create encounter features
        self.create_encounter_features(encounter_cols)
        
        # Create interaction features
        self.create_seasonal_interaction_features()
        
        # Analyze patterns
        self.analyze_seasonal_patterns()
        
        # Return final feature set
        return self.get_final_feature_set()


if __name__ == "__main__":
    # Example usage
    logger.info("BatSeasonalFeatureEngineer module loaded successfully")