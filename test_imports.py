#!/usr/bin/env python3
"""
Test script to isolate import issues
"""

print("Starting test imports...")

try:
    import pandas as pd
    print("✅ pandas imported successfully")
except Exception as e:
    print(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except Exception as e:
    print(f"❌ numpy import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported successfully")
except Exception as e:
    print(f"❌ matplotlib import failed: {e}")

try:
    import seaborn as sns
    print("✅ seaborn imported successfully")
except Exception as e:
    print(f"❌ seaborn import failed: {e}")

try:
    from sklearn.linear_model import LinearRegression
    print("✅ scikit-learn imported successfully")
except Exception as e:
    print(f"❌ scikit-learn import failed: {e}")

try:
    import yaml
    print("✅ yaml imported successfully")
except Exception as e:
    print(f"❌ yaml import failed: {e}")

try:
    import plotly
    print("✅ plotly imported successfully")
except Exception as e:
    print(f"❌ plotly import failed: {e}")

print("\nTesting custom module imports...")

try:
    from src.data.data_processor import DataProcessor
    print("✅ DataProcessor imported successfully")
except Exception as e:
    print(f"❌ DataProcessor import failed: {e}")

try:
    from src.features.feature_engineer import BatSeasonalFeatureEngineer
    print("✅ BatSeasonalFeatureEngineer imported successfully")
except Exception as e:
    print(f"❌ BatSeasonalFeatureEngineer import failed: {e}")

try:
    from src.models.regression_models import BatBehaviorModeler
    print("✅ BatBehaviorModeler imported successfully")
except Exception as e:
    print(f"❌ BatBehaviorModeler import failed: {e}")

try:
    from src.visualization.plots import BatBehaviorVisualizer
    print("✅ BatBehaviorVisualizer imported successfully")
except Exception as e:
    print(f"❌ BatBehaviorVisualizer import failed: {e}")

print("\nAll import tests completed!")