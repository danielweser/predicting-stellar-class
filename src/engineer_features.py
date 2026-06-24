from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
import gc

def manual_feature_engineering(df, save_path):    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'class' in df.columns:
        y = df.pop('class')

    # Extra precaution: grab only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Store all new columns in a dictionary
    new_features = {}
    
    print(f"Generating combinations for {len(numeric_cols)} columns...")
    
    # 1. Single Column Transforms (Log and Square Root)
    for col in numeric_cols:
        vals = np.abs(df[col].values)
        new_features[f"SQRT({col})"] = np.sqrt(vals)
        new_features[f"LOG({col})"]  = np.log1p(vals)
        
    # 2. Pairwise Transforms (Add, Multiply, Divide)
    for col1, col2 in combinations(numeric_cols, 2):
        val1 = df[col1].values
        val2 = df[col2].values
        
        # Addition & Multiplication
        new_features[f"{col1} + {col2}"] = val1 + val2
        new_features[f"{col1} * {col2}"] = val1 * val2
        
        new_features[f"{col1} / {col2}"] = np.divide(val1, val2, out=np.zeros_like(val1, dtype=float), where=(val2 != 0))
        new_features[f"{col2} / {col1}"] = np.divide(val2, val1, out=np.zeros_like(val2, dtype=float), where=(val1 != 0))
        
    print("Concatenating new features...")
    # Convert dictionary of arrays into DataFrame
    engineered_df = pd.DataFrame(new_features, index=df.index)
    
    # Combine the original columns, new columns, and target class
    final_df = pd.concat([df, engineered_df], axis=1)
    try:
        final_df['class'] = y.values
    except NameError:
        pass

    print("Writing data...")
    # Write data
    final_df.to_parquet(save_path, index=False)
    print(f"Saved successfully to {save_path}.")
    
    print("-" * 50)
    print(f"Original Columns: {len(numeric_cols)}")
    print(f"New Features:     {engineered_df.shape[1]}")
    print(f"Total Columns:    {final_df.shape[1] - 1}") # -1 to not count target
    print("-" * 50)

if __name__ == "__main__":
    data_dir = Path("../Data")
    train_path = data_dir / "train_data.csv"
    train_save_path = data_dir / "train_data_engineered.parquet"
    test_path = data_dir / "test_data.csv"
    test_save_path = data_dir / "test_data_engineered.parquet"

    print("Loading data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    manual_feature_engineering(df_train, train_save_path)
    manual_feature_engineering(df_test, test_save_path)