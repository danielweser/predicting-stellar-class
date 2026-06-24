import kagglehub
from pathlib import Path
import pandas as pd
import json
import shutil
from engineer_features import manual_feature_engineering

def main():
    """
    1. Download and process dataset from Kaggle competition
    2. Engineer features for XGBoost training
    """
    delete_comp_dir = True  # Optional: delete competition dir after processing

    # Download competition directory from Kaggle
    comp_dir = Path("../competition")
    kagglehub.login()
    kagglehub.competition_download('playground-series-s6e6', output_dir=comp_dir)

    print("Data downloaded.")
    print("Processing data...")

    train_path = comp_dir / "train.csv"
    test_path = comp_dir / "test.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Map columns of string values to integers
    unique_spectra = df_train['spectral_type'].unique().tolist()
    unique_galaxypop = df_train['galaxy_population'].unique().tolist()
    unique_classes = df_train['class'].unique().tolist()

    spectra_dict = {unique_spectra[i]: i for i in range(len(unique_spectra))}
    galaxypop_dict = {unique_galaxypop[i]: i for i in range(len(unique_galaxypop))}
    class_dict = {unique_classes[i]: i for i in range(len(unique_classes))}
    label_dict = {'spectral_type': spectra_dict, 'galaxy_population': galaxypop_dict, 'class': class_dict}

    df_train['spectral_type'] = df_train['spectral_type'].map(spectra_dict)
    df_test['spectral_type'] = df_test['spectral_type'].map(spectra_dict)
    df_train['galaxy_population'] = df_train['galaxy_population'].map(galaxypop_dict)
    df_test['galaxy_population'] = df_test['galaxy_population'].map(galaxypop_dict)
    df_train['class'] = df_train['class'].map(class_dict)

    print("Writing data...")
    # Write data to new directory
    data_dir = Path("../data")
    train_data_path = data_dir / "train_data.csv"
    test_data_path = data_dir / "test_data.csv"
    label_dict_path = data_dir / "label_dict.json"

    data_dir.mkdir(parents=True)
    df_train.to_csv(train_data_path, index=False)
    df_test.to_csv(test_data_path, index=False)

    # Write dictionary of strings & labels
    with open(label_dict_path, "w") as file:
        json.dump(label_dict, file, indent=4)

    print("Data writing complete.")

    # Delete competition directory
    if delete_comp_dir:
        shutil.rmtree(comp_dir, ignore_errors=False)
        print("Deleted competition directory.")

    print("Engineering new features...")
    # Engineer new data features
    train_engineered_path = data_dir / "train_data_engineered.parquet"
    test_engineered_path = data_dir / "test_data_engineered.parquet"
    manual_feature_engineering(df_train, train_engineered_path)
    manual_feature_engineering(df_test, test_engineered_path)

if __name__ == "__main__":
    main()