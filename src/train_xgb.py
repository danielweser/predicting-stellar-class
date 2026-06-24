import os
import gc
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

def main():
    os.makedirs("../models", exist_ok=True)
    
    print("Loading engineered dataset...")
    df = pd.read_parquet("../data/train_data_engineered.parquet")
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    y = df.pop('class').values
    X = df.values
    
    num_classes = len(np.unique(y))
    oof_probs = np.zeros((len(X), num_classes))
    
    xgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.06980087900851036,
        'max_depth': 9,
        'min_child_weight': 8,
        'subsample': 0.7780490432635381,
        'colsample_bytree': 0.7565108185040311,
        'gamma': 1.3784772488088997,
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda'
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Initiating XGBoost 5-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
        model.fit(
            X_train, y_train, 
            sample_weight=sample_weights, 
            eval_set=[(X_val, y_val)], 
            verbose=False
        )
        
        model.save_model(f"../models/xgb_fold_{fold}.json")
        
        fold_probs = model.predict_proba(X_val)
        oof_probs[val_idx] = fold_probs
        
        fold_preds = np.argmax(fold_probs, axis=1)
        score = balanced_accuracy_score(y_val, fold_preds)
        print(f"Fold {fold} BAS: {score:.4f}")

        del model, X_train, X_val, y_train, y_val
        gc.collect()

    final_score = balanced_accuracy_score(y, np.argmax(oof_probs, axis=1))
    print(f"XGBoost OOF BAS: {final_score:.4f}")
    
    np.save("../data/xgb_oof_probs.npy", oof_probs)
    np.save("../data/y_true.npy", y)

if __name__ == "__main__":
    main()