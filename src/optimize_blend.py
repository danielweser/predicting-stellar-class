import os
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

def evaluate_blend(weights, xgb_probs, mlp_probs, y_true):
    weights = weights / np.sum(weights)
    blended_probs = (weights[0] * xgb_probs) + (weights[1] * mlp_probs)
    blended_preds = np.argmax(blended_probs, axis=1)
    # Return negative BAS for minimization
    return -balanced_accuracy_score(y_true, blended_preds)

def main():
    os.makedirs("../models", exist_ok=True)
    print("Loading serialized OOF probabilities...")
    
    try:
        xgb_probs = np.load("../data/xgb_oof_probs.npy")
        mlp_probs = np.load("../data/mlp_oof_probs.npy")
        y_true = np.load("../data/y_true.npy")
    except FileNotFoundError as e:
        print(f"Error loading dependencies: {e}")
        return

    print("Optimizing ensemble weights...")
    initial_weights = [0.5, 0.5]
    bounds = ((0, 1), (0, 1))
    
    result = minimize(
        evaluate_blend, 
        initial_weights, 
        args=(xgb_probs, mlp_probs, y_true), 
        method='Nelder-Mead',
        bounds=bounds
    )
    
    best_weights = result.x / np.sum(result.x)
    best_score = -result.fun
    
    print("-" * 40)
    print(f"Optimal XGBoost Weight: {best_weights[0]:.4f}")
    print(f"Optimal PyTorch Weight: {best_weights[1]:.4f}")
    print(f"Optimized Ensemble BAS: {best_score:.4f}")
    print("-" * 40)
    
    np.save("../models/optimal_blend_weights.npy", best_weights)

if __name__ == "__main__":
    main()