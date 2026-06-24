import os
import gc
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TabularMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TabularMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Mish(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Mish(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

def main():
    os.makedirs("../models", exist_ok=True)
    
    print("Loading raw dataset...")
    df = pd.read_csv("../data/train_data.csv")
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    y = df.pop('class').values
    X = df.values
    
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    oof_probs = np.zeros((len(X), num_classes))
    
    class_counts = np.bincount(y)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    batch_size = 16384 
    
    print("Initiating MLP 5-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        
        joblib.dump(scaler, f"../models/scaler_fold_{fold}.pkl")
        
        # Load fold data directly into VRAM to bypass host-to-device bottlenecks
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y[train_idx], dtype=torch.long).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y[val_idx], dtype=torch.long).to(device)
        
        model = TabularMLP(num_features, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_fold_bas = 0.0
        patience, patience_counter = 10, 0
        best_model_state = None
        best_val_probs = None
        num_samples = len(X_train_t)
        
        for epoch in range(100):
            model.train()
            permutation = torch.randperm(num_samples)
            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i+batch_size]
                batch_X, batch_y = X_train_t[indices], y_train_t[indices]
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                outputs = model(X_val_t)
                probs = torch.softmax(outputs, dim=1).cpu().numpy() 
                preds = np.argmax(probs, axis=1)
            
            epoch_bas = balanced_accuracy_score(y_val_t.cpu().numpy(), preds)
            scheduler.step(epoch_bas)
            
            if epoch_bas > best_fold_bas:
                best_fold_bas = epoch_bas
                patience_counter = 0
                best_val_probs = probs
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        torch.save(best_model_state, f"../models/mlp_fold_{fold}.pth")
        oof_probs[val_idx] = best_val_probs
        print(f"Fold {fold} BAS: {best_fold_bas:.4f}")
        
        del model, X_train_t, y_train_t, X_val_t, y_val_t
        torch.cuda.empty_cache()
        gc.collect()
        
    final_score = balanced_accuracy_score(y, np.argmax(oof_probs, axis=1))
    print(f"PyTorch OOF BAS: {final_score:.4f}")
    
    np.save("../data/mlp_oof_probs.npy", oof_probs)

if __name__ == "__main__":
    main()