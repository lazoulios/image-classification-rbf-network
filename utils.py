import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_data():
    """
    Loads FULL CIFAR-10 dataset (50,000 train images).
    Scales and applies PCA (>90% variance).
    """
    print("\n[Data] Loading FULL CIFAR-10 dataset...")
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Flatten images (32x32x3 -> 3072)
    x_train_flat = x_train.reshape(x_train.shape[0], -1).astype('float32')
    x_test_flat = x_test.reshape(x_test.shape[0], -1).astype('float32')
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Scaling (Standardization)
    print("[Data] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)

    # PCA Implementation (>90% variance)
    print("[Data] Applying PCA (retaining >90% variance)...")
    # Σημείωση: Στα 50.000 δείγματα, το PCA θα πάρει 1-2 λεπτά.
    pca = PCA(n_components=0.90, random_state=42)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    
    print(f"[Data] Features reduced from {x_train_flat.shape[1]} to {x_train_pca.shape[1]}")
    
    return (x_train_pca, y_train), (x_test_pca, y_test)

def save_model(model, filename):
    path = os.path.join(MODELS_DIR, filename)
    print(f"[Utils] Saving model to {path}...")
    joblib.dump(model, path)

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        print(f"[Utils] Loading model from {path}...")
        return joblib.load(path)
    return None