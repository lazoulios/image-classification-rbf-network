import time
import utils
import numpy as np
from sklearn.metrics import accuracy_score
from rbf import RBFNetwork  

def train_and_evaluate(model, model_name, filename, x_train, y_train, x_test, y_test):
    print(f"\n--- Processing: {model_name} ---")
    
    # Load or Fit logic
    clf = utils.load_model(filename)
    
    if clf is None:
        print(f"  Training {model_name}...")
        start_time = time.time()
        clf = model
        clf.fit(x_train, y_train)
        train_time = time.time() - start_time
        print(f"  Training completed in {train_time:.2f} seconds.")
        utils.save_model(clf, filename)
    else:
        print("  Model loaded from disk.")

    # Evaluate
    print("  Evaluating on Test set...")
    start_test = time.time()
    y_pred = clf.predict(x_test)
    test_time = time.time() - start_test
    acc = accuracy_score(y_test, y_pred)
    
    print(f"  > Accuracy: {acc*100:.2f}%")
    print(f"  > Prediction Time: {test_time:.2f}s")
    
    # Save predictions for visualization
    np.save(f"models/{filename}_preds.npy", y_pred)
    
    return acc

def main():
    # 1. Load FULL Data with PCA
    (x_train, y_train), (x_test, y_test) = utils.load_data()
    
    # --- MODEL: RBF Neural Network ---
    # Αυξάνω λίγο τα κέντρα (200) επειδή έχουμε Full Dataset (50k εικόνες)
    # για να πιάσει καλύτερη απόδοση.
    print("\nInitializing RBF Network...")
    rbf_net = RBFNetwork(n_centers=200, gamma=0.001, random_state=42)
    
    acc_rbf = train_and_evaluate(rbf_net, "RBF Network (200 Centers)", "rbf_net_full_cifar10.pkl", 
                                 x_train, y_train, x_test, y_test)

    # --- Final Summary (With your existing results) ---
    print("\n" + "="*45)
    print("FINAL COMPARISON RESULTS (Full Dataset)")
    print("="*45)
    print(f"RBF Network (Ours):     {acc_rbf*100:.2f}%")
    # Αυτά είναι τα νούμερα από την εικόνα που έστειλες:
    print(f"KNN (K=1) [Baseline]:   37.38% (Reference)")
    print(f"KNN (K=3) [Baseline]:   35.34% (Reference)")
    print(f"Nearest Centroid:       27.75% (Reference)")
    print("="*45)

main()