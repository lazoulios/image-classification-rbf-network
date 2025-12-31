import matplotlib.pyplot as plt
import seaborn as sns
import utils
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.datasets import cifar10
import os

if not os.path.exists("images"):
    os.makedirs("images")

# Load raw data for plotting images
(_, _), (x_test_img, y_test_true) = cifar10.load_data()
y_test_true = y_test_true.ravel()

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"images/{filename}")
    plt.close()
    print(f"Saved {filename}")

def main():
    # Load PCA data just to ensure consistency (y_test needed)
    _, (_, y_test) = utils.load_data()
    
    # Load RBF Predictions
    pred_file = "models/rbf_net_full_cifar10.pkl_preds.npy"
    
    try:
        y_pred_rbf = np.load(pred_file)
        
        # 1. Confusion Matrix for RBF
        plot_confusion_matrix(y_test, y_pred_rbf, "RBF Network Confusion Matrix", "rbf_cm.png")
        
        # 2. Example Predictions for RBF
        plt.figure(figsize=(12, 6))
        # Pick 10 random images
        indices = np.random.choice(len(x_test_img), 10, replace=False)
        
        for i, idx in enumerate(indices):
            plt.subplot(2, 5, i+1)
            plt.imshow(x_test_img[idx])
            true_label = class_names[y_test_true[idx]]
            pred_label = class_names[y_pred_rbf[idx]]
            
            # Green title if correct, Red if wrong
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig("images/rbf_examples.png")
        print("Saved rbf_examples.png")
        
    except FileNotFoundError:
        print(f"Error: Could not find {pred_file}. Run main.py first!")

if __name__ == "__main__":
    main()