import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

def load_data():
    """Load validation data and predictions"""
    # Load validation data
    val_data = pd.read_csv('processed_data/val_rf.csv')
    y_true = val_data['ClassId']
    return y_true

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix with proper labels and formatting
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axes with larger size
    plt.figure(figsize=(20, 16))
    
    # Plot confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    # Add labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Traffic Sign Classification')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
    return cm

def analyze_errors(cm, y_true):
    """
    Analyze and print most common misclassifications
    """
    n_classes = cm.shape[0]
    
    # Calculate per-class accuracy
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    
    # Find worst performing classes
    worst_classes = np.argsort(class_accuracy)[:5]
    print("\nWorst performing classes:")
    for cls in worst_classes:
        print(f"Class {cls}: {class_accuracy[cls]*100:.2f}% accuracy")
        
        # Find most common misclassifications for this class
        misclassifications = [(i, cm[cls,i]) for i in range(n_classes) if i != cls]
        misclassifications.sort(key=lambda x: x[1], reverse=True)
        
        print("Most common misclassifications:")
        for pred_cls, count in misclassifications[:3]:
            if count > 0:
                print(f"  Predicted as class {pred_cls}: {count} times")
        print()

def main():
    # Load validation data
    y_true = load_data()
    
    # Load your best model's predictions
    # You'll need to modify this based on where you saved your predictions
    try:
        y_pred = pd.read_csv('best_model_val_predictions.csv')['Prediction']
    except FileNotFoundError:
        print("Please ensure you have saved your best model's validation predictions as 'best_model_val_predictions.csv'")
        return
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred)
    
    # Analyze errors
    analyze_errors(cm, y_true)
    
    print("Analysis complete! Check confusion_matrix.png for the visualization.")

if __name__ == "__main__":
    main() 