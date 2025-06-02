'''
Test script for evaluating the LiteSkinLesionClassifier on a test dataset.
'''
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from model import LiteSkinLesionClassifier
from preprocessing import get_transforms, ISICDataset
import yaml
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score

def test_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.
    Args:
        model (LiteSkinLesionClassifier): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
    Returns:
        tuple: Test accuracy, ROC-AUC score, true labels, and predicted labels.
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, auc, all_labels, all_preds

if __name__ == "__main__":
    with open("../config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    image_size = (config['image_w'], config['image_h'])
    batch_size = config['batch_size']
    model_name = config['architecture']
    pretrained = config['pretrained']
    test_csv = config['test_csv']
    model_save_path = config['model_save_path']

    # Load the test dataset
    df_test = pd.read_csv('../data/splits/test_dataset.csv')
    test_ds = ISICDataset(df_test, transform=get_transforms(image_size)[1])
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=config['num_workers'], shuffle=False)

    # Load the best model saved during training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiteSkinLesionClassifier(model_name=model_name, pretrained=pretrained).to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_path, "best_model_efficientnet_b0.pt")))

    test_acc, test_auc, all_labels, all_preds = test_model(model, test_loader, device)


    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    # Sensitivity = Recall for the positive class
    sensitivity = recall
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) if tn + fp != 0 else 0

    cr = classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'], output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(os.path.join('../experiments/results', "classification_report.csv"))
    
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
    print(confusion_matrix(all_labels, all_preds))
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test Sensitivity: {sensitivity:.4f}")
    print(f"Test Specificity: {specificity:.4f}")

    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('../experiments/results', "confusion_matrix.png"))
