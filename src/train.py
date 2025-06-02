'''
Training utilities for dermoscopic melanoma classification.
'''
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from model import LiteSkinLesionClassifier
from preprocessing import get_transforms, ISICDataset
from train_utils import entropy_weighted_loss, set_seed

def train_one_fold(model, train_loader, val_loader, device, epochs, lr, use_entropy_loss):
    """
    Train the model for one fold of cross-validation.
    Args:
        model (LiteSkinLesionClassifier): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for the optimizer.
        use_entropy_loss (bool): Whether to use entropy-weighted loss.
    Returns:
        tuple: Best ROC-AUC score and accuracy on the validation set.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_auc = 0
    patience, patience_counter = 10, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            
            if use_entropy_loss:
                loss = entropy_weighted_loss(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        print(f"Val Accuracy: {acc:.4f} - ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(config['model_save_path'], f"best_model_{config['architecture']}.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    return best_auc, acc

if __name__ == "__main__":
    with open("../config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    csv_path = config['train_val_csv']
    image_size = (config['image_w'], config['image_h'])
    batch_size = config['batch_size']
    epochs = config['num_epochs']
    lr = config['learning_rate']
    k = config['k_fold']
    model_name = config['architecture']
    pretrained = config['pretrained']
    use_entropy_loss = config['uncertainty_loss']

    df = pd.read_csv(csv_path)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config['seed'])

    all_aucs, all_accs = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n Fold {fold + 1}/{k} ------------------")

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val   = df.iloc[val_idx].reset_index(drop=True)

        train_ds = ISICDataset(df_train, transform=get_transforms(image_size,
                                                                  horizontal_flip=config['data_augmentation']['horizontal_flip'],
                                                                  vertical_flip=config['data_augmentation']['vertical_flip'],
                                                                  rotation=config['data_augmentation']['rotation'])[0])
        val_ds   = ISICDataset(df_val, transform=get_transforms(image_size)[1])

        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=config['num_workers'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=config['num_workers'], shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LiteSkinLesionClassifier(model_name=model_name, pretrained=pretrained).to(device)

        auc, acc = train_one_fold(model, train_loader, val_loader, device, epochs, lr, use_entropy_loss)
        all_aucs.append(auc)
        all_accs.append(acc)

        print(f"Fold {fold+1} ROC-AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    print("\nK-Fold Results Summary:")
    print(f"Mean ROC-AUC: {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    print(f"Mean Accuracy: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
