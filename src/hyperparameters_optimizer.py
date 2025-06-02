'''
Hyperparameter optimization script using Optuna for skin lesion classification.
'''
import optuna
from functools import partial
import torch
from torch.utils.data import DataLoader
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
from train_utils import set_seed
from model import LiteSkinLesionClassifier
from preprocessing import ISICDataset, get_transforms
from train import train_one_fold

def objective(trial, df, image_size, device, config):
    """
    Optuna objective function to optimize hyperparameters for skin lesion classification.
    Args:
        trial (optuna.Trial): The trial object for hyperparameter optimization.
        df (pd.DataFrame): DataFrame containing image paths and labels.
        image_size (tuple): The target image size for transformations.
        device (torch.device): The device to run the model on (CPU or GPU).
        config (dict): Configuration dictionary containing model parameters.
    Returns:
        float: The AUC score for the validation set.
    """
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    dropout = trial.suggest_float('dropout_rate', 0.2, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

    train_idx, val_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, df['label']))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    train_ds = ISICDataset(df_train, transform=get_transforms(image_size)[0])
    val_ds = ISICDataset(df_val, transform=get_transforms(image_size)[1])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = LiteSkinLesionClassifier(model_name=config['architecture'], pretrained=True, dropout_rate=dropout).to(device)

    auc, _ = train_one_fold(model, train_loader, val_loader, device, epochs=10, lr=lr, use_entropy_loss=False)
    return auc


if __name__ == "__main__":
    with open("../config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    df = pd.read_csv(config['train_val_csv'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (config['image_w'], config['image_h'])

    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, df=df, image_size=image_size, device=device, config=config), n_trials=20)

    print("En iyi AUC:", study.best_value)
    print("En iyi parametreler:", study.best_params)
