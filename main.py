# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

from gitdb.util import write

from covid import COVID19Dataset

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# network
from network import My_Model

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap
import optuna
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'select_all': False,  # Whether to use all features.
    'select_k': 24,  # num of selected features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 3000,  # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-3,
    'T_max': 2000,  # Number of epochs to restart learning rate.
    'weight_decay': 1e-4,  # L2 regularization strength.
    'grad_norm_max': 10.0,  # Gradient clipping.
    'early_stop': 500,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

features = pd.read_csv('./data/covid.train.csv')
x_data, y_data = features.iloc[:, 0:117], features.iloc[:, 117]

#try choose your k best features
selector = SelectKBest(score_func=f_regression, k=config['select_k'])
result = selector.fit(x_data, y_data)

#result.scores_ inclues scores for each features
#np.argsort sort scores in ascending order by index, we reverse it to make it descending.
idx = np.argsort(result.scores_)[::-1]
print(f"Top {config['select_k']} Best feature score ")
print(result.scores_[idx[:config['select_k']]])

print(f"\nTop {config['select_k']} Best feature index ")
print(idx[:config['select_k']])

print(f"\nTop {config['select_k']} Best feature name ")
print(x_data.columns[idx[:config['select_k']]])

selected_idx = list(np.sort(idx[:config['select_k']]))
print(selected_idx)
print(x_data.columns[selected_idx])


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = selected_idx  # TODO: Select suitable feature columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


def auto_select_feat(train_data, valid_data, test_data, top_k_after_shap=30):
    """
    两步自动特征选择：
    1) LassoCV 稀疏化
    2) LightGBM+SHAP 再挑非线性重要特征
    返回已切片好的 x_train / x_valid / x_test
    """
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    X_train, X_valid, X_test = train_data[:, 1:-1], valid_data[:, 1:-1], test_data[:, 1:]

    # ---------- 第 1 步：LassoCV ----------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    lasso = LassoCV(cv=5, random_state=5201314, max_iter=10000, n_jobs=-1)
    lasso.fit(X_train_s, y_train)

    mask_lasso = lasso.coef_ != 0
    idx_lasso = np.where(mask_lasso)[0]
    if len(idx_lasso) == 0:
        raise ValueError("Lasso 把所有特征都干掉了！调大 alpha 或换方法。")

    X_train_lasso = X_train[:, idx_lasso]
    X_valid_lasso = X_valid[:, idx_lasso]
    X_test_lasso = X_test[:, idx_lasso]

    print(f"LassoCV 保留 {len(idx_lasso)}/{X_train.shape[1]} 个特征")

    # ---------- 第 2 步：LightGBM + SHAP ----------
    lgb_train = lgb.Dataset(X_train_lasso, y_train)
    params = dict(objective='regression', verbosity=-1, seed=5201314)
    gbm = lgb.train(params, lgb_train, num_boost_round=200)

    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(X_train_lasso)
    importance = np.abs(shap_values).mean(axis=0)
    idx_shap = np.argsort(importance)[-top_k_after_shap:]  # Top-K

    print(f"SHAP 再选 {len(idx_shap)} 个特征")

    # ---------- 返回最终切片 ----------
    return (X_train_lasso[:, idx_shap],
            X_valid_lasso[:, idx_shap],
            X_test_lasso[:, idx_shap],
            y_train,
            y_valid)


if torch.cuda.is_available():
    device = 'cuda'
    print("GPU is available. Using CUDA.")
else:
    print("GPU is not available. Using CPU.")
    device = 'cpu'

# Set seed for reproducibility
same_seed(config['seed'])

# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('./data/covid.train.csv').values, pd.read_csv('./data/covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])
# x_train, x_valid, x_test, y_train, y_valid = auto_select_feat(train_data, valid_data, test_data)

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
    COVID19Dataset(x_valid, y_valid), \
    COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                     T_max=config['T_max'],
                                                                     eta_min=config['learning_rate'] / 100)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                                                  T_0=2, T_mult=2, eta_min=config['learning_rate']/100)

    writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        lr = scheduler. get_last_lr()[0]

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_norm_max'])
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
            writer.add_scalar('grad_norm', grad_norm, step)

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('Learning Rate', lr, step)

        scheduler.step()

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            writer.add_hparams(hparam_dict=config, metric_dict={'hparam/valid_loss': best_loss})
            writer.flush()
            writer.close()
            print(f'\nModel is not improving, so we halt the training session with the best_loss: {best_loss:.5f}')
            return
    else:
        writer.add_hparams(hparam_dict=config, metric_dict={'hparam/valid_loss': best_loss})
        writer.flush()
        writer.close()
        print('{best_loss:.5f} \n')

model = My_Model(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred.csv')

print(torch.__version__)
print(torch.cuda.is_available())
