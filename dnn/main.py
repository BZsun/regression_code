import os
import json
import pandas as pd
import numpy as np
import torch
import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.signal import medfilt
from torch.utils.data import (
    Dataset,
    DataLoader, 
    TensorDataset
)
import torch.optim as optim
from network import MyDataset, MyModel
import eval_function
from ipdb import set_trace

def save_checkpoint(model, model_name):
    torch.save(model.state_dict(), model_name)
    print('------ Checkpoint saved to {} ------'.format(model_name))


def denoise_function(df, feature_denoise, gen_denoise):
    window_size = 3  # 窗口大小
    for ind, col in enumerate(feature_denoise):
        df[gen_denoise[ind]] = medfilt(df[col], window_size)

    return df

def down_sample(df):
    out = pd.DataFrame([])
    f1_key = list(set(df['f1']))
    f2_key = list(set(df['f2']))
    for _f1 in f1_key:
        for _f2 in f2_key:
            data = df[(df['f1']==_f1) & (df['f2']==_f2)]
            if not data.empty:
                data = data.drop(['f1', 'f2'], axis=1)
                data = data.sort_values('time')
                data['time'] = pd.to_datetime(data['time'])
                data = data.set_index('time')
                data = data.resample('15min').mean()
                data.dropna(inplace=True)
                data.insert(0, "f2", [_f2] * data.shape[0])
                data.insert(0, "f1", [_f1] * data.shape[0])
                out = pd.concat([out, data])
    out.reset_index(level='time', inplace=True)
    return out

def preporcess_function(filename):
    data = pd.read_csv(filename)
    data = data.dropna(subset=['time'], how='any')
    data = data.dropna(how='all')
    data = data.dropna(how='all', axis=1)
    data.fillna(0, inplace=True)
    # 1. 离散化处理
    data['f8'] = pd.to_numeric(data['f8'], errors='coerce')
    data['f8'].fillna(0)
    data['f32'] = pd.cut(data['f8'], bins=4, labels=False)
    # 2. 降噪
    data = denoise_function(data, feature_denoise, gen_denoise)
    # 3. 下采样
    data = down_sample(data)
    # 4. 降维
    pca = PCA(n_components=4)
    dim_reduce = data[feature_dim_reduce]
    new_data = pca.fit_transform(dim_reduce)
    new_df = pd.DataFrame(new_data, columns=gen_dim_reduce)
    for col in gen_dim_reduce:
        data[col] = new_df[col]
    data.to_csv('./data/x_test_data_new.csv', index=False)
    
    return data

def data_polomerize(df):
    out = []
    names = ['f2']
    columns = df.columns.tolist()
    columns.pop(columns.index('time'))
    columns.pop(columns.index('f1'))
    columns.pop(columns.index('f2'))
    f2_key = list(set(df['f2']))
    for key in f2_key:
        mid = [key]
        data = df[df['f2']==key]
        for col in columns:
            _avg = col + '_Avg'
            _std = col + '_Std'
            _diff = col + '_Diff'
            v_avg = np.mean(data[col])
            v_std = np.std(data[col])
            v_diff = np.max(data[col]) - np.min(data[col])
            mid.extend([v_avg, v_std, v_diff])
            if _avg not in names and _std not in names and _diff not in names:
                names.extend([_avg, _std, _diff])
        out.append(mid)
    res = pd.DataFrame(out, columns=names)
    res.to_csv('./data/x_test_data_describe.csv')

def load_loader(df_x, df_y):
    merged_df = pd.merge(df_x, df_y, on='f2')
    inp_df = merged_df.iloc[:, 3:]
    x_train, x_test, y_train, y_test = train_test_split(
                                                        inp_df.iloc[:, :-1], 
                                                        inp_df["y"], 
                                                        test_size=0.3, 
                                                        random_state=3
                                                        )
    corpus_train = pd.concat([x_train, y_train], axis=1)
    corpus_eval = pd.concat([x_test, y_test], axis=1)
    train_set = MyDataset(corpus_train)
    eval_set = MyDataset(corpus_eval)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_set, batch_size=64, shuffle=True, num_workers=4)

    return merged_df, train_loader, eval_loader

def evaluate(model, loss_fn, eval_loader):
    model = model.to(device)
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(eval_loader, desc="Eval"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            total += len(inputs)
        _loss = total_loss / total
    return _loss

def train_model(train_loader, eval_loader, device):
    input_dim = 36
    hidden_dim = 72
    output_dim = 1
    num_epochs = 10
    best_loss = 100
    model = MyModel(input_dim, hidden_dim, output_dim)
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    loss_fn = eval_function.mse_score
    length = len(train_loader)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in tqdm.tqdm(train_loader, desc="Train"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        _loss = evaluate(model, loss_fn, eval_loader)
        if _loss < best_loss:
            best_loss = _loss
            save_checkpoint(model, 'regression_model.pth')
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss/length))

def execute_infer(infer_file):
    input_dim = 36
    hidden_dim = 72
    output_dim = 1
    model = MyModel(input_dim, hidden_dim, output_dim)
    model_name = './regression_model.pth'
    model.load_state_dict(torch.load(model_name))
    model = model.to(device)
    model.eval()
    inf_df = pd.read_csv(infer_file).iloc[:, 3:-1]
    inf_features = torch.tensor(inf_df.values, dtype=torch.float32)
    inf_set = TensorDataset(inf_features)
    inf_loader = DataLoader(inf_set, batch_size=64, shuffle=False, num_workers=4)
    out = []
    with torch.no_grad():
        for batch_x in tqdm.tqdm(inf_loader, desc="Infer"):
            batch_x = batch_x[0].to(device)
            pred = model(batch_x).squeeze().tolist()
            out.extend(pred)
    inf_df['pred'] = out
    inf_df.to_csv('result.csv', index=False)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feature_dispersed = ['f8']
    gen_dispersed = ['f32']
    feature_denoise = ['f9','f10','f11']
    gen_denoise = ['f33', 'f34', 'f35']
    feature_dim_reduce = ['f12','f13', 'f14', 'f15','f16','f17','f18','f19','f20','f21']
    gen_dim_reduce = ['f36', 'f37', 'f38', 'f39']
    input_X_file = './data/x_test_data.csv'
    pre_dataframe = preporcess_function(input_X_file)
    # data_polomerize(pre_dataframe)
    df_y = pd.read_csv('./data/y_test_data.csv')
    merged_df, train_loader, eval_loader = load_loader(pre_dataframe, df_y)
    train_model(train_loader, eval_loader, device)
    merged_df.to_csv('./data/infer_file.csv', index=False)
    execute_infer('./data/infer_file.csv')