"""
@ models.py: Netual Models, Validation and Evaluation.
@ Thesis: Geographical Data and Predictions of Windmill Energy Production
@ Weisi Li
@ liwb@itu.dk, liweisi8121@hotmail.com
"""

import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score

from constant import model_path, error_path


class MLP_Regression(nn.Module):
    """
    MLP Regression Model
    """
    def __init__(self, input_size, hidden_size, f_active, bias=0):
        super(MLP_Regression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1.bias.data.fill_(bias)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.d = nn.Dropout(p=0.5)

        f_active = F.sigmoid
        f_active = F.tanh if f_active == "tanh" else f_active
        f_active = F.leaky_relu if f_active == "leaky_relu" else f_active
        f_active = F.relu if f_active == "relu" else f_active
        self.f_active = f_active

    def forward(self, x):
        x = self.f_active(self.fc1(x))
        # x = self.f_active(self.fc2(x))
        x = self.fc3(x)
        return x


class LSTM_Regression(nn.Module):
    """
    LSTM Regression Model
    """

    def __init__(self, input_size, hidden_size, seq_len, n_layers=2):
        super(LSTM_Regression, self).__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=0.5
            )

        self.linear = nn.Linear(in_features=n_hidden, out_features=1)


    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred


def create_sequences(data, seq_length):
    """
    Create Sequences for LSTM
    """
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def train_model(
    model, 
    lr,
    num_epochs,
    X_train, 
    y_train, 
    X_test=None, 
    y_test=None,
    path=None
    ):
    loss_fn = torch.nn.SmoothL1Loss()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = num_epochs

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        if model is LSTM_Regression:
            model.reset_hidden_state()
 
        y_pred = model(X_train)
        loss = loss_fn(y_pred.float(), y_train)

        if X_test is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()

            if t % 10 == 0 or t == num_epochs-1:  
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')

        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    if path is not None:
        torch.save(model, path)

    return model.eval(), train_hist, test_hist



def train_model_flexible(
    input_size,
    hidden_size,
    f_active,
    lr,
    num_epochs,
    X_train, 
    y_train, 
    X_test=None, 
    y_test=None,
    loss_record=True, 
    path=None
    ):

    model = MLP_Regression(
            input_size=input_size,
            hidden_size=hidden_size,
            f_active = f_active
            )
    
    model.share_memory()

    loss_fn = torch.nn.SmoothL1Loss()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = num_epochs

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        if model is LSTM_Regression:
            model.reset_hidden_state()
 
        y_pred = model(X_train)
        loss = loss_fn(y_pred.float(), y_train)

        if loss_record:
            if X_test is not None:
                with torch.no_grad():
                    y_test_pred = model(X_test)
                    test_loss = loss_fn(y_test_pred.float(), y_test)
                test_hist[t] = test_loss.item()

                if t % 10 == 0 or t == num_epochs-1:  
                    print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')

            elif t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()}')

            train_hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    if path is not None:
        torch.save(model, path)

    return model.eval(), train_hist, test_hist



def timeseries_kfold_validation_training(df, features, target, n_groups, model, lr=0.001, num_epochs=30):
    """
    KFold Validation
    """
    st = time.time()

    dtype = torch.float
    # train_hists, test_hists = [], []
    k_scores = [] 

    start = df["TIME_CET"].min()
    end = df["TIME_CET"].max()
    gap = (end - start)/n_groups

    for i in range(1, n_groups):
        pivot = start + gap * i
        end = pivot + gap
        if i == n_groups-1:
            end = df["TIME_CET"].max()
        train_index = (df[df["TIME_CET"] <= pivot].index).append(df[df["TIME_CET"] > end].index)
        test_index = df[df["TIME_CET"] > pivot][df["TIME_CET"] <= end].index

        train = df.iloc[train_index]
        test = df.iloc[test_index]
        
        x_train, y_train = train[features].values, train[target].values
        x_test, y_test = test[features].values, test[target].values

        x_train_tensor = torch.tensor(x_train, dtype = dtype)
        y_train_tensor = torch.tensor(y_train, dtype = dtype)
        x_test_tensor = torch.tensor(x_test, dtype = dtype)
        y_test_tensor = torch.tensor(y_test, dtype = dtype)

        _, _, test_hist = train_model(model, 
                                        lr,
                                        num_epochs,
                                        x_train_tensor, 
                                        y_train_tensor, 
                                        x_test_tensor, 
                                        y_test_tensor
                                        )
        
        k_scores.append(np.mean(test_hist.tolist()))

    et = time.time()
    print("NN k-fold-validation time: ", et - st)

    return k_scores


def train_test_validation(df, features, model, lr=0.001, num_epochs=30):
    st = time.time()

    dtype = torch.float
    train_hists, test_hists = [], []

    days = np.unique(df["TIME_CET"].astype(str).apply(lambda x: x[:10]))
    days = np.random.choice(days, int(len(days)*0.8))
    
    train_index = df[df["TIME_CET"].isin(days)].index
    test_index = df[~df["TIME_CET"].isin(days)].index

    train = df.iloc[train_index]
    test = df.iloc[test_index]
    
    x_train, y_train = train[features].values, train["VAERDI"].values
    x_test, y_test = test[features].values, test["VAERDI"].values

    x_train_tensor = torch.tensor(x_train, dtype = dtype)
    y_train_tensor = torch.tensor(y_train, dtype = dtype)
    x_test_tensor = torch.tensor(x_test, dtype = dtype)
    y_test_tensor = torch.tensor(y_test, dtype = dtype)

    _, _, test_hist = train_model(model, 
                                    lr,
                                    num_epochs,
                                    x_train_tensor, 
                                    y_train_tensor, 
                                    x_test_tensor, 
                                    y_test_tensor
                                    )
        
    train_hists.extend(train_hist.tolist())
    test_hists.extend(test_hist.tolist())
        
    et = time.time()
    print("NN k-fold-validation time: ", et - st)

    return train_hists, test_hists


RMSE_all = lambda df :(((df["VAERDI"] - df["pred"]) ** 2).sum() / len(df) )** 0.5
NRMSE_all = lambda df : (((df["VAERDI"] - df["pred"]) ** 2).sum() / len(df) )** 0.5 * len(df)/ df["VAERDI"].sum()

def model_evaluation(dfevl, features, model, days=True, ahead=None, path=None):
    """
    Evaluation model in hourly or daily
    """
    df = dfevl.copy()
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(df[features].values, dtype = torch.float)
        y_pred_tensor = model(x_test_tensor)
        df["pred"] = y_pred_tensor.detach().flatten().numpy() 


    max_VAERDI = df["max_VAERDI"][0]

    # Convert to megawatt (A megawatt hour (Mwh) is equal to 1,000 Kilowatt hours (Kwh))
    df["pred"] = df["pred"] * max_VAERDI / 1000
    df["VAERDI"] = df["VAERDI"] * max_VAERDI / 1000
    df["TIME_CET"] = df["TIME_CET"].astype(str)
    
    df = df.groupby("TIME_CET", as_index=False).agg({"VAERDI" : lambda x : x.sum(), "pred" : lambda x : x.sum()})
    df.columns = ["TIME_CET", "VAERDI", "pred"]
    
    if type(ahead) == int:
        df["pred"][ahead:len(df)] = df["pred"][0:len(df)-1]
        df = df.drop(df.tail(1).index)

    if days:
        df["TIME_CET"] = df["TIME_CET"].apply(lambda x :x[:10])
        df = df.groupby("TIME_CET", as_index=False).agg({"VAERDI" : lambda x : x.sum(), "pred" : lambda x : x.sum()})
        df.columns = ["TIME_CET", "VAERDI", "pred"]
 
    df["NMAE"] = (abs(df["VAERDI"] - df["pred"])).cumsum()/ df["VAERDI"].cumsum()
    df["NMSE"] = ((abs(df["VAERDI"] - df["pred"])) ** 2).cumsum()/ df["VAERDI"].cumsum()
    df["RMSE"] = (((df["VAERDI"] - df["pred"]) ** 2).cumsum()/ len(df) )** 0.5
    df["NRMSE"] = (((df["VAERDI"] - df["pred"]) ** 2).cumsum() / df["VAERDI"].cumsum() )** 0.5
    # df["NRMSE"] = (((df["VAERDI"] - df["pred"]) ** 2).cumsum() / len(df) )** 0.5 * len(df)/ df["VAERDI"].cumsum()
    

    if path is not None:
        df.to_csv(path, index=False)
    
    return df


def model_improvement(errors:dict, col):
    """
    Compare the improvements of models
    """
    Imp = {x:[] for x in errors.keys()}
    r2 = {x:[] for x in errors.keys()}
    
    for ecname in errors.keys():
        df = pd.DataFrame([], columns=[x for x in errors.keys() if x != ecname])
        ecrefs = errors.copy()
        ec = ecrefs.pop(ecname)
        for ecrefname, ecref in ecrefs.items():
            I = 100 * (ecref[col] - ec[col]) / (ecref[col])
            df[ecrefname] = I
        Imp[ecname] = df

        r2[ecname] = round(r2_score(ec["VAERDI"], ec["pred"]), 3)
    return Imp, r2


def Grid_Search(x_train_tensor, y_train_tensor, df_evl, df_train, paras, input_size, cols):
    grid_results = pd.DataFrame(columns=["NRMSE_train_all","NRMSE_evl_all", "NRMSE_train_var", "NRMSE_evl_var", "R2_train", "R2_evl", "time"])
    for i, para in enumerate(paras):
        start = time.time()
        model, _, _ = train_model_flexible(input_size = input_size,
                                hidden_size = para[0],
                                f_active = para[1],
                                lr = para[2],
                                num_epochs = para[3],
                                X_train = x_train_tensor, 
                                y_train = y_train_tensor,
                                loss_record=False)
        end = time.time()

        model_train_errs = model_evaluation(df_train, cols, model, days=True)
        model_evl_errs = model_evaluation(df_evl, cols, model, days=True)
        
        r2_train = round(r2_score(model_train_errs["VAERDI"], model_train_errs["pred"]), 2)
        r2_evl = round(r2_score(model_evl_errs["VAERDI"], model_evl_errs["pred"]), 2)

        grid_results = grid_results.append({"NRMSE_train_all": NRMSE_all(model_train_errs),
                                            "NRMSE_evl_all": NRMSE_all(model_evl_errs),
                                            "NRMSE_train_var": model_train_errs["NRMSE"].std(), 
                                            "NRMSE_evl_var": model_evl_errs["NRMSE"].std(), 
                                            "R2_train": r2_train, 
                                            "R2_evl": r2_evl,
                                            "time": end - start}, 
                                            ignore_index=True)
        print(i, " Done")
    return grid_results