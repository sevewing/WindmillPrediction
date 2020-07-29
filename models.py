import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from constant import model_path, error_path


class MLP_Regression(nn.Module):

  def __init__(self, input_size, hidden_size):
    super(MLP_Regression, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    self.d = nn.Dropout(p=0.5)

  def forward(self, x):
    x = F.leaky_relu(self.fc1(x))
    # x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))

    return x


class LSTM_Regression(nn.Module):

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
    y_test=None
    ):
    loss_fn = torch.nn.SmoothL1Loss()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = num_epochs

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        if model is LSTM_Regression:
            print("passssss")
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

    return model, train_hist, test_hist



def timeseries_kfold_validation_training(df, features, target, n_groups, model, lr=0.001, num_epochs=30, save_name=None):
    dtype = torch.float
    train_hists, test_hists = [], []

    start = df["TIME_CET"].min()
    end = df["TIME_CET"].max()
    gap = (end - start)/n_groups

    for i in range(1, n_groups):
        pivot = start + gap * i
        end = pivot + gap
        if i == n_groups-1:
            end = df["TIME_CET"].max()
        train_index = df[df["TIME_CET"] <= pivot].index
        test_index = df[df["TIME_CET"] > pivot][df["TIME_CET"] <= end].index

        train = df.iloc[train_index]
        test = df.iloc[test_index]
        
        x_train, y_train = train[features].values, train[target].values
        x_test, y_test = test[features].values, test[target].values

        x_train_tensor = torch.tensor(x_train, dtype = dtype)
        y_train_tensor = torch.tensor(y_train, dtype = dtype)
        x_test_tensor = torch.tensor(x_test, dtype = dtype)
        y_test_tensor = torch.tensor(y_test, dtype = dtype)

        model, train_hist, test_hist = train_model(
                                        model, 
                                        lr,
                                        num_epochs,
                                        x_train_tensor, 
                                        y_train_tensor, 
                                        x_test_tensor, 
                                        y_test_tensor
                                        )
        
        train_hists.extend(train_hist.tolist())
        test_hists.extend(test_hist.tolist())

    if save_name is not None:
        torch.save(model, model_path+save_name)

    return model.eval(), train_hists, test_hists


def model_evaluation(df, features, model, save_name=None):
    with torch.no_grad():
        x_test_tensor = torch.tensor(df[features].values, dtype = torch.float)
        y_pred_tensor = model(x_test_tensor)
        df["pred"] = y_pred_tensor.detach().flatten().numpy() 

    df = df.groupby("TIME_CET").agg({"VAERDI" : lambda x : x.sum(),
                                        "pred" : lambda x : x.sum()})
    df.columns = ["VAERDI", "pred"]
    
    df = df.reset_index()
    # df = df.reset_index(drop = True)
    df = df.reset_index()
    df["index"] = df["index"] + 1
    df["VAERDI_cumsum"] = df["VAERDI"].cumsum()
    df["NBIAS"] = (df["VAERDI"] - df["pred"]).cumsum() / df["VAERDI_cumsum"]
    df["NMAE"] = (abs(df["VAERDI"] - df["pred"])).cumsum() / df["VAERDI_cumsum"]
    df["NMSE"] = ((abs(df["VAERDI"] - df["pred"])).cumsum() ** 2) / df["VAERDI_cumsum"]
    df["NRMSE"] = ((((abs(df["VAERDI"] - df["pred"])).cumsum() ** 2) * df["index"]) ** 0.5 ) / df["VAERDI_cumsum"] 
    df["accuracy"] = round((1 - abs(df["VAERDI"] - df["pred"] + 1) / (df["VAERDI"] + 1)),4) * 100

    df = df.drop(["index"], axis=1)

    # df = df.groupby("TIME_CET").agg({"accuracy" : lambda x :round(x.mean(), 1),
    #                                     "NBIAS" : lambda x :round(x.mean(), 1),
    #                                     "NMAE" : lambda x :round(x.mean(), 1),
    #                                     "NMSE" : lambda x :round(x.mean(), 1),
    #                                     "NRMSE" : lambda x :round(x.mean(), 1),
    #                                     "VAERDI" : lambda x :round(x.sum(), 1),
    #                                     "pred" : lambda x :round(x.sum(), 1)})
    # df.columns = ["accuracy", "NBIAS", "NMAE", "NMSE", "NRMSE", "VAERDI", "pred"]
    # df = df.reset_index()
    if save_name is not None:
        df.to_csv(error_path + save_name, index=False)
    
    return df


def model_improvement(ref_error, target_error):
    I = 100 * (ref_error["NMAE"] - target_error["NMAE"]) / ref_error["NMAE"]

    r2_ref = r2_score(ref_error["VAERDI"], ref_error["pred"])
    r2_tar = r2_score(target_error["VAERDI"], target_error["pred"])
    return I, {"r2_ref": r2_ref, "r2_tar": r2_tar}