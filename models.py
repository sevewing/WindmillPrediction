import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from myplot import timelines_plot

from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit


class MLP(nn.Module):

  def __init__(self, input_size, hidden_size, output_size):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)
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
            print(f'Epoch {t+1} train loss: {loss.item()}')

        train_hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return model.eval(), train_hist, test_hist



def timeseries_kfold_validation_training(df, features, target, n_groups, model, lr=0.001, num_epochs=30):
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

    return model, train_hists, test_hists



def Evaluation(df, name):
    df = df.reset_index(drop = True)
    df = df.reset_index()
    df["index"] = df["index"] + 1

    df["VAERDI_cumsum"] = df["VAERDI"].cumsum()
    df["NBIAS"] = (df["VAERDI"] - df["pred"]).cumsum() / df["VAERDI_cumsum"]
    df["NMAE"] = (abs(df["VAERDI"] - df["pred"])).cumsum() / df["VAERDI_cumsum"]
    df["NMSE"] = ((abs(df["VAERDI"] - df["pred"])).cumsum() ** 2) / df["VAERDI_cumsum"]
    df["NRMSE"] = ((((abs(df["VAERDI"] - df["pred"])).cumsum() ** 2) * df["index"]) ** 0.5 ) / df["VAERDI_cumsum"] 
    
    # df[name+"_R^2"] = [r2_score(df["VAERDI"][:i+1], df["pred"][:i+1]) for i in range(len(df))]

    timelines_plot(df["TIME_CET"], 
            {"NBIAS":df["NBIAS"], 
            "NMAE":df["NMAE"],
            "NMSE":df["NMSE"],
            "NRMSE":df["NRMSE"]})
    df = df.drop(["index"], axis=1)

    return df


def model_improvement(ref_error, target_error):
    I = 100 * (ref_error["NMAE"] - target_error["NMAE"]) / ref_error["NMAE"]
    # R_2 = (ref_error["NMSE"] - target_error["NMSE"]) / ref_error["NMSE"]
    return I