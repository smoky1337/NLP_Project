import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from main import DEVICE as device
from torch import nn

class GRURegressor(nn.Module):
    def __init__(self, features= 2, hidden_units= 32):
        super(GRURegressor, self).__init__()
        self.hidden_units = hidden_units
        self.features = features
        self.n_layers = 2
        self.GRU = nn.GRU(input_size=features, hidden_size=self.hidden_units, num_layers=self.n_layers, batch_first=True
                          ,dropout= 0.3)
        self.linear = nn.Linear(self.hidden_units, 1)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_units).requires_grad_().to(device)

        output, _ = self.GRU(x, h0)
        output = output[:, -1, :].to(device)
        return self.linear(output)

class Custom_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_hat):
        loss = nn.MSELoss()(y_true,y_hat)
        if y_true < 0 < y_hat or y_hat < 0 < y_true:
            loss *= 10
        return loss


def neural_network_sentiment(filename, data,SYMBOL):


    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import Dataset, DataLoader

    SEQUENCE_LENGTH = 32
    BATCH_SIZE = 1

    class SequenceDataset(Dataset):
        def __init__(self, dataframe, target, features, sequence_length=5):
            self.features = features
            self.target = target
            self.sequence_length = sequence_length
            self.y = torch.tensor(dataframe[target].values).float()
            self.X = torch.tensor(dataframe[features].values).float()

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                x = self.X[i_start:(i + 1), :]
            else:
                padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
                x = self.X[0:(i + 1), :]
                x = torch.cat((padding, x), 0)

            return x, self.y[i]


    split = int(len(data) / 10)
    train_data = data[:-split]
    scaler = MinMaxScaler()
    train_data["close_value"] = scaler.fit_transform(train_data["close_value"].diff().fillna(0).to_numpy().reshape(-1,1))

    test_data = data[-split:]
    test_data["close_value"] = scaler.transform(test_data["close_value"].diff().fillna(0).to_numpy().reshape(-1,1))
    train_data = train_data.reset_index()


    train_dataset = SequenceDataset(
        train_data,
        target="close_value",
        features=["close_value", SYMBOL],
        sequence_length=SEQUENCE_LENGTH
    )
    test_dataset = SequenceDataset(
        test_data,
        target="close_value",
        features=["close_value", SYMBOL],
        sequence_length=SEQUENCE_LENGTH
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last= True)


    learning_rate = 5e-4
    num_hidden_units = 128
    evals = []

    model = GRURegressor(features=2, hidden_units=num_hidden_units)
    model.to(device)
    loss_function = Custom_loss()

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    def train_model(data_loader, model, loss_function, optimizer):
        t = time.time()
        num_batches = len(data_loader)
        total_loss = 0
        model.train()

        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            output = output.to(device)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Train loss: {avg_loss}, Elapsed time in s: {np.round(time.time()-t,2)}")

    def test_model(data_loader, model, loss_function):
        t = time.time()

        num_batches = len(data_loader)
        total_loss = 0

        model.to(device)
        model.eval()

        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(device)
                y = y.to(device)
                output = model(X)
                output = output.to(device)
                total_loss += loss_function(output, y).item()

        avg_loss = total_loss / num_batches
        evals.append(avg_loss)
        print(f"Test loss: {avg_loss}, Elapsed time in s: {np.round(time.time()-t,2)}")

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()
    start = time.time()
    for ix_epoch in range(15):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_model(test_loader, model, loss_function)
        print(f"Elapsed time in s: {np.round(time.time()-start,2)}")

    torch.save(model.state_dict(), f"{filename}_sentiment_model.pt")

    for X,y in test_loader:
        if "predictions" not in locals():
            predictions = model(X).detach().numpy().squeeze()
            real = y.detach().numpy()
        predictions = np.hstack((predictions, model(X).squeeze().detach().numpy()))
        real = np.hstack((real.squeeze(),y.detach().numpy()))
    y_hat = scaler.inverse_transform(predictions.reshape(((-1,1))))
    y_true = scaler.inverse_transform(real.reshape((-1,1)))
    plt.plot(y_hat)
    plt.plot(y_true)
    plt.title(f"Model with sentiment MSE {evals[-1]}")
    plt.legend(["Preds", "Y_true"])
    plt.savefig(f"{filename}_sentiment_model_results.png")
    plt.show()

def neural_network(filename, data):


    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import Dataset, DataLoader

    SEQUENCE_LENGTH = 32
    BATCH_SIZE = 1

    class SequenceDataset(Dataset):
        def __init__(self, dataframe, target, features, sequence_length=5):
            self.features = features
            self.target = target
            self.sequence_length = sequence_length
            self.y = torch.tensor(dataframe[target].values).float()
            self.X = torch.tensor(dataframe[features].values).float()

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                x = self.X[i_start:(i + 1), :]
            else:
                padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
                x = self.X[0:(i + 1), :]
                x = torch.cat((padding, x), 0)

            return x, self.y[i]


    split = int(len(data) / 10)
    train_data = data[:-split]
    scaler = MinMaxScaler()
    train_data["close_value"] = scaler.fit_transform(train_data["close_value"].diff().fillna(0).to_numpy().reshape(-1,1))

    test_data = data[-split:]
    test_data["close_value"] = scaler.transform(test_data["close_value"].diff().fillna(0).to_numpy().reshape(-1,1))
    train_data = train_data.reset_index()


    train_dataset = SequenceDataset(
        train_data,
        target="close_value",
        features=["close_value"],
        sequence_length=SEQUENCE_LENGTH
    )
    test_dataset = SequenceDataset(
        test_data,
        target="close_value",
        features=["close_value"],
        sequence_length=SEQUENCE_LENGTH
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    learning_rate = 5e-4
    num_hidden_units = 128

    model = GRURegressor(features=1, hidden_units=num_hidden_units)
    model.to(device)
    loss_function = AdjMSELoss2()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    evals = []

    def train_model(data_loader, model, loss_function, optimizer):
        t = time.time()
        num_batches = len(data_loader)
        total_loss = 0
        model.train()

        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            output = output.to(device)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Train loss: {avg_loss}, Elapsed time in s: {np.round(time.time()-t,2)}")

    def test_model(data_loader, model, loss_function):
        t = time.time()

        num_batches = len(data_loader)
        total_loss = 0

        model.to(device)
        model.eval()

        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(device)
                y = y.to(device)
                output = model(X)
                output = output.to(device)
                total_loss += loss_function(output, y).item()

        avg_loss = total_loss / num_batches
        evals.append(avg_loss)
        print(f"Test loss: {avg_loss}, Elapsed time in s: {np.round(time.time()-t,2)}")

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()
    start = time.time()
    for ix_epoch in range(15):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_model(test_loader, model, loss_function)
        print(f"Elapsed time in s: {np.round(time.time()-start,2)}")

    torch.save(model.state_dict(), f"{filename}_model.pt")

    for X,y in test_loader:
        if "predictions" not in locals():
            predictions = model(X).detach().numpy().squeeze()
            real = y.detach().numpy()
        predictions = np.hstack((predictions, model(X).squeeze().detach().numpy()))
        real = np.hstack((real.squeeze(),y.detach().numpy()))
    y_hat = scaler.inverse_transform(predictions.reshape(((-1,1))))
    y_true = scaler.inverse_transform(real.reshape((-1,1)))
    plt.plot(y_hat)
    plt.plot(y_true)
    plt.legend(["Preds", "Y_true"])
    plt.title(f"Model without sentiment. MSE {evals[-1]}")
    plt.savefig(f"{filename}_model_results.png")
    plt.show()
if __name__ == "__main__":
    k = GRURegressor()
    print(k)