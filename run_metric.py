import os
import warnings
import sklearn
import matplotlib.pyplot as plt
from NeuralNetworkApproach import *
from utils import *
from SkForecastApproach import *
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.use('TkAgg')  # !IMPORTANT

MODEL_SENTIMENT_PATH = os.path.join("results_NNA_diff_64hu_TSLA", "1677011291.3394449_sentiment_model.pt")
MODEL_PATH = os.path.join("results_NNA_diff_64hu_TSLA", "1677011291.3394449_model.pt")

def direction_metric(true, pred):
    bool_function = lambda x: True if x >= 0 else False
    diff = np.subtract(true,pred)
    true = np.array([bool_function(x) for x in true])
    pred = np.array([bool_function(x) for x in pred])

    m = sklearn.metrics.confusion_matrix(true, pred)
    print(f"# True Positives:  {m[1, 1]} ({(m[1, 1]/m.sum(axis=1)[1]*100):.2f}%)\n"
          f"# False Positives: {m[1, 0]} ({(m[1, 0]/m.sum(axis=1)[1]*100):.2f}%)\n"
          f"# True Negatives:  {m[0, 0]} ({(m[0, 0]/m.sum(axis=1)[0]*100):.2f}%)\n"
          f"# False Negatives: {m[0, 1]} ({(m[0, 1]/m.sum(axis=1)[0]*100):.2f}%)\n#")
    print(f"# Accuracy:  {(np.sum(true==pred)/len(true)*100):.2f}%")
    precision = m[1, 1] / (m[1, 1] + m[1, 0])
    print(f"# Precision: {precision:.2f}")
    recall = m[1, 1]/(m[1, 1] + m[0, 1])
    print(f"# Recall:    {recall:.2f}")
    f1 = 2*((precision * recall) / (precision + recall))
    print(f"# F1-Score:  {f1:.2f}")
    '''
    fig, ax = plt.subplots()
    ax.bar(range(len(diff)),diff)
    plt.show()
    '''



# BEGIN CONFIG GENERAL #
SYMBOL = "AMZN"
START = "2015-01-01"
END = "2019-12-31"
# BEGIN CONFIG NNA #
SEQUENCE_LENGTH = 32
BATCH_SIZE = 1
LR = 5e-4
HU = 64
DEVICE = "cpu"
# END CONFIG NNA #
# END CONFIG

# Prepare data
sentiment = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"), index_col="day_date")
sentiment.index = pd.to_datetime(sentiment.index)
stocks = prepare_stock_data()
data = align_stock_sentiment(stocks, sentiment, symbol=SYMBOL, start=START, end=END)
data = score_momentum(data, SYMBOL)

split = int(len(data) / 10)
train_data = data[:-split].fillna(0)
scaler = MinMaxScaler()
train_data["close_value"] = scaler.fit_transform(train_data["close_value"].diff().fillna(0).to_numpy().reshape(-1, 1))

test_data = data[-split:].fillna(0)
test_data["close_value"] = scaler.transform(test_data["close_value"].diff().fillna(0).to_numpy().reshape(-1, 1))
train_data = train_data.reset_index()

#----------model with sentiment----------#
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

model = GRURegressor(features=2, hidden_units=HU)
model.load_state_dict(torch.load(MODEL_SENTIMENT_PATH))
model = model.eval()

true = []
pred = []
for x, y in train_loader:
    pred.append(model(x).tolist()[0][0])
    true.append(y.tolist()[0])

print("#---Model with sentiment---#")
direction_metric([x[0] for x in scaler.inverse_transform(np.array(true).reshape(-1, 1))],
                 [x[0] for x in scaler.inverse_transform(np.array(pred).reshape(-1, 1))])
print("#--------------------------#\n")

#----------model withOUT sentiment----------#
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

model = GRURegressor(features=1, hidden_units=HU)
model.load_state_dict(torch.load(MODEL_PATH))

model = model.eval()

true = []
pred = []
for x, y in train_loader:
    pred.append(model(x).tolist()[0][0])
    true.append(y.tolist()[0])
print("#-------Model withOUT------#")
direction_metric([x[0] for x in scaler.inverse_transform(np.array(true).reshape(-1, 1))],
                 [x[0] for x in scaler.inverse_transform(np.array(pred).reshape(-1, 1))])
print("#--------------------------#")



