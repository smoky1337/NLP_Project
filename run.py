#import the library
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tqdm
from sklearn.metrics import log_loss, f1_score, accuracy_score
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from xgboost import XGBRegressor
import torch
from torch import nn

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv(os.path.join(os.getcwd(),"tweet_data", "Tweet_Stocks_Sentiment.csv"), index_col="tweet_id")
symbols = ["AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA"]


#calculate the negative, positive, neutral and compound scores, plus verbal evaluation
def sentiment_vader(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"

    return compound

def create_split_data(data, symbol, columns, start = None, end = None, train = 0.8, return_weekdays = True):
    if start is None:
        start = data.index.min()
    if end is None:
        end = data.index.max()
    s = data.loc[start:end][columns + [symbol]].copy()
    if return_weekdays:
        t = pd.get_dummies(s.index.weekday,"week_day")
        t.index = s.index
        s = s.join(t)
    s = s.asfreq("D")
    split = int(np.floor(len(s)*train))
    val = split + int(np.floor((len(s) - split) / 3))
    return s.iloc[:split], s.iloc[split:val],s.iloc[val:]


def main():
    return 0
    # Create Sentiment Data
    # Create Daily Sentiment for stock
    data = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Tweet_Stocks_Sentiment.csv"), index_col="tweet_id")
    symbols = ["AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA"]
    date_score = pd.DataFrame()
    date_score.index = data.post_date.unique()
    for s in symbols:
        group = data[data[s] > 0]
        avg = group.groupby('post_date')["score"].mean().rename(s)
        date_score = date_score.join(avg, lsuffix="")

    #create avergae
    date_score["avg_score"] = date_score.sum(axis=1) / 6
    # Save to disk
    date_score.to_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"))

#create score momentum
def score_momentum(df, symbol, intervals = ("3d", "1w")):
    s = df.copy()
    for i in intervals:
        s[f"{i}_momentum"] = s[symbol].resample(i).mean()
        s[f"{i}_momentum"].iloc[0] = s[symbol].iloc[0]
        s[f"{i}_momentum"].interpolate(inplace=True)

    return s


def create_sentiment_scores():
    data = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Tweet.csv"), index_col="tweet_id")
    data["post_date"] = pd.to_datetime(data["post_date"], unit="s")
    symbol = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Company_Tweet.csv"), index_col="tweet_id")
    s = pd.get_dummies(symbol, prefix_sep="", prefix="").max(level=0)
    data = data.join(s)
    data = data.reset_index()
    scores = [sentiment_vader(t) for t in tqdm.tqdm(data["body"])]
    data["score"] = scores
    data.to_csv(os.path.join(os.getcwd(), "tweet_data", "Tweet_Stocks_Sentiment.csv"))

def create_daily_sentiment():
    data = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Tweet_Stocks_Sentiment.csv"), index_col="tweet_id")
    symbols = ["AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA"]
    date_score = pd.DataFrame()
    date_score.index = data.post_date.unique()
    for s in symbols:
        group = data[data[s] > 0]
        avg = group.groupby('post_date')["score"].mean()
        date_score = date_score.join(avg, lsuffix=s)

    date_score.score = np.sum(date_score[:, :-1])
    # create average:
    date_score.score = (date_score.sum(axis=1) - date_score.score) / 5
    # Save to disk
    date_score.to_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"))


def compare_sentiment_to_avg(data, symbols = ["AAPL"], save=False, path="SentimentComp.png"):
    plt.plot(data[symbols + ["avg_score"]].resample("1m").mean())
    plt.legend(symbols + ["Average"])
    plt.suptitle("Monthly Sentiment by Stock")
    plt.title("Sentiment between very negativ (-1) and very positiv (+1)")
    plt.xlabel("Date")
    plt.ylabel("Sentiment")
    if save:
        plt.savefig(path)
    plt.show()

def prepare_stock_data():
    stocks = pd.read_csv(os.path.join(os.getcwd(), "stock_data", "Stocks.csv"), index_col=["day_date", "ticker_symbol"])
    stocks.index = stocks.index.set_levels([pd.to_datetime(stocks.index.levels[0]), stocks.index.levels[1]])
    return stocks

def align_stock_sentiment(stock_data,sentiment_data, symbol = "AAPL", start = "2015-01-01", end = "2019-12-31"):
    if symbol not in sentiment_data.columns:
        raise KeyError(f"{symbol} not in data.")
    s = stock_data.copy(deep=True)
    s = s.loc[pd.IndexSlice[start:end,symbol],:]
    t = sentiment_data.copy(deep=True)
    t = t.loc[start:end][symbol]
    s = s.join(t)
    s["change"] = s.close_value.pct_change()
    s["change"][0] = 0
    s = s.reset_index()
    s.index = s["day_date"]
    s = s.drop(["ticker_symbol", "day_date"], axis=1)
    return s



def show_stock_sentiment(data, symbol,save = False, path = "Stock_Sentiment.png"):
    plt.plot(data.index.get_level_values(0), data["change"], c = "r")

    plt.ylabel("Close Price Difference (%)")
    plt.legend(["% Change"])

    plt.twinx()
    plt.plot(data.index.get_level_values(0),data[symbol])
    plt.ylabel("Average Sentiment Score")
    plt.xlabel("Days")
    plt.suptitle("Stock Price Change vs Sentiment")
    plt.title(f"From {data.index.levels[0].min().date()} to {data.index.levels[0].max().date()} for symbol {symbol}")

    plt.legend(["Sentiment"])
    if save:
        plt.savefig(path)
    else:
        plt.show()



def create_train_forecaster(symbol, start, end, exog, task, columns, target,steps = 1, verbose = False):
    #CONFIG
    SYMBOL = symbol
    START = start
    END = end
    COLUMNS = [target] + columns
    TRAIN = 0.8
    EXO = exog

    sentiment = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"), index_col="day_date")
    sentiment.index = pd.to_datetime(sentiment.index)
    #compare_sentiment_to_avg(sentiment, symbols= ["AAPL", "GOOG"])
    stocks = prepare_stock_data()

    data = align_stock_sentiment(stocks, sentiment, symbol= SYMBOL, start=START,end=END)
    data = score_momentum(data,SYMBOL)
    data_train, data_val, data_test = create_split_data(data, SYMBOL, COLUMNS, start = START, end = END, train=TRAIN, return_weekdays=EXO)

    if task == "regression":
        regressor = XGBRegressor(random_state=123)
    elif task == "classification":
        regressor = XGBClassifier(random_state=123)
    else:
        raise ValueError(f"{task} is not in set (regression, classification")
    # Create and train forecaster
    # ==============================================================================
    # Create forecaster
    forecaster = ForecasterAutoreg(
                     regressor     = regressor,
                     lags          = 7,
                     transformer_y = StandardScaler()
                 )
    if verbose:
        print(forecaster)

    param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
    }

    # Lags used as predictors
    lags_grid = [24, 48, 72, [1, 2, 3, 23, 24, 25, 71, 72, 73]]
    if exog:
        exog = [column for column in data_train.columns if column.startswith('week')]
        exog = pd.concat([data_train[exog],data_val[exog]])
    else:
        exog = None

    results_grid = grid_search_forecaster(
        forecaster         = forecaster,
        y                  = pd.concat([data_train[target],data_val[target]]), # Train and validation data
        exog               = exog,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        steps              = steps,
        refit              = False,
        metric             = 'mean_squared_error' if task == "regression" else accuracy_score,
        initial_train_size = len(data_train), # Model is trained with trainign data
        fixed_train_size   = False,
        return_best        = True,
        verbose            = verbose
        )
    print(results_grid)
    metric, predictions = backtesting_forecaster(
        forecaster = forecaster,
        y          = pd.concat([data_train[target],data_val[target],data_test[target]]),
        initial_train_size = len(data_train)+ len(data_val),
        fixed_train_size   = False,
        steps      = steps,
        refit      = False,
        metric     = 'mean_squared_error',
        verbose    = verbose # Change to True to see detailed information
    )

    print(f"Backtest error: {metric}")
    # Plot of predictions
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(11, 4))
    data_test[COLUMNS[0]].plot(ax=ax, label='test')
    predictions['pred'].plot(ax=ax, label='predictions')
    ax.set_title(f"Overall Backtest Error {metric}")
    ax.legend()
    plt.savefig(f"{symbol}_{task}")
    plt.show()

DEVICE = "cpu"
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = DEVICE
    # CONFIG
    SYMBOL = "TSLA"
    START = "2015-01-01"
    END = "2019-12-31"

    sentiment = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"), index_col="day_date")
    sentiment.index = pd.to_datetime(sentiment.index)
    stocks = prepare_stock_data()
    data = align_stock_sentiment(stocks, sentiment, symbol=SYMBOL, start=START, end=END)
    data = score_momentum(data, SYMBOL)
    filename = f"{time.time()}"
    neural_network_sentiment(filename,data,SYMBOL)
    neural_network(filename,data)

    exit()


    #CONFIG
    SYMBOL = "TSLA"
    START = "2016-01-01"
    END = "2019-12-31"


    sentiment = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"), index_col="day_date")
    sentiment.index = pd.to_datetime(sentiment.index)
    #compare_sentiment_to_avg(sentiment, symbols= ["AAPL", "GOOG"])
    stocks = prepare_stock_data()

    data = align_stock_sentiment(stocks, sentiment, symbol= SYMBOL, start=START,end=END)
    show_stock_sentiment(data, SYMBOL, False)