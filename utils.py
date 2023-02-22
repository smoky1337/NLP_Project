import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sentiment_vader(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05:
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05:
        overall_sentiment = "Negative"

    else:
        overall_sentiment = "Neutral"

    return compound


def create_split_data(data, symbol, columns, start=None, end=None, train=0.8, return_weekdays=True):
    if start is None:
        start = data.index.min()
    if end is None:
        end = data.index.max()
    s = data.loc[start:end][columns + [symbol]].copy()
    if return_weekdays:
        t = pd.get_dummies(s.index.weekday, "week_day")
        t.index = s.index
        s = s.join(t)
    s = s.asfreq("D")
    split = int(np.floor(len(s) * train))
    val = split + int(np.floor((len(s) - split) / 3))
    return s.iloc[:split], s.iloc[split:val], s.iloc[val:]


def score_momentum(df, symbol, intervals=("3d", "1w")):
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


def compare_sentiment_to_avg(data, symbols=["AAPL"], save=True, path="SentimentComp"):
    plt.plot(data[symbols + ["avg_score"]].resample("1m").mean())
    plt.legend(symbols + ["Average"])
    plt.suptitle("Monthly Sentiment by Stock")
    plt.title("Sentiment between very negativ (-1) and very positiv (+1)")
    plt.xlabel("Date")
    plt.ylabel("Sentiment")
    if save:
        plt.savefig("".join((path, *symbols, ".png")))
    plt.show()


def prepare_stock_data():
    stocks = pd.read_csv(os.path.join(os.getcwd(), "stock_data", "Stocks.csv"), index_col=["day_date", "ticker_symbol"])
    stocks.index = stocks.index.set_levels([pd.to_datetime(stocks.index.levels[0]), stocks.index.levels[1]])
    return stocks


def align_stock_sentiment(stock_data, sentiment_data, symbol="AAPL", start="2015-01-01", end="2019-12-31"):
    if symbol not in sentiment_data.columns:
        raise KeyError(f"{symbol} not in data.")
    s = stock_data.copy(deep=True)
    s = s.loc[pd.IndexSlice[start:end, symbol], :]
    t = sentiment_data.copy(deep=True)
    t = t.loc[start:end][symbol]
    s = s.join(t)
    s["change"] = s.close_value.pct_change()
    s["change"][0] = 0
    s = s.reset_index()
    s.index = s["day_date"]
    s = s.drop(["ticker_symbol", "day_date"], axis=1)
    return s


def show_stock_sentiment(data, symbol, save=False, path="Stock_Sentiment.png"):
    plt.plot(data.index.get_level_values(0), data["change"], c="r")

    plt.ylabel("Close Price Difference (%)")
    plt.legend(["% Change"])

    plt.twinx()
    plt.plot(data.index.get_level_values(0), data[symbol])
    plt.ylabel("Average Sentiment Score")
    plt.xlabel("Days")
    plt.suptitle("Stock Price Change vs Sentiment")
    plt.title(f"From {data.index.levels[0].min().date()} to {data.index.levels[0].max().date()} for symbol {symbol}")

    plt.legend(["Sentiment"])
    if save:
        plt.savefig(path)
    else:
        plt.show()
