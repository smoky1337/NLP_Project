#import the library


import os
import warnings

from NeuralNetworkApproach import *
from utils import *
from SkForecastApproach import *

warnings.filterwarnings('ignore')
data = pd.read_csv(os.path.join(os.getcwd(),"tweet_data", "Tweet_Stocks_Sentiment.csv"), index_col="tweet_id")
symbols = ["AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA"]


#calculate the negative, positive, neutral and compound scores, plus verbal evaluation





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

# BEGIN CONFIG #
FILENAME = f"{time.time()}"

# BEGIN CONFIG GENERAL #
SYMBOL = "AAPL"
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


if __name__ == '__main__':
    device = DEVICE

    sentiment = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"), index_col="day_date")
    sentiment.index = pd.to_datetime(sentiment.index)
    stocks = prepare_stock_data()
    data = align_stock_sentiment(stocks, sentiment, symbol=SYMBOL, start=START, end=END)
    data = score_momentum(data, SYMBOL)


    # BEGIN TRAINING #
    neural_network_sentiment(data,SYMBOL)
    neural_network(data)
    # END TRAINING #


    sentiment = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"), index_col="day_date")
    sentiment.index = pd.to_datetime(sentiment.index)
    #compare_sentiment_to_avg(sentiment, symbols= ["AAPL", "GOOG"])
    stocks = prepare_stock_data()

    data = align_stock_sentiment(stocks, sentiment, symbol= SYMBOL, start=START,end=END)
    show_stock_sentiment(data, SYMBOL, False)