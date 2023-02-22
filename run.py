#import the library


import os
import warnings

from NeuralNetworkApproach import *
from utils import *
from SkForecastApproach import *

warnings.filterwarnings('ignore')


# BEGIN CONFIG #
FILENAME = f"{time.time()}"
SYMBOLS = ["AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA"]
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

    # BEGIN DATA CREATION #
    # Warning: Takes around 5 hours and overwrites current files! #
    # create_sentiment_scores()
    # create_daily_sentiment()
    # END DATA CREATION #

    # BEGIN DATA WRANGLING #
    data = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Tweet_Stocks_Sentiment.csv"), index_col="tweet_id")
    sentiment = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"), index_col="day_date")
    sentiment.index = pd.to_datetime(sentiment.index)
    stocks = prepare_stock_data()
    data = align_stock_sentiment(stocks, sentiment, symbol=SYMBOL, start=START, end=END)
    data = score_momentum(data, SYMBOL)
    # END DATA WRANGLING #

    # BEGIN TRAINING #
    # neural_network_sentiment(data,SYMBOL)
    # neural_network(data)
    # END TRAINING #

    compare_sentiment_to_avg(sentiment, symbols= ["AAPL", "GOOG"])
    show_stock_sentiment(data, SYMBOL, False)
    create_train_forecaster(SYMBOL, START, END, exog=False, task = "r", columns = None, target="close_value", steps=1,
                            SEQUENCE_LENGTH=SEQUENCE_LENGTH, filename=FILENAME, verbose=False)
