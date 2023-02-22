from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import log_loss, f1_score, accuracy_score


from utils import *


def create_train_forecaster(symbol, start, end, exog, task, columns, target, steps=1, SEQUENCE_LENGTH = 24, filename = "",metric = None,verbose=False):
    # CONFIG
    SYMBOL = symbol
    START = start
    END = end
    COLUMNS = [target] + columns if columns is not None else [target]
    TRAIN = 0.9
    EXO = exog

    sentiment = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"), index_col="day_date")
    sentiment.index = pd.to_datetime(sentiment.index)
    # compare_sentiment_to_avg(sentiment, symbols= ["AAPL", "GOOG"])
    stocks = prepare_stock_data()

    data = align_stock_sentiment(stocks, sentiment, symbol=SYMBOL, start=START, end=END)
    data = score_momentum(data, SYMBOL)
    if target == "close_value":
        data["close_value"] = data["close_value"].diff().fillna(0)

    data_train, data_val, data_test = create_split_data(data, SYMBOL, COLUMNS, start=START, end=END, train=TRAIN,
                                                        return_weekdays=EXO)
    # Choose Task, use Gradient Boosted Trees
    if task == "regression" or task == "r":
        regressor = XGBRegressor(random_state=123)
    elif task == "classification" or task == "c":
        regressor = XGBClassifier(random_state=123)
    else:
        raise ValueError(f"{task} is not in set (regression (r), classification (c)")
    # Create and train forecaster
    # ==============================================================================
    # Create forecaster
    forecaster = ForecasterAutoreg(
        regressor=regressor,
        lags=SEQUENCE_LENGTH,
        transformer_y=MinMaxScaler()
    )
    if verbose:
        print(forecaster)

    param_grid = {
        'n_estimators': [10,100],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1]
    }

    if exog:
        exog = [column for column in data_train.columns if column.startswith('week')]
        exog = pd.concat([data_train[exog], data_val[exog]])
    else:
        exog = None

    if not metric:
        metric = 'mean_squared_error' if task == "regression" or "r" else accuracy_score
    mname = metric if type(metric) == str else metric.__name__


    results_grid = grid_search_forecaster(
        forecaster=forecaster,
        y=pd.concat([data_train[target], data_val[target]]),  # Train and validation data
        exog=exog,
        param_grid=param_grid,
        steps=steps,
        refit=False,
        metric=metric,
        initial_train_size=len(data_train),  # Model is trained with trainign data
        fixed_train_size=False,
        return_best=True,
        verbose=verbose
    )


    error, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=pd.concat([data_train[target], data_val[target], data_test[target]]),
        initial_train_size=len(data_train) + len(data_val),
        fixed_train_size=False,
        steps=steps,
        refit=False,
        metric=metric,
        verbose=verbose  # Change to True to see detailed information
    )

    print(f"Backtest error {mname}: {error}")
    # Plot of predictions
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(11, 4))
    data_train[COLUMNS[0]].plot(ax=ax, label="train")
    data_test[COLUMNS[0]].plot(ax=ax, label='test')
    predictions['pred'].plot(ax=ax, label='predictions')
    ax.set_title(f"Overall Backtest Error {mname}{error}")
    ax.legend()
    plt.savefig(f"{filename}_{symbol}_{task}_{SEQUENCE_LENGTH}_{mname}.png")
    plt.show()
