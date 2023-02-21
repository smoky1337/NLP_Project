from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import log_loss, f1_score, accuracy_score


from utils import *


def create_train_forecaster(symbol, start, end, exog, task, columns, target, steps=1, verbose=False):
    # CONFIG
    SYMBOL = symbol
    START = start
    END = end
    COLUMNS = [target] + columns
    TRAIN = 0.8
    EXO = exog

    sentiment = pd.read_csv(os.path.join(os.getcwd(), "tweet_data", "Daily_Sentiment.csv"), index_col="day_date")
    sentiment.index = pd.to_datetime(sentiment.index)
    # compare_sentiment_to_avg(sentiment, symbols= ["AAPL", "GOOG"])
    stocks = prepare_stock_data()

    data = align_stock_sentiment(stocks, sentiment, symbol=SYMBOL, start=START, end=END)
    data = score_momentum(data, SYMBOL)
    data_train, data_val, data_test = create_split_data(data, SYMBOL, COLUMNS, start=START, end=END, train=TRAIN,
                                                        return_weekdays=EXO)

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
        regressor=regressor,
        lags=7,
        transformer_y=StandardScaler()
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
        exog = pd.concat([data_train[exog], data_val[exog]])
    else:
        exog = None

    results_grid = grid_search_forecaster(
        forecaster=forecaster,
        y=pd.concat([data_train[target], data_val[target]]),  # Train and validation data
        exog=exog,
        param_grid=param_grid,
        lags_grid=lags_grid,
        steps=steps,
        refit=False,
        metric='mean_squared_error' if task == "regression" else accuracy_score,
        initial_train_size=len(data_train),  # Model is trained with trainign data
        fixed_train_size=False,
        return_best=True,
        verbose=verbose
    )
    print(results_grid)
    metric, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=pd.concat([data_train[target], data_val[target], data_test[target]]),
        initial_train_size=len(data_train) + len(data_val),
        fixed_train_size=False,
        steps=steps,
        refit=False,
        metric='mean_squared_error',
        verbose=verbose  # Change to True to see detailed information
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
