import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from core.data_funcs import (
    filter_date_range,
    StandardScalerPD,
    DataClipperStd,
    pd_type
)

from performance import sharpe_from_returns


def process_data_sets(market_returns, features, train_start, train_end, test_end):
    """Filter data sets based on the training and testing periods."""
    clipper = DataClipperStd(mul=3.0)
    scaler = StandardScalerPD()

    train_ret = filter_date_range(market_returns, train_start, train_end)
    train_features = filter_date_range(features, train_start, train_end)

    test_ret = filter_date_range(market_returns, train_end, test_end)
    test_features = filter_date_range(features, train_end, test_end)

    output = {
        'y_train': train_ret,  # no scaling for returns
        'y_test': test_ret,
        'X_train': scaler.fit_transform(clipper.fit_transform(train_features)),
        'X_test': scaler.transform(clipper.transform(test_features)),
    }

    return output


def penalised_accuracy(y_true, y_pred, lambda_penalty=0.01):
    """Penalised accuracy function for frequent prediction changes."""
    changes = np.sum(y_pred[1:] != y_pred[:-1])
    return np.mean(y_true == y_pred) - lambda_penalty * changes


def run_strategy(returns: pd_type, trade_signals: pd_type, freq='B', transaction_cost=0.001):
    """Run backtest on the model."""
    signals = trade_signals.copy()
    signals.dropna(inplace=True, how='all')
    returns = returns.loc[signals.index].copy()

    signal_changes = signals != signals.shift(1)
    signal_changes = signal_changes.loc[signals.index].copy()

    model_returns = returns * signals
    model_returns -= np.where(signal_changes, transaction_cost, 0)
    model_returns.dropna(inplace=True, how='all')

    sharpe = sharpe_from_returns(model_returns, freq=freq)

    return sharpe, model_returns


def get_xgb_model(model_name):
    """Dummy for live pipeline pulling of index data."""
    json_model = XGBClassifier()
    json_model.load_model(f'models/{model_name}_xgb_model.json')
    return json_model


def single_window_forecast(datasets, model, smoothing: int, threshold: float = 0.5):
    """Train the XGB model and forecast regimes."""
    X_train = datasets['X_train']
    X_test = datasets['X_test']
    y_train = datasets['y_train']
    y_test = datasets['y_test']

    model.fit(X_train, y_train)

    if len(X_test) != 1:
        y_proba = pd.Series(model.predict_proba(X_test)[:, 0], index=X_test.index).shift(1).dropna()
        y_proba_smoothed = y_proba.rolling(smoothing).mean()
        y_pred = (y_proba_smoothed >= threshold).astype(int)
        y_pred.dropna(inplace=True)
        y_proba = y_proba.loc[y_pred.index]
        return y_pred, y_test, y_proba

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return y_pred[0], y_test, y_proba[0]


def rolling_window_backtest(
    returns_series: pd_type,
    features: pd_type,
    test_periods: list,
    train_window: int,
    val_window: int,
    xgb_model,
    freq,
    transaction_cost=0.002,
    smoothing=8
):
    """Run rolling window backtest on multiple windows."""
    forecast_signals = pd.Series(index=returns_series.index)
    forecast_probabilities = pd.Series(index=returns_series.index)

    print('Running multi-window backtest')

    val_window_start = train_window + val_window

    for i, period in enumerate(test_periods[val_window_start + 1:], start=val_window_start):
        train_start = test_periods[i - train_window]
        train_end = test_periods[i]
        test_end = period

        # Extend testing window if near the end of the dataset
        if len(test_periods) - i <= 2:
            test_end = test_periods[-1]

        print(f'Training period from {train_start} to {train_end}')

        model_validation_start = test_periods[i - val_window_start]
        model_validation_end = test_periods[i - val_window]

        validation_datasets = process_data_sets(
            returns_series, features, model_validation_start, model_validation_end, train_end
        )

        backtest_data = process_data_sets(
            returns_series, features, train_start, train_end, test_end
        )

        y_pred, y_test, y_proba = single_window_forecast(backtest_data, xgb_model, smoothing)

        date_index_list = y_test.index
        forecast_signals.loc[date_index_list] = y_pred
        forecast_probabilities.loc[date_index_list] = y_proba

        if len(test_periods) - i <= 1:
            break

    return forecast_signals, forecast_probabilities
