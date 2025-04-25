import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.rolling import RollingOLS

from researchlab.core.utils import check_make_folder
from researchlab.core.data_funcs import *

# ------------------------------
# Utility Functions
# ------------------------------

def Z_Score(current, period_history):
    return (current - period_history.mean(axis=0)) / period_history.std(axis=0)


def ewmaVol(ret_series, decay=0.95, invert=0):
    """
    Calculate exponentially weighted volatility, with optional reverse weighting.
    """
    window = len(ret_series)
    ewm_weights = [(1 - decay) * (decay ** (window - i)) for i in range(window)]
    if invert:
        ewm_weights.reverse()
    sq_ret = ret_series ** 2
    ewm_vol = (np.array(ewm_weights) * sq_ret).sum() ** 0.5
    return ewm_vol

def olsAnalysis(X: pd.DataFrame, y: pd.Series, save_model=True, result_name='experiment'):
    """OLS regression without intercept; optionally saves model and summary"""
    model = OLS(y, X).fit()
    print(model.summary())

    check_make_folder('data/model_results')

    if save_model:
        model.save(f'data/model_results/{result_name}.pkl')
        with open(f'data/model_results/{result_name}.txt', 'w') as f:
            f.write(model.summary().as_text())

    return model

def rollingOlsAnalysis(X: pd.DataFrame, y: pd.Series, window: int = 30, save_model=True, result_name='experiment'):
    """Rolling OLS; returns coefficients and residuals"""
    endog_factors = X.columns.tolist()
    rolling_model = RollingOLS(y, X, window=window).fit()
    rolling_betas = rolling_model.params.copy()
    rolling_betas.index = X.index

    # Residuals
    rolling_betas['Residuals'] = y - (rolling_betas[X.columns] * X).sum(axis=1)

    fig = rolling_model.plot_recursive_coefficient(variables=endog_factors, figsize=(14, 18))

    if save_model:
        check_make_folder('data/model_results')
        fig.savefig(f'data/model_results/{result_name}_rolling_ols.png')
        rolling_betas.to_csv(f'data/model_results/{result_name}_rolling_ols.csv')

    return rolling_betas

def custom_zscore_signal(indicators_data: pd.DataFrame, offset_map: List[tuple], weights=None, name=None):
    """Compute Z-score based signal from historical indicators"""
    datetimeindex = indicators_data.index
    current = datetimeindex[-1]
    date_cols = [current] + [datetimeindex[offset[0]] for offset in offset_map]

    fvt_fixed = indicators_data.loc[date_cols].T.round(2)
    fvt_fixed.columns = ['Current'] + [offset[1] for offset in offset_map]

    zscores = [
        Z_Score(indicators_data.loc[date_cols[0]], indicators_data.loc[date_cols[i + 1]:date_cols[0]])
        for i in range(len(offset_map))
    ]
    fvt_z = pd.DataFrame(zscores).T.round(4)
    fvt_z.columns = [f'{offset[1]} Z-score' for offset in offset_map]

    if weights is None:
        weights = [1 / len(offset_map)] * len(offset_map)

    fvt_z['Weighted Z-score'] = fvt_z.mul(weights).sum(axis=1).round(4)

    result = pd.concat([fvt_fixed, fvt_z], axis=1)
    result.index.name = name
    return result

def get_last_3m(data: pd.DataFrame):
    """Returns last 3 months of data (approx. 63 trading days)"""
    return data.iloc[-63:]

def create_dashboard_table(data: pd.DataFrame, offset_map: List[tuple], name: str = ''):
    """Format time series snapshots into a dashboard table"""
    dashboard = data.iloc[[offset[0] for offset in offset_map]]
    dashboard.index = [offset[1] for offset in offset_map]
    dashboard = dashboard.T
    dashboard.index.name = name
    return dashboard

def pca_analysis(features: pd.DataFrame, scale=True, experiment='exp'):
    """Principal Component Analysis and visualizations"""
    X = features.copy()
    if scale:
        scaler = StandardScaler().set_output(transform='pandas')
        X = scaler.fit_transform(features)

    pca = PCA(n_components=0.95).set_output(transform='pandas')
    X_pca = pca.fit_transform(X)

    loadings = pd.DataFrame(
        pca.components_, columns=X.columns,
        index=[f"PCA {i + 1}" for i in range(pca.n_components_)]
    )

    check_make_folder('data/plots')

    # Variance plot
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum())
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance vs Number of Components")
    plt.savefig(f'data/plots/PCA Analysis for {experiment}.png')
    plt.show()

    # Loadings heatmap
    plt.figure(figsize=(12, 16))
    sns.heatmap(loadings, cmap="coolwarm", annot=False, fmt='.2f')
    plt.title("PCA Loadings (Feature Contributions)")
    plt.savefig(f'data/plots/PCA Loadings for {experiment}.png')
    plt.show()

    print("Figures saved in data/plots")
    return loadings

def rolling_qcut(series: pd.Series, window: int):
    """Quartile label of most recent value in rolling window"""
    def apply_qcut(x):
        if x.nunique() < 4:
            return np.nan
        return pd.qcut(x, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']).iloc[-1]

    return series.rolling(window=window).apply(apply_qcut, raw=False)

def quartile_analysis(target: pd.Series, features: pd.DataFrame, experiment_name='experiment'):
    """Analyze how target varies across quartiles of features"""
    features_qcut = features.apply(lambda x: pd.qcut(x, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']))
    df = pd.concat([features_qcut, target], axis=1)

    results = {}
    for feature in features_qcut:
        stats = df.groupby(feature)[target.name].agg(['mean', 'mode', 'median', 'min', 'max'])
        df.groupby(feature)[target.name].plot(kind='hist', legend=True, subplots=True,
                                              title=f'{target.name} distribution by {feature}')
        plt.savefig(f"data/plots/Distribution analysis for {experiment_name}.png")
        results[feature] = stats

    return results

def stationarity_test(features: pd.DataFrame):
    """Perform ADF stationarity tests and plot time series"""
    for col in features.columns:
        series = features[col]
        print(f'ADF for {col}')
        result = adfuller(series)
        print(f'Test Statistic: {result[0]:.4f}')
        print(f'P-value: {result[1]:.4f}')
        print('Critical Values:', result[4])
        print('------------------------\n')

        # Time series plot
        plt.figure(figsize=(6, 4))
        plt.plot(series)
        plt.title(col)
        plt.grid(True)
        plt.show()

def vif_test(features: pd.DataFrame):
    """Variance Inflation Factor diagnostic"""
    vif = pd.DataFrame()
    vif["feature"] = features.columns
    vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    print(vif)
    return vif
