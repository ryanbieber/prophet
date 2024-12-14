# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
from tqdm.auto import tqdm
from copy import deepcopy
import concurrent.futures

import numpy as np
import polars as pl
import re
from datetime import timedelta

logger = logging.getLogger('prophet')


def generate_cutoffs(df, horizon, initial, period):
    """Generate cutoff dates

    Parameters
    ----------
    df: pl.DataFrame with historical data.
    horizon: pl.Timedelta forecast horizon.
    initial: pl.Timedelta window of the initial forecast period.
    period: pl.Timedelta simulated forecasts are done with this period.

    Returns
    -------
    list of pl.Datetime
    """
    # Last cutoff is 'latest date in data - horizon' date
    cutoff = df['ds'].max() - horizon
    if cutoff < df['ds'].min():
        raise ValueError('Less data than horizon.')
    result = [cutoff]
    df_min_ds = df['ds'].min()
    while result[-1] >= df_min_ds + initial:
        cutoff -= period
        # If data does not exist in data range (cutoff, cutoff + horizon]
        mask = (df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)
        has_data = df.filter(mask).height > 0
        if not has_data:
            # Next cutoff point is 'last date before cutoff in data - horizon'
            if cutoff > df_min_ds:
                closest_date = df.filter(pl.col('ds') <= cutoff)['ds'].max()
                cutoff = closest_date - horizon
            # else no data left, leave cutoff as is, it will be dropped.
        result.append(cutoff)
    result = result[:-1]
    if len(result) == 0:
        raise ValueError(
            'Less data than horizon after initial window. '
            'Make horizon or initial shorter.'
        )
    logger.info('Making {} forecasts with cutoffs between {} and {}'.format(
        len(result), result[-1], result[0]
    ))
    return list(reversed(result))


def parse_duration(duration_str):
    regex = r"(\d+)\s*(days?|hours?|minutes?|seconds?)"
    matches = re.findall(regex, duration_str)
    if not matches:
        raise ValueError("Invalid duration string: {}".format(duration_str))

    kwargs = {}
    for (value, unit) in matches:
        value = int(value)
        unit = unit.rstrip('s')  # Remove plural 's' if present
        kwargs[unit] = value
    return timedelta(**kwargs)

def cross_validation(model, horizon, period=None, initial=None, parallel=None, cutoffs=None, disable_tqdm=False, extra_output_columns=None):
    """Cross-Validation for time series.

    Computes forecasts from historical cutoff points, which user can input.
    If not provided, begins from (end - horizon) and works backwards, making
    cutoffs with a spacing of period until initial is reached.

    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .

    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model.
    horizon: string with timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    period: string with timedelta compatible style. Simulated forecast will
        be done at every this period. If not provided, 0.5 * horizon is used.
    initial: string with timedelta compatible style. The first training
        period will include at least this much data. If not provided,
        3 * horizon is used.
    cutoffs: list of datetime objects specifying cutoffs to be used during
        cross validation. If not provided, they are generated as described
        above.
    parallel : {None, 'processes', 'threads', 'dask', object}
        How to parallelize the forecast computation. By default no parallelism
        is used.

        * None : No parallelism.
        * 'processes' : Parallelize with concurrent.futures.ProcessPoolExecutor.
        * 'threads' : Parallelize with concurrent.futures.ThreadPoolExecutor.
            Note that some operations currently hold Python's Global Interpreter
            Lock, so parallelizing with threads may be slower than training
            sequentially.
        * 'dask': Parallelize with Dask.
           This requires that a dask.distributed Client be created.
        * object : Any instance with a `.map` method. This method will
          be called with :func:`single_cutoff_forecast` and a sequence of
          iterables where each element is the tuple of arguments to pass to
          :func:`single_cutoff_forecast`

    disable_tqdm: if True it disables the progress bar that would otherwise show up when parallel=None
    extra_output_columns: A String or List of Strings e.g. 'trend' or ['trend'].
         Additional columns to 'yhat' and 'ds' to be returned in output.

    Returns
    -------
    A pl.DataFrame with the forecast, actual value and cutoff.
    """
    if model.history is None:
        raise Exception('Model has not been fit. Fitting the model provides contextual parameters for cross validation.')
    
    df = model.history.clone()
    horizon = parse_duration(horizon)
    predict_columns = ['ds', 'yhat']
        
    if model.uncertainty_samples:
        predict_columns.extend(['yhat_lower', 'yhat_upper'])

    if extra_output_columns is not None:
        if isinstance(extra_output_columns, str):
            extra_output_columns = [extra_output_columns]
        predict_columns.extend([c for c in extra_output_columns if c not in predict_columns])
        
    # Identify largest seasonality period
    period_max = 0.
    for s in model.seasonalities.values():
        period_max = max(period_max, s['period'])
    seasonality_dt = timedelta(days=period_max)    

    if cutoffs is None:
        # Set period
        period = 0.5 * horizon if period is None else parse_duration(period)

        # Set initial
        initial = (
            max(3 * horizon, seasonality_dt) if initial is None
            else parse_duration(initial)
        )

        # Compute Cutoffs
        cutoffs = generate_cutoffs(df, horizon, initial, period)
    else:
        # Validate cutoffs
        if min(cutoffs) <= df['ds'].min(): 
            raise ValueError("Minimum cutoff value is not strictly greater than min date in history")
        end_date_minus_horizon = df['ds'].max() - horizon 
        if max(cutoffs) > end_date_minus_horizon: 
            raise ValueError("Maximum cutoff value is greater than end date minus horizon, no value for cross-validation remaining")
        initial = cutoffs[0] - df['ds'].min()
        
    if initial < seasonality_dt:
        msg = 'Seasonality has period of {} days '.format(period_max)
        msg += 'which is larger than initial window. '
        msg += 'Consider increasing initial.'
        logger.warning(msg)

    if parallel:
        valid = {"threads", "processes", "dask"}

        if parallel == "threads":
            pool = concurrent.futures.ThreadPoolExecutor()
        elif parallel == "processes":
            pool = concurrent.futures.ProcessPoolExecutor()
        elif parallel == "dask":
            try:
                from dask.distributed import get_client
            except ImportError as e:
                raise ImportError("parallel='dask' requires the optional "
                                  "dependency dask.") from e
            pool = get_client()
            df, model = pool.scatter([df, model])
        elif hasattr(parallel, "map"):
            pool = parallel
        else:
            msg = ("'parallel' should be one of {} or an instance with a "
                   "'map' method".format(', '.join(valid)))
            raise ValueError(msg)

        iterables = ((df, model, cutoff, horizon, predict_columns)
                     for cutoff in cutoffs)
        iterables = zip(*iterables)

        logger.info("Applying in parallel with %s", pool)
        predicts = pool.map(single_cutoff_forecast, *iterables)
        if parallel == "dask":
            predicts = pool.gather(predicts)

    else:
        predicts = [
            single_cutoff_forecast(df, model, cutoff, horizon, predict_columns) 
            for cutoff in (tqdm(cutoffs) if not disable_tqdm else cutoffs)
        ]

    # Combine all predicted pl.DataFrame into one pl.DataFrame
    return pl.concat(predicts)

def single_cutoff_forecast(df, model, cutoff, horizon, predict_columns):
    """Forecast for single cutoff. Used in cross validation function
    when evaluating for multiple cutoffs either sequentially or in parallel.

    Parameters
    ----------
    df: pl.DataFrame.
        DataFrame with history to be used for single cutoff forecast.
    model: Prophet model object.
    cutoff: pl.Datetime cutoff date.
        Simulated Forecast will start from this date.
    horizon: timedelta forecast horizon.
    predict_columns: List of strings e.g. ['ds', 'yhat'].
        Columns with date and forecast to be returned in output.

    Returns
    -------
    A pl.DataFrame with the forecast, actual value, and cutoff.
    """

    # Generate new object with copying fitting options
    m = prophet_copy(model, cutoff)
    # Train model
    history_c = df.filter(pl.col('ds') <= cutoff)
    if history_c.height < 2:
        raise Exception(
            'Less than two datapoints before cutoff. Increase initial window.'
        )
    m.fit(history_c, **model.fit_kwargs)
    # Calculate yhat
    columns = ['ds']
    if m.growth == 'logistic':
        columns.append('cap')
        if m.logistic_floor:
            columns.append('floor')
    columns.extend(m.extra_regressors.keys())
    columns.extend([
        props['condition_name']
        for props in m.seasonalities.values()
        if props['condition_name'] is not None
    ])
    future_df = df.filter(
        (pl.col('ds') > cutoff) & (pl.col('ds') <= cutoff + horizon)
    ).select(columns)
    yhat = m.predict(future_df)
    # Merge yhat(predicts), y(df, original data) and cutoff
    yhat_selected = yhat.select(predict_columns)
    df_y = df.filter(
        (pl.col('ds') > cutoff) & (pl.col('ds') <= cutoff + horizon)
    ).select('y')
    result = yhat_selected.with_columns([
        df_y['y'],
        pl.lit(cutoff).alias('cutoff')
    ])
    return result


def prophet_copy(m, cutoff=None):
    """Copy Prophet object.

    Parameters
    ----------
    m: Prophet model.
    cutoff: pl.Datetime or None, default None.
        Cutoff timestamp for changepoints member variable.
        Changepoints are only retained if 'changepoints <= cutoff'

    Returns
    -------
    Prophet class object with the same parameters as the model variable.
    """
    if m.history is None:
        raise Exception('This is for copying a fitted Prophet object.')

    if m.specified_changepoints:
        changepoints = m.changepoints
        if cutoff is not None:
            # Filter change points '< cutoff'
            last_history_date = m.history.filter(pl.col('ds') <= cutoff)['ds'].max()
            changepoints = changepoints[changepoints < last_history_date]
    else:
        changepoints = None

    # Auto seasonalities are set to False because they are already set in m.seasonalities.
    m2 = m.__class__(
        growth=m.growth,
        n_changepoints=m.n_changepoints,
        changepoint_range=m.changepoint_range,
        changepoints=changepoints,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=m.holidays,
        holidays_mode=m.holidays_mode,
        seasonality_mode=m.seasonality_mode,
        seasonality_prior_scale=m.seasonality_prior_scale,
        changepoint_prior_scale=m.changepoint_prior_scale,
        holidays_prior_scale=m.holidays_prior_scale,
        mcmc_samples=m.mcmc_samples,
        interval_width=m.interval_width,
        uncertainty_samples=m.uncertainty_samples,
        stan_backend=(
            m.stan_backend.get_type() if m.stan_backend is not None else None
        ),
    )
    m2.extra_regressors = deepcopy(m.extra_regressors)
    m2.seasonalities = deepcopy(m.seasonalities)
    m2.country_holidays = deepcopy(m.country_holidays)
    return m2


PERFORMANCE_METRICS = dict()


def register_performance_metric(func):
    """Register custom performance metric.

    Parameters
    ----------
    func: Function to register.

    Returns
    -------
    The registered function.
    """
    PERFORMANCE_METRICS[func.__name__] = func
    return func


def performance_metrics(df, metrics=None, rolling_window=0.1, monthly=False):
    """Compute performance metrics from cross-validation results.

    Parameters
    ----------
    df: The dataframe returned by cross_validation.
    metrics: A list of performance metrics to compute.
    rolling_window: Proportion of data to use in each rolling window for computing the metrics.
    monthly: If True, compute horizons as numbers of calendar months from the cutoff date.

    Returns
    -------
    Dataframe with a column for each metric and column 'horizon'.
    """
    valid_metrics = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']
    if metrics is None:
        metrics = valid_metrics
    if ('yhat_lower' not in df.columns or 'yhat_upper' not in df.columns) and ('coverage' in metrics):
        metrics.remove('coverage')
    if len(set(metrics)) != len(metrics):
        raise ValueError('Input metrics must be a list of unique values')
    if not set(metrics).issubset(set(PERFORMANCE_METRICS)):
        raise ValueError('Valid values for metrics are: {}'.format(valid_metrics))
    df_m = df.clone()
    if monthly:
        df_m = df_m.with_column(
            (
                pl.col('ds').dt.truncate('1mo').cast(int) - pl.col('cutoff').dt.truncate('1mo').cast(int)
            ).alias('horizon')
        )
    else:
        df_m = df_m.with_column((pl.col('ds') - pl.col('cutoff')).alias('horizon'))
    df_m = df_m.sort('horizon')
    min_abs_y = df_m.select(pl.abs(pl.col('y')).min()).item()
    if 'mape' in metrics and min_abs_y < 1e-8:
        logger.info('Skipping MAPE because y close to 0')
        metrics.remove('mape')
    if len(metrics) == 0:
        return None
    w = int(rolling_window * df_m.height)
    if w >= 0:
        w = max(w, 1)
        w = min(w, df_m.height)
    # Compute all metrics
    dfs = {}
    for metric in metrics:
        dfs[metric] = PERFORMANCE_METRICS[metric](df_m, w)
    res = dfs[metrics[0]]
    for i in range(1, len(metrics)):
        res_m = dfs[metrics[i]]
        if not res['horizon'].series_equal(res_m['horizon']):
            raise ValueError("Horizons are not aligned")
        res = res.hstack([res_m.select(metrics[i])])
    return res


def rolling_mean_by_h(x, h, w, name):
    """Compute a rolling mean of x, after first aggregating by h.

    Parameters
    ----------
    x: Array.
    h: Array of horizon for each value in x.
    w: Integer window size (number of elements).
    name: Name for metric in result dataframe.

    Returns
    -------
    Dataframe with columns horizon and name, the rolling mean of x.
    """
    # Aggregate over h
    df = pl.DataFrame({'x': x, 'h': h})
    df2 = df.groupby('h').agg([
        pl.col('x').sum().alias('x_sum'),
        pl.col('x').count().alias('x_count')
    ]).sort('h')
    xs = df2['x_sum'].to_numpy()
    ns = df2['x_count'].to_numpy()
    hs = df2['h'].to_numpy()

    trailing_i = len(df2) - 1
    x_sum = 0
    n_sum = 0
    res_x = np.empty(len(df2))

    for i in range(len(df2) - 1, -1, -1):
        x_sum += xs[i]
        n_sum += ns[i]
        while n_sum >= w:
            excess_n = n_sum - w
            excess_x = excess_n * xs[i] / ns[i]
            res_x[trailing_i] = (x_sum - excess_x) / w
            x_sum -= xs[trailing_i]
            n_sum -= ns[trailing_i]
            trailing_i -= 1

    res_h = hs[(trailing_i + 1):]
    res_x = res_x[(trailing_i + 1):]
    return pl.DataFrame({'horizon': res_h, name: res_x})


def rolling_median_by_h(x, h, w, name):
    """Compute a rolling median of x, after first aggregating by h.

    Parameters
    ----------
    x: Array.
    h: Array of horizon for each value in x.
    w: Integer window size (number of elements).
    name: Name for metric in result dataframe.

    Returns
    -------
    Dataframe with columns horizon and name, the rolling median of x.
    """
    df = pl.DataFrame({'x': x, 'h': h})
    grouped = df.groupby('h')
    hs = grouped.first().sort('h')['h'].to_numpy()
    res_h = []
    res_x = []
    i = len(hs) - 1
    total_xs = []
    while i >= 0:
        h_i = hs[i]
        xs = df.filter(pl.col('h') == h_i)['x'].to_list()
        total_xs.extend(xs)
        while len(total_xs) >= w:
            res_h.append(h_i)
            res_x.append(np.median(total_xs[-w:]))
            total_xs = total_xs[:-1]
        i -= 1
    res_h.reverse()
    res_x.reverse()
    return pl.DataFrame({'horizon': res_h, name: res_x})


@register_performance_metric
def mse(df, w):
    """Mean squared error."""
    se = (df['y'] - df['yhat']) ** 2
    if w < 0:
        return pl.DataFrame({'horizon': df['horizon'], 'mse': se})
    return rolling_mean_by_h(
        x=se.to_numpy(), h=df['horizon'].to_numpy(), w=w, name='mse'
    )


@register_performance_metric
def rmse(df, w):
    """Root mean squared error."""
    res = mse(df, w)
    res = res.with_column(pl.sqrt(pl.col('mse')).alias('rmse'))
    res = res.drop(['mse'])
    return res


@register_performance_metric
def mae(df, w):
    """Mean absolute error."""
    ae = (df['y'] - df['yhat']).abs()
    if w < 0:
        return pl.DataFrame({'horizon': df['horizon'], 'mae': ae})
    return rolling_mean_by_h(
        x=ae.to_numpy(), h=df['horizon'].to_numpy(), w=w, name='mae'
    )


@register_performance_metric
def mape(df, w):
    """Mean absolute percent error."""
    ape = ((df['y'] - df['yhat']).abs() / df['y']).abs()
    if w < 0:
        return pl.DataFrame({'horizon': df['horizon'], 'mape': ape})
    return rolling_mean_by_h(
        x=ape.to_numpy(), h=df['horizon'].to_numpy(), w=w, name='mape'
    )


@register_performance_metric
def mdape(df, w):
    """Median absolute percent error."""
    ape = ((df['y'] - df['yhat']).abs() / df['y']).abs()
    if w < 0:
        return pl.DataFrame({'horizon': df['horizon'], 'mdape': ape})
    return rolling_median_by_h(
        x=ape.to_numpy(), h=df['horizon'].to_numpy(), w=w, name='mdape'
    )


@register_performance_metric
def smape(df, w):
    """Symmetric mean absolute percentage error."""
    sape = (df['y'] - df['yhat']).abs() / ((df['y'].abs() + df['yhat'].abs()) / 2)
    sape = sape.fill_null(0)
    if w < 0:
        return pl.DataFrame({'horizon': df['horizon'], 'smape': sape})
    return rolling_mean_by_h(
        x=sape.to_numpy(), h=df['horizon'].to_numpy(), w=w, name='smape'
    )


@register_performance_metric
def coverage(df, w):
    """Coverage."""
    is_covered = ((df['y'] >= df['yhat_lower']) & (df['y'] <= df['yhat_upper'])).cast(pl.Float64)
    if w < 0:
        return pl.DataFrame({'horizon': df['horizon'], 'coverage': is_covered})
    return rolling_mean_by_h(
        x=is_covered.to_numpy(), h=df['horizon'].to_numpy(), w=w, name='coverage'
    )
