# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import numpy as np
import polars as pl
from prophet.forecaster import Prophet

about = {}
here = Path(__file__).parent.resolve()
with open(here / "__version__.py", "r") as f:
    exec(f.read(), about)

SIMPLE_ATTRIBUTES = [
    'growth', 'n_changepoints', 'specified_changepoints', 'changepoint_range',
    'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
    'seasonality_mode', 'seasonality_prior_scale', 'changepoint_prior_scale',
    'holidays_prior_scale', 'mcmc_samples', 'interval_width', 'uncertainty_samples',
    'y_scale', 'y_min', 'scaling', 'logistic_floor', 'country_holidays', 'component_modes',
    'holidays_mode'
]
PL_SERIES = ['changepoints', 'history_dates', 'train_holiday_names']
PL_TIMESTAMP = ['start']
PL_DURATION = ['t_scale']
PL_DATAFRAME = ['holidays', 'history', 'train_component_cols']
NP_ARRAY = ['changepoints_t']
ORDEREDDICT = ['seasonalities', 'extra_regressors']

def model_to_dict(model):
    """Convert a Prophet model to a dictionary suitable for JSON serialization.
    Model must be fitted. Skips Stan objects that are not needed for predict.
    Can be reversed with model_from_dict.
    """
    if model.history is None:
        raise ValueError(
            "This can only be used to serialize models that have already been fit."
        )
    model_dict = {
        attribute: getattr(model, attribute) for attribute in SIMPLE_ATTRIBUTES
    }
    # Handle attributes of non-core types
    for attribute in PL_SERIES:
        if getattr(model, attribute) is None:
            model_dict[attribute] = None
        else:
            model_dict[attribute] = getattr(model, attribute).to_series().to_list()

    for attribute in PL_TIMESTAMP:
        model_dict[attribute] = getattr(model, attribute).timestamp()

    for attribute in PL_DURATION:
        model_dict[attribute] = getattr(model, attribute).total_seconds()

    for attribute in PL_DATAFRAME:
        if getattr(model, attribute) is None:
            model_dict[attribute] = None
        else:
            # Convert the dataframe to JSON
            model_dict[attribute] = getattr(model, attribute).write_json()
    
    for attribute in NP_ARRAY:
        model_dict[attribute] = getattr(model, attribute).tolist()

    for attribute in ORDEREDDICT:
        model_dict[attribute] = [
            list(getattr(model, attribute).keys()),
            getattr(model, attribute),
        ]
    
    # Other attributes with special handling
    fit_kwargs = deepcopy(model.fit_kwargs)
    if 'init' in fit_kwargs:
        for k, v in fit_kwargs['init'].items():
            if isinstance(v, np.ndarray):
                fit_kwargs['init'][k] = v.tolist()
            elif isinstance(v, np.floating):
                fit_kwargs['init'][k] = float(v)
    model_dict['fit_kwargs'] = fit_kwargs

    model_dict['params'] = {k: v.tolist() for k, v in model.params.items()}

    model_dict['__prophet_version'] = about["__version__"]
    return model_dict

def model_to_json(model):
    """Serialize a Prophet model to json string.
    Model must be fitted. Skips Stan objects that are not needed for predict.
    Can be deserialized with model_from_json.
    """
    model_json = model_to_dict(model)
    return json.dumps(model_json)

def _handle_simple_attributes_backwards_compat(model_dict):
    """Handle backwards compatibility for SIMPLE_ATTRIBUTES."""
    if 'scaling' not in model_dict:
        model_dict['scaling'] = 'absmax'
        model_dict['y_min'] = 0.
    if 'holidays_mode' not in model_dict:
        model_dict['holidays_mode'] = model_dict['seasonality_mode']

def model_from_dict(model_dict):
    """Recreate a Prophet model from a dictionary.
    Recreates models that were converted with model_to_dict.
    """
    model = Prophet()
    _handle_simple_attributes_backwards_compat(model_dict)
    for attribute in SIMPLE_ATTRIBUTES:
        setattr(model, attribute, model_dict[attribute])

    for attribute in PL_SERIES:
        if model_dict[attribute] is None:
            setattr(model, attribute, None)
        else:
            s = pl.Series(name=attribute, values=model_dict[attribute])
            setattr(model, attribute, s)

    for attribute in PL_TIMESTAMP:
        setattr(model, attribute, pl.datetime.from_timestamp(model_dict[attribute]))

    for attribute in PL_DURATION:
        setattr(model, attribute, pl.Duration(seconds=model_dict[attribute]))

    for attribute in PL_DATAFRAME:
        if model_dict[attribute] is None:
            setattr(model, attribute, None)
        else:
            df = pl.read_json(model_dict[attribute])
            setattr(model, attribute, df)

    for attribute in NP_ARRAY:
        setattr(model, attribute, np.array(model_dict[attribute]))

    for attribute in ORDEREDDICT:
        key_list, unordered_dict = model_dict[attribute]
        od = OrderedDict()
        for key in key_list:
            od[key] = unordered_dict[key]
        setattr(model, attribute, od)

    model.fit_kwargs = model_dict['fit_kwargs']
    model.params = {k: np.array(v) for k, v in model_dict['params'].items()}
    model.stan_backend = None
    model.stan_fit = None
    return model

def model_from_json(model_json):
    """Deserialize a Prophet model from json string.
    Deserializes models that were serialized with model_to_json.
    """
    model_dict = json.loads(model_json)
    return model_from_dict(model_dict)