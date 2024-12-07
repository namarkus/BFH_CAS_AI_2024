# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# fix python path if working locally

import numpy as np
import pandas as pd
import datetime
import warnings

import matplotlib.pyplot as plt

from _tft_darts_helpers import TftDatasetMetadata
from _tft_darts_helpers import TftPreprocessor
from _tft_darts_models import TftModel 

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.dataprocessing import Pipeline
from darts.models import TFTModel, CatBoostModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression


from typing import Union, Optional
from dataclasses import dataclass

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #locale.setlocale(locale.LC_ALL, 'de_DE')
    dataset_meta_data = TftDatasetMetadata(name=["data/CH_grouped.csv"], #  "data/DE_grouped.csv", "data/AT_grouped.csv", 
                                        target_cols=["temperature"],
                                        header_time="date",
                                        group_cols=['station_id', 'observation_type'],
                                        past_cov_cols=['precipitation', 'snowfall', 'snowdepth'],
                                        format_time="%Y%m%d",
                                        freq=1,
                                        static_cols=['latitude', 'longitude', 'elevation'])
    preprocessor = TftPreprocessor(dataset_meta_data)
    preprocessor.load_data()
    #####
    model = TftModel(preprocessor, 2, 3)
    model.transform(StaticCovariatesTransformer(), Scaler(verbose=False, n_jobs=-1, name="TftScaler"))
    print(model.train_target_transformed[0].static_covariates)
    model.fit()
    predictions_list = model.predict(1, model.train_target_transformed, model.train_past_cov_transformed)
    predictions_list = model.train_target_scaler.inverse_transform(predictions_list)
    prediction_df = pd.concat([prediction.pd_dataframe() for prediction in predictions_list])

    gp = preprocessor.data.groupby(dataset_meta_data.group_cols)
    groups_df = pd.DataFrame(gp.groups.keys(), index=prediction_df.index)
    prediction_df = pd.concat([groups_df, prediction_df], axis=1)

    print(prediction_df)
