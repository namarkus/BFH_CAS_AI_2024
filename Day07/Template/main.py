# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# fix python path if working locally

import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.dataprocessing import Pipeline
from darts.models import TFTModel, CatBoostModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

import warnings
from typing import Union, Optional
from dataclasses import dataclass

warnings.filterwarnings("ignore")


@dataclass
class DatasetMetadata:
    # name of the dataset file, including extension
    name: str
    # used to indicate the target
    target_cols: Union[str, list[str]]
    # used to parse the dataset file
    header_time: Optional[str]
    # used to create group series
    group_cols: Union[str, list[str]]
    # used to select past covariates
    past_cov_cols: Union[str, list[str]]
    # used to select static cols
    static_cols: Union[str, list[str]] = None
    # used to convert the string date to pd.Datetime
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    format_time: Optional[str] = None
    # used to indicate the freq when we already know it
    freq: Optional[str] = None
    # multivariate
    multivariate: Optional[bool] = None
    # cutoff
    training_cutoff: [float] = 0.5


class Preprocessor:
    def __init__(self, metadata: DatasetMetadata):
        self.metadata = metadata
        # Read data
        self.data = pd.read_csv(self.metadata.name)
        self.data['Static_1'] = self.data['Agency']
        self.data['Static_2'] = self.data['SKU']

        self.data = self._format_time_column(self.data)
        self.past_cov_series_list = self._to_timeseries(self.metadata.past_cov_cols)
        self.target_series_list = self._to_timeseries(self.metadata.target_cols)

    def _format_time_column(self, df):
        df[self.metadata.header_time] = df[self.metadata.header_time].apply(
            lambda x: datetime.datetime.strptime(str(x), self.metadata.format_time))
        return df

    def _to_timeseries(self, value_cols):
        past_cv_ts_list = TimeSeries.from_group_dataframe(
            df=self.data,
            group_cols=self.metadata.group_cols,
            value_cols=value_cols,
            time_col=self.metadata.header_time,
            freq=self.metadata.freq,
            fill_missing_dates=True,
            static_cols=self.metadata.static_cols,
            drop_group_cols=self.metadata.group_cols,

        )

        return past_cv_ts_list

    def split(self):
        pass

    def get_description(self):
        pass


class Model:
    def __init__(self, preprocessor: Preprocessor, version: str, model_name="TFT"):
        assert model_name == "TFT" or model_name == "CatBoost", 'Use "TFT" or "CatBoost" for model name'

        self.model_name = model_name
        self.preprocessor = preprocessor
        self.version = version
        # train
        self.train_target_transformed = None
        self.val_target_transformed = None
        # validation
        self.train_past_cov_transformed = None
        self.val_past_cov_transformed = None

        self.input_chunk_length = 3
        self.forecast_horizon = 2

        self.train_target_scaler = None
        self.train_past_cov_scaler = None
        self.train_static_transformer = None

        if model_name == "TFT":
            # before starting, we define some constants
            # default quantiles for QuantileRegression
            quantiles = [
                0.01,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                0.99,
            ]

            self.model = TFTModel(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.forecast_horizon,
                hidden_size=64,
                lstm_layers=1,
                num_attention_heads=4,
                dropout=0.1,
                batch_size=16,
                n_epochs=1,
                add_relative_index=True,
                add_encoders=None,
                likelihood=QuantileRegression(
                    quantiles=quantiles
                ),  # QuantileRegression is set per default
                # loss_fn=MSELoss(),
                random_state=42,
                use_static_covariates=True
            )

        if model_name == "CatBoost":
            self.model = CatBoostModel(
                lags=self.input_chunk_length,
                lags_past_covariates=self.input_chunk_length,
                lags_future_covariates=None,
                output_chunk_length=self.forecast_horizon
            )

    def transform(self):

        target_series_list = self.preprocessor.target_series_list
        past_cov_series_list = self.preprocessor.past_cov_series_list

        # use StaticCovariatesTransformer to encode categorical static covariates into numeric data
        self.train_static_transformer = StaticCovariatesTransformer()
        target_series_list = self.train_static_transformer.fit_transform(target_series_list)
        past_cov_series_list = self.train_static_transformer.fit_transform(past_cov_series_list)

        # Create training and validation sets:
        # target(s)
        target_series_list_split = [x.split_after(self.preprocessor.metadata.training_cutoff) for x in
                                    target_series_list]
        train_target_series_list = [x[0] for x in target_series_list_split]
        val_target_series_list = [x[1] for x in target_series_list_split]

        # past covariates:
        past_cov_series_list_split = [x.split_after(self.preprocessor.metadata.training_cutoff) for x in
                                      past_cov_series_list]
        train_past_cov_series_list = [x[0] for x in past_cov_series_list_split]
        val_past_cov_series_list = [x[1] for x in past_cov_series_list_split]

        # Normalize the time series (note: we avoid fitting the transformer on the validation set)

        self.train_target_scaler = Scaler(verbose=False, n_jobs=-1, name="Target_Scaling")
        self.train_past_cov_scaler = Scaler(verbose=False, n_jobs=-1, name="Past_Cov_Scaling")

        # TODO: Pipeline for several transformations at ones
        # train_pipeline = Pipeline([train_filler,
        #                           static_cov_transformer,
        #                           log_transformer,
        #                           train_scaler])
        self.train_target_transformed = self.train_target_scaler.fit_transform(train_target_series_list)
        self.val_target_transformed = self.train_target_scaler.transform(val_target_series_list)

        self.train_past_cov_transformed = self.train_past_cov_scaler.fit_transform(train_past_cov_series_list)
        self.val_past_cov_transformed = self.train_past_cov_scaler.fit_transform(val_past_cov_series_list)

    def fit(self):
        self.model.fit(self.train_target_transformed,
                       past_covariates=self.train_past_cov_transformed,
                       val_series=self.val_target_transformed,
                       val_past_covariates=self.val_past_cov_transformed,
                       verbose=True)

    def predict(self, n: int, target_series: TimeSeries, past_covariates: TimeSeries):
        forecast = self.model.predict(n=n, series=target_series, past_covariates=past_covariates, )
        return forecast

    def validate(self, on_finished_projects):
        backtest_series_transformed = model.generate_backtest_series(on_finished_projects=on_finished_projects)

        backtest_series = model.train_target_scaler.inverse_transform(backtest_series_transformed)
        if on_finished_projects:
            target_series = model.val_target_transformed
            data = model.preprocessor.val_data
        else:
            target_series = model.test_target_transformed
            data = model.preprocessor.test_data
        target_series = model.train_target_scaler.inverse_transform(target_series)
        gp = data.groupby(model.preprocessor.metadata.group_cols)
        results = []
        for backtest, target, group in zip(backtest_series, target_series, gp.groups.keys()):
            print(f"Group Keys: {group}")
            target_df = target.pd_dataframe()
            backtest_df = concatenate(backtest).pd_dataframe()
            # print(backtest_df.index)

            merge = pd.merge(target_df, backtest_df, how='left', left_index=True, right_index=True,
                             suffixes=('_true', '_forecast'))
            merge['residuals'] = target_df - backtest_df
            merge['integration_id'] = group[0]
            merge['stage'] = group[1]
            merge['milestone'] = group[2]

            results.append(merge)
        result_df = pd.concat(results, axis=0)
        result_df.reset_index(drop=False, inplace=True)
        return result_df


    def save(self, directory: str):
        self.model.save()

    def load(self, directory: str):
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_meta_data = DatasetMetadata(name="data/sales_data/price_sales_promotion.csv",
                                        target_cols=["Sales"],
                                        header_time="YearMonth",
                                        group_cols=['Agency', 'SKU'],
                                        past_cov_cols=['Price', 'Promotions'],
                                        format_time="%Y%m",
                                        static_cols=['Static_1', 'Static_2'])
    # dataset_meta_data.freq = 'M'
    preproc = Preprocessor(dataset_meta_data)
    model = Model(preproc, '1', 'TFT')
    model.transform()
    print(model.train_target_transformed[0].static_covariates)
    model.fit()
    predictions_list = model.predict(1, model.train_target_transformed, model.train_past_cov_transformed)

    predictions_list = model.train_target_scaler.inverse_transform(predictions_list)
    prediction_df = pd.concat([prediction.pd_dataframe() for prediction in predictions_list])

    gp = preproc.data.groupby(dataset_meta_data.group_cols)
    groups_df = pd.DataFrame(gp.groups.keys(), index=prediction_df.index)

    prediction_df = pd.concat([groups_df, prediction_df], axis=1)

    print(prediction_df)
