# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# fix python path if working locally

import os
import numpy as np
import pandas as pd
import datetime
import warnings

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


from typing import Union, Optional
from dataclasses import dataclass

warnings.filterwarnings("ignore")


@dataclass
class TftDatasetMetadata:   
    name: Union[str, list[str]]  # name of the dataset files, including directory and extension 
    target_cols: Union[str, list[str]]     # used to indicate the target
    header_time: Optional[str]     # used to parse the dataset file
    group_cols: Union[str, list[str]]     # used to create group series
    past_cov_cols: Union[str, list[str]]     # used to select past covariates
    static_cols: Union[str, list[str]] = None     # used to select static cols
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    format_time: Optional[str] = None     # used to convert the string date to pd.Datetime
    freq: Optional[str] = None     # used to indicate the freq when we already know it
    multivariate: Optional[bool] = None     # multivariate
    year_cutback: Optional[int] = None    # Limitieren der zu bearbeitenden Daten auf die letzen x Jahre
    training_cutoff: Optional[float] = 0.5     # cutoff

class TftPreprocessor:
    def __init__(self, metadata: TftDatasetMetadata):
        self.metadata = metadata
        self.data = None

    def load_data(self):
        data = pd.DataFrame()
        for filename in self.metadata.name:
            if not os.path.isfile(filename):
                raise FileNotFoundError(f'Datei {filename} nicht gefunden.')
            print (f'Lade Datei {filename} ...')
            data = pd.concat([data, pd.read_csv(filename)], ignore_index=True)
        data['date'] = pd.to_datetime(data['date'], format=self.metadata.format_time)
        data.fillna(value={"precipitation": 0, "snowfall": 0, "snowdepth": 0}, inplace=True)
        if self.metadata.year_cutback:
            oldest_required_date = pd.Timestamp.now() - pd.DateOffset(years=self.metadata.year_cutback)
            print(f'Filtere auf die letzten {self.metadata.year_cutback} Jahre des Datensatzes. Alles vor dem {oldest_required_date:%d.%m.%Y} wird ignoriert.')
            data = data[data['date'] >= oldest_required_date]
        self.data = data
        print (f'Daten geladen. Dataset enthält {self.data.shape[0]:,} Zeilen und {self.data.shape[1]} Spalten.')        
        #self.data = self._format_time_column(self.data)
        self.__past_covariant_timeseries = self._as_timeseries(self.metadata.past_cov_cols)
        print (f'Past Covariates erstellt. Dataset enthält {len(self.__past_covariant_timeseries):,} Zeitreihen.')
        self.__target_timeseries = self._as_timeseries(self.metadata.target_cols)
        print (f'Target erstellt. Dataset enthält {len(self.__target_timeseries):,} Zeitreihen.')

    def get_train_data(self):
        if self.data is None:
            self.load_data()
        return self.data[self.data[self.metadata.header_time] < self.metadata.training_cutoff]
    
    @property
    def target_timeseries(self):
        if self.data is None:
            self.load_data()
        return self.__target_timeseries
    
    @property
    def past_covariant_timeseries(self):
        if self.data is None:
            self.load_data()
        return self.__past_covariant_timeseries

    def _format_time_column(self, df):
        df[self.metadata.header_time] = df[self.metadata.header_time].apply(
            lambda x: datetime.datetime.strptime(str(x), self.metadata.format_time))
        return df
    

    def _as_timeseries(self, value_cols):
        past_cv_ts_list = TimeSeries.from_group_dataframe(
            df=self.data,
            group_cols=self.metadata.group_cols,
            value_cols=value_cols,
            time_col=self.metadata.header_time,
            freq=self.metadata.freq,
            fill_missing_dates=True,
            static_cols=self.metadata.static_cols,
            drop_group_cols=self.metadata.group_cols
        )
        return past_cv_ts_list

    def split(self):
        pass

    def get_description(self):
        pass
