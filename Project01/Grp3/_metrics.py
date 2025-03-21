#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _metrics.py ]__________________________________________________________
"""
Diverse Mechanismen für die zentrale Sammlung von Metriken.

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Funktionen zur Initialiserung, Konfiguration und Verwendung von Metriken
bereit. 
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
from datetime import datetime
from _configs import VbcConfig
from lightning.pytorch import loggers as pl_loggers

class TensorBoardMonitor:

    def __init__(self, config: VbcConfig):
        run_id=f"vbc_{config.action.value}"
        run_version = config.as_profile_label() + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = pl_loggers.TensorBoardLogger(save_dir="logs/", name=run_id,  version=run_version)
        self.tensorboard.log_hyperparams(config.as_hyperparams())

    def log_metrics(self, metrics: dict):
        self.tensorboard.log_metrics(metrics)

    def log_hyperparams(self, hyperparams: dict):
        self.tensorboard.log_hyperparams(hyperparams) 

    def log_graph(self, model):
        self.tensorboard.log_graph(model)

    def log_images(self, tag: str, images: list):  
        self.tensorboard.log_images(tag, images)

    def log_text(self, tag: str, text: str):
        self.tensorboard.log_text(tag, text)
