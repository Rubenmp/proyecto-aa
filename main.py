#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROYECTO FINAL DE APRENDIZAJE AUTOMÁTICO
Autores:
- Rubén Morales Pérez
- Francisco Javier Morales Piqueras
"""

from dataset import *

import pandas as pd

data_folder = "./datos"

print("Problema de clasificación APS Failure at Scania Trucks Data Set\n")

# Clasification data
print("Preprocesamiento de datos para clasificación")
train_f, test_f = "aps_failure_training_set.csv", "aps_failure_test_set.csv" 
ds = DataSet(data_folder + "/" + train_f, data_folder + "/" + test_f)
