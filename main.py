#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRABAJO 3. 
Nombre Estudiante: Rubén Morales Pérez
"""

from dataset import *

import pandas as pd

data_folder = "./datos"

##################
# CLASSIFICATION #
##################

print("Problema de clasificación APS Failure at Scania Trucks Data Set\n")

# Clasification data
print("Preprocesamiento de datos para clasificación")
train_f, test_f = "aps_failure_training_set.csv", "aps_failure_test_set.csv" 
ds = dataset(data_folder + "/" + train_f, data_folder + "/" + test_f, cont_output=False)
#ds.preprocessing()
