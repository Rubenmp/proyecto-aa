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


print("Problema de clasificación APS Failure at Scania Trucks Data Set\n")

# Clasification data
ds = get_dataset(small=True)
ds.preprocess()
