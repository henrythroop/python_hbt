# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:30:01 2014

@author: throop
"""

file_csv = '/Users/throop/Downloads/gv_table_planets_throop_28a10_3244.txt'

from pandas import DataFrame, Series
import pandas as pd; import numpy as np

from numpy import genfromtxt
my_data = genfromtxt(file_csv, delimiter=',')

#DataFrame.from_csv(path, header=0, sep=', ', index_col=0, parse_dates=True, encoding=None, tupleize_cols=False, infer_datetime_format=False)

d = DataFrame.from_csv(file_csv, header=0, sep=', ', index_col=0, parse_dates=True, encoding=None, tupleize_cols=False, infer_datetime_format=False)
