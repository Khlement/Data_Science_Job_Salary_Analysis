# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 00:48:18 2023

@author: khlement
"""

import scraper as s
import pandas as pd

path = "C:/Users/khlement/Documents/ds_salary_proj/chromedriver"

df = s.get_jobs('data scients', 15, False, path, 15)