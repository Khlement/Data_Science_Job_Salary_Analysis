# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:35:08 2023

@author: khlement
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

df.columns

#selecting relevant columns
df_model = df[['avg_salary','Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 
               'Revenue', 'num_comp', 'hourly', 'employer_provided', 'job_state', 'same_state', 
               'age', 'python_yn', 'spark', 'aws', 'excel', 'job_simp', 'seniority', 'desc_len',]]

#get dummy data
df_dum = pd.get_dummies(df_model)

# train test split

from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis = 1)
y = df_dum.avg_salary.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
fitting = model.fit().summary()


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)


def cross_val(model):
    result = np.mean(cross_val_score(model, X_train, y_train,scoring = 'neg_mean_absolute_error', cv=3))
    print(result)

Linear_val = cross_val(lm)

#lasso regression
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train, y_train)
Lasso_val = cross_val(lm_l)

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha = (i/100))
    Lasso_val2 = cross_val(lml)
    error.append(Lasso_val2)


plt.plot(alpha,error)  


err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
best_alpha = df_err[df_err.error == max(df_err.error)]

#print(best_alpha)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf_val = cross_val(rf)
#print(rf_val)

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': range(10, 300, 10), 'criterion': ('squared_error', 'absolute_error'), 'max_features': ('auto', 'sqrt', 'log2')}

gd = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gd.fit(X_train, y_train)
gd.best_score_
gd.best_estimator_

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf =gd.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test, tpred_rf)

mean_absolute_error(y_test, (tpred_lm + tpred_rf)/2)