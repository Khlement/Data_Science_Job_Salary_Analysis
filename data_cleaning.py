# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 00:44:40 2023

@author: khlement
"""

import pandas as pd

df = pd.read_csv("glassdoor_jobs.csv")

## SALARY PARSING


#Creating new column based on hourly wages or employer provided salary in the "Salry Estimate" column
df["Hourly"] = df["Salary Estimate"].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df["Employer_provided_salary"] = df["Salary Estimate"].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)
df.head()

# remove all salry estimates with value '-1'. '-1' is a placeholder for unavailable salary estimate
df = df[df['Salary Estimate'] != '-1']

# removing "(Glassdoor estimate)" from "Salary Estimate" column
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x: x.replace('K', '').replace('$',''))


min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:',''))

df['Min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['Max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['Average_salary'] = (df.Min_salary + df.Max_salary)/2


## COMPANY NAME TEXT ONLY
df['Company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)


##STATE FIELD
df['Job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df.Job_state.value_counts()

df['Same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)


## AGE OF COMPANY
df['Age'] = df.Founded.apply(lambda x: x if x < 1 else 2023 - x)

## PARSING JOB DESCRIPTION (python, etc)

df['Job Description'][0]

#python
df['Python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
print(df.Python_yn.value_counts())

#R studio 
df['RStudio_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
print(df.RStudio_yn.value_counts())

#spark
df['Spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
print(df.Spark.value_counts())

#aws
df['AWS'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
print(df.AWS.value_counts())

#excel
df['Excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
print(df.Excel.value_counts())

df_out = df.drop(['Unnamed: 0'], axis = 1)

df_out.to_csv('salary_data_cleaned.csv', index = False)