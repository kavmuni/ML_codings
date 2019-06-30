# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 13:54:27 2019

@author: Muniappan
"""
"""import os
import requests
from requests import session
from dotenv import load_dotenv, find_dotenv
dotenv_path=find_dotenv() # F:\\Data_science_with_python\\ML_from_pluralsight\\.env
load_dotenv(dotenv_path)
KAGGLE_USERNAME=os.environ.get('KAGGLE_USERNAME')
KAGGLE_PASSWORD=os.environ.get('KAGGLE_PASSWORD')

## payload for post
payload={'acrion':'login',
         'username':KAGGLE_USERNAME,
         'password':KAGGLE_PASSWORD}
url='https://www.kaggle.com/c/titanic/download/train.csv'
# https://www.kaggle.com/c/3136/download-all

# session function is used
c=requests.session()
c.post('https://www.kaggle.com/account/login', data=payload)
response=c.get('https://www.kaggle.com/c/titanic/download/train.csv')
cookies = response.cookies.get_dict()
print(response.url)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path_train=r'F:\Data_science_with_python\ML_from_pluralsight\titanic_data_set\train.csv'
path_test=r'F:\Data_science_with_python\ML_from_pluralsight\titanic_data_set\test.csv'

# index_col is the unique column present in the data set, like primary key
data_train=pd.read_csv(path_train, index_col='PassengerId')
data_test=pd.read_csv(path_test, index_col='PassengerId')

data_train.columns.tolist()
data_train['PassengerId']
data_train.info()
data_train.Name
data_train.groupby('Survived').agg({'Survived':'size'})
# 0 died, 1 survived

data_test.info()
# Survived column is not present in the test data set
data_test_with_survived=pd.read_csv(path_test, index_col='PassengerId')
data_test_with_survived['Survived']=-999
data_train_test=pd.concat((data_train, data_test_with_survived), sort=True)

data_train_test.info()
data_train_test.head(2)

data_train_test['Age'].unique()

# label based index
#data_train_test.loc[index of rows: column range]
data_train_test.loc[5:10,'Age':'Embarked']

# position based index
# data_train_test.iloc[row positions,column positions]
data_train_test.iloc[3:5,3:5]

data_train_test.describe()

data_train_test_male=data_train_test.query("Sex == 'male'").groupby('Age').agg({'Age':'size'})
data_train_test_malegroupby=data_train_test.query("Sex == 'male'").groupby('Age')
data_train_test_malegroupby.first()
data_train_test_malegroupby['Age'].first()
len(data_train_test_male)
data_train_test_male.info()
data_train_test_male.shape # (96, 1)
data_train_test_male['Age'] == 3
data_train_test_male.query("Age == 3")

data_train_test_malegroupby.get_group(0.33)
data_train_test_male.get_group(0.33)

data_train_test_male_data=data_train_test.query("Sex == 'male'")
data_train_test_male_data_age_unique=data_train_test_male_data['Age'].unique()

age_group_dict={}
for i in data_train_test_male_data_age_unique:
    x_len_value=len(data_train_test_male_data['Age'] == i)
    print(str(i)+ " - " + str(x_len_value))
    #age_group_dict.update(i:x_len_value)
    
data_train_test_male_data['Age']    

data_train_test_male.columns.tolist()

data_train_test_male_age_unique=data_train_test_male.unique()
len(data_train_test_male_age_unique)

data_train_test['Age'].unique()

plt.bar([0-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80,80-90])
x_min_max=[0,10,20,3,40,50,60,70,80,90]
age_groups_count={}

def fun_age_count(x,x_min,x_max):
    x_count=0
    if x >= x_min and x < x_max:
        x_count=x_count+1
    return x_count






