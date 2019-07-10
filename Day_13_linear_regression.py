# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:14:33 2019

@author: Muniappan
"""

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence  import variance_inflation_factor
from patsy import dmatrices

path=r'F:\Data_science_with_python\Day_13\dm.csv'

dm_data=pd.read_csv(path)

dm_data.head(5)

dm_data.info()
"""
dm_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 11 columns):
Age            1000 non-null object
Gender         1000 non-null object
OwnHome        1000 non-null object
Married        1000 non-null object
Location       1000 non-null object
Salary         1000 non-null int64
Children       1000 non-null int64
History        697 non-null object
Catalogs       1000 non-null int64
AmountSpent    1000 non-null int64
Cust_Id        1000 non-null int64
dtypes: int64(5), object(6)
memory usage: 86.0+ KB
"""

# 3 months down the line how much will the customer spend
dm_data.isnull().sum()
"""
dm_data.isnull().sum()
Out[8]: 
Age              0
Gender           0
OwnHome          0
Married          0
Location         0
Salary           0
Children         0
History        303
Catalogs         0
AmountSpent      0
Cust_Id          0
dtype: int64
"""

# fill the null values with a values
dm_data['History']=dm_data['History'].fillna('NewCustomer')

dm_data.info()
"""
dm_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 11 columns):
Age            1000 non-null object
Gender         1000 non-null object
OwnHome        1000 non-null object
Married        1000 non-null object
Location       1000 non-null object
Salary         1000 non-null int64
Children       1000 non-null int64
History        1000 non-null object
Catalogs       1000 non-null int64
AmountSpent    1000 non-null int64
Cust_Id        1000 non-null int64
dtypes: int64(5), object(6)
memory usage: 86.0+ KB
"""

# OLS - Ordinary Least Squares regression

# Amount spedn is a function of salary
AmountSpent_mod=smf.ols("AmountSpent~Salary", data=dm_data).fit()
print(AmountSpent_mod.summary())
"""
print(AmountSpent_mod.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            AmountSpent   R-squared:                       0.489
Model:                            OLS   Adj. R-squared:                  0.489
Method:                 Least Squares   F-statistic:                     956.7
Date:                Fri, 05 Jul 2019   Prob (F-statistic):          7.50e-148
Time:                        11:27:22   Log-Likelihood:                -7950.4
No. Observations:                1000   AIC:                         1.590e+04
Df Residuals:                     998   BIC:                         1.591e+04
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -15.3178     45.374     -0.338      0.736    -104.358      73.722
Salary         0.0220      0.001     30.930      0.000       0.021       0.023
==============================================================================
Omnibus:                      189.162   Durbin-Watson:                   2.003
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              704.029
Skew:                           0.868   Prob(JB):                    1.32e-153
Kurtosis:                       6.726   Cond. No.                     1.33e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.33e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
# -15.3178 constant
# 0.0220 is beta value
# amount spent = constant+beta value(salary)

dm_data['Gender'].unique()
# Out[20]: array(['Female', 'Male'], dtype=object)
Gender_mod=smf.ols("AmountSpent~C(Gender)", data=dm_data).fit()
print(Gender_mod.summary())
"""
print(Gender_mod.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            AmountSpent   R-squared:                       0.041
Model:                            OLS   Adj. R-squared:                  0.040
Method:                 Least Squares   F-statistic:                     42.32
Date:                Fri, 05 Jul 2019   Prob (F-statistic):           1.22e-10
Time:                        11:30:38   Log-Likelihood:                -8265.7
No. Observations:                1000   AIC:                         1.654e+04
Df Residuals:                     998   BIC:                         1.655e+04
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
Intercept          1025.3399     41.868     24.490      0.000     943.181    1107.499
C(Gender)[T.Male]   387.5103     59.568      6.505      0.000     270.617     504.404
==============================================================================
Omnibus:                      297.759   Durbin-Watson:                   1.924
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              822.189
Skew:                           1.520   Prob(JB):                    2.91e-179
Kurtosis:                       6.238   Cond. No.                         2.60
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

dm_data['History'].unique()
# Out[19]: array(['High', 'Low', 'Medium', 'NewCustomer'], dtype=object)
history_mod=smf.ols("AmountSpent~C(History)", data=dm_data).fit()
print(history_mod.summary())
"""
R-squared = (TSS - RSS)/TSS

R-squared = 0.46 (46% of data is correctly represented correct)

We should consider the column in model calculation when P values is low.

print(history_mod.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            AmountSpent   R-squared:                       0.460
Model:                            OLS   Adj. R-squared:                  0.459
Method:                 Least Squares   F-statistic:                     283.2
Date:                Fri, 05 Jul 2019   Prob (F-statistic):          6.52e-133
Time:                        11:32:43   Log-Likelihood:                -7978.0
No. Observations:                1000   AIC:                         1.596e+04
Df Residuals:                     996   BIC:                         1.598e+04
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
=============================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------
Intercept                  2186.1373     44.277     49.374      0.000    2099.249    2273.025
C(History)[T.Low]         -1829.0503     64.297    -28.447      0.000   -1955.223   -1702.877
C(History)[T.Medium]      -1235.7363     65.716    -18.804      0.000   -1364.695   -1106.778
C(History)[T.NewCustomer]  -946.2363     60.087    -15.748      0.000   -1064.147    -828.325
==============================================================================
Omnibus:                      343.012   Durbin-Watson:                   1.946
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1494.832
Skew:                           1.559   Prob(JB):                         0.00
Kurtosis:                       8.115   Cond. No.                         4.78
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
# high category is neglected
# compared to high, low spend differ by -1829.0503

history_salary_mod=smf.ols("AmountSpent~C(History)+Salary", data=dm_data).fit()
print(history_salary_mod.summary())
"""
print(history_salary_mod.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            AmountSpent   R-squared:                       0.607
Model:                            OLS   Adj. R-squared:                  0.605
Method:                 Least Squares   F-statistic:                     383.8
Date:                Fri, 05 Jul 2019   Prob (F-statistic):          6.96e-200
Time:                        11:37:38   Log-Likelihood:                -7819.9
No. Observations:                1000   AIC:                         1.565e+04
Df Residuals:                     995   BIC:                         1.567e+04
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
=============================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------
Intercept                   940.3952     74.977     12.543      0.000     793.265    1087.525
C(History)[T.Low]         -1064.3412     67.789    -15.701      0.000   -1197.366    -931.316
C(History)[T.Medium]       -815.6504     60.226    -13.543      0.000    -933.835    -697.466
C(History)[T.NewCustomer]  -470.1233     56.975     -8.251      0.000    -581.928    -358.319
Salary                        0.0149      0.001     19.242      0.000       0.013       0.016
==============================================================================
Omnibus:                      274.043   Durbin-Watson:                   1.972
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1197.653
Skew:                           1.218   Prob(JB):                    8.57e-261
Kurtosis:                       7.776   Cond. No.                     3.80e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.8e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
# amount spent=constant+b1*salary+b2*Low+b3*Medium+b4*Newcustomer

"""
RSS - Residual Sum of Squares

Minimize the RSS by changing beta and constant value to find the model with least error

Differential calculus is use to minimize the equation and arrive at the solution
RSS is found by calculating the error for each row, square it and add the whole entries


Alt + left square, then backspace
"""
# the estimation is done on the sample of data from the whole actual data

plt.scatter(dm_data['Salary'], dm_data['AmountSpent'])

dm_data.describe() # will give central tendency of the numeric variable
"""
              Salary    Children     Catalogs  AmountSpent      Cust_Id
count    1000.000000  1000.00000  1000.000000  1000.000000  1000.000000
mean    56103.900000     0.93400    14.682000  1216.770000   643.930000
std     30616.314826     1.05107     6.622895   961.068613   369.047166
min     10100.000000     0.00000     6.000000    38.000000    12.000000
25%     29975.000000     0.00000     6.000000   488.250000   316.750000
50%     53700.000000     1.00000    12.000000   962.000000   636.000000
75%     77025.000000     2.00000    18.000000  1688.500000   955.500000
max    168800.000000     3.00000    24.000000  6217.000000  1297.000000
"""
dm_data.groupby('Age').agg({'AmountSpent':'mean'})
"""
        AmountSpent
Age                
Middle  1501.690945
Old     1432.126829
Young    558.623693
"""

dm_data.boxplot(by='Age', column='AmountSpent')

dm_data.boxplot(by='Gender', column='AmountSpent')

dm_data.boxplot(by='OwnHome', column='AmountSpent')

dm_data.boxplot(by='Married', column='AmountSpent')

dm_data.boxplot(by='Location', column='AmountSpent')

dm_data.boxplot(by='History', column='AmountSpent')

dm_data.boxplot(by='Catalogs', column='AmountSpent')

dm_data['Age_new']=dm_data['Age'].map(lambda x:"Old-Middle" if x != "Young" else "Young")

# for categorical use box plot, for continuous use scatter plot

dm_data['Children_01']=dm_data['Children'].map(lambda x:1 if x<=1 else 0)
dm_data['Children_23']=dm_data['Children'].map(lambda x:1 if x>1 else 0)

dm_data.dtypes

## before we model we seggregate the data into 2 parts. 1 part for building and training the model and the other is used to test the model


# sampling the data
dm_data_train=dm_data.sample(frac=0.7, random_state=200) # 70% is training

dm_data_test_0_3=dm_data.sample(frac=0.3, random_state=200)

dm_data_test=dm_data.drop(dm_data_train.index)

dm_data.columns

all_columns_model=smf.ols("AmountSpent~Gender+OwnHome+Married+Location+Salary+History+Catalogs+Age_new+Children_01+Children_23", data=dm_data).fit()
print(all_columns_model.summary())

all_columns_model_train=smf.ols("AmountSpent~Gender+OwnHome+Married+Location+Salary+History+Catalogs+Age_new+Children_01+Children_23", data=dm_data_train).fit()
print(all_columns_model_train.summary())
"""
We should consider the column in model calculation when P values is low.

                            OLS Regression Results                            
==============================================================================
Dep. Variable:            AmountSpent   R-squared:                       0.750
Model:                            OLS   Adj. R-squared:                  0.746
Method:                 Least Squares   F-statistic:                     187.8
Date:                Wed, 10 Jul 2019   Prob (F-statistic):          9.16e-199
Time:                        11:07:17   Log-Likelihood:                -5324.6
No. Observations:                 700   AIC:                         1.067e+04
Df Residuals:                     688   BIC:                         1.073e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
==========================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------
Intercept               -216.6788     88.203     -2.457      0.014    -389.858     -43.499
Gender[T.Male]           -40.2653     39.191     -1.027      0.305    -117.214      36.684
OwnHome[T.Rent]          -50.6079     44.038     -1.149      0.251    -137.074      35.858
Married[T.Single]         -3.0776     50.889     -0.060      0.952    -102.994      96.839
Location[T.Far]          435.2792     43.438     10.021      0.000     349.993     520.565
History[T.Low]          -464.4192     77.078     -6.025      0.000    -615.755    -313.083
History[T.Medium]       -487.2177     63.118     -7.719      0.000    -611.145    -363.290
History[T.NewCustomer]   -67.5273     61.232     -1.103      0.270    -187.751      52.696
Age_new[T.Young]          -5.4165     55.076     -0.098      0.922    -113.553     102.720
Salary                     0.0180      0.001     15.271      0.000       0.016       0.020
Catalogs                  41.3310      2.941     14.056      0.000      35.558      47.104
Children_01               82.4486     42.265      1.951      0.051      -0.535     165.432
Children_23             -299.1274     56.252     -5.318      0.000    -409.573    -188.681
==============================================================================
Omnibus:                      206.494   Durbin-Watson:                   2.100
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              909.738
Skew:                           1.286   Prob(JB):                    2.84e-198
Kurtosis:                       7.958   Cond. No.                     8.09e+19
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 4.38e-28. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
"""

all_columns_model_train=smf.ols("AmountSpent~Gender+OwnHome+Married+Location+Salary+History+Catalogs+Age_new+Children_01", data=dm_data_train).fit()
print(all_columns_model_train.summary())

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            AmountSpent   R-squared:                       0.750
Model:                            OLS   Adj. R-squared:                  0.746
Method:                 Least Squares   F-statistic:                     187.8
Date:                Wed, 10 Jul 2019   Prob (F-statistic):          9.16e-199
Time:                        11:14:55   Log-Likelihood:                -5324.6
No. Observations:                 700   AIC:                         1.067e+04
Df Residuals:                     688   BIC:                         1.073e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
==========================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------
Intercept               -515.8062    141.781     -3.638      0.000    -794.181    -237.431
Gender[T.Male]           -40.2653     39.191     -1.027      0.305    -117.214      36.684
OwnHome[T.Rent]          -50.6079     44.038     -1.149      0.251    -137.074      35.858
Married[T.Single]         -3.0776     50.889     -0.060      0.952    -102.994      96.839
Location[T.Far]          435.2792     43.438     10.021      0.000     349.993     520.565
History[T.Low]          -464.4192     77.078     -6.025      0.000    -615.755    -313.083
History[T.Medium]       -487.2177     63.118     -7.719      0.000    -611.145    -363.290
History[T.NewCustomer]   -67.5273     61.232     -1.103      0.270    -187.751      52.696
Age_new[T.Young]          -5.4165     55.076     -0.098      0.922    -113.553     102.720
Salary                     0.0180      0.001     15.271      0.000       0.016       0.020
Catalogs                  41.3310      2.941     14.056      0.000      35.558      47.104
Children_01              381.5761     46.059      8.285      0.000     291.144     472.008
==============================================================================
Omnibus:                      206.494   Durbin-Watson:                   2.100
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              909.738
Skew:                           1.286   Prob(JB):                    2.84e-198
Kurtosis:                       7.958   Cond. No.                     5.73e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.73e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

formula="AmountSpent~Location+Salary+Catalogs+History_low+History_medium"

dm_data_train['History_low']=dm_data_train['History'].map(lambda x:1 if x=='Low' else 0)
dm_data_train['History_medium']=dm_data_train['History'].map(lambda x:1 if x=='Medium' else 0)
dm_data_test['History_low']=dm_data_test['History'].map(lambda x:1 if x=='Low' else 0)
dm_data_test['History_medium']=dm_data_test['History'].map(lambda x:1 if x=='Medium' else 0)

all_columns_model_train=smf.ols(formula, data=dm_data_train).fit()
print(all_columns_model_train.summary())

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            AmountSpent   R-squared:                       0.720
Model:                            OLS   Adj. R-squared:                  0.718
Method:                 Least Squares   F-statistic:                     356.3
Date:                Wed, 10 Jul 2019   Prob (F-statistic):          6.77e-189
Time:                        11:22:03   Log-Likelihood:                -5364.9
No. Observations:                 700   AIC:                         1.074e+04
Df Residuals:                     694   BIC:                         1.077e+04
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
Intercept        -278.8466     72.263     -3.859      0.000    -420.727    -136.966
Location[T.Far]   412.2709     44.641      9.235      0.000     324.623     499.918
Salary              0.0175      0.001     23.610      0.000       0.016       0.019
Catalogs           43.5490      3.027     14.388      0.000      37.606      49.492
History_low      -569.9114     55.241    -10.317      0.000    -678.371    -461.452
History_medium   -485.5025     50.489     -9.616      0.000    -584.632    -386.373
==============================================================================
Omnibus:                      195.303   Durbin-Watson:                   2.101
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              854.215
Skew:                           1.212   Prob(JB):                    3.23e-186
Kurtosis:                       7.838   Cond. No.                     2.82e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.82e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
fitted=all_columns_model_train.fittedvalues
residual=all_columns_model_train.resid

plt.scatter(fitted, residual)

plt.plot(fitted, residual,"*")


formula_new="np.log(AmountSpent)~Location+Salary+Catalogs+History_low+History_medium"

all_columns_model_train_new=smf.ols(formula_new, data=dm_data_train).fit()
print(all_columns_model_train_new.summary())

fitted=all_columns_model_train_new.fittedvalues
residual=all_columns_model_train_new.resid

plt.scatter(fitted, residual)

plt.plot(fitted, residual,"*")

# VIF - Varience Inflation Factory
# to check if the variables are inter related - correlation coefficient

# if VIF > 5, variable is related to others and we should remove the varible

y,x=dmatrices(formula,dm_data_train,return_type="dataframe")

x.head(5)

# VIF for Location[T.Far]
variance_inflation_factor(x.values,1) # 1.049167042504158

# VIF for Salary
variance_inflation_factor(x.values,2) # 1.2938583635964613

# VIF for Catalogs
variance_inflation_factor(x.values,3) # 1.0529766496268975

# VIF for History_low
variance_inflation_factor(x.values,4) # 1.049167042504158

# VIF for History_medium
variance_inflation_factor(x.values,5) # 1.163433747567432

variance_inflation_factor(x.values,0) # 13.637016132070277

all_columns_model_train_new.predict(dm_data_test).head(5)

# since log is used in model, we use exponential to get actual values
np.exp(all_columns_model_train_new.predict(dm_data_test)).head(5)
"""
0      847.985101
2      274.439325
6      677.032738
7     1408.577979
10     781.578178
dtype: float64
"""


