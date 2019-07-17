#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Logistic regression is used in predicting the categorical varaiable
import sklearn.metrics as mt


# In[4]:


path=r'D:\Data_science_course\17072019_morning\dm.csv'


# In[5]:


dm_data=pd.read_csv(path)


# In[6]:


dm_data.columns


# In[7]:


dm_data.sample(5)


# In[8]:


dm_data.head(5)


# In[9]:


## instead of predicting amount spent column, now predict who will be spending more vs spending less
# define average amount spent
avg_amount_spent=dm_data.agg({'AmountSpent':'mean'})


# In[10]:


avg_amount_spent


# In[12]:


dm_data['Target']=dm_data['AmountSpent'].map(lambda x: 1 if x>1216.77 else 0 )


# In[13]:


dm_data.sample(5)


# In[15]:


dm_data_new=dm_data.drop(['AmountSpent','Cust_Id'], axis=1)


# In[16]:


dm_data_new.sample(5)


# In[17]:


dm_data_new['History']=dm_data_new['History'].fillna('NewCustomer')


# In[18]:


dm_data_new['AgeNew']=dm_data_new['Age'].map(lambda x: 1 if x!='Young' else 0)


# In[20]:


dm_data_new['ChildrenNew']=dm_data_new['Children'].map(lambda x: 1 if x<=1 else 0)


# In[21]:


dm_data_new.sample(5)


# In[23]:


dm_data_new_train=dm_data_new.sample(frac=0.7, random_state=200)


# In[24]:


dm_data_new_test=dm_data_new.drop(dm_data_new_train.index)


# In[26]:


import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[28]:


formula='Target~Gender+OwnHome+Married+Location+Salary+AgeNew+ChildrenNew+History+Catalogs'
# glm = to build all inds of models
train_model=smf.glm(formula,data=dm_data_new_train,family=sm.families.Binomial()).fit() # Binomial is for logistic regression


# In[29]:


print(train_model.summary())


# In[30]:


# high P-value, drop the column from modelling
dm_data_new_train['History_dummy']=dm_data_new_train['History'].map(lambda x:1 if x=='Medium' else 0)
dm_data_new_test['History_dummy']=dm_data_new_test['History'].map(lambda x:1 if x=='Medium' else 0)


# In[31]:


dm_data_new_train.sample(4)


# In[32]:


formula_new='Target~Location+History_dummy+Salary+AgeNew+ChildrenNew+Catalogs'
train_model_2=smf.glm(formula_new,data=dm_data_new_train,family=sm.families.Binomial()).fit() 
print(train_model_2.summary())


# In[33]:


## multicollinaerity --> compute VIF (Varience)
# ROC and AUC
Probabily_of_high_spending=train_model_2.predict(dm_data_new_train)


# In[34]:


mt.roc_auc_score(dm_data_new_train['Target'],Probabily_of_high_spending)


# In[35]:


fpr, tpr, thresholds=mt.roc_curve(dm_data_new_train['Target'],Probabily_of_high_spending) # fpr - false possitve rate, tpr - true positive rate


# In[36]:


thresholds


# In[37]:


plt.plot(fpr, tpr,'-');


# In[ ]:




