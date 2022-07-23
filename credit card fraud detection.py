#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the libraries
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score 
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import average_precision_score, precision_recall_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import AdaBoostClassifier

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Data description and wrangling
# 

# In[ ]:


file = '/Users/olayinkafaniran/Documents/AIDA/COURSE WORK/DISSERTATION/DISSERTAION DATA/DATAOPTION1.csv'
df = pd.read_csv(file)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.dtypes


# In[ ]:


time_conv = pd.to_timedelta(df['unix_time'], unit = 's')
df['Day_Time'] = (time_conv.dt.components.days).astype(int)
df['Hour_Time'] = (time_conv.dt.components.hours).astype(int)
df['Min_Time'] = (time_conv.dt.components.minutes).astype(int)


# In[ ]:


select = ['credit_card_number','amount','zip_code','latitude','longitude',
          'city_population','unix_time','merchant_lat','Day_Time','Hour_Time','Min_Time',
         'merchant_long','fraud']
df = df.loc[:,select]
df.head()


# In[ ]:


df.drop(['Day_Time','Min_Time', 'unix_time'], axis = 1, inplace= True)


# In[ ]:


df.head()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


valid = df[df['fraud']==0]
fraud = df[df['fraud']==1]


# In[ ]:


valid['amount'].describe()


# In[ ]:


fraud['amount'].describe()


# In[ ]:


df.groupby('fraud').mean()


# In[ ]:


valid.shape


# In[ ]:


fraud.shape


# In[ ]:


valid.isnull().value_counts().sum()


# In[ ]:


fraud.isnull().value_counts().sum()


# In[ ]:


valid.describe()


# In[ ]:


fraud.describe()


# In[ ]:


# valid_sample = valid.sample(n=5412)
# valid_sample.head()
# df_new = pd.concat([valid_sample, fraud], axis = 0)
# df_new.head()


# In[ ]:


df.shape


# In[ ]:


df['fraud'].value_counts()


# In[ ]:


df.groupby('fraud').mean()


# In[ ]:


correlation = df.corr()
correlation


# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(correlation, linecolor= 'red', cmap='coolwarm')
plt.show()


# Feature selection

# In[ ]:


#drop columns that have multicolinearity features
df = df.drop(['zip_code','merchant_lat'], axis = 1)


# In[ ]:


X = df.drop('fraud', axis = 1)
Y = df['fraud']


# In[ ]:


cols = list(X.columns.values)

valid_trx = df.fraud == 0
fraud_trx = df.fraud == 1

plt.figure(figsize=(20, 70))
for n, col in enumerate(cols):
  plt.subplot(10,3,n+1)
  sns.distplot(X[col][valid_trx], color='blue')
  sns.distplot(X[col][fraud_trx], color='red')
  plt.title(col, fontsize=17)
plt.show()


# # MACHINE LEARNING MODELS

# In[ ]:


X.head()


# Logistic Regression

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 2)


# In[ ]:


#accuracy
#auc_roc
#threshold
#use GridsearchCV
#Kfold cross val
#Store results in a Data frame 

#NORMAL MODEL
#create a dataframe to store results
df_model_results = pd.DataFrame(columns = ['Methodology','Model','Train Accuracy', 'Test Accuracy','Roc_value','Threshold'])

#train the model
model = LogisticRegression()
model.fit(x_train, y_train)

#obtain the model score for training data
y_predict = model.predict(x_train)
model_train_accuracy_score = accuracy_score(y_predict, y_train)

#obtain accuracy score for test data
y_predict_test = model.predict(x_test)
model_accuracy_score = accuracy_score(y_predict_test, y_test)

#confusion matrix
c_matrix = confusion_matrix(y_test, y_predict_test)

#Classification report for model
report = classification_report(y_predict_test, y_test)
print('Classification report for Logistic regression model is: \n',report)
  
#Find predicted probabilities
y_pred_prob = model.predict_proba(x_test)[:,1] 

#ROC and AUC and threshold determination
roc_value = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
threshold = thresholds[np.argmax(tpr-fpr)]

#ROC-AUC curve plot
roc_auc = metrics.auc(fpr, tpr)
print("ROC for the test dataset",roc_auc)
plt.plot(fpr,tpr,label="Test, auc="+str(roc_auc))
plt.legend(loc=4)
plt.show()

print('Result table for the Model is as follows: ')

df_model_results = df_model_results.append(pd.DataFrame({'Methodology': 'Basic traing and testing',
                                                         'Train Accuracy': model_train_accuracy_score,
                                                         'Test Accuracy': model_accuracy_score,
                                                         'Roc_value': roc_value,
                                                         'Threshold': threshold,
                                                         'Accuracy' : model_accuracy_score,
                                                         'Model':'Logistic Regression'},
                                                        index = [0]),ignore_index= True)
df_model_results


# # LOGISTIC REGRESSION WITH CROSS VALIDATION USING KFOLD AND HYPER PARAMETER TUNING

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import KFold

