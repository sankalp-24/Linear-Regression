#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[15]:


matplotlib.rcParams['figure.figsize'] = (10, 6)


# In[17]:


import os

os.chdir(r'/Users/sankalpsingh/Downloads/PGA Imarticus/ML/Linear Regression & OLS')


# In[19]:


health_df = pd.read_csv("Insurance Charges Prediction.csv")
health_df.head(30)


# In[21]:


health_df.shape


# In[23]:


health_df.info()


# In[25]:


health_df.describe()


# In[27]:


health_df.describe(include='object')


# In[29]:


# EDA from this step
# Univariate Analysis
health_df.hist()
plt.show()


# In[31]:


# Bivariate Analysis
# Visualization of the distribution of medical charges in connection with other factors like "sex" and "region"

sns.boxplot(x = 'sex', y = 'charges', data = health_df)
plt.show()


# In[33]:


sns.boxplot(x = 'region', y = 'charges', data = health_df)
plt.show()


# In[35]:


health_df.smoker.value_counts()


# In[37]:


sns.countplot(x = 'smoker', hue = 'sex', data = health_df)
plt.show()


# In[39]:


plt.scatter(x="age", y="charges", data=health_df)
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()


# In[41]:


plt.scatter(x="bmi", y="charges", data=health_df)
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()


# In[43]:


sns.violinplot(x="children", y="charges", data=health_df)
plt.show()


# In[45]:


sns.barplot(x = 'sex',y = 'charges',hue = "smoker",data = health_df)
plt.show()


# In[47]:


#import warnings
#warnings.filterwarnings('ignore')

health_df.corr(numeric_only=True)


# In[49]:


sns.heatmap(health_df.corr(numeric_only=True), cmap='Blues', annot=True)
plt.show()


# In[51]:


# check the distribution of target variable
health_df.charges.hist(color = 'maroon')

# add plot and axes labels
# set text size using 'fontsize'
plt.title('Distribution of Target Variable (Charges)', fontsize = 15)
plt.xlabel('Charges', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

# display the plot
plt.show()


# In[53]:


health_df.charges.skew()


# In[55]:


#If the data is not normally distributed, use log transformation to reduce the skewness and get a near normally distributed data
health_df['log_charges'] = np.log(health_df["charges"])

# display the top 5 rows of the data
health_df.head()


# In[57]:


# check the distribution of target variable
health_df.log_charges.hist(color = 'maroon')

# add plot and axes labels
# set text size using 'fontsize'
plt.title('Distribution of Target Variable (Charges)', fontsize = 15)
plt.xlabel('Charges', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

# display the plot
plt.show()


# In[59]:


health_df.log_charges.skew()


# In[61]:


#Dummy Encode the Categorical Variables

# filter out the categorical variables and consider only the numeric variables using (include=np.number)
df_numeric_features = health_df.select_dtypes(include=np.number)

# display the numeric features
df_numeric_features.columns


# In[63]:


# filter out the numerical variables and consider only the categorical variables using (include=object)
df_categoric_features = health_df.select_dtypes(include = object)

# display categorical features
df_categoric_features.columns


# In[65]:


#Dummy encode the catergorical variables
# to create the dummy variables  we use 'get_dummies()' from pandas 
# to create (n-1) dummy variables we use 'drop_first = True' 
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# In[67]:


# concatenate the numerical and dummy encoded categorical variables column-wise
df_health_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_health_dummy.head()


# In[69]:


df_health_dummy.shape


# In[71]:


import statsmodels
import statsmodels.api as sm
#from statsmodels.compat import lzip
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.tools.eval_measures import rmse
#from statsmodels.stats.outliers_influence import variance_inflation_factor

# 'Scikit-learn' (sklearn) emphasizes various regression, classification and clustering algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing


# In[72]:


#Linear Regression (OLS)
#Split the data into training and test sets
# add the intercept column using 'add_constant()'

df_health_dummy = sm.add_constant(df_health_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
X=df_health_dummy.drop(['charges','log_charges'],axis=1)
y=df_health_dummy[['charges','log_charges']]


# In[73]:


X.columns


# In[77]:


# split data into train data and test data 
# what proportion of data should be included in test data is passed using 'test_size'
# set 'random_state' to get the same data each time the code is executed 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# In[79]:


#Build model using sm.OLS().fit()
linreg_logmodel_full = sm.OLS(y_train['log_charges'], X_train.astype(int)).fit()

# print the summary output
print(linreg_logmodel_full.summary())

#Charges=7.1+0.03*age+0.01*bmi+0.09*children


# In[81]:


#Predict the values using test set
# predict the 'log_charges' using predict()
linreg_logmodel_full_predictions = linreg_logmodel_full.predict(X_test)


# In[83]:


# take the exponential of predictions using np.exp()
predicted_charges = np.exp(linreg_logmodel_full_predictions.astype(int))

# extract the 'Property_Sale_Price' values from the test data
actual_charges = y_test['charges']


# In[85]:


#Compute accuracy measures
# calculate rmse using rmse()
linreg_logmodel_full_rmse = rmse(actual_charges, predicted_charges)

# calculate R-squared using rsquared
linreg_logmodel_full_rsquared = linreg_logmodel_full.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_logmodel_full_rsquared_adj = linreg_logmodel_full.rsquared_adj 


# In[87]:


#Tabulate the results
# create the result table for all accuracy scores
# accuracy measures considered for model comparision are RMSE, R-squared value and Adjusted R-squared value
# create a list of column names
cols = ['Model', 'RMSE', 'R-Squared', 'Adj. R-Squared']

# create a empty dataframe of the columns
# columns: specifies the columns to be selected
result_tabulation = pd.DataFrame(columns = cols)

# compile the required information
linreg_logmodel_full_metrics = pd.Series({'Model': "Linreg full model with log of target variable ",
                     'RMSE':linreg_logmodel_full_rmse,
                     'R-Squared': linreg_logmodel_full_rsquared,
                     'Adj. R-Squared': linreg_logmodel_full_rsquared_adj     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels. If you want the concatenation to ignore existing
# indices, you can set the argument ignore_index=True. Then the resulting DataFrame
# index will be labeled with 0,1,2,....n-1.     
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation._append(linreg_logmodel_full_metrics, ignore_index = True)

# print the result table
result_tabulation


# In[ ]:




