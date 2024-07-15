#!/usr/bin/env python
# coding: utf-8

# # Predicting Mobile Prices

# Importing Libraries

# In[1]:


get_ipython().system('pip install dataprep')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import dataprep
from dataprep.eda import create_report


# # Data Exploration

# Extracting Data

# In[3]:


#importing dataset

mob = pd.read_excel("C:/Users/vipin/OneDrive/Documents/Desktop/Project4/Processed_Flipdata.xlsx")
mob


# In[4]:


#printing top 10 Entries

mob.head(10)


# In[5]:


#printing last 10 Entries

mob.tail(10)


# In[6]:


#counting rows and columns

mob.shape


# In[7]:


#Getting DataType Info

mob.info()


# In[8]:


#Generate descriptive statistics

mob.describe()


# # Data Cleaning and Preprocessing

# Missing Values

# In[9]:


# Handling Missing Values

mob.isnull().sum().head(12)


# Separating numerical and object columns

# In[30]:


# Identifying numerical columns
numerical_columns = mob.select_dtypes(include=['float','int']).columns
mob[numerical_columns]


# In[11]:


# Identifying categorical columns
categorical_columns = mob.select_dtypes(include=['object']).columns.tolist()
mob[categorical_columns]


# In[12]:


# Handle missing values using SimpleImputer

imputer = SimpleImputer(strategy='mean')
mob[numerical_columns] = imputer.fit_transform(mob[numerical_columns])


# In[13]:


# Drop Unnamed columns

columns_to_drop = ['Unnamed: 0']
mob= mob.drop(columns=columns_to_drop, axis=1)


# In[14]:


mob


# # Feature Engineering

# In[15]:


#perform one_hot_encoding

one_hot_encoded_data = pd.get_dummies(mob, columns=['Model', 'Colour', 'Rear Camera', 'Front Camera', 'Processor_'])


# In[16]:


pd.set_option('display.max_columns', None)

# Print the resulting
one_hot_encoded_data


# In[17]:


mob.head()


# In[ ]:





# # Outliers

# In[18]:


Q1 = mob['Prize'].quantile(0.25)
Q3 = mob['Prize'].quantile(0.75)
IQR = Q3-Q1
outliers = mob[(mob['Prize'] < (Q1 - 1.5*IQR)) | (mob['Prize'] > (Q3 + 1.5*IQR))]

outliers


# In[19]:


#Outlier Detection overall

plt.figure(figsize = (15,3), dpi = 400)
outliers.plot(kind = 'box', color = 'green')
plt.xticks(rotation = 90)
plt.title('Total Outliers', color = 'red')
plt.show()


# In[20]:


# Box Plot for Outlier Detection in 'Prize'
plt.figure(figsize=(10, 6))
sns.boxplot(x=mob['Prize'])
plt.title('Boxplot of Prize')
plt.show()


# In[21]:


# Log Transformation of Price
mob['Prize'] = np.log1p(mob['Prize'])

# Visualizing the transformed Prize
sns.histplot(mob['Prize'], kde=True)
plt.title('Log-Transformed Prize Distribution')
plt.show()


# # Removing Outliers

# In[22]:


#using Z score for outlier removal

z_scores = np.abs(stats.zscore(mob['Prize']))
mob = mob[(z_scores < 3)]


# In[23]:


mob


# # EXPLORATORY DATA ANALYSIS (EDA)

# In[24]:


plt.figure(figsize=(16, 14))
plotnumber = 1
mobile = pd.DataFrame(mob)
for column in mobile:
    if plotnumber <= 12:
        plt.subplot(4, 3, plotnumber)
        sns.histplot(mobile[column], kde=True)  # Use histplot
        plt.xlabel(column, fontsize=15)
        plt.ylabel('Values', fontsize=15)
    plotnumber += 1
plt.tight_layout()
plt.show()


# In[31]:


plt.figure(figsize=(16, 14))
plotnumber = 1

for column in numerical_columns:
    if column != 'Prize' and plotnumber <= 12:
        plt.subplot(4, 3, plotnumber)
        sns.scatterplot(x=mob[column], y=mob['Prize'])
        plt.xlabel(column, fontsize=15)
        plt.ylabel('Prize', fontsize=15)
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[26]:


plt.figure(figsize=(16, 14))
plotnumber = 1

for column in categorical_columns:
    if column != 'Prize' and plotnumber <= 12:
        plt.subplot(4, 3, plotnumber)
        sns.scatterplot(x=mob[column], y=mob['Prize'])
        plt.xlabel(column, fontsize=10)
        plt.ylabel('Prize', fontsize=10)
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[27]:


# Histogram of Price Distribution
plt.figure(figsize=(10, 8))
sns.histplot(mob['Prize'], bins=20, kde=True, color='green')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()


# In[28]:


# Compute the correlation matrix

corr_matrix = mob.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# #  Violin plots

# In[32]:


ax = sns.violinplot(x = 'Memory', y = 'Prize', data = mob)


# In[33]:


ax = sns.violinplot(x = 'RAM', y = 'Prize', data = mob)


# In[34]:


ax = sns.violinplot(x = 'AI Lens', y = 'Prize', data = mob)


# In[35]:


mob.var()


# # Important Features that influence Mobile Price

# In[36]:


mob.corr()['Prize'].sort_values(ascending= False)[:10]


# In[37]:


important_features = corr_matrix['Prize'][abs(corr_matrix['Prize']) > 0.3].index.tolist()


# # Model Training

# In[38]:


# Model Building
X = mob[important_features].drop('Prize', axis=1)
y = mob['Prize']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)



# In[39]:


X_test


# In[40]:


y_train


# In[41]:


y_test


# # Model Evaluation

# In[42]:


# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)


print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


print(f"Cross-Validation RMSE: {cv_rmse.mean()} ± {cv_rmse.std()}")


# # Feature Importance Analysis

# In[43]:


# Get feature importances from the trained model
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

feature_importances


# In[44]:


# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.xlabel('Feature Importance Score')
plt.show()


# In[ ]:





# In[45]:


# Assuming model.predict(X_test) gives predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, model.predict(X_test), alpha=0.5)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line y=x
plt.show()


# In[46]:


create_report(mob)


# In[ ]:





# In[47]:


Conclusion = """
\033[1m Data Quality:\033[0m  The dataset was initially explored to understand its structure and content. 
This included checking for missing values, handling them using SimpleImputer, and dropping unnecessary columns (Unnamed: 0).
Categorical variables were transformed using one-hot encoding to prepare them for modeling.

\033[1m Outlier Handling and Transformation:\033[0m Outliers in the Prize (price) column were detected using statistical methods (IQR and Z-score). 
To mitigate their impact, a logarithmic transformation (np.log1p()) was applied, which improved the distribution of Prize for modeling purposes.

\033[1m Exploratory Data Analysis (EDA): \033[0m  EDA involved visualizing distributions, relationships, and correlations between features and the target variable (Prize). 
Histograms, scatter plots, and violin plots provided insights into how different features influence mobile prices.
Key features such as RAM, Memory, and specific camera attributes (Rear and Front Camera quality) emerged as significant factors affecting price variability.

\033[1m Model Training and Evaluation: \033[0m A Random Forest Regressor model was chosen for its ability to handle complex relationships and provide feature importance rankings.
The model was trained on a subset of important features identified through correlation analysis and evaluated using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).
Cross-validation ensured the model's reliability and generalization ability, yielding consistent performance metrics across different folds.

\033[1m Feature Importance Analysis: \033[0m Feature importance analysis highlighted that RAM, Memory, and specific camera features were among the most influential in determining mobile prices. 
This aligns with consumer expectations, where these attributes often drive purchasing decisions."""


# In[48]:


print(Conclusion)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




