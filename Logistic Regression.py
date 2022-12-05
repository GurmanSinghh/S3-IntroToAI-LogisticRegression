#!/usr/bin/env python
# coding: utf-8

# In[10]:



#Import the necessary libraries and methods
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve


#Load dataset
filename = 'titanic.csv'
path = '/Users/gurman/Desktop'
fullpath = os.path.join(path,filename)

titanic_gurman = pd.read_csv(fullpath)


#Data exploration
titanic_gurman.head(3)
titanic_gurman.shape
titanic_gurman.dtypes
titanic_gurman.info()

#Display Unique values
titanic_gurman['Sex'].value_counts()

titanic_gurman['Pclass'].value_counts()

#Plot bar chart of Survived vs Passenger Class
pclass_survived = pd.crosstab(titanic_gurman['Survived'],titanic_gurman['Pclass'])
pclass_survived.plot(kind='bar')
plt.title('Class_Survivors_Gurman')
plt.ylabel('Number of deceased & survivors per sex')

#Plot bar chart of Survived vs Sex
sex_survived = pd.crosstab(titanic_gurman['Survived'], titanic_gurman['Sex'])
sex_survived.plot(kind='bar')
plt.title('Gender_Survivors_Gurman')
plt.ylabel('Number of deceased and survivors per gender')

#Investigate the number of unique variables in 'Survived'
titanic_gurman.Survived.value_counts()

#Using a scatter matrix to analyze the relationships between variables
pd.plotting.scatter_matrix(titanic_gurman[['Survived', 'Sex', 'Fare', 'Pclass', 'SibSp', 'Parch']], figsize=(17,15), hist_kwds={'bins':10}, alpha=0.1)

#Using Seaborn Pairplot in an attempt to have a better view of the relationship between variables
sns.pairplot(titanic_gurman[['Survived', 'Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']], hue='Survived',height=3, aspect=0.75,diag_kind="hist", dropna=True, plot_kws={'alpha': 0.2})

#Drop features that will not have statistical value for our model
titanic_gurman.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace = True)

#Fill na values in the 'Embarked' feature with the forward value
titanic_gurman['Embarked'].fillna(method='ffill', inplace=True)

titanic_gurman.Embarked.isna().sum()

#Convert 'Sex' and 'Embarked' to Numerical
dummies = pd.get_dummies(titanic_gurman[['Sex', 'Embarked']], drop_first=True)

dummies.head()

#Concatenate the dummies variables to the titanic data frame
titanic_b = pd.concat([titanic_gurman, dummies], axis = 1)

#Drop redundant features
titanic_b.drop(['Sex', 'Embarked'], axis = 1, inplace = True)

titanic_b.head()

#Inspect 'Age' for NAN
titanic_b['Age'].isna().sum()

titanic_b['Age'].fillna(titanic_b['Age'].mean(), inplace = True)

titanic_b.describe()
titanic_b.info()

#Convert data types to float
titanic_b = titanic_b.astype(float)

titanic_b.info()


#Function to normalize data
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return df_norm

#Applying normalization on the dataset
titanic_gurman_norm = min_max_scaling(titanic_b)

#Displaying the first two records
titanic_gurman_norm.head(2)

#Plot a histogram of the variables
titanic_gurman_norm.hist(figsize=(9,10), bins=10)

#Split the data set into independent and dependet variables
X_gurman = titanic_gurman_norm.drop('Survived', axis = 1)
y_gurman = titanic_gurman_norm['Survived']


#Use Train_Test_Split to randomly split the dataset into train and test datasets
X_train_gurman, X_test_gurman, y_train_gurman, y_test_gurman = train_test_split(
    X_gurman, y_gurman, test_size=0.3, random_state=86)

#Instantiate and train a Logistic model
gurman_model = LogisticRegression()
gurman_model.fit(X_train_gurman, y_train_gurman)

#Print the coefficients of all predictors
print(pd.DataFrame(zip(X_train_gurman.columns, np.transpose(gurman_model.coef_)), columns=['Predictor', 'Coefficients']))

#Use cross validation
scores = cross_val_score(gurman_model, X_train_gurman, y_train_gurman, cv=10)
print(scores)


#Running cross-validation with different splits
mean_score = []

for i in np.arange(0.10, 0.55, 0.05):
    X_train, X_test, y_train, y_test = train_test_split(
    X_gurman, y_gurman, test_size=i, random_state=86)
    
    gurman_model.fit(X_train, y_train)
    scores = cross_val_score(gurman_model, X_train, y_train, cv=10)
    mean_score.append(scores.mean())
    print(f"Split {i}\nMinimum Accuracy: {scores.min()}\nMean Accuracy: {scores.mean()}\nMaximum Accuracy: {scores.max()}\n ")
    

#Split the data set into 70% train and 30% test
X_train_gurman, X_test_gurman, y_train_gurman, y_test_gurman = train_test_split(
    X_gurman, y_gurman, test_size=0.3, random_state=86)

#Retrain the model
gurman_model.fit(X_train_gurman, y_train_gurman)

#Calculating the probability of each prediction
y_pred_gurman = gurman_model.predict_proba(X_test_gurman)

#Print probabilities
print(y_pred_gurman[:,1])

#Set the threshold to 50% and determine the outputs for the labels
y_pred_gurman_flag = np.where(y_pred_gurman[:,1]>0.5,1,0)
print(y_pred_gurman_flag)

#Calculate the accuracy score
accuracy_score(y_test_gurman, y_pred_gurman_flag)

# Null accuracy: accuracy that could be achieved by always predicting the most frequent class
y_test_gurman.value_counts()
print(f'The accuracy that should be achieved by always predicting the most frequent class is: \n{max(y_test_gurman.mean(), 1-y_test_gurman.mean())}')

#Confusion Matrix
cf_matrix_log_gurman = confusion_matrix(y_test_gurman, y_pred_gurman_flag)
print(cf_matrix_log_gurman)



#Print classification report
print(classification_report(y_test_gurman, y_pred_gurman_flag))
type(y_test_gurman)

#Making predictions with the threshold set to 5%
y_pred_gurman_flag2 = np.where(y_pred_gurman[:,1]>0.75,1,0)
print(y_pred_gurman_flag2)

#Calculate the accuracy score with the threshold set to 75%
accuracy_score(y_test_gurman, y_pred_gurman_flag2)

#Confusion Matrix
cf_matrix = confusion_matrix(y_test_gurman, y_pred_gurman_flag2)
print(cf_matrix)

#Classification report
print(classification_report(y_test_gurman, y_pred_gurman_flag2))


X_train_gurman.info()

#Making predictions with the training dataset
y_pred_gurman_train = gurman_model.predict_proba(X_train_gurman)
y_pred_gurman_flag_train = np.where(y_pred_gurman_train[:,1]>0.75,1,0)
print(y_pred_gurman_flag_train)

#Calculating the accuracy score
accuracy_score(y_train_gurman, y_pred_gurman_flag_train)

#Confusion Matrix
cf = confusion_matrix(y_train_gurman, y_pred_gurman_flag_train)
print(cf)

#Print the classification report
print(classification_report(y_train_gurman, y_pred_gurman_flag_train))


# In[ ]:




