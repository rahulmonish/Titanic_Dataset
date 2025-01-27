#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:06:59 2020

@author: rahul
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from numpy import set_printoptions
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import OneHotEncoder 
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import seaborn as sns
import xgboost as xgb







from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def rename_columns(df):
    list1= df.columns
    for i in list1:
        df.rename(columns={i: str(i)})
    
    return df

def check_column(df, df2):
    list1= df.columns
    list2= df2.columns
    
    for i in list2:
        if i not in list1:
            df[i]=0
    for i in list1:
        if i not in list2:
            df2[i]=0
    
    return df,df2

def change_age(df):
    for i in range(0,len(df)):
        df['Age'].at[i]= int(df['Age'].at[i]/5)
    
    return df

def convert_Dataframe(df):
    df= df.drop(['PassengerId', 'Fare', 'Name','Ticket', 'Cabin'], axis=1)
    
    df['Sex']= label_encoder.fit_transform(df['Sex'])
    
    Embarked= pd.get_dummies(df['Embarked'], drop_first= True)
    Age= pd.get_dummies(df['Age'], drop_first= True)
    Pclass= pd.get_dummies(df['Pclass'], drop_first= True)
    
    df= pd.concat([df, Age], axis=1)
    df= pd.concat([df, Pclass], axis=1)
    df= pd.concat([df, Embarked], axis=1)
    
    df= df.drop(['Embarked','Age','Pclass'], axis=1)
    return df



def preprocess(df):
    rich_mean= np.mean(df[df['Pclass']==1]['Age'])
    avg_mean= np.mean(df[df['Pclass']==2]['Age'])
    poor_mean= np.mean(df[df['Pclass']==3]['Age'])
    df['Age']= df['Age'].fillna(0)
    
    for i in range(0,len(df)):
        
        if df['Pclass'].at[i]==1 and df['Age'].at[i]==0:
            df['Age'].at[i]= int(rich_mean)
        if df['Pclass'].at[i]==2 and df['Age'].at[i]==0:
            df['Age'].at[i]= int(avg_mean)
        if df['Pclass'].at[i]==3 and df['Age'].at[i]==0:
            df['Age'].at[i]= int(poor_mean)
    
    return df


df= pd.read_csv('train.csv')
df.isnull().sum()
correlation_plot= df.corr(method= 'pearson')
print("before correlation: " ,df['Age'].corr(df['Survived']))
sns.distplot(df['Age'].dropna())
sns.countplot(x='Survived', hue= 'Sex', data= df)


df= preprocess(df)
df= change_age(df)
df= convert_Dataframe(df)


#print("after correlation: " ,df['Fare'].corr(df['Pclass']))


label_encoder = preprocessing.LabelEncoder() 


y= df['Survived']
df= df.drop('Survived', axis=1)

#x= df

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

count=1
# =============================================================================
# for g in columns:
#     #g= 'PassengerId'
#     mean= np.mean(df[g])
#     sd= np.std(df[g])
#     x_axis= df[g]
#     plt.subplot(3,3,count)
#     count+=1
#     plt.plot(x_axis, norm.pdf(x_axis,mean,sd))
#     plt.show()
# =============================================================================

model= xg_reg = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1, alpha = 10, n_estimators = 10)
#model= LogisticRegression()
#model = svm.SVC()
#model= DecisionTreeClassifier()
#model= RandomForestClassifier()
# =============================================================================
# model = Sequential()
# model.add(Dense(100, input_dim=8, activation='relu'))
# model.add(Dense(10,  activation='relu'))
# model.add(Dense(1,  activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x,y, epochs=100)
# =============================================================================
#model.fit(x,y)
# =============================================================================
# test = SelectKBest(score_func=f_classif, k=3)
# fit = test.fit(x, y)
# # summarize scores
# set_printoptions(precision=3)
# print(fit.scores_)    
# features = fit.transform(x)
# # summarize selected features
# print(features[0:5,:])
# =============================================================================

test_set= pd.read_csv('test.csv')
passenger= test_set['PassengerId']

test_set= preprocess(test_set)
test_set= change_age(test_set)
test_set= convert_Dataframe(test_set)

df, test_set= check_column(df, test_set)
df = df.reindex(columns=list(test_set.columns))
# =============================================================================
# df = rename_columns(df)
# test_set= rename_columns(test_set)
# df = df.reindex(sorted(df.columns), axis=1)
# test_set = test_set.reindex(sorted(test_set.columns), axis=1)
# columns= df.columns
# columns
# =============================================================================
x=df
model.fit(x,y)


test_set= test_set.fillna(0)

x_predict= test_set

final= model.predict(x_predict)

#accuracy = log_loss(y_test, final)
test_set['Survived']= final
test_set['PassengerId']= passenger
test_set= test_set[['PassengerId', 'Survived']]
convert_dict = {'Survived': int} 


for i in range(0,len(test_set)):
    if test_set['Survived'].at[i]< .5:
       test_set['Survived'].at[i]= int(0)
       print(type(test_set['Survived'].at[i]))
    else:
        test_set['Survived'].at[i]=int(1)

test_set = test_set.astype(convert_dict) 
test_set.to_csv('attempt1.csv',  index=False)
sns.countplot(test_set['Survived'])
test_set.isnull().sum()