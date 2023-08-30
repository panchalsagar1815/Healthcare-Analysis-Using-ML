import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

raw_data1=pd.read_csv(r'../input/healthcare-analytics/Train/First_Health_Camp_Attended.csv')
raw_data2=pd.read_csv(r'../input/healthcare-analytics/Train/Second_Health_Camp_Attended.csv')
raw_data3=pd.read_csv(r'../input/healthcare-analytics/Train/Third_Health_Camp_Attended.csv')

raw_data1 = raw_data1.drop(['Donation', 'Unnamed: 4'], axis=1)
raw_data3 = raw_data3.drop('Last_Stall_Visited_Number', axis=1)

hsbinary1=[]
i=0
for i in range(0, raw_data1.shape[0]):
    if raw_data1['Health_Score'][i] > raw_data1['Health_Score'].mean():
        hsbinary1.append(1)
    else:
        hsbinary1.append(0)

hsbinary2=[]
j=0
for j in range(0, raw_data2.shape[0]):
    if raw_data2['Health Score'][j] > raw_data2['Health Score'].mean():
        hsbinary2.append(1)
    else:
        hsbinary2.append(0)

hsbinary3=[]
k=0
for k in range(0, raw_data3.shape[0]):
    if raw_data3['Number_of_stall_visited'][k] == 0:
        hsbinary3.append(0)
    else:
        hsbinary3.append(1)

raw_data1['HS Binary']=hsbinary1
raw_data2['HS Binary']=hsbinary2
raw_data3['HS Binary']=hsbinary3

raw_data1=raw_data1.drop('Health_Score', axis=1)
new_data1=raw_data1.copy()
raw_data2=raw_data2.drop('Health Score', axis=1)
new_data2=raw_data2.copy()
raw_data3=raw_data3.drop('Number_of_stall_visited', axis=1)
new_data3=raw_data3.copy()

all_atd_data=pd.concat([new_data1, new_data2, new_data3], axis=0).reset_index().drop('index', axis=1)
train_data=pd.read_csv(r'../input/healthcare-analytics/Train/Train.csv')
collected_data = pd.merge(all_atd_data, train_data, on=['Patient_ID', 'Health_Camp_ID'], how='inner')
data_no_mv = collected_data.dropna(axis=0).reset_index().drop('index', axis=1)
 
datee=pd.to_datetime(data_no_mv['Registration_Date'], format='%d-%b-%y')

months=[]
m=0
for m in range(0, data_no_mv.shape[0]):
    months.append(datee[m].month)

data_no_mv['Months']=months
data_no_mv = data_no_mv.drop('Registration_Date', axis=1)
unscaled_data=data_no_mv[['Patient_ID', 'Health_Camp_ID', 'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Months', 'HS Binary']]

unscaled_input=unscaled_data.iloc[:, 2:-1]
target=unscaled_data.iloc[:, -1:]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator,TransformerMixin): 
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

columns_to_omit = ['Months']
columns_to_scale = [x for x in unscaled_input.columns.values if x not in columns_to_omit]

healthcare_scaler=CustomScaler(columns_to_scale)
healthcare_scaler.fit(unscaled_input)
scaled_input=healthcare_scaler.transform(unscaled_input)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_input, target, train_size=0.80, random_state=20)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg=LogisticRegression()
logreg.fit(x_train, y_train)
print(logreg.score(x_train, y_train))

predict_data=pd.read_csv(r'../input/healthcare-analytics/Train/test.csv')
datee2=pd.to_datetime(predict_data['Registration_Date'], format='%d-%b-%y')

months2=[]
m2=0
for m2 in range(0, predict_data.shape[0]):
    months2.append(datee2[m2].month)

predict_data['Months']=months2
clean_predict_data=predict_data[['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Months']]

predicted_proba = logreg.predict_proba(clean_predict_data)

submission = pd.read_csv(r'../input/healthcare-analytics/sample_submmission.csv')
submission['Outcome']=predicted_proba[:, 1:]
submission.to_csv(r'./submission.csv', index = False)