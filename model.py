# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  
dataset = pd.read_csv('ML Case Study - Data.csv')
dataset.drop(['ID','ZIP Code'], axis=1, inplace=True)
df_dummies0 = pd.get_dummies(dataset, columns=['Education'])
x = df_dummies0.drop('Personal Loan', axis = 1)
y = df_dummies0['Personal Loan']
XT, Xt, YT, Yt = train_test_split(x, y, test_size=0.30)
DT = DecisionTreeClassifier(criterion="entropy",)
DT.fit(XT,YT)
print(DT.score(XT,YT),DT.score(Xt,Yt))

# Saving model to disk
pickle.dump(DT, open('modelDT.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('modelDT.pkl','rb'))
print(model.predict(pd.DataFrame([[29,3,40,1,1.90,0,0,0,1,0,0,0,1]])))
