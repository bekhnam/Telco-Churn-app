import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df_churn = pd.read_csv('dataset/Telco-Customer-Churn.csv')
df_churn = df_churn[['gender', 'PaymentMethod', 'MonthlyCharges', 'tenure', 'Churn']]
df = df_churn.copy()
df.fillna(0, inplace=True)

# create dummy variables
encode = ['gender', 'PaymentMethod']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df['Churn'] = np.where(df['Churn']=='Yes', 1, 0)

# define dataset
X = df.drop('Churn', axis=1)
y = df['Churn']

# define RandomForestClassifier model
clf = RandomForestClassifier().fit(X, y)

# save model
pickle.dump(clf, open('churn_clf.pkl', 'wb'))