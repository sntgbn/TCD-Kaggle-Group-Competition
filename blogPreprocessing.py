import pandas as pd 
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.target_encoder import TargetEncoder
from datetime import datetime

def replace_nan_with_unknown(X, X_test,cols):
    for i in cols:
        X[i] = X[i].replace('\\N', )
        X[i] = X[i].astype(str)
        X_test[i] = X_test[i].replace('\\N')
        X_test[i] = X_test[i].astype(str)
    return (X,X_test)

def replace_nan_with_mean(X, X_test,cols):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    for i in cols:
        X[i] = X[i].replace("\\N", np.nan)
        X[i] = X[i].replace("nA", np.nan)
        X[i] = imputer.fit_transform(X[i].values.reshape(-1,1))
        X_test[i] = X_test[i].replace("\\N", np.nan)
        X_test[i] = X_test[i].replace("nA", np.nan)
        X_test[i] = imputer.transform(X_test[i].values.reshape(-1,1))
    return (X,X_test)


#Using StandardScaler
def scale(X, X_test, cols):
    ss = StandardScaler()
    for i in cols:
        X[i] = ss.fit_transform(X[i].values.reshape(-1,1))
        X_test[i] = ss.transform(X_test[i].values.reshape(-1,1))
    return (X, X_test)
        


def cat_encode(X, X_test,cols,y):
    ce = CatBoostEncoder(cols=cols)
    X = ce.fit_transform(X,y)
    X_test = ce.transform(X_test)
    return (X,X_test)
        

def label_encode(X, X_test,cols):
    le = LabelEncoder()
    for i in cols:
        print(i)
        X[i] = le.fit_transform(X[i].values.reshape(-1,1))
        X_test[i] = le.transform(X_test[i].values.reshape(-1,1))
    return (X,X_test)

def one_hot_encode(X, X_test, cols):
    for c in cols:
        cats = X[c].unique()
        print("Dummies: ", pd.get_dummies(X[c]).columns)
        print("Cats: " , cats)
        X = pd.concat([X,pd.get_dummies(X[c]).T.reindex(cats).T.fillna(0)],axis=1)
        X.drop([c],axis=1, inplace=True)
        X_test = pd.concat([X_test,pd.get_dummies(X_test[c]).T.reindex(cats).T.fillna(0)],axis=1)
        X_test.drop([c],axis=1, inplace=True)
    return (X,X_test)

def target_encode(X, X_test, cols, y):
    te = TargetEncoder(cols=cols, return_df=True)
    X = te.fit_transform(X,y)
    X_test = te.transform(X_test)
    return (X,X_test)


def date_to_int(X, X_test, cols, dayZero):
    for c in cols:
        X[c] = pd.to_datetime(X[c], format="%d/%m/%Y")
        X[c] = X[c] - dayZero
        X_test[c] = pd.to_datetime(X_test[c], format="%d/%m/%Y")
        X_test[c] = X_test[c] - dayZero
    return (X,X_test)

def set_non_zero_to_one(X, X_test, cols):
    for c in cols:
        X[c].loc[X[c] != 0] = 1
        X_test[c].loc[X_test[c] != 0] = 1
    return (X,X_test)

def set_binary(X, X_test,cols, val):
    for c in cols:
        X[c].loc[X[c] != val] = 0
        X_test[c].loc[X_test[c] != val] = 0
    return (X,X_test)

def remove_time_from_date(X, X_test, cols):
    print(cols)
    for c in cols:
        X[c] = X[c].astype(str)
        X[c] = X[c].str.slice(start = 0, stop = 10, step = 1)
        X_test[c] = X_test[c].astype(str)
        X_test[c] = X_test[c].str.slice(start = 0, stop = 10, step = 1)
    return (X,X_test)
    

def find_most_common_timezone_num(X, prefixCol, numCol):
    timezones = {}
    prefColX = X[prefixCol].replace(r"\N", np.nan)
    numColX = X[numCol].replace(r"\N", np.nan)

    for i in range(prefColX.size):
        if prefColX.iloc[i] is not np.nan:
            if prefColX.iloc[i] not in timezones:
                timezones[prefColX.iloc[i]] = {}
                timezones[prefColX.iloc[i]][numColX.iloc[i]] = 1
            else:
                if numColX.iloc[i] not in timezones[prefColX.iloc[i]]:
                    timezones[prefColX.iloc[i]][numColX.iloc[i]] = 1
                else:
                    timezones[prefColX.iloc[i]][numColX.iloc[i]] = timezones[prefColX.iloc[i]][numColX.iloc[i]] + 1


    print("\nThe numerical timezones associated with each prefix and their frequencies: ", timezones)
    reducedTimezones = {}
    for i in timezones:
        mostFreqNum = 0
        mostFreqNumOccurences = 0
        for j in timezones[i]:
            if timezones[i][j] is not np.nan and timezones[i][j] > mostFreqNumOccurences:
                mostFreqNum = j
                mostFreqNumOccurences = timezones[i][j]
        reducedTimezones[i] = mostFreqNum 

    print("\n\nReduced timezones: \n", reducedTimezones)
    return reducedTimezones

#might have an issue if timezones exist in test but not training
def replace_with_most_common_timezone_num(X, X_test, cols, timezoneDict):
    X[cols] = X[cols].replace(r'\N', np.nan)
    X_test[cols] = X_test[cols].replace(r'\N', np.nan)
    print(X.size)
    for i in range(X[cols[1]].size):
        if X[cols[1]].iloc[i] is np.nan:
            if X[cols[0]].iloc[i] is np.nan:
                X[cols[1]].iloc[i] = 200
            else:
                X[cols[1]].iloc[i] = timezoneDict[X[cols[0]].iloc[i]]

    for i in range(X_test[cols[1]].size):
        if X_test[cols[1]].iloc[i] is np.nan:
            if X_test[cols[0]].iloc[i] is np.nan:
                X_test[cols[1]].iloc[i] = 200
            else:
                X_test[cols[1]].iloc[i] = timezoneDict[X_test[cols[0]].iloc[i]]
    
    return (X,X_test)


        


        



