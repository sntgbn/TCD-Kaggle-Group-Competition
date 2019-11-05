# General purpose imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

# Machine Learning models from SKLEARN
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from catboost import CatBoostRegressor

# Importing CSV and generating a pandas dataframe from it
print('Loading CSV files')
training_data = pd.read_csv('../tcdml1920-rec-click-pred--training.csv')
prediction_data = pd.read_csv('../tcdml1920-rec-click-pred--test.csv')
submission_data = pd.read_csv('../tcdml1920-rec-click-pred--submission file.csv')

# Getting Jabref data
training_data_jabref = training_data.loc[training_data['query_identifier'] == 'Withheld for privacy']

# Getting NOT Jabref data, fetching myvolts and blog from it
training_data_myvolts_blog = training_data.loc[training_data['query_identifier'] != 'Withheld for privacy']
training_data_myvolts = training_data_myvolts_blog.loc[training_data_myvolts_blog['user_id'] == (r'\N' or np.NaN)]
training_data_blog = training_data_myvolts_blog.loc[training_data_myvolts_blog['user_id'] != (r'\N' or np.NaN)]

if training_data.shape[0] == training_data_jabref.shape[0]+training_data_myvolts.shape[0]+training_data_blog.shape[0]:
    print('Data Sets Sizes Match, continue')
else:
    raise Exception('Data sets sizes do not match')

# Deleting unuseed PD Data Frames
del training_data_jabref
del training_data

# MyVolts Training Data

# Joeran's Training Data
# End goal is to predict set clicked
# JabRef (recommendations for research articles)
# MyVolts (recommendations for e-commerce products)
# Joeran Beel's homepage (recommendations for blog articles). 


# model=CatBoostRegressor(iterations=10000, depth=5, learning_rate=0.1)
# model.fit(x_train, y_train, cat_features=category_indexes, eval_set=(x_test, y_test),plot=True)
# submission_data['Income'] = model.predict(prediction_data)
# submission_data.to_csv('tcd_ml_final_submission.csv', index=False)

