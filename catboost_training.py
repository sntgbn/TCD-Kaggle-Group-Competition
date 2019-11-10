# Catboost trainig

def train_catboost(training_data, prediction_data):
    # Catboost and train_test_split importing
    from catboost import CatBoostRegressor
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split 

    # Importing submission CSV
    submission_data = pd.read_csv('submission.csv')

    # Generating training data set and splitting it
    training_x = training_data.copy()
    training_y = training_data['set_clicked'].copy()
    del training_x['set_clicked']

    # Getting X/Y train/test        
    x_train, x_test, y_train, y_test = train_test_split(training_x, training_y, train_size=0.7, random_state=905732)

    #Only categorical variables are Profession and Country
    category_indexes = np.where(training_data.dtypes == np.object)[0]
    model=CatBoostRegressor(iterations=1000, depth=5, learning_rate=0.1)
    model.fit(x_train, y_train, cat_features=category_indexes, eval_set=(x_test, y_test), plot=True)
    submission_data['set_clicked'] = model.predict(prediction_data)[0:9145].astype(int)
    submission_data.to_csv('submission.csv', index=False)