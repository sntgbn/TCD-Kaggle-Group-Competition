import catboost_training as cbt
import pandas as pd
import numpy as np
import myvolts_processing as mvp

'''
Comments:
replace all missing timezone_by_ip nans with most common country/timezone_ip numerical pair
then replace all non_numerical timezone_by_ip vals with mode country/timezone_ip numerical pair
user_timezone - dependent on other two timezone columns
remove organisation id




'''
#Returns list of Columns that have all Nans - uses columnNumber from original Data
#Must be altered if planning on calling after removing columns
def inspect_missing_data(X):
    print("Inspecting the Missing Data:")
    listOfAllNans = []
    columnList = (list(data.columns))[:-1]
    i = 0
    for col in columnList:
        backNCount = 0
        notProvidedCount = 0
        backN = "\\N"
        notProvided = "Not Provided"
        for j in X[:,i]:
            if j == backN:
                backNCount = backNCount + 1
            if j == notProvided:
                notProvidedCount = notProvidedCount + 1
        print("There are", (pd.isnull(X[:,i]).sum() + backNCount + notProvidedCount),"nan values in", col, "column")
        i = i + 1
        if X[:,i].size == backNCount:
            listOfAllNans.append(col)
    return listOfAllNans

#Returns list of Columns that have all Nans - uses columnNumber from original Data
#Must be altered if planning on calling after removing columns
def inspect_timezone(X):
    usertimezone = X[:,(list(data.columns).index("user_timezone"))]
    timezoneip = X[:,(list(data.columns).index("timezone_by_ip"))]
    for i in range(0, usertimezone.size):
        if usertimezone[i] != "\\N" and usertimezone[i] != "Not provided":
            if usertimezone[i] == timezoneip[i]:
                print("Same value for user_timezone and time_zone_ip")
            else:
                print("Usertimezone val: ", usertimezone[i])
                print("timezoneip val: ", timezoneip[i])

#TODO:
def only_one_val_in_columns(X):
    columnList = (list(data.columns))[:-1]
    i = 0
    for col in columnList:
        ul = np.unique(X[:,i])
        print(ul)
        if (ul.size == 1):
            print("Only one value in ", col)
        i = i + 1
        print("Number of unique vals in ", col, ": ", ul.size)

data = pd.read_csv('training.csv')
print("Matrix shape: ", data.shape)
print("Data columns: ", data.columns)


blogX = data.loc[data['organization_id'] == 8].copy()
jabRefX = data.loc[data['organization_id'] == 1].copy()
myVoltX = data.loc[data['organization_id'] == 4].copy()

prediction_data = pd.read_csv('testing.csv')
print("Matrix shape: ", prediction_data.shape)
print("Data columns: ", prediction_data.columns)
predictionBlogX = prediction_data.loc[prediction_data['organization_id'] == 8].copy()
predictionJabRefX = prediction_data.loc[prediction_data['organization_id'] == 1].copy()
predictionMyVoltX = prediction_data.loc[prediction_data['organization_id'] == 4].copy()

# print("\nInspect missing Joeran Blog Data:")
# allNans = inspect_missing_data(blogX.values)
# print("Columns that have all Nans: ", allNans)
# #print("Inspecting timezone: ")
# #inspect_timezone(blogX)
# #print("Checking for single value columns: ")
# #only_one_val_in_columns(blogX)
# print("Joeran Dataset Size: ", blogX.shape)

# print("\nInspecting missing jabRef Data:")
# allNans = inspect_missing_data(jabRefX.values)
# print("Columns that have all Nans: ", allNans)
# #print("Inspecting timezone: ")
# #inspect_timezone(jabRefX)
# #print("Checking for single value columns: ")
# #only_one_val_in_columns(jabRefX)
# print("jabRef Dataset Size: ", jabRefX.shape)

print("\nInspect missing myVolt Data: ")
allNans = inspect_missing_data(myVoltX.values)
print("Columns that have all Nans: ", allNans)
#print("Inspecting timezone:")
#inspect_timezone(myVoltX)
#print("Checking for single value columns: ")
#only_one_val_in_columns(myVoltX)
print("myVolt Dataset Size: ", myVoltX.shape)

print("\nInspect missing myVolt Prediction Data: ")
predictionAllNans = inspect_missing_data(predictionMyVoltX.values)
print("Columns that have all Nans: ", predictionAllNans)
#print("Inspecting timezone:")
#inspect_timezone(myVoltX)
#print("Checking for single value columns: ")
#only_one_val_in_columns(myVoltX)
print("myVolt Dataset Size: ", predictionMyVoltX.shape)

# Processing all data as if it were myvolts
# myvoltX = data
# Deleting NAN columns
myVoltX = mvp.delete_nan_columns(data, allNans).copy()
predictionMyVoltX = mvp.delete_nan_columns(prediction_data, predictionAllNans).copy()
# Deleting Irrelevant Columns
myVoltX = mvp.delete_irrelevant_myvolts_columns(myVoltX)
predictionMyVoltX = mvp.delete_irrelevant_myvolts_columns(predictionMyVoltX)
# Deleting NANs for constants
myVoltX = mvp.replace_nan_constant(myVoltX)
predictionMyVoltX = mvp.replace_nan_constant(predictionMyVoltX)
# Replacing NANs for Mean values
myVoltX = mvp.replace_nan_mean(myVoltX)
predictionMyVoltX = mvp.replace_nan_mean(predictionMyVoltX)
# # Converting floats to INTs for catboost
# myVoltX = mvp.convert_float_int(myVoltX)
# predictionMyVoltX = mvp.convert_float_int(predictionMyVoltX)
# training_data.isnull().any()
# Creating training set
cbt.train_catboost(myVoltX, predictionMyVoltX)