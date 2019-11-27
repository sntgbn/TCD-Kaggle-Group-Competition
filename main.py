import pandas as pd
import numpy as np
import blogPreprocessing as bp
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import time

#Load data
data = pd.read_csv('training.csv')
data_test = pd.read_csv('test.csv')
data_test = data_test.iloc[0:9145, :]

#Print data summary
print("Training shape: ", data.shape)
print("Training columns: ", data.columns)
print("Test shape: ", data_test.shape)
print("Test columns: " , data_test.columns)
    
#Training Data
volts = data.loc[data['organization_id'] == 1]
blog = data.loc[data['organization_id'] == 8]
blogVoltsX = pd.concat([volts, blog]) #Volts #1, Blog #2
blogVoltsY = blogVoltsX['set_clicked']
blogVoltsX = blogVoltsX.drop(columns=['recommendation_set_id','local_time_of_request', 'local_hour_of_request', 'organization_id', 'response_delivered','user_id','session_id',
    'document_language_provided','year_published','number_of_authors','first_author_id','num_pubs_by_first_author','app_version','app_lang',
    'user_os','user_os_version','user_java_version','user_timezone', 'application_type', 'item_type','abstract_detected_language'
    ,'number_of_recs_in_set','set_clicked', 'time_recs_recieved', 'time_recs_displayed', 'time_recs_viewed'])

jabX = data.loc[data['organization_id'] == 1]
jabY = jabX['set_clicked']
jabX = jabX.drop(columns=['query_identifier', 'recommendation_set_id','local_time_of_request', 'local_hour_of_request', 'organization_id', 'response_delivered','user_id','session_id',
    'document_language_provided','app_version','app_lang',
    'user_os','user_os_version','user_java_version','user_timezone', 'item_type','abstract_detected_language'
    ,'number_of_recs_in_set','set_clicked', 'time_recs_recieved', 'time_recs_displayed', 'time_recs_viewed'])

#Test Data
voltsX_test = data_test.loc[data_test['organization_id'] == 1]
blogX_test = data_test.loc[data_test['organization_id'] == 8]
blogVoltsX_test = pd.concat([voltsX_test, blogX_test])
blogVoltsX_test = blogVoltsX_test.drop(columns=['recommendation_set_id','local_time_of_request', 'local_hour_of_request', 'organization_id', 'response_delivered','user_id','session_id',
    'document_language_provided','year_published','number_of_authors','first_author_id','num_pubs_by_first_author','app_version','app_lang',
    'user_os','user_os_version','user_java_version','user_timezone', 'application_type', 'item_type','abstract_detected_language'
    ,'number_of_recs_in_set','set_clicked', 'time_recs_recieved', 'time_recs_displayed', 'time_recs_viewed'])

jabX_test = data_test.loc[data['organization_id'] == 1]
jabX_test = jabX_test.drop(columns=['query_identifier', 'recommendation_set_id','local_time_of_request', 'local_hour_of_request', 'organization_id', 'response_delivered','user_id','session_id',
    'document_language_provided','app_version','app_lang',
    'user_os','user_os_version','user_java_version','user_timezone', 'item_type','abstract_detected_language'
    ,'number_of_recs_in_set','set_clicked', 'time_recs_recieved', 'time_recs_displayed', 'time_recs_viewed'])


print("blogVolts columns: ", blogVoltsX.columns)
print("blogVolts shape: " , blogVoltsX.shape)
print("jab columns: " , jabX.columns)
print("jab shahpe: " , jabX.shape)


cat_cols = [ 'query_identifier', 'query_detected_language', 'query_document_id', 'recommendation_algorithm_id_used', 'algorithm_class',
            'cbf_parser', 'search_title', 'search_keywords', 'search_abstract', 'clicks', 'ctr']
num_cols = ['query_word_count', 'query_char_count', 'abstract_word_count', 'abstract_char_count', 'request_received', 'hour_request_received', 'rec_processing_time', 'timezone_by_ip']
ss_cols = ['query_word_count', 'query_char_count', 'abstract_word_count', 'abstract_char_count', 'hour_request_received', 'rec_processing_time', 'timezone_by_ip', 'request_received']
le_cols = ['query_detected_language', 'algorithm_class', 'cbf_parser', 'search_title', 'search_keywords', 'search_abstract', 'clicks', 'ctr']
te_cols = ['query_detected_language', 'algorithm_class', 'cbf_parser', 'query_identifier', 'query_document_id', 'recommendation_algorithm_id_used', 'algorithm_class', 'cbf_parser', 'query_detected_language']
date_cols = ['request_received']
sb_cols = ['clicks', 'ctr']
SB_VAL = 0
ohe_cols = [ 'query_detected_language', 'algorithm_class', 'cbf_parser' ]



jab_num_cols = ['query_word_count', 'query_char_count', 'abstract_word_count', 'abstract_char_count', 'request_received', 'hour_request_received', 'rec_processing_time', 'timezone_by_ip',
                'year_published', 'number_of_authors', 'num_pubs_by_first_author']
jab_cat_cols = ['application_type', 'first_author_id', 'query_detected_language', 'query_document_id', 'recommendation_algorithm_id_used', 'algorithm_class',
            'cbf_parser', 'search_title', 'search_keywords', 'search_abstract', 'clicks', 'ctr']
jab_sb_cols = ['clicks', 'ctr']
JAB_SB_VAL = 0
jab_date_cols = ['request_received']
jab_le_cols = ['query_detected_language', 'algorithm_class', 'cbf_parser', 'search_title', 'search_keywords', 'search_abstract', 'clicks', 'ctr']
jab_ss_cols = ['num_pubs_by_first_author', 'number_of_authors', 'year_published','query_word_count', 'query_char_count', 'abstract_word_count', 'abstract_char_count', 'hour_request_received', 'rec_processing_time', 'timezone_by_ip', 'request_received']
jab_te_cols = ['query_detected_language', 'algorithm_class', 'cbf_parser', 'application_type', 'first_author_id', 'query_document_id', 'recommendation_algorithm_id_used', 'algorithm_class', 'cbf_parser', 'query_detected_language']
jab_ohe_cols = ['query_detected_language', 'algorithm_class', 'cbf_parser']

#Timezones
print("Starting jab timezones..")
print("Time 0")
start = time.time()
jab_timezones = bp.find_most_common_timezone_num(jabX, 'country_by_ip', 'timezone_by_ip')
timeone = time.time()
print("Time: ", timeone - start)
(jabX, jabX_test) = bp.replace_with_most_common_timezone_num(jabX, jabX_test, ['country_by_ip', 'timezone_by_ip'], jab_timezones)
timetwo = time.time()
print("Time two: " , timetwo - timeone)
print("Starting volts timezones..")
timezones = bp.find_most_common_timezone_num(blogVoltsX, 'country_by_ip', 'timezone_by_ip')
(blogVoltsX, blogVoltsX_test) = bp.replace_with_most_common_timezone_num(blogVoltsX, blogVoltsX_test, ['country_by_ip', 'timezone_by_ip'], timezones)

#Drop country_by_ip
blogVoltsX = blogVoltsX.drop(columns=['country_by_ip'])
blogVoltsX_test = blogVoltsX_test.drop(columns=['country_by_ip'])
jabX = jabX.drop(columns=['country_by_ip'])
jabX_test = jabX_test.drop(columns=['country_by_ip'])


print("Starting jab preprocess..")
(jabX, jabX_test) = bp.remove_time_from_date(jabX, jabX_test, jab_date_cols)
firstDate = jabX['request_received'].iloc[0]
dayZero = datetime.strptime(firstDate, '%d/%m/%Y')
(jabX, jabX_test) = bp.date_to_int(jabX, jabX_test, jab_date_cols, dayZero)
(jabX, jabX_test) = bp.replace_nan_with_unknown(jabX, jabX_test, jab_cat_cols)
(jabX, jabX_test) = bp.replace_nan_with_mean(jabX, jabX_test, jab_num_cols)
(jabX, jabX_test) = bp.set_binary(jabX, jabX_test, jab_sb_cols, JAB_SB_VAL)
(jabX, jabX_test) = bp.target_encode(jabX, jabX_test, jab_te_cols, jabY)
(jabX, jabX_test) = bp.scale(jabX, jabX_test, jab_ss_cols)
#(jabX, jabX_test) = bp.label_encode(jabX, jabX_test, jab_le_cols)
#(jabX, jabX_test) = bp.one_hot_encode(jabX, jabX_test, jab_ohe_cols)



print("Starting myvolts preprocess..")
(blogVoltsX, blogVoltsX_test) = bp.remove_time_from_date(blogVoltsX, blogVoltsX_test, date_cols)
firstDate = blogVoltsX['request_received'].iloc[0]
dayZero = datetime.strptime(firstDate, '%d/%m/%Y')
(blogVoltsX, blogVoltsX_test) = bp.date_to_int(blogVoltsX, blogVoltsX_test, date_cols, dayZero)
(blogVoltsX, blogVoltsX_test) = bp.replace_nan_with_unknown(blogVoltsX, blogVoltsX_test, cat_cols)
(blogVoltsX, blogVoltsX_test) = bp.replace_nan_with_mean(blogVoltsX, blogVoltsX_test, num_cols)
(blogVoltsX, blogVoltsX_test) = bp.set_binary(blogVoltsX, blogVoltsX_test, sb_cols, SB_VAL)
(blogVoltsX, blogVoltsX_test) = bp.target_encode(blogVoltsX, blogVoltsX_test, te_cols, blogVoltsY)
(blogVoltsX, blogVoltsX_test) = bp.scale(blogVoltsX, blogVoltsX_test, ss_cols)
#(blogVoltsX, blogVoltsX_test) = bp.label_encode(blogVoltsX, blogVoltsX_test, le_cols)
#(blogVoltsX, blogVoltsX_test) = bp.one_hot_encode(blogVoltsX, blogVoltsX_test, ohe_cols)
print("End of preprocessing...")


print("Saving processed training data")
blogVoltsX.to_csv("blogVolts_pro_train.csv")
jabX.to_csv("jab_proc_train.csv")


print("Starting blogVolts nn...")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15), random_state=1)
clf.fit(blogVoltsX, blogVoltsY)
res = clf.predict(blogVoltsX_test)


print("Starting jab nn...")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15), random_state=1)
clf.fit(jabX, jabY)
jab_res = clf.predict(jabX_test)

ones = 0
for i in res:
    if i == 1:
        ones = ones + 1
for i in jab_res:
    if i == 1:
        ones = ones + 1

print("Ones: ", ones)

np.savetxt("blogVolts_results.txt", res, fmt= "%d",newline='\n')
np.savetxt("jab_results.txt", jab_res, fmt= "%d",newline='\n')

