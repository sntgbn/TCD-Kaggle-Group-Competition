
# MyVolts Processing

# Already deleted columns:
# ['document_language_provided', 'year_published', 'number_of_authors', 'first_author_id', 'num_pubs_by_first_author', 'app_version', 'user_os', 'user_os_version', 'user_java_version', 'user_timezone']
def delete_nan_columns(data_frame, list_all_nans):
    for column in list_all_nans:
        del data_frame[column]
    return data_frame

def delete_irrelevant_myvolts_columns(data_frame):
    irrelevant_columns = ['user_id', 'session_id',\
                          'query_document_id', 'organization_id',\
                          'application_type', 'response_delivered',\
                          'number_of_recs_in_set', 'clicks', 'ctr', 'rec_processing_time']
    # Language fields were mostly english
    # Application type was mostly e-commerce
    # response delivered seemed irrelevant
    # Should recommended algorithm ID used be removed?
    for column in irrelevant_columns:
        del data_frame[column]
    return data_frame

def replace_nan_constant(data_frame):
    import numpy as np
    # Query Identifier CLeanup
    data_frame['query_identifier'] = data_frame['query_identifier'].replace(r'\N', 'Unknown')
    data_frame['query_identifier'] = data_frame['query_identifier'].replace(np.NaN, 'Unknown')
    data_frame['query_identifier'] = data_frame['query_identifier'].astype(str)
    # Item Type Cleanup
    data_frame['item_type'] = data_frame['item_type'].replace(r'\N', 'Unknown')
    data_frame['item_type'] = data_frame['item_type'].replace(np.NaN, 'Unknown')
    data_frame['item_type'] = data_frame['item_type'].astype(str)
    # Country by IP Cleanup
    data_frame['country_by_ip'] = data_frame['country_by_ip'].replace(r'\N', 'DE')
    data_frame['country_by_ip'] = data_frame['country_by_ip'].replace(np.NaN, 'DE')
    data_frame['country_by_ip'] = data_frame['country_by_ip'].astype(str)
    # CBF Parser Cleanup
    data_frame['cbf_parser'] = data_frame['cbf_parser'].replace(r'\N', 'Uknown')
    data_frame['cbf_parser'] = data_frame['cbf_parser'].replace(np.NaN, 'Uknown')
    data_frame['cbf_parser'] = data_frame['cbf_parser'].astype(str)
    # Language Columns Parser Cleanup
    # Should these be removed instead?
    data_frame['query_detected_language'] = data_frame['query_detected_language'].replace(r'\N', 'Unknown')
    data_frame['query_detected_language'] = data_frame['query_detected_language'].replace(np.NaN, 'Unknown')
    data_frame['query_detected_language'] = data_frame['query_detected_language'].astype(str)
    data_frame['abstract_detected_language'] = data_frame['abstract_detected_language'].replace(r'\N', 'Unknown')
    data_frame['abstract_detected_language'] = data_frame['abstract_detected_language'].replace(np.NaN, 'Unknown')
    data_frame['abstract_detected_language'] = data_frame['abstract_detected_language'].astype(str)
    data_frame['app_lang'] = data_frame['app_lang'].replace(r'\N', 'Unknown')
    data_frame['app_lang'] = data_frame['app_lang'].replace(np.NaN, 'Unknown')
    data_frame['app_lang'] = data_frame['app_lang'].astype(str)
    # TIME Columns Parser Cleanup
    # Should these be removed instead?
    data_frame['local_time_of_request'] = data_frame['local_time_of_request'].replace(r'\N', 'Unknown')
    data_frame['local_time_of_request'] = data_frame['local_time_of_request'].replace(np.NaN, 'Unknown')
    data_frame['local_time_of_request'] = data_frame['local_time_of_request'].astype(str)
    data_frame['local_hour_of_request'] = data_frame['local_hour_of_request'].replace(r'\N', 'Unknown')
    data_frame['local_hour_of_request'] = data_frame['local_hour_of_request'].replace(np.NaN, 'Unknown')
    data_frame['local_hour_of_request'] = data_frame['local_hour_of_request'].astype(str)
    data_frame['timezone_by_ip'] = data_frame['timezone_by_ip'].replace(r'\N', 'Unknown')
    data_frame['timezone_by_ip'] = data_frame['timezone_by_ip'].replace(np.NaN, 'Unknown')
    data_frame['timezone_by_ip'] = data_frame['timezone_by_ip'].astype(str)
    data_frame['time_recs_recieved'] = data_frame['time_recs_recieved'].replace(r'\N', 'Unknown')
    data_frame['time_recs_recieved'] = data_frame['time_recs_recieved'].replace(np.NaN, 'Unknown')
    data_frame['time_recs_recieved'] = data_frame['time_recs_recieved'].astype(str)
    data_frame['time_recs_displayed'] = data_frame['time_recs_recieved'].replace(r'\N', 'Unknown')
    data_frame['time_recs_recieved'] = data_frame['time_recs_recieved'].replace(np.NaN, 'Unknown')
    data_frame['time_recs_displayed'] = data_frame['time_recs_displayed'].astype(str)
    data_frame['time_recs_viewed'] = data_frame['time_recs_viewed'].replace(r'\N', 'Unknown')
    data_frame['time_recs_viewed'] = data_frame['time_recs_viewed'].replace(np.NaN, 'Unknown')
    data_frame['time_recs_viewed'] = data_frame['time_recs_viewed'].astype(str)
    data_frame['request_received'] = data_frame['request_received'].replace(r'\N', 'Unknown')
    data_frame['request_received'] = data_frame['request_received'].replace(np.NaN, 'Unknown')
    data_frame['request_received'] = data_frame['request_received'].astype(str)

    data_frame['algorithm_class'] = data_frame['algorithm_class'].replace(r'\N', 'unknown')
    data_frame['algorithm_class'] = data_frame['algorithm_class'].replace(np.NaN, 'unknown')
    data_frame['algorithm_class'] = data_frame['algorithm_class'].astype(str)

    data_frame['search_title'] = data_frame['search_title'].replace(r'\N', 'Unknown')
    data_frame['search_title'] = data_frame['search_title'].replace(np.NaN, 'Unknown')
    data_frame['search_title'] = data_frame['search_title'].astype(str)

    data_frame['search_keywords'] = data_frame['search_keywords'].replace(r'\N', 'Unknown')
    data_frame['search_keywords'] = data_frame['search_keywords'].replace(np.NaN, 'Unknown')
    data_frame['search_keywords'] = data_frame['search_keywords'].astype(str)

    data_frame['search_abstract'] = data_frame['search_abstract'].replace(r'\N', 'Unknown')
    data_frame['search_abstract'] = data_frame['search_abstract'].replace(np.NaN, 'Unknown')
    data_frame['search_abstract'] = data_frame['search_abstract'].astype(str)
    return data_frame

def replace_nan_mean(data_frame):
    import numpy as np
    from sklearn.impute import SimpleImputer
    # Nan to Mean SimpleImputer
    imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
    # Char Count        eanup
    data_frame['query_char_count'] = data_frame['query_char_count'].replace(r'\N', np.NaN)
    data_frame['query_char_count'] = imputer.fit_transform(data_frame['query_char_count'].values.reshape(-1,1))
    data_frame['query_char_count'] = data_frame['query_char_count'].astype(int)
    # Word Count cleanup
    data_frame['query_word_count'] = data_frame['query_word_count'].replace(r'\N', np.NaN)
    data_frame['query_word_count'] = imputer.fit_transform(data_frame['query_word_count'].values.reshape(-1,1))
    data_frame['query_word_count'] = data_frame['query_word_count'].astype(int)
    # Abstract Char Count cleanup
    data_frame['abstract_char_count'] = data_frame['abstract_char_count'].replace(r'\N', np.NaN)
    data_frame['abstract_char_count'] = imputer.fit_transform(data_frame['abstract_char_count'].values.reshape(-1,1))
    data_frame['abstract_char_count'] = data_frame['abstract_char_count'].astype(int)
    # Abstract Word Count cleanup
    data_frame['abstract_word_count'] = data_frame['abstract_word_count'].replace(r'\N', np.NaN)
    data_frame['abstract_word_count'] = imputer.fit_transform(data_frame['abstract_word_count'].values.reshape(-1,1))
    data_frame['abstract_word_count'] = data_frame['abstract_word_count'].astype(int)
    # Should these be removed instead?
    data_frame['time_recs_viewed'] = imputer.fit_transform(data_frame['abstract_word_count'].values.reshape(-1,1))
    data_frame['abstract_word_count'] = data_frame['abstract_word_count'].astype(int)
    # Recommendation algorithm ID
    data_frame['recommendation_algorithm_id_used'] = data_frame['recommendation_algorithm_id_used'].replace(r'\N', np.NaN)
    data_frame['recommendation_algorithm_id_used'] = imputer.fit_transform(data_frame['recommendation_algorithm_id_used'].values.reshape(-1,1))
    data_frame['recommendation_algorithm_id_used'] = data_frame['recommendation_algorithm_id_used'].astype(int)
    # Hour request received
    data_frame['hour_request_received'] = data_frame['hour_request_received'].replace(r'\N', np.NaN)
    data_frame['hour_request_received'] = imputer.fit_transform(data_frame['hour_request_received'].values.reshape(-1,1))
    data_frame['hour_request_received'] = data_frame['hour_request_received'].astype(int)
    # Times REC Viewed
    data_frame['time_recs_viewed'] = data_frame['time_recs_viewed'].replace(r'\N', np.NaN)
    data_frame['time_recs_viewed'] = imputer.fit_transform(data_frame['time_recs_viewed'].values.reshape(-1,1))
    data_frame['time_recs_viewed'] = data_frame['time_recs_viewed'].astype(int)
    return data_frame

def convert_float_int(data_frame):
    data_frame['hour_request_received'] = data_frame['hour_request_received'].astype(int)
    data_frame['time_recs_viewed'] = data_frame['time_recs_viewed'].astype(int)