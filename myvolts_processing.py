
# MyVolts Processing

# Already deleted columns:
# ['document_language_provided', 'year_published', 'number_of_authors', 'first_author_id', 'num_pubs_by_first_author', 'app_version', 'user_os', 'user_os_version', 'user_java_version', 'user_timezone']
def delete_nan_columns(data_frame, list_all_nans):
    for column in list_all_nans:
        del data_frame[column]
    return data_frame

def delete_irrelevant_myvolts_columns(data_frame):
    irrelevant_columns = ['recommendation_set_id', 'user_id', 'session_id',\
                          'query_document_id', 'query_detected_language',\
                          'abstract_detected_language', 'organization_id',\
                          'application_type', 'response_delivered',\
                          'app_lang']
    # Language fields were mostly english
    # Application type was mostly e-commerce
    # response delivered seemed irrelevant
    for column in irrelevant_columns:
        del data_frame[column]
    return data_frame

def replace_nan_constant(data_frame):
    import numpy as np
    # Query Identifier CLeanup
    data_frame['query_identifier'] = data_frame['query_identifier'].replace(r'\N', 'Unknown')
    data_frame['query_identifier'] = data_frame['query_identifier'].astype(str)
    data_frame['item_type'] = data_frame['item_type'].replace(r'\N', 'Unknown')
    data_frame['item_type'] = data_frame['item_type'].astype(str)
    # Next Column cleanup
    return data_frame

def replace_nan_mean(data_frame):
    import numpy as np
    from sklearn.impute import SimpleImputer
    # Nan to Mean SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Char Count cleanup
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
    return data_frame