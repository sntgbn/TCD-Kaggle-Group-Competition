# MyVolts Processing
def delete_columns(data_frame, list_all_nans):
    for column in list_all_nans:
        del data_frame[column]
    return data_frame