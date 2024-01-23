import numpy as np
import csv
import pandas as pd
import os

tag_type_to_sr = {
    'AX': 25,
    'AY': 25, 
    'AZ': 25, 
    'GX': 25, 
    'GY': 25, 
    'GZ': 25, 
    'MX': 25, 
    'MY': 25, 
    'MZ': 25,
    'PI': 25, 
    'PG': 25, 
    'PR': 25,
    'T1': 7.5, 
    'TH': 7.5,
    'EA': 15, 
    'EL': 15, 
    'HR': 15
}

file_types_to_process = {'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'MX', 'MY', 'MZ',
                        'PI', 'PG',  'PR', 'T1', 'TH', 'EA', 'EL', 'LM', 'SA', 'SR', 'SF', 'HR'}
frequency_to_upsample = 25
scr = {'SA', 'SR', 'SF'}

def interpolate_data(raw_df, frequency):
    raw_df = raw_df.copy()
    if raw_df['EmotiBitTimestamp'].duplicated().any():
        raw_df = raw_df.drop_duplicates(subset='EmotiBitTimestamp')
    emotibit_time = list(raw_df['EmotiBitTimestamp'])
    emotibit_time = [int(time) for time in emotibit_time]

    # Create a new index at 25 Hz frequency (every 40 milliseconds)
    new_index = np.arange(emotibit_time[0], emotibit_time[-1], 1000/frequency)
    raw_df.set_index('EmotiBitTimestamp', inplace=True)
    df_interpolated = raw_df.reindex(new_index)
    df_interpolated = df_interpolated.interpolate(method='linear')
    return df_interpolated

def try_convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return None

def check_str(value):
    try:
        if not value.replace('.', '', 1).isdigit():
            return value
    except:
        return None

def process_emotibit_data(emotibit_data_path, pupil_path):
    # first id all of the data files and load them in
    files_in_directory = os.listdir(emotibit_data_path)
    data_files = [file for file in files_in_directory if file.endswith('.csv')]
    filtered_files = []
    for file in data_files:
        if file.split('.')[0][-2:] in file_types_to_process and 'LC' not in file:
            filtered_files.append(file)

    data_stream_to_df = dict()

    for file in filtered_files:
        df = pd.read_csv(os.path.join(emotibit_data_path, file))
        data = file.split('.')[0][-2:]
        data_stream_to_df[data] = df

    interpolated_data = dict()
    for stream in data_stream_to_df:
        if stream in tag_type_to_sr and tag_type_to_sr[stream] < frequency_to_upsample:
            df = data_stream_to_df[stream]
            interpolated_df = interpolate_data(df, frequency_to_upsample)
            interpolated_data[stream] = interpolated_df
        elif stream not in scr:
            interpolated_df = data_stream_to_df[stream].copy()
            interpolated_df.set_index('EmotiBitTimestamp', inplace=True)
            interpolated_data[stream] = interpolated_df
    
    df = interpolated_data['T1']
    drop_cols = ['ProtocolVersion', 'DataReliability', 'LslMarkerSourceTimestamp', 
                'LocalTimestamp', 'DataLength', 'TypeTag', 'PacketNumber']
    merged_df = df.drop(drop_cols, axis=1)
    merged_df = merged_df.reset_index()
    merged_df = merged_df.sort_values('EmotiBitTimestamp')
    merged_df['EmotiBitTimestamp'] = merged_df['EmotiBitTimestamp'].astype(int)
    for stream in interpolated_data:
        if stream != 'LM' and stream != 'T1':
            df = interpolated_data[stream].drop(drop_cols, axis=1)
            df = df.reset_index()
            df['EmotiBitTimestamp'] = df['EmotiBitTimestamp'].astype(int)
            merged_df = merged_df.sort_values('EmotiBitTimestamp')
            df = df.sort_values('EmotiBitTimestamp')
            merged_df = pd.merge_asof(merged_df, df, on='EmotiBitTimestamp', direction='nearest')
    
    df = interpolated_data['LM'].copy()  # Create a copy of the DataFrame
    df['sparse_labels'] = df['LD'].apply(lambda x: check_str(x))

    # Try to convert the string to a float; if it fails, return None
    df['sparse_pupil_time'] = df['LD'].apply(lambda x: try_convert_to_float(x))
    drop_cols = ['ProtocolVersion', 'DataReliability', 'LslMarkerSourceTimestamp', 
                'LocalTimestamp', 'DataLength', 'TypeTag', 'PacketNumber', 'LR', 'LM','LC', 'LD', 'Unnamed: 12']
    df = df.drop(drop_cols, axis=1)
    merged_df = pd.merge_asof(merged_df, df, on='EmotiBitTimestamp', tolerance=25, direction='nearest')
    # merged_df#.pupil_time.unique()
    merged_df['sparse_pupil_time'] = pd.to_numeric(merged_df['sparse_pupil_time'], errors='coerce')

    merged_df['estimated_pupil_time'] = merged_df['sparse_pupil_time'].interpolate(method='spline', order=2)
    first_valid_index = merged_df['estimated_pupil_time'].first_valid_index()

    # Remove all rows up to and including the row with the first valid timestamp
    if first_valid_index is not None:
        # +1 because slicing is exclusive of the endpoint, and we want to include the row at first_valid_index
        merged_df = merged_df.loc[first_valid_index + 1:]

    pupil_data = pd.read_csv(pupil_path)

    label_columns = ['pupil_timestamp', 'label', 'is_question', 'is_verbal', 'remembered', 'resp_confidence', 'image']  # Replace with actual label column names
    other_sensor_labels = pupil_data[label_columns]

    # Sort both dataframes by their timestamp columns
    current_sensor_df = merged_df.sort_values('estimated_pupil_time')
    other_sensor_labels = other_sensor_labels.sort_values('pupil_timestamp')

    # Perform the asof merge
    merged = pd.merge_asof(current_sensor_df, other_sensor_labels, left_on='estimated_pupil_time', right_on='pupil_timestamp', direction='nearest')

    # Drop the other sensor's timestamp column if it's included in the merge
    if 'pupil_timestamp' in merged.columns:
        merged.drop(columns='pupil_timestamp', inplace=True)

    merged.drop(columns=['sparse_labels', 'sparse_pupil_time'], inplace=True)

        #add in the SCR files
    for stream in scr:
        df = data_stream_to_df[stream]
        df = df[['EmotiBitTimestamp', stream]]
        df = df.sort_values('EmotiBitTimestamp')
        df['EmotiBitTimestamp'] = df['EmotiBitTimestamp'].astype(int)
        merged = merged.sort_values('EmotiBitTimestamp')
        merged = pd.merge_asof(merged, df, on='EmotiBitTimestamp', tolerance=25, direction='nearest')
    
    return merged

def run(parsed_loc, pupil_data_loc, save_loc, subject_nums):
    for subject_id in subject_nums:
        emotibit_data_path = os.path.join(parsed_loc, str(subject_id), 'parsed_emotibit')
        pupil_path = os.path.join(pupil_data_loc, str(subject_id), 'processed_pupil', f'processed_pupil_{subject_id}_eye_right.csv')
        processed_emotibit = process_emotibit_data(emotibit_data_path, pupil_path)

        save_path = os.path.join(save_loc, str(subject_id))
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'processed_emotibit_{subject_id}.csv')
        processed_emotibit.to_csv(save_path)

if __name__=='__main__':
    num_subjects = 32
    subject_nums = list(range(101, 100+num_subjects+1))
    parsed_loc = '/Users/monaabd/Desktop/pupil_exports/'
    pupil_data_loc = '/Users/monaabd/Desktop/pupil_processed_new_updated/'
    save_loc = '/Users/monaabd/Desktop/emotibit_processed_new/'
    run(parsed_loc, pupil_data_loc, save_loc, subject_nums)
