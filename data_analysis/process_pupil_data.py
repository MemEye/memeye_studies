import numpy as np
import csv
import pandas as pd
from pyplr import utils
import os

def reformat_annos(annos, label_name):
    df = annos.copy()
    df['pair'] = [i // 2 for i in range(len(df))]  

    # Group by the pair and label, then get the first 'times' as start and last 'times' as end for each block
    grouped = df.groupby(['pair', label_name])
    result_df = pd.DataFrame({
        'timestamp': grouped['timestamp'].first(),
        label_name: grouped[label_name].first(),
        'time_end': grouped['timestamp'].last()
    }).reset_index(drop=True)

    return result_df

def merge_events(p):
    # Empty list to store the new rows
    new_rows = []
    i = 0
    while i < len(p) - 1:
        x = p.iloc[i]['label']
        if p.iloc[i + 1]['label'] == x + ' questions':
            # Merge the two rows
            new_row = {
                'timestamp': p.iloc[i]['timestamp'],
                'label': x,
                'time_end': p.iloc[i + 1]['time_end'],
            }
            new_rows.append(new_row)
            i += 2
        else:
            # Include the current row if it's not part of a merge
            new_row = {
                'timestamp': p.iloc[i]['timestamp'],
                'label': p.iloc[i]['label'],
                'time_end': p.iloc[i]['time_end']
            }
            new_rows.append(new_row)
            i += 1

    new_row = {
            'timestamp': p.iloc[-1]['timestamp'],
            'label': p.iloc[-1]['label'],
            'time_end': p.iloc[-1]['time_end']
            }
    new_rows.append(new_row)
    p = pd.DataFrame(new_rows)

    new_rows = []
    i = 0
    while i < len(p) - 1:
        x = p.iloc[i]['label']
        if p.iloc[i + 1]['label'] == x + ' verbal':
            # Merge the two rows
            new_row = {
                'timestamp': p.iloc[i]['timestamp'],
                'label': x,
                'time_end': p.iloc[i + 1]['time_end'],
            }
            new_rows.append(new_row)
            i += 2
        else:
            # Include the current row if it's not part of a merge
            new_row = {
                'timestamp': p.iloc[i]['timestamp'],
                'label': p.iloc[i]['label'],
                'time_end': p.iloc[i]['time_end']
            }
            new_rows.append(new_row)
            i += 1

    p = pd.DataFrame(new_rows)
    return p

def process_pupil_data(rec_dir, sample_rate, eye_id):
    s = utils.new_subject(
    rec_dir, export='000', out_dir_nm=f'processed')

    # Load pupil data
    samples = utils.load_pupil(
        s['data_dir'], eye_id=eye_id, method='3d')
    
    events = utils.load_annotations(s['data_dir'])
    
    #TODO: load fixations
    #TODO: see if we can load both eyes?
    #TODO: load blinks
    #TODO: load gazes
    events.reset_index(inplace=True)
    events = events.iloc[14:]
    events = events[['timestamp', 'label']]

    events_jpg = events[events.label.str.contains('jpg')]
    events_jpg = events_jpg.rename(columns= {'label': 'image'})

    events_global_label = events[events.label.str.contains('recall') | 
                                events.label.str.contains('recognition') |
                                events.label.str.contains('learning') |
                                events.label.str.contains('baseline') |
                                events.label.str.contains('break')]

    events_question = events[events.label.str.contains('question')]
    events_question = events_question.rename(columns= {'label': 'is_question'})

    events_verbal = events[events.label.str.contains('verbal')]
    events_verbal = events_verbal.rename(columns= {'label': 'is_verbal'})

    events_rem_q = events[events.label.str.contains('Yes') | events.label.str.contains('No')]
    events_rem_q = events_rem_q.rename(columns= {'label': 'remembered'})

    events_conf = events[events.label.str.contains('High') | events.label.str.contains('Low')]
    events_conf = events_conf.rename(columns= {'label': 'resp_confidence'})
    # global label, question, verbal, is practice, picture, yes/no, confidence

    samples = samples.reset_index()
    p = reformat_annos(events_global_label.sort_values(['timestamp', 'label']), 'label')
    p = merge_events(p)
    samples = samples.sort_values('pupil_timestamp')
    p = p.sort_values('timestamp')

    merged_df = pd.merge_asof(samples.sort_values('pupil_timestamp'),
                          p.sort_values('timestamp'),
                          left_on='pupil_timestamp',
                          right_on='timestamp',
                          direction='backward')

    # Now filter out the rows where the 'timestamp' is greater than the 'time_end'
    merged_df['label'] = np.where((merged_df['pupil_timestamp'] >= merged_df['timestamp']) & (merged_df['pupil_timestamp'] < merged_df['time_end']), merged_df['label'], np.nan)

    first_label_index = merged_df['label'].first_valid_index()
    buffer_size = sample_rate * 10
    if first_label_index is not None and first_label_index >= buffer_size:
        merged_df.loc[buffer_size:first_label_index - buffer_size, 'label'] = 'baseline'

    merged_df = merged_df.drop(columns = ['timestamp', 'time_end'])

    question = reformat_annos(events_question.sort_values(['timestamp', 'is_question']), 'is_question')
    merged_df = pd.merge_asof(merged_df.sort_values('pupil_timestamp'),
                            question.sort_values('timestamp'),
                            left_on='pupil_timestamp',
                            right_on='timestamp',
                            direction='nearest')

    # Now filter out the rows where the 'timestamp' is greater than the 'end_times'
    merged_df['is_question'] = np.where((merged_df['pupil_timestamp'] >= merged_df['timestamp']) & (merged_df['pupil_timestamp'] < merged_df['time_end']), merged_df['is_question'], np.nan)
    merged_df = merged_df.drop(columns = ['timestamp', 'time_end'])

    q_map = {'practice recognition questions':'y', 
             'practice recall name questions': 'y',
             'recognition questions': 'y', 
             'recall name questions': 'y'}

    merged_df['is_question'] = merged_df['is_question'].map(q_map)

    verbal = reformat_annos(events_verbal.sort_values(['timestamp', 'is_verbal']), 'is_verbal')
    merged_df = pd.merge_asof(merged_df.sort_values('pupil_timestamp'),
                            verbal.sort_values('timestamp'),
                            left_on='pupil_timestamp',
                            right_on='timestamp',
                            direction='nearest')

    # Now filter out the rows where the 'timestamp' is greater than the 'end_times'
    merged_df['is_verbal'] = np.where((merged_df['pupil_timestamp'] >= merged_df['timestamp']) & (merged_df['pupil_timestamp'] < merged_df['time_end']), merged_df['is_verbal'], np.nan)
    merged_df = merged_df.drop(columns = ['timestamp', 'time_end'])

    q_map = {'practice recall name verbal': 'y', 
             'recall name verbal': 'y'}

    merged_df['is_verbal'] = merged_df['is_verbal'].map(q_map)

    merged_df = pd.merge_asof(merged_df.sort_values('pupil_timestamp'),
                          events_rem_q.sort_values('timestamp'),
                          left_on='pupil_timestamp',
                          right_on='timestamp',
                          direction='nearest')
    merged_df['remembered'] = np.where(merged_df['is_question'].isna(), np.nan, merged_df['remembered'])
    merged_df = merged_df.drop(columns = ['timestamp'])

    merged_df = pd.merge_asof(merged_df.sort_values('pupil_timestamp'),
                          events_conf.sort_values('timestamp'),
                          left_on='pupil_timestamp',
                          right_on='timestamp',
                          direction='nearest')
    merged_df['resp_confidence'] = np.where(merged_df['is_question'].isna(), np.nan, merged_df['resp_confidence'])
    merged_df = merged_df.drop(columns = ['timestamp'])

    merged_df = pd.merge_asof(merged_df.sort_values('pupil_timestamp'),
                            events_jpg.sort_values('timestamp'),
                            left_on='pupil_timestamp',
                            right_on='timestamp',
                            direction='nearest')
    merged_df['image'] = np.where(merged_df['label'].isna() | (merged_df['label'] == 'baseline'), np.nan, merged_df['image'])
    merged_df['image'] = np.where(merged_df['label'].str.contains('break'), np.nan, merged_df['image'])
    merged_df = merged_df.drop(columns = ['timestamp'])

    return merged_df

def load_extra_csvs(data_loc, subject_num):
    extra_data_loc = os.path.join(data_loc, str(subject_num), 'exports/000')
    fixations_path = os.path.join(extra_data_loc, 'fixations.csv')
    blinks_path = os.path.join(extra_data_loc, 'blinks.csv')
    gazes_path = os.path.join(extra_data_loc, 'gaze_positions.csv')

    fixations_df = pd.read_csv(fixations_path)
    fixations_df = fixations_df.add_prefix('fixations_')
    blinks_df = pd.read_csv(blinks_path)
    blinks_df = blinks_df.add_prefix('blinks_')
    gazes_df = pd.read_csv(gazes_path)
    gazes_new_mapping = {
        'world_index': 'gaze_world_index', 
        'confidence': 'gaze_confidence',
        'norm_pos_x': 'gaze_norm_pos_x', 
        'norm_pos_y': 'gaze_norm_pos_y', 
        'base_data': 'gaze_base_data',
        'eye_center0_3d_x': 'gaze_eye_center0_3d_x',
        'eye_center0_3d_y': 'gaze_eye_center0_3d_y', 
        'eye_center0_3d_z': 'gaze_eye_center0_3d_z',
        'eye_center1_3d_x': 'gaze_eye_center1_3d_x', 
        'eye_center1_3d_y': 'gaze_eye_center1_3d_y',
        'eye_center1_3d_z': 'gaze_eye_center1_3d_z'
    }
    gazes_df.rename(columns = gazes_new_mapping, inplace=True)
    return fixations_df, blinks_df, gazes_df


def process_extra_data(processed_pupil_eye_left, processed_pupil_eye_right, data_loc, subject_num):
    fixations_df, blinks_df, gazes_df = load_extra_csvs(data_loc, subject_num)
    blinks_df = blinks_df.loc[blinks_df['blinks_confidence'] >= 0.4] # filter out any blinks with confidence of less than 0.40
    fixations_df['fixations_end_timestamp'] = fixations_df['fixations_start_timestamp'] + fixations_df['fixations_duration']

    fixations_df, blinks_df, gazes_df = load_extra_csvs(data_loc, subject_num)
    blinks_df = blinks_df.loc[blinks_df['blinks_confidence'] >= 0.4] # filter out any blinks with confidence of less than 0.40
    fixations_df['fixations_end_timestamp'] = fixations_df['fixations_start_timestamp'] + fixations_df['fixations_duration']

    processed_pupil_both = pd.concat([processed_pupil_eye_left, processed_pupil_eye_right])
    processed_pupil_both.sort_values('pupil_timestamp', inplace=True)
    processed_pupil_both = pd.merge_asof(processed_pupil_both, gazes_df, left_on='pupil_timestamp',
                                right_on='gaze_timestamp', direction='nearest')
    processed_pupil_eye_left = processed_pupil_both.loc[processed_pupil_both.eye_id==1]
    processed_pupil_eye_right = processed_pupil_both.loc[processed_pupil_both.eye_id==0]

    cols_to_drop_left = ['gaze_eye_center0_3d_x', 'gaze_eye_center0_3d_y', 'gaze_eye_center0_3d_z', 
                        'gaze_normal0_x', 'gaze_normal0_y', 'gaze_normal0_z', 'gaze_timestamp', 'gaze_world_index']

    cols_to_drop_right = ['gaze_eye_center1_3d_x', 'gaze_eye_center1_3d_y', 'gaze_eye_center1_3d_z', 
                        'gaze_normal1_x', 'gaze_normal1_y', 'gaze_normal1_z', 'gaze_timestamp', 'gaze_world_index']

    processed_pupil_eye_left.drop(columns=cols_to_drop_left, inplace=True)
    processed_pupil_eye_right.drop(columns=cols_to_drop_right, inplace=True)

    processed_pupil_left = pd.merge_asof(processed_pupil_eye_left, blinks_df, left_on='pupil_timestamp',
                            right_on='blinks_start_timestamp', direction='nearest')

    processed_pupil_right = pd.merge_asof(processed_pupil_eye_right, blinks_df, left_on='pupil_timestamp',
                                right_on='blinks_start_timestamp', direction='nearest')

    processed_pupil_left['blinks_confidence'] = np.where((processed_pupil_left['pupil_timestamp'] >= processed_pupil_left['blinks_start_timestamp']) & (processed_pupil_left['pupil_timestamp'] < processed_pupil_left['blinks_end_timestamp']), processed_pupil_left['blinks_confidence'], np.nan)
    processed_pupil_right['blinks_confidence'] = np.where((processed_pupil_right['pupil_timestamp'] >= processed_pupil_right['blinks_start_timestamp']) & (processed_pupil_right['pupil_timestamp'] < processed_pupil_right['blinks_end_timestamp']), processed_pupil_right['blinks_confidence'], np.nan)

    cols_to_drop = ['blinks_id', 'blinks_start_timestamp', 'blinks_duration', 'blinks_end_timestamp', 'blinks_start_frame_index', 
                    'blinks_index', 'blinks_end_frame_index', 'blinks_filter_response', 'blinks_base_data']

    processed_pupil_left.drop(columns=cols_to_drop, inplace=True)
    processed_pupil_right.drop(columns=cols_to_drop, inplace=True)

    processed_pupil_left['is_blink'] = np.where(processed_pupil_left['blinks_confidence'].notna(), 'y', '')
    processed_pupil_right['is_blink'] = np.where(processed_pupil_right['blinks_confidence'].notna(), 'y', '')

    processed_pupil_left = pd.merge_asof(processed_pupil_left, fixations_df, left_on='pupil_timestamp',
                            right_on='fixations_start_timestamp', direction='nearest')

    processed_pupil_right = pd.merge_asof(processed_pupil_right, fixations_df, left_on='pupil_timestamp',
                                right_on='fixations_start_timestamp', direction='nearest')
    for col in fixations_df.columns:
        processed_pupil_left[col] = np.where((processed_pupil_left['pupil_timestamp'] >= processed_pupil_left['fixations_start_timestamp']) & (processed_pupil_left['pupil_timestamp'] < processed_pupil_left['fixations_end_timestamp']), processed_pupil_left[col], np.nan)
        processed_pupil_right[col] = np.where((processed_pupil_right['pupil_timestamp'] >= processed_pupil_right['fixations_start_timestamp']) & (processed_pupil_right['pupil_timestamp'] < processed_pupil_right['fixations_end_timestamp']), processed_pupil_right[col], np.nan)

    cols_to_drop = ['fixations_id', 'fixations_start_timestamp', 'fixations_end_timestamp', 'fixations_duration', 'fixations_start_frame_index', 
                    'fixations_end_frame_index', 'fixations_base_data']

    processed_pupil_left.drop(columns=cols_to_drop, inplace=True)
    processed_pupil_right.drop(columns=cols_to_drop, inplace=True)

    return processed_pupil_left, processed_pupil_right



def run(data_loc, save_loc, subject_nums):
    for i in subject_nums:
        subject_loc = os.path.join(data_loc, str(i))
        subject_save_loc = os.path.join(save_loc, str(i))
        sample_rate = 120 if i < 120 else 200
        processed_pupil_eye_left = process_pupil_data(subject_loc, sample_rate, 'left') # eye_id = 1
        processed_pupil_eye_right = process_pupil_data(subject_loc, sample_rate, 'right') # eye_id = 0

        processed_pupil_left, processed_pupil_right = process_extra_data(processed_pupil_eye_left, 
                                                                         processed_pupil_eye_right, 
                                                                         data_loc, i)

        os.makedirs(os.path.join(subject_save_loc, f'processed_pupil'), exist_ok=True)
        save_path_left = os.path.join(subject_save_loc, f'processed_pupil', f'processed_pupil_{i}_eye_left.csv')
        save_path_right = os.path.join(subject_save_loc, f'processed_pupil', f'processed_pupil_{i}_eye_right.csv')
        print(save_path_left, save_path_right)
        processed_pupil_left.to_csv(save_path_right)
        processed_pupil_right.to_csv(save_path_left)


if __name__=='__main__':
    num_subjects = 32
    subject_nums = list(range(101, 100+num_subjects+1))
    data_loc = '/Users/monaabd/Desktop/pupil_exports_new/'
    save_loc = '/Users/monaabd/Desktop/pupil_processed_new/'
    run(data_loc, save_loc, subject_nums)
