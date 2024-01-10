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

def process_pupil_data(rec_dir, sample_rate):
    s = utils.new_subject(
    rec_dir, export='000', out_dir_nm=f'processed')

    # Load pupil data
    samples = utils.load_pupil(
        s['data_dir'], eye_id='best', method='3d')
    
    events = utils.load_annotations(s['data_dir'])
    #TODO: load fixations
    #TODO: load blinks
    #TODO: load gazes
    #TODO: higher level feature processing (vergence, saccade duration, etc)
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


def run(data_loc, subject_nums):
    for i in subject_nums:
        subject_loc = os.path.join(data_loc, str(i))
        sample_rate = 120 if i < 120 else 200
        processed_pupil = process_pupil_data(subject_loc, sample_rate)
        os.makedirs(os.path.join(subject_loc, f'processed'), exist_ok=True)
        save_path = os.path.join(subject_loc, f'processed', f'processed_pupil_{i}.csv')
        print(save_path)
        processed_pupil.to_csv(save_path)


if __name__=='__main__':
    num_subjects = 32
    subject_nums = list(range(101, 100+num_subjects+1))
    data_loc = '/Users/monaabd/Desktop/pupil_exports/'
    run(data_loc, subject_nums)
