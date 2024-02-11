#Creating a pytorch dataset out of the processed_data
import torch
import pandas as pd 
import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self,start,end):
        self.data=[]
        for x in ['emotibit_segmented_new/*']:
            for file in glob.glob(x):
                if int(file[-3:])>=start and int(file[-3:])<=end:
                    for f in glob.glob(file+'/*'):
                        l=f.split('/')[-1]
                        if l in ['negative','learning','recall','recognition_familar']:
                            for csvs in glob.glob(f+'/*.csv'):
                                label=l
                                df = pd.read_csv(csvs)
                                # x=df['remembered'].value_counts()
                                # if label=='recall':
                                #     if 'Yes' in x:
                                #         if 'No' in x:
                                #             if x['Yes']<x['No']:
                                #                 label='recall_fail'
                                #             else:
                                #                 label='recall_correct'
                                #         else:
                                #             label='recall_correct'
                                #     else:
                                #         label='recall_fail'
                                # elif label=='recognition_familar':
                                #     if 'Yes' in x:
                                #         if 'No' in x:
                                #             if x['Yes']<x['No']:
                                #                 label='recognition_fail'
                                #             else:
                                #                 label='recognition_correct'
                                #         else:
                                #             label='recognition_correct'
                                #     else:
                                #         label='recognition_fail'
                                # elif label=='recognition_new':
                                #     if 'No' in x:
                                #         if 'Yes' in x:
                                #             if x['No']>x['Yes']:
                                #                 label='recognition_fail'
                                #             else:
                                #                 continue
                                #         else:
                                #             continue
                                if df['is_question'].isna().all() and df['is_verbal'].isna().all():
                                    pass
                                else:
                                    for index,row in df.iterrows():
                                        if not pd.isna(row['is_question']) or not pd.isna(row['is_verbal']):
                                            break
                                    df=df[:index]
                                df=df[['T1','TH','EA','EL','PI','PR','PG','SF','SR','SA']]
                                for col in df.columns:
                                    df[col]=(df[col]-df[col].mean())/df[col].std()
                                df=df.fillna(0)
                                df=df.values
                                df=torch.tensor(df,dtype=torch.float32)
                                #Create one hot encoding for labels

                                if label=='negative':
                                    label=torch.tensor([1,0,0,0],dtype=torch.float32)
                                elif label=='learning':
                                    label=torch.tensor([0,1,0,0],dtype=torch.float32)
                                elif label=='recall':
                                    label=torch.tensor([0,0,1,0],dtype=torch.float32)
                                elif label=='recognition_familar':
                                    label=torch.tensor([0,0,0,1],dtype=torch.float32)
                                else:
                                    continue
                                self.data.append([df,label])
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    
#Save the dataset in train.pt
train_dataset=Dataset(101,120)
torch.save(train_dataset,'train.pt')
val_dataset=Dataset(121,123)
torch.save(val_dataset,'val.pt')
test_dataset=Dataset(124,132)
torch.save(test_dataset,'test.pt')

