#Creating a pytorch dataset out of the processed_data
import torch
import pandas as pd 
import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self,start,end):
        self.data=[]
        for x in ['processed_data 1/*','processed_data 2/*','processed_data 3/*','processed_data 4/*']:
            for file in glob.glob(x):
                if int(file[-3:])>=start and int(file[-3:])<=end:
                    for f in glob.glob(file+'/processed/segmented/emotibit/*'):
                        label=f.split('/')[-1]
                        if label=='recognition_new':
                            continue
                        if label in ['negative','learning','recall','recognition_familar']:
                            for csvs in glob.glob(f+'/*.csv'):
                                df = pd.read_csv(csvs)
                                if label=='recall' or label=='recognition_familar':
                                    x=df['remembered'].value_counts().to_dict()
                                    if 'Yes' not in x:
                                        label=label+'_no'
                                    elif 'No' not in x:
                                        label=label+'_yes'
                                    elif x['Yes']<x['No']:
                                        label=label+'_yes'
                                    else:
                                        label=label+'_no'
                                df=df[['T1','TH','EA','EL','PI','PR','PG','SF','SR','SA']]
                                for col in df.columns:
                                    df[col]=(df[col]-df[col].mean())/df[col].std()
                                df=df.fillna(0)
                                df=df.values
                                df=torch.tensor(df,dtype=torch.float32)
                                #Create one hot encoding for labels
                                if label=='negative':
                                    label=torch.tensor([1,0,0,0,0,0],dtype=torch.float32)
                                elif label=='learning':
                                    label=torch.tensor([0,1,0,0,0,0],dtype=torch.float32)
                                elif label=='recall_yes':
                                    label=torch.tensor([0,0,1,0,0,0],dtype=torch.float32)
                                elif label=='recall_no':
                                    label=torch.tensor([0,0,0,1,0,0],dtype=torch.float32)
                                elif label=='recognition_familar_yes':
                                    label=torch.tensor([0,0,0,0,1,0],dtype=torch.float32)
                                elif label=='recognition_familar_no':
                                    label=torch.tensor([0,0,0,0,0,1],dtype=torch.float32)
                                self.data.append([df,label])
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    
#Save the dataset in train.pt
train_dataset=Dataset(101,120)
torch.save(train_dataset,'train_6.pt')
val_dataset=Dataset(121,123)
torch.save(val_dataset,'val_6.pt')
test_dataset=Dataset(124,132)
torch.save(test_dataset,'test_6.pt')

