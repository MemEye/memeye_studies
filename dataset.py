#Creating a pytorch dataset out of the processed_data
import torch
import pandas as pd 
import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self,start,end):
        self.data=[]
        for x in ['emotibit_segmented/*']:
            for file in glob.glob(x):
                if int(file[-3:])>=start and int(file[-3:])<=end:
                    for f in glob.glob(file+'/*'):
                        label=f.split('/')[-1]
                        if label=='recognition_new':
                            continue
                        if label in ['negative','learning','recall','recognition_familar']:
                            for csvs in glob.glob(f+'/*.csv'):
                                df = pd.read_csv(csvs)
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
                                self.data.append([df,label])
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    
#Save the dataset in train.pt
train_dataset=Dataset(101,120)
torch.save(train_dataset,'train4.pt')
val_dataset=Dataset(121,123)
torch.save(val_dataset,'val4.pt')
test_dataset=Dataset(124,132)
torch.save(test_dataset,'test4.pt')

