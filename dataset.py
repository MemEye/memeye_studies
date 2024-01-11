#Creating a pytorch dataset out of the processed_data
import torch
import pandas as pd 
import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data=[]
        for x in ['processed_data 1/*','processed_data 2/*','processed_data 3/*','processed_data 4/*']:
            for file in glob.glob(x):
                if int(file[-3:])>120:
                    for f in glob.glob(file+'/processed/segmented/emotibit/*'):
                        label=f.split('/')[-1]
                        if label in ['negative','learning','recall','recognition_familar','recognition_new']:
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
                                    label=torch.tensor([1,0,0,0,0],dtype=torch.float32)
                                elif label=='learning':
                                    label=torch.tensor([0,1,0,0,0],dtype=torch.float32)
                                elif label=='recall':
                                    label=torch.tensor([0,0,1,0,0],dtype=torch.float32)
                                elif label=='recognition_familar':
                                    label=torch.tensor([0,0,0,1,0],dtype=torch.float32)
                                elif label=='recognition_new':
                                    label=torch.tensor([0,0,0,0,1],dtype=torch.float32)
                                self.data.append([df,label])
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    
#Save the dataset in train.pt
dataset=Dataset()
torch.save(dataset,'test.pt')

