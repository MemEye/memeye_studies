import torch 
from dataset import Dataset
from torch.utils.data import DataLoader

def get_dataloader(seq_length=64, batch_size=64):
    train_dataset = torch.load('train.pt')
    test_dataset = torch.load('test.pt')
    #Make a dataloader by breaking into sequence of 64 length
    train_dataset_new=[]
    for x in train_dataset:
        if len(x[0])%seq_length!=0:
            x[0]=torch.cat((x[0],torch.zeros(seq_length-len(x[0])%seq_length,10)))
        for i in range(0,len(x[0]),seq_length):
            train_dataset_new.append([x[0][i:i+seq_length],x[1]])
    test_dataset_new=[]
    for x in test_dataset:
        if len(x[0])%seq_length!=0:
            x[0]=torch.cat((x[0],torch.zeros(seq_length-len(x[0])%seq_length,10)))
        for i in range(0,len(x[0]),seq_length):
            test_dataset_new.append([x[0][i:i+seq_length],x[1]])
    print(len(train_dataset_new),len(test_dataset_new))
    train_dataloader = DataLoader(train_dataset_new, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset_new, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, test_dataloader = get_dataloader()
    torch.save(train_dataloader,'train_dataloader.pt')
    torch.save(test_dataloader,'test_dataloader.pt')
