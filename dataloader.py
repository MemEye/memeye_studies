import torch 
from dataset import Dataset
from torch.utils.data import DataLoader

def get_dataloader(sec=2, batch_size=64):
    seq_length=int(sec*25)
    train_dataset = torch.load('train4.pt')
    val_dataset = torch.load('val4.pt')
    test_dataset = torch.load('test4.pt')
    #Make a dataloader by breaking into sequence of 64 length
    train_dataset_new=[]
    count=0
    no_of_labels=[0,0,0,0]  
    for x in train_dataset:
        for i in range(0,len(x[0]),seq_length):
            label=torch.argmax(x[1])
            no_of_labels[label]+=1
    for x in train_dataset:
        if len(x[0])%seq_length!=0:
            x[0]=torch.cat((x[0],torch.zeros(seq_length-len(x[0])%seq_length,10)))
        for i in range(0,len(x[0]),seq_length):
            label=torch.argmax(x[1])
            if label==0:
                count+=1
            if label==0 and count>=max(no_of_labels[2],max(no_of_labels[3]*2,no_of_labels[1]*2)):
                continue
            if label==1 or label==3:
                train_dataset_new.append([x[0][i:i+seq_length],x[1]])
            train_dataset_new.append([x[0][i:i+seq_length],x[1]])

    val_dataset_new=[]
    for x in val_dataset:
        if len(x[0])%seq_length!=0:
            x[0]=torch.cat((x[0],torch.zeros(seq_length-len(x[0])%seq_length,10)))
        for i in range(0,len(x[0]),seq_length):
            val_dataset_new.append([x[0][i:i+seq_length],x[1]])
    test_dataset_new=[]
    for x in test_dataset:
        if len(x[0])%seq_length!=0:
            x[0]=torch.cat((x[0],torch.zeros(seq_length-len(x[0])%seq_length,10)))
        for i in range(0,len(x[0]),seq_length):
            test_dataset_new.append([x[0][i:i+seq_length],x[1]])
    print(len(train_dataset_new),len(val_dataset_new),len(test_dataset_new))
    train_dataloader = DataLoader(train_dataset_new, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset_new, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset_new, batch_size=batch_size, shuffle=True)
    return train_dataloader,val_dataloader,test_dataloader

if __name__ == "__main__":
    for sec in [2,2.25,2.5,2.75,3,3.25,3.5]:
        train_dataloader,val_dataloader,test_dataloader = get_dataloader(sec=sec)
        torch.save(train_dataloader,'train'+str(sec)+'_dataloader.pt')
        torch.save(val_dataloader,'val'+str(sec)+'_dataloader.pt')
        torch.save(test_dataloader,'test'+str(sec)+'_dataloader.pt')
