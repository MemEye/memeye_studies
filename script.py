#Create a script which takes a model and dataset and n which is the number of seconds and outputs the prediction in a csv file
import torch
import glob
import pandas as pd

def script(model_name,folder_name, n):
    seq_length=int(n*25)
    model=torch.load(model_name)
    model.eval()
    f1=open('negative.csv','w')
    f1.write("file_name,label,length\n")
    f2=open('learning.csv','w')
    f2.write("file_name,label,length\n")
    f3=open('recall.csv','w')
    f3.write("file_name,label,length\n")
    f4=open('recognition_familar.csv','w')
    f4.write("file_name,label,length\n")
    for file in glob.glob(folder_name+"/*"):
        class_name = file.split("/")[-1]
        for csv in glob.glob(file+"/*.csv"):
            csv_name = csv.split("/")[-1]
            df = pd.read_csv(csv)
            df=df[['T1','TH','EA','EL','PI','PR','PG','SF','SR','SA']]
            for col in df.columns:
                df[col]=(df[col]-df[col].mean())/df[col].std()
            df=df.fillna(0)
            df=df.values
            df=torch.tensor(df,dtype=torch.float32)
            df=df.unsqueeze(0)
            count=[0,0,0,0]
            for i in range(0,len(df[0]),seq_length):
                x=df[0][i:i+seq_length]
                if len(x)<seq_length:
                    x=torch.cat((x,torch.zeros(seq_length-len(x),10)))
                x=x.unsqueeze(0)
                output=model(x)
                output=torch.argmax(output)
                count[output]+=1
            f1.write(csv+","+str(count[0]/sum(count))+","+str(seq_length)+"\n")
            f2.write(csv+","+str(count[1]/sum(count))+","+str(seq_length)+"\n")
            f3.write(csv+","+str(count[2]/sum(count))+","+str(seq_length)+"\n")
            f4.write(csv+","+str(count[3]/sum(count))+","+str(seq_length)+"\n")
    f1.close()
    f2.close()
    f3.close()
    f4.close()

if __name__ == "__main__":
    script('model3/3_16/model39.pt','emotibit_segmented/101',3)

