#Create a script which takes a model and dataset and n which is the number of seconds and outputs the prediction in a csv file
import torch
import glob
import pandas as pd

def script(model_name,folder_name,n):
    seq_length=int(n*25)
    model=torch.load(model_name)
    model.eval()
    f1=open('prediction.csv','w')
    f1.write("file_name,length,pred_negative,pred_learning,pred_recall,pred_recognition_familiar,true_label\n")
    for file in glob.glob(folder_name+"/*"):
        class_name = file.split("/")[-1]
        if class_name not in ["negative","learning","recall","recognition_familar"]:
            continue
        if class_name=="negative":
            class_name="0"
        elif class_name=="learning":
            class_name="1"
        elif class_name=="recall":
            class_name="2"
        elif class_name=="recognition_familar":
            class_name="3"
        for csv in glob.glob(file+"/*.csv"):
            csv_name = csv.split("/")[-1]
            csv_name=csv_name[:-4]
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
            x=sum(count)
            count=[i/x for i in count]
            f1.write(csv_name+","+str(seq_length)+","+str(count[0])+","+str(count[1])+","+str(count[2])+","+str(count[3])+","+class_name+"\n")
    f1.close()

if __name__ == "__main__":
    script('model3/3_16/model39.pt','emotibit_segmented/101',3)

