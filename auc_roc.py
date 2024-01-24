#Plot the ROC curve and calculate the AUC for the model

import torch
train_dataloader= torch.load('train3.5_dataloader.pt')
val_dataloader= torch.load('val3.5_dataloader.pt')    
test_dataloader= torch.load('test3.5_dataloader.pt')
model= torch.load('model50.pt')
dict=['negative','learning','recall','recognition']
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

model.eval()
y_score=[[],[],[],[]]
y_true = []
func=torch.nn.Softmax(dim=1)
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_dataloader):
        labels=torch.argmax(labels,1)
        outputs = model(inputs)
        outputs=func(outputs)
        y_score[0].extend(outputs[:,0].data.cpu().numpy())
        y_score[1].extend(outputs[:,1].data.cpu().numpy())
        y_score[2].extend(outputs[:,2].data.cpu().numpy())
        y_score[3].extend(outputs[:,3].data.cpu().numpy())
        y_true.extend(labels.data.cpu().numpy())

# Compute ROC curve and ROC area for each class
for class_num in range(4):
    y=[]
    for i in range(len(y_true)):
        if y_true[i]==class_num:
            y.append(1)
        else:
            y.append(0)
    fpr, tpr, _ = roc_curve(y, y_score[class_num])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(dict[class_num], roc_auc))   
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of Class {0} for 3.5 second, 16 hidden dim, 4 layer model'.format(dict[class_num]))
    plt.legend(loc="lower right")
    plt.savefig('roc'+str(class_num)+'.png')
    plt.show()
    plt.close()
    print('AUC of class {0} is {1:0.2f}'.format(class_num, roc_auc))


