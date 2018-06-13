import torch
import torch
import torch.utils.data as Data
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
import models
import dataprocessing

def cal_accu(outputVal,yvalVariable):
    
    def compare(array1,array2):
        accu=0.
        if len(array1)!=len(array2):
            print ('len error')
            return
        for i,a1 in enumerate(array1):
    #         print (a1,array2[i])
            if a1==array2[i]:
                accu+=1.
        return accu/len(array1)


    _,outputVal=torch.max(outputVal, 1)
    return compare(outputVal.data.numpy(),yvalVariable.data.numpy())
    

BATCH_SIZE=256
EPOCH=50



def trainIter(xtrain,ytrain,xval,yval,cnnAudio,optimizer,loss_func,modelName,batch_size=BATCH_SIZE,epochNum=EPOCH,loadModels=False):
    modelSaveFilePath='./{}modelTrain.pkl'.format(modelName)
    x_train_tensor,y_train_tensor=torch.from_numpy(xtrain),torch.from_numpy(ytrain)
    x_val_tensor,y_val_tensor=torch.from_numpy(xval),torch.from_numpy(yval)
            
#   xvalVariable=autograd.Variable(x_val_tensor)
#   yvalVariable=autograd.Variable(y_val_tensor) 
    
    torch_dataset=Data.TensorDataset(x_train_tensor,y_train_tensor)

    loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    if loadModels :
        print ('model loaded')
        cnnAudio.load_state_dict(torch.load(modelSaveFilePath))


    for epoch in range(epochNum):
        print ('**********EPOCH',epoch,'*************')
        for step,(x_,y_) in enumerate(loader):
            # print step
            bx=autograd.Variable(x_)
            by=autograd.Variable(y_)
          

            output=cnnAudio.forwarding(bx)
          
            loss=loss_func(output,by)
            accu=cal_accu(output,by)
            
            if step%10==0:
#               print ('training loss:',loss.data.numpy()[0])
#               print ('training accuracy:',accu)
                valindices = list(range(0,xval.shape[0]))
                np.random.shuffle(valindices)
                valx=xval[valindices][:batch_size]
                valy=yval[valindices][:batch_size]
                valxVariable=Variable(torch.from_numpy(valx))
                valyVariable=Variable(torch.from_numpy(valy))
                
                outputVal=cnnAudio.forwarding(valxVariable,isTrain=False)
                accuval=cal_accu(outputVal,valyVariable) 
                print ('training accuracy:',accu,'val accuracy:',accuval,'val pos:neg--',len(valy[valy==1])/len(valy))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

#       print ('val accuracy:',accuval)

        
        torch.save(cnnAudio.state_dict(),modelSaveFilePath)


if __name__ == '__main__':
    modelName=sys.argv[1]

    wanted_words='trainnoisydata15db,trainnoise15db'
    trainx,trainy,valx,valy,model_settings,_,__=dataprocessing.returnData(datadir='../../data/selfbuildData15dB/',\
        wanted_words=wanted_words)

    model=models.selectingModel(modelName,model_settings,classN=len(wanted_words.split(',')))


    
    optimizer=torch.optim.Adam(model.parameters(),0.001)
    loss_func=nn.CrossEntropyLoss()
    trainIter(trainx,trainy,valx,valy,model,optimizer,loss_func,modelName)
       




