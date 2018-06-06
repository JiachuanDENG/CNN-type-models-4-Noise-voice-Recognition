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
import time
from train import cal_accu as cal_accu



def forwardingTime(model,X,y):
    xVariable=Variable(torch.from_numpy(X))
    yVariable=Variable(torch.from_numpy(y))

    start = time.clock()
    outputVariable=model.forwarding(xVariable,isTrain=False)
    print ('Time Consumption :{}'.format(time.clock() - start))
    print ('accuracy:{}'.format(cal_accu(outputVariable,yVariable)),'val pos:neg--',len(y[y==1])/len(y))
    perf_measure(yVariable,outputVariable)

def perf_measure(y_actualVariable, y_hatVariable):
    _,y_hatVariable=torch.max(y_hatVariable, 1)
    y_actual=y_actualVariable.data.numpy()
    y_hat=y_hatVariable.data.numpy()


    # noisy voice label 0
    # noise label 1
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==0 and y_actual[i]==y_hat[i]:
           TP += 1 # pred voice, actual voice
        if y_hat[i]==0  and y_actual[i]!=y_hat[i]:
           FP += 1 # pred voice , actual noise
        if y_actual[i]==1 and y_actual[i]==y_hat[i]:
           TN += 1 # pred noise, actual noise
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FN += 1 # pred noise, actual voice

    print ("TP (pred voice, actual voice):{}\nFP (pred voice , actual noise):{}\nTN (pred noise, actual noise):{}\nFN (pred noise, actual voice):{}".format(TP, FP, TN, FN))



if __name__ == '__main__':
    modelNames=sys.argv[1]
    testPercentage=sys.argv[2]
    modelNameList=modelNames.split(',')
    wanted_words='testnoisydata15dbnorm,testnoise15db'
    trainx,trainy,valx,valy,model_settings=dataprocessing.returnData(datadir='../../data/realtest',\
        wanted_words=wanted_words)
    
    testx=np.concatenate((trainx,valx),axis=0)
    testy=np.concatenate((trainy,valy),axis=0)

    for modelName in modelNameList:
        modelSaveFilePath='./{}modelTrain.pkl'.format(modelName)
        model=models.selectingModel(modelName,model_settings,classN=len(wanted_words.split(',')))
        print ('model loaded')
        model.load_state_dict(torch.load(modelSaveFilePath))
        print ('*'*10,modelName,'*'*10)
        # forwardingTime(model,testx[:1],testy[:1])
        forwardingTime(model,testx[:int(float(testPercentage)*testx.shape[0])],testy[:int(float(testPercentage)*testx.shape[0])])



