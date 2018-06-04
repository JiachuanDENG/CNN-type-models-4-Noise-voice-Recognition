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



if __name__ == '__main__':
    modelNames=sys.argv[1]
    testPercentage=sys.argv[2]
    modelNameList=modelNames.split(',')
    wanted_words='testnoisydata15db,testnoise15db'
    trainx,trainy,valx,valy,model_settings=dataprocessing.returnData(datadir='../../data/selfbuildDataTest15dB/',\
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



