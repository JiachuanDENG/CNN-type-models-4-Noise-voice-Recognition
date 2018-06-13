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
import os
import real_sampledDataprocessing


def forwardingTime(model,X,y,filetrack,trackingMap,threshold):
    xVariable=Variable(torch.from_numpy(X))
    yVariable=Variable(torch.from_numpy(y))

    start = time.clock()
    outputVariable=model.forwarding(xVariable,isTrain=False)
    print ('Time Consumption :{}'.format(time.clock() - start))
    # print ('accuracy:{}'.format(cal_accu(outputVariable,yVariable)),'val pos:neg--',len(y[y==1])/len(y))
    # perf_measure(yVariable,outputVariable)
    classifyFiles(outputVariable,filetrack,trackingMap,threshold)


def classifyFiles(outputVal,filesTrack,trackingMap,threshold):
    def cperrorfiles(errorfilenames,errorporb,outputdir='../../data/recordTest/errorfiles'):
        if not os.path.exists(outputdir):
            os.system('mkdir {}'.format(outputdir))
        idx=0
        for errorfile in errorfilenames:
            filename=errorfile.split('/')[-1].split('.wav')[0]
            os.system('cp {} {}/{}'.format(errorfile,outputdir,filename+'_'+errorporb[idx]+'.wav'))
            idx+=1


    def classify(array1Variable,filesTrack,trackingMap,threshold):
        _,outputVal=torch.max(array1Variable, 1)
        array1=outputVal.data.numpy()
        noisefiles,voicefiles,noisefilesProb,voicefilesProb=[],[],[],[]
        for i,a1 in enumerate(array1):

            p=torch.nn.functional.softmax(array1Variable[i]).data.numpy()
            prob='{}_{}'.format(p[0],p[1])
        
            if array1[i]==0:
                #  predict to be voice
                voicefiles.append(trackingMap[filesTrack[i]])
                voicefilesProb.append(prob)

            if array1[i]==1 and p[1]>=threshold:
                #predict to be noise

                noisefiles.append(trackingMap[filesTrack[i]])
                noisefilesProb.append(prob)



        cperrorfiles(voicefiles,voicefilesProb,'../../data/recordTest/predictVoice') 
        cperrorfiles(noisefiles,noisefilesProb,'../../data/recordTest/predictNoise') 


    return classify (outputVal,filesTrack,trackingMap,threshold)




if __name__ == '__main__':
    modelNames=sys.argv[1]
    filename=sys.argv[2]
    threshold=float(sys.argv[3]) 

    wnd=50
    targetdb=-26
    targetDir='../../data/recordTest/'

    realdataprocessor=real_sampledDataprocessing.RealDataProcessor('./','./',wnd,targetdb)

    wavtrackingMap=realdataprocessor.processingfile(filename,targetDir+'testNoisyData15dBNorm/',targetDir+'testNoise15dB/','../../data/origwavDir/')

    # print (wavtrackingMap)

    testPercentage=1.0
    modelNameList=modelNames.split(',')
    wanted_words='testnoisydata15dbnorm,testnoise15db'
    trainx,trainy,valx,valy,model_settings,trainfiletrack,valfiletrack=dataprocessing.returnData(datadir='../../data/recordTest',\
        wanted_words=wanted_words)
    
    testx=np.concatenate((trainx,valx),axis=0)
    testy=np.concatenate((trainy,valy),axis=0)
    filetrack=trainfiletrack+valfiletrack

    for modelName in modelNameList:
        modelSaveFilePath='./{}modelTrain.pkl'.format(modelName)
        model=models.selectingModel(modelName,model_settings,classN=len(wanted_words.split(',')))
        print ('model loaded')
        model.load_state_dict(torch.load(modelSaveFilePath))
        print ('*'*10,modelName,'*'*10)
        # forwardingTime(model,testx[:1],testy[:1])
        # print (filetrack)
        forwardingTime(model,testx[:int(float(testPercentage)*testx.shape[0])],testy[:int(float(testPercentage)*testx.shape[0])],filetrack[:int(float(testPercentage)*testx.shape[0])],wavtrackingMap,threshold)



