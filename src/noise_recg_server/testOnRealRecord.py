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
from pydub import AudioSegment


def forwardingTime(model,X,y,filetrack,trackingMap,threshold):
    xVariable=Variable(torch.from_numpy(X))
    yVariable=Variable(torch.from_numpy(y))

    start = time.clock()
    outputVariable=model.forwarding(xVariable,isTrain=False)
    print ('Time Consumption :{}'.format(time.clock() - start))
    # classify audio segments and save segments into different directories
    classifyFiles(outputVariable,filetrack,trackingMap,threshold)


# function used for regenerate complete audio series using predicted noise/voice segments
def mixingSegments(segmentDir,name):
        def comparatorHelper(s1):
            return float(s1.split('_')[0])
        segmentfiles=[f for f in os.listdir(segmentDir) if '.wav' in f]
        segmentfiles=sorted(segmentfiles,key=comparatorHelper) 
        s=AudioSegment.empty()
        for i,file in enumerate(segmentfiles):
            if i%1000==0:
                print ('mixed {}'.format(i))

            
            if 'noise' in file:
                sound=AudioSegment.silent(250)
            else:
                sound=AudioSegment.from_file(segmentDir+file)
            s+=sound
        print ('writing to file....')
        s.export('../../data/recordTest/'+name+'.wav', format="wav")

def classifyFiles(outputVal,filesTrack,trackingMap,threshold):
    def cperrorfiles(errorfilenames,errorporb,tag,outputdir='../../data/recordTest/errorfiles'):
        if not os.path.exists(outputdir):
            os.system('mkdir {}'.format(outputdir))
        idx=0
        for errorfile in errorfilenames:
            filename=errorfile.split('/')[-1].split('.wav')[0]
            os.system('cp {} {}/{}'.format(errorfile,outputdir,filename+'_'+errorporb[idx]+'_'+tag+'.wav'))
            idx+=1

    def cpBlankFIles(filenames,outputdir,length=250):
        for f in [f_ for f_ in filenames if '.wav' in f_]:
            sound=AudioSegment.silent(duration=length)
            sound.export(outputdir+f,format='wav')

    def cpfiles(origdir,outputdir):
        for f in [f_ for f_ in os.listdir(origdir) if 'wav' in f_]:
            os.system('cp {}/{} {}'.format(origdir,f,outputdir))
    
    

    def classify(array1Variable,filesTrack,trackingMap,threshold):
        _,outputVal=torch.max(array1Variable, 1)
        array1=outputVal.data.numpy()
        noisefiles,voicefiles,noisefilesProb,voicefilesProb=[],[],[],[]
        for i,a1 in enumerate(array1):

            p=torch.nn.functional.softmax(array1Variable[i]).data.numpy()
            prob='{}_{}'.format(p[0],p[1])
        
            

            if array1[i]==1 and p[1]>=threshold:
                #predict to be noise
                noisefiles.append(trackingMap[filesTrack[i]])
                noisefilesProb.append(prob)
            else:
                #  predict to be voice
                voicefiles.append(trackingMap[filesTrack[i]])
                voicefilesProb.append(prob)


        # save predicted voice file into voice directory
        cperrorfiles(voicefiles,voicefilesProb,'voice','../../data/recordTest/predictVoice')
        # save predicted noise file into noise directory
        cperrorfiles(noisefiles,noisefilesProb,'noise','../../data/recordTest/predictNoise') 
        
        # copy noise files into voice files directory, so that easier to regenerate a complete audio sereies.
        cpfiles('../../data/recordTest/predictNoise','../../data/recordTest/predictVoice')

    return classify (outputVal,filesTrack,trackingMap,threshold)




if __name__ == '__main__':
    modelNames=sys.argv[1]
    filename=sys.argv[2]
    threshold=float(sys.argv[3]) 
 
    wnd=50
    targetdb=-26
    targetDir='../../data/recordTest/'

    # object used for processing real record audio
    realdataprocessor=real_sampledDataprocessing.RealDataProcessor('./','./',wnd,targetdb)

    # map between normalized audio and original audio
    wavtrackingMap=realdataprocessor.processingfile(filename,targetDir+'testNoisyData15dBNorm/',targetDir+'testNoise15dB/','../../data/origwavDir/')



    testPercentage=1.0

    # model(s) to be tested
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
     
        # predict voice and noise, and compute the time used for forwarding part
        forwardingTime(model,testx[:int(float(testPercentage)*testx.shape[0])],testy[:int(float(testPercentage)*testx.shape[0])],filetrack[:int(float(testPercentage)*testx.shape[0])],wavtrackingMap,threshold)
        
        # regenerate audio, recognized to be noise part will be replaced by silent segment
        mixingSegments('../../data/recordTest/predictVoice/','voice')
        

