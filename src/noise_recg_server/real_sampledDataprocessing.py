
import os
from pydub import AudioSegment
import sys
import random
class RealDataProcessor(object):
	def __init__(self,realVoiceDir,realNoiseDir,wnd,targetdb):
		self.realVoiceDir=realVoiceDir
		self.realNoiseDir=realNoiseDir
		self.wnd=wnd
		self.targetdb=targetdb

	def getAllFiles(self,dir):
		return [dir+f for f in os.listdir(dir) if '.wav' in f]

	def sliceWav(self,wavfile):
		newAudios=[]
		audio=AudioSegment.from_wav(wavfile)
		print('audio length: ',len(audio))
		t1,t2=0,len(audio)//250

		while t1<t2:
			newAudios.append(audio[t1*250:(t1+1)*250])
			t1+=1
		print ('number of segments: {}, should have:{}'.format(len(newAudios),t1))
		return newAudios

	def caldbs(self,audio):
	    dbs=[]
	    wnd=self.wnd
	    for i in range(len(audio)//wnd):
	        dbs.append(audio[i*wnd:(i+1)*wnd].dBFS)
	    return dbs

	def normalize(self,audio):
		normalizedAudio=AudioSegment.empty()
		maxdb=max(self.caldbs(audio))
		gaindb=self.targetdb-maxdb
		for i in range(len(audio)//self.wnd):
			normalizedAudio+=audio[i*self.wnd:(i+1)*self.wnd].apply_gain(gaindb)
		return normalizedAudio

	# since audio need to be normalize before sending to model,
	# normalized audio is not friendly for testing manually
	# this function will return a map between original audio and normalized audio
	def saveAudios(self,audios,origAudios,targetDir1,targetDir2,origAudioDir):
		trackingMap=dict()
		if not os.path.exists(targetDir1):
			os.system('mkdir {}'.format(targetDir1))
		if not os.path.exists(targetDir2):
			os.system('mkdir {}'.format(targetDir2))
		if not os.path.exists(origAudioDir):
			os.system('mkdir {}'.format(origAudioDir))
		for i,audio in enumerate(audios):
			targetDir=random.choice([targetDir1,targetDir2])
			audio.export(targetDir+str(i)+'.wav',format='wav')
			origAudios[i].export(origAudioDir+str(i*250/1000)+'.wav',format='wav')
			trackingMap[targetDir+str(i)+'.wav']=origAudioDir+str(i*250/1000)+'.wav'
		return trackingMap


	

	def processingfile(self,file,targetDir1,targetDir2,origAudioDir):
		allAudios=[]
		origAudios=self.sliceWav(file)
		return self.saveAudios( [self.normalize(audio) for audio in origAudios],  origAudios,targetDir1,targetDir2,origAudioDir)

		



	















