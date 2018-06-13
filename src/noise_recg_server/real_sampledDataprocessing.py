
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
		t1,t2=0,len(audio)//500
		while t1<t2:
			newAudios.append(audio[t1*500:(t1+1)*500])
			t1+=1
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

	def saveAudios(self,audios,targetDir1,targetDir2):
		if not os.path.exists(targetDir1):
			os.system('mkdir {}'.format(targetDir1))
		if not os.path.exists(targetDir2):
			os.system('mkdir {}'.format(targetDir2))
		for i,audio in enumerate(audios):
			targetDir=random.choice([targetDir1,targetDir2])
			audio.export(targetDir+str(i)+'.wav',format='wav')

	

	def processingfile(self,file,targetDir1,targetDir2):
		allAudios=[]
		self.saveAudios( [self.normalize(audio) for audio in self.sliceWav(file)], targetDir1,targetDir2)

		



	















