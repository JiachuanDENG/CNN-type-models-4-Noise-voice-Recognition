
import os
from pydub import AudioSegment
import sys
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
		t1,t2=0,len(audio)//1000
		while t1<t2:
			newAudios.append(audio[t1*1000:(t1+1)*1000])
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

	def saveAudios(self,audios,targetDir):
		if not os.path.exists(targetDir):
			os.system('mkdir {}'.format(targetDir))
		for i,audio in enumerate(audios):
			audio.export(targetDir+str(i)+'.wav',format='wav')

	

	def processingfile(self,file,targetDir):
		allAudios=[]
		self.saveAudios( [self.normalize(audio) for audio in self.sliceWav(file)], targetDir)

		

if __name__ == '__main__':
	filename=sys.argv[1]
	wnd=int(sys.argv[2])
	targetdb=int(sys.argv[3])
	targetDir=sys.argv[4]

	realdataprocessor=RealDataProcessor('./','./',wnd,targetdb)

	realdataprocessor.processingfile(filename,targetDir)


	















