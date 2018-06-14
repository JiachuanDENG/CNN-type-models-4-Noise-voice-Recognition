
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_dataV2 as input_data
import models
from tensorflow.python.platform import gfile
import pickle as pkl
# FLAGS = None

def shuffleSamples(trainXFull,trainyFull):
  trainXFull=trainXFull.astype('float32')
  trainyFull=trainyFull.astype('int')
  indices = list(range(0,trainXFull.shape[0]))
  np.random.shuffle(indices)
  trainXFull=trainXFull[indices,:]
  trainyFull=trainyFull[indices]
  return trainXFull,trainyFull
def datatypechange(trainXFull,trainyFull):
  trainXFull=trainXFull.astype('float32')
  trainyFull=trainyFull.astype('int')
 
  return trainXFull,trainyFull


def getAllProcessedData(FLAGS):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()
  

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
  print ('^'*50)
  if ('trainingX.npy' in os.listdir(FLAGS.data_dir) ) and \
  ('trainingy.npy' in os.listdir(FLAGS.data_dir) ) and \
  ('valX.npy' in os.listdir(FLAGS.data_dir) )  and \
  ('trainingy.npy' in os.listdir(FLAGS.data_dir) ) and \
  ('testX.npy' in os.listdir(FLAGS.data_dir)) and \
  ('testy.npy' in os.listdir(FLAGS.data_dir)) :
    print ('loading data...')
    trainXFull,trainyFull=np.load(FLAGS.data_dir+'/trainingX.npy'),np.load(FLAGS.data_dir+'/trainingy.npy')
    valXFull,valyFull=np.load(FLAGS.data_dir+'/valX.npy'),np.load(FLAGS.data_dir+'/valy.npy')
    testXFull,testyFull=np.load(FLAGS.data_dir+'/testX.npy'),np.load(FLAGS.data_dir+'/testy.npy')
  
  else:
    print ('constructing data...')
    audio_processor = input_data.AudioProcessor(
         FLAGS.data_dir, 
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)

    
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    # Figure out the learning rates for each training phase. Since it's often
    # effective to have high learning rates at the start of training, followed by
    # lower levels towards the end, the number of steps and learning rates can be
    # specified as comma-separated lists to define the rate at each stage. For
    # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
    # will run 13,000 training loops in total, with a rate of 0.001 for the first
    # 10,000, and 0.0001 for the final 3,000.
    
    

    trainXFull, trainyFull,trainFilesTrack = audio_processor.get_data(
          -1, 0, model_settings,time_shift_samples, 'training', sess)
    valXFull, valyFull,valFilesTrack= (
            audio_processor.get_data(-1, 0, model_settings,0, 'validation', sess))
    testXFull, testyFull,testFilesTrack= (
            audio_processor.get_data(-1, 0, model_settings,0, 'testing', sess))

    trainXFull=np.concatenate((trainXFull,testXFull),axis=0)
    trainyFull=np.concatenate((trainyFull,testyFull),axis=0)

    trainFilesTrack=trainFilesTrack+testFilesTrack

    trainXFull,trainyFull=datatypechange(trainXFull,trainyFull)
    valXFull,valyFull=datatypechange(valXFull,valyFull)
    


    # trainlen=int(0.7*trainXFull.shape[0])
    # trainx,trainy=trainXFull[:trainlen],trainyFull[:trainlen]
    # valx,valy=trainXFull[trainlen:],trainyFull[trainlen:]

    np.save(FLAGS.data_dir+'/trainingX.npy',trainXFull)
    
    np.save(FLAGS.data_dir+'/trainingy.npy',trainyFull)

    np.save(FLAGS.data_dir+'/valX.npy',valXFull)

    np.save(FLAGS.data_dir+'/valy.npy',valyFull)
    # pkl.dump(trainFilesTrack,open(FLAGS.data_dir+'/trainfilesTrack.pkl','wb'))
    
    


  return trainXFull,trainyFull,valXFull,valyFull,model_settings,trainFilesTrack,valFilesTrack
  # return train_fingerprints,train_ground_truth,model_settings
  


def returnData(datadir='../../data/selfbuildDataTest/',wanted_words='testnoisydata,noise'):
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--data_dir',
    type=str,
    default=datadir,
    help="""\
    Where to download the speech training data to.
    """)

  parser.add_argument(
    '--time_shift_ms',
    type=float,
    default=20,    #---- previously 100
    help="""\
    Range to randomly shift the training audio by in time.
    """)
  parser.add_argument(
    '--testing_percentage',
    type=int,
    default=10,
    help='What percentage of wavs to use as a test set.')
  parser.add_argument(
    '--validation_percentage',
    type=int,
    default=10,
    help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
    '--sample_rate',
    type=int,
    default=16000,
    help='Expected sample rate of the wavs',)
  parser.add_argument(
    '--clip_duration_ms',
    type=int,
    default=250,
    help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
    '--window_size_ms',
    type=float,
    default=30.0,
    help='How long each spectrogram timeslice is.',)
  parser.add_argument(
    '--window_stride_ms',
    type=float,
    default=10.0,
    help='How far to move in time between spectogram timeslices.',)
  parser.add_argument(
    '--dct_coefficient_count',
    type=int,
    default=40,
    help='How many bins to use for the MFCC fingerprint',)

  parser.add_argument(
    '--wanted_words',
    type=str,
    default=wanted_words,
    help='Words to use (others will be added to an unknown label)',)

  FLAGS, unparsed = parser.parse_known_args()
  # print ('hello',FLAGS)
  # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  trainx,trainy,valx,valy,model_settings,trainFilesTrack,valFilesTrack=getAllProcessedData(FLAGS)
  return  trainx,trainy,valx,valy,model_settings,trainFilesTrack,valFilesTrack
