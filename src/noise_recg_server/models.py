# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import torch
import torch
import torch.utils.data as Data
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }

class CNNAudio(nn.Module):
    def __init__(self,model_setting,classN,dropoutP=0.5):
        super(CNNAudio,self).__init__()
        
        self.classN=classN
        self.model_setting=model_setting
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(20,8)
            ),
            nn.ReLU()
            )
        self.dropout1=nn.Dropout(dropoutP)
        
        self.maxpool1=nn.MaxPool2d(2,stride=2,padding=1)
        
        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(10,4)
            ),
            nn.ReLU()
        )
        self.dropout2=nn.Dropout(dropoutP)
        
        self.FC=nn.Linear(5504,self.classN)
        
        
    def forwarding(self,x,isTrain=True):
        x=x.view(-1,
                 1,
#                  self.model_setting['spectrogram_length'],
                 self.model_setting['dct_coefficient_count'],
                self.model_setting['spectrogram_length']
                )
        x=self.conv1(x)
        if isTrain:
            x=self.dropout1(x)
        x=self.maxpool1(x)
        
        x=self.conv2(x)
        if isTrain:
            x=self.dropout2(x)
            
        x=x.view(x.size(0),-1)
        x=self.FC(x)
        return x
        
class CNNAudioLowLatency(nn.Module):
    def __init__(self, model_settings,classN,dropoutP=0.5):
        super(CNNAudioLowLatency,self).__init__()
        self.model_settings=model_settings
        self.classN=classN
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=186,
                stride=1,
                kernel_size=(
                            self.model_settings['spectrogram_length'],
                            8
                            )
                    ),
            nn.ReLU()
            
        )
        
        self.dropout1=nn.Dropout(dropoutP)
        
        self.FC1=nn.Linear(186*33,128)
        
        self.dropout2=nn.Dropout(dropoutP)
        
        self.FC2=nn.Linear(128,128)
        
        self.dropout3=nn.Dropout(dropoutP)
        
        self.FC3=nn.Linear(128,self.classN)
        
    def forwarding(self,x,isTrain=True):
        x=x.view(-1,
                 1,
                self.model_settings['spectrogram_length'],
                self.model_settings['dct_coefficient_count']
                )
        x=self.conv1(x)
        if isTrain:
            x=self.dropout1(x)
        
        x=x.view(x.size(0),-1)
#         print (x.size())
        
        x=self.FC1(x)
        if isTrain:
            x=self.dropout2(x)
        x=self.FC2(x)
        if isTrain:
            x=self.dropout3(x)
        x=self.FC3(x)
        
        return x    

class CNNAudioMobile(nn.Module):
    def __init__(self,model_setting,classN,dropoutP=0.5):
        super(CNNAudioMobile,self).__init__()
        self.classN=classN
        self.model_setting=model_setting
#         self.conv1=nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=64,
#                 kernel_size=(20,8)
#             ),
#             nn.ReLU()
#             )
        self.conv1=nn.Sequential(
            
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(20,8)
            ),
            
            nn.BatchNorm2d(1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
                
                
            
        )
        self.dropout1=nn.Dropout(dropoutP)
        
        self.maxpool1=nn.MaxPool2d(2,stride=2,padding=1)
        
        
        
        
        self.conv2=nn.Sequential(
            
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=(10,4)
            ),
            
            nn.BatchNorm2d(1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
                
                
            
        )
        # self.conv2=nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=64,
        #         out_channels=64,
        #         kernel_size=(10,4)
        #     ),
        #     nn.ReLU()
        # )
        self.dropout2=nn.Dropout(dropoutP)
        
        self.FC=nn.Linear(5504,self.classN)
        
        
    def forwarding(self,x,isTrain=True):
        x=x.view(-1,
                 1,
#                  self.model_setting['spectrogram_length'],
                 self.model_setting['dct_coefficient_count'],
                self.model_setting['spectrogram_length']
                )
        x=self.conv1(x)
        if isTrain:
            x=self.dropout1(x)
        x=self.maxpool1(x)
        
        x=self.conv2(x)
        if isTrain:
            x=self.dropout2(x)
            
        x=x.view(x.size(0),-1)
        x=self.FC(x)
        return x
        
class CNNAudioLowLatencyMobile(nn.Module):
    def __init__(self, model_settings,classN,dropoutP=0.5):
        super(CNNAudioLowLatencyMobile,self).__init__()
        self.model_settings=model_settings
        self.classN=classN

        self.conv1=nn.Sequential(
            
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(
                            self.model_settings['spectrogram_length'],
                            8
                            )
            ),
            
            nn.BatchNorm2d(1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=1,
                      out_channels=186,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(186),
            nn.ReLU()
                
                
            
        )
        
        self.dropout1=nn.Dropout(dropoutP)
        
        self.FC1=nn.Linear(186*33,128)
        
        self.dropout2=nn.Dropout(dropoutP)
        
        self.FC2=nn.Linear(128,128)
        
        self.dropout3=nn.Dropout(dropoutP)
        
        self.FC3=nn.Linear(128,self.classN)
        
    def forwarding(self,x,isTrain=True):
        x=x.view(-1,
                 1,
                self.model_settings['spectrogram_length'],
                self.model_settings['dct_coefficient_count']
                )
        x=self.conv1(x)
        if isTrain:
            x=self.dropout1(x)
        
        x=x.view(x.size(0),-1)
#         print (x.size())
        
        x=self.FC1(x)
        if isTrain:
            x=self.dropout2(x)
        x=self.FC2(x)
        if isTrain:
            x=self.dropout3(x)
        x=self.FC3(x)
        
        return x

class CNNAudioOneFpool3(nn.Module):
    def __init__(self, model_settings,classN,dropoutP=0.5):
        super(CNNAudioOneFpool3,self).__init__()
        self.model_settings=model_settings
        self.classN=classN
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=54,
                stride=1,
                kernel_size=(
                            self.model_settings['spectrogram_length'],
                            4
                            )
                    ),
            nn.ReLU()
            
            
        )
        
        
        self.dropout1=nn.Dropout(dropoutP)
        
        self.maxpool1=nn.MaxPool2d(kernel_size=(1,3))
        
        self.FC1=nn.Linear(648,32) #648
        
        self.dropout2=nn.Dropout(dropoutP)
        
        self.FC2=nn.Linear(32,128)
        
        self.dropout3=nn.Dropout(dropoutP)
        
        self.FC3=nn.Linear(128,self.classN)
        
       
        
    def forwarding(self,x,isTrain=True):
        x=x.view(-1,
                 1,
                 self.model_settings['spectrogram_length'],
                 self.model_settings['dct_coefficient_count']
                )
        x=self.conv1(x)
        if isTrain:
            x=self.dropout1(x)
        x=self.maxpool1(x)
        x=x.view(x.size(0),-1)
        # print (x.size())
        
        x=self.FC1(x)
        if isTrain:
            x=self.dropout2(x)
        x=self.FC2(x)
        if isTrain:
            x=self.dropout3(x)
        x=self.FC3(x)
        
        return x

class CNNAudioOneFpool3Mobile(nn.Module):
    def __init__(self, model_settings,classN,dropoutP=0.5):
        super(CNNAudioOneFpool3Mobile,self).__init__()
        self.model_settings=model_settings
        self.classN=classN

        self.conv1=nn.Sequential(
            
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(
                            model_settings['spectrogram_length'],
                            8
                            )
            ),
            
            nn.BatchNorm2d(1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=1,
                      out_channels=54,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(54),
            nn.ReLU()
                
                
            
        )
        
        self.dropout1=nn.Dropout(dropoutP)
        
        self.maxpool1=nn.MaxPool2d(kernel_size=(1,3))
        
        self.FC1=nn.Linear(594,32)
        
        self.dropout2=nn.Dropout(dropoutP)
        
        self.FC2=nn.Linear(32,128)
        
        self.dropout3=nn.Dropout(dropoutP)
        
        self.FC3=nn.Linear(128,self.classN)
        
       
        
    def forwarding(self,x,isTrain=True):
        x=x.view(-1,
                 1,
                 self.model_settings['spectrogram_length'],
                 self.model_settings['dct_coefficient_count']
                )
        x=self.conv1(x)
        if isTrain:
            x=self.dropout1(x)
        x=self.maxpool1(x)
        x=x.view(x.size(0),-1)
#         print (x.size())
        
        x=self.FC1(x)
        if isTrain:
            x=self.dropout2(x)
        x=self.FC2(x)
        if isTrain:
            x=self.dropout3(x)
        x=self.FC3(x)
        
        return x

def selectingModel(modelName,model_settings,classN):
  if modelName=='cnn':
    model=CNNAudio(model_settings,classN)
  elif modelName=='cnnMobile':
    model=CNNAudioMobile(model_settings,classN)
  elif modelName=='cnnLowLatency':
    model=CNNAudioLowLatency(model_settings,classN)
  elif modelName=='cnnLowLatencyMobile':
    model=CNNAudioLowLatencyMobile(model_settings,classN)
  elif modelName=='cnnOneFpool3':
    model=CNNAudioOneFpool3(model_settings,classN)
  elif modelName=='cnnOneFpool3Mobile':
    model=CNNAudioOneFpool3Mobile(model_settings,classN)
  else:
    print ('you should select model name from:')
    print ('1. cnn\n2. cnnMobile\n3. cnnLowLatency\n4. cnnLowLatencyMobile\n 5. cnnOneFpool3\n6. cnnOneFpool3Mobile\n')
    return 
  return model







