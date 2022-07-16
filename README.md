# CNN-type-models-4-Noise-voice-Recognition
Real-time Noise-voice recognition task with CNN-type models

## Dataset:
**Noise**: self labeled noises from webist freesound. Including several common noises appeared in online meeting: crowd, dog, keyboard, lawnmower, mouse click, passing car</br>
**Voice**: additive noise type noisy voice generated by mixing noise above with clean voice sampled from clean dataset Aurora4

## Models:
CNN-type pytorch implemented models in paper https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43969.pdf</br>
</br>
With [MobileNet](https://arxiv.org/pdf/1704.04861.pdf) Tricks applied, so that models may be more reasonable applied in real-time tasks. (Still needs more data to let this trick work)

## Packages Required:
**tensorflow** version >= 1.8.0 some useful audio processing tool are used for audio data preprocessing</br>
**pytorch** used for model training</br>
**pydub** used for audio preprocessing (audio segmentation + normalization) 

## Run codes:
**training**: python3 train.py [model name] </br>
**testing**: python3 test.py [model name] [percentage of whole test data want to use]</br>
**testing on real record wav**: python3 testOnRealRecord.py [model name] [wav file path]


