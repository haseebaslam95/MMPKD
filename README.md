# Multimodal Priviliged Knowledge Distillation (M2PKD)



----------------
----------------

# 0. Table of Contents
#### This manual has has two parts
### 1- Installation
Provides installation guide.
### 2- Running the code
Details on how to run the code.

----------------
# 1. Installation
##### 1.2 Pytorch(1.10.0) which can be installed with details provided here: https://pytorch.org/get-started/locally/
For most users, ```pip3 install torch torchvision``` should work.
If you are using Anaconda, ```conda install pytorch torchvision -c pytorch``` should work. 

Create a virtual environment using Conda or Virtualenv and use the requirement.txt file to install all the dependencies


# 2. Running The code
### 2.1 Important Files In codebase: 
#### 2.1.1 `models_kd.py` Creates and defines all neural networks.
#### 2.2.2 `mmpkd.py` The main code. Trains the teacher model and student models based on the function call.	
#### 2.1.3 `data.py` Creates datasets and dataloaders to be used by mmpkd file.
#### 2.1.4 `audio_extract.py` This file creates the audio file segments synchronized with annotations and visual frames
#### 2.1.5 `audio_melspec.py` Reads the audio segments and creates log-mel spectrograms.
#### 2.1.6 `face_detector.py` Used to extract faces in raw videos and saves extracted images. Use face_detector_frameRetention.py to use Frame Retention.
#### 2.1.7 'config.py' Use this file to set the paths and hyperparameters for training 
#### 2.1.8 'plot_ccc.py' Use this file to get the final CCC value. The required post-processing steps are provided in this file.

### 2.2 Configs
#### 2.2.1 Training Settings
These settings are needed during training
###### 2.2.1.1 epochs
Number of training epochs.
###### 2.2.1.2 batch-size
mini batch size used in training.
###### 2.2.1.3 learning-rate
Initial learning rate for SGD optimizer. Depending on models in might be changed during training.



## 2.3 Running
The first step is to create the folder heirarchy.

-Cropped_Aligned
--dev_1
--dev_2
.
.
.
All 9 videos 

-specs
--dev_1
--dev_2
.
.
.
All 9 videos 

-Annotations
--Train_Set
--Val_Set
 
1) Run the 'face_detector.py' file to extract facial frames from raw videos and save them in the 'Cropped _Aligned' folder.
2) Run the 'audio_extract.py' file to extract audio segments from raw audios and save them into separate folders.
3) Run the 'audio_melspec.py' file to generate log-mel spectrograms of synchronized audio segments and save them in 'specs' folder.
4) You may need to create csv files having 'image path', 'arousal', and 'valence' as header. Use the Goldstandard values to generate these csv files.
You have to run the code using `mmpkd.py`. The config file will automatically set all the parameters for running the code.
 

------


### 3.1 (Baseline Lower Bound)
You should use the 'Visual Training Call' in the 'mmpkd.py' file to get the results for visual only baseline.

### 3.2 (Multimodal Upper Bound)
You should use the 'Teacher Training Call' in the 'mmpkd.py' file to get the results for Multimodal teacher for upperbound.

### 3.3 (Visual Student)
You should use the 'Student Training Call' in the 'mmpkd.py' file to get the results for Unimodal student that uses MMPKD.



### Reference

```
