from typing import Counter
import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis

# to play the audio files
from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
#from keras.utils import np_utils
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import keras  # High-level neural networks API
#from tensorflow.keras.utils import to_categorical  # Utility for one-hot encoding
from keras.models import Sequential  # Sequential model for stacking layers
from keras.layers import *  # Different layers for building neural networks
#from keras.optimizers import rmsprop  # Optimizer for training the model

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

##############################################################################################################################################################

print("SUCCESS - loading libraries")

### load data
Ravdess = "DataSets/RAVDESS/audio_speech_actors_01-24/"
Crema = "DataSets/CREMA-D/AudioWAV/"
Tess = "DataSets/TESS/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data/"
Savee = "DataSets/SAVEE/ALL/"
print("SUCCESS - loading data")

##############################################################################################################################################################
### dataframe

##################################     RAVDESS 
ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
print(Ravdess_df.head())

##################################     CREMA 
#crema_directory_list = os.listdir(Crema)
#
#file_emotion = []
#file_path = []
#
#for file in crema_directory_list:
#    # storing file paths
#    file_path.append(Crema + file)
#    # storing file emotions
#    part=file.split('_')
#    if part[2] == 'SAD':
#        file_emotion.append('sad')
#    elif part[2] == 'ANG':
#        file_emotion.append('angry')
#    elif part[2] == 'DIS':
#        file_emotion.append('disgust')
#    elif part[2] == 'FEA':
#        file_emotion.append('fear')
#    elif part[2] == 'HAP':
#        file_emotion.append('happy')
#    elif part[2] == 'NEU':
#        file_emotion.append('neutral')
#    else:
#        file_emotion.append('Unknown')
#        
## dataframe for emotion of files
#emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
#
## dataframe for path of files.
#path_df = pd.DataFrame(file_path, columns=['Path'])
#Crema_df = pd.concat([emotion_df, path_df], axis=1)
#print(Crema_df.head())
#
#
##################################     TESS
tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part=='ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
print(Tess_df.head()) 

###################################     SAVEE
#savee_directory_list = os.listdir(Savee)
#
#file_emotion = []
#file_path = []
#
#for file in savee_directory_list:
#    file_path.append(Savee + file)
#    part = file.split('_')[1]
#    ele = part[:-6]
#    if ele=='a':
#        file_emotion.append('angry')
#    elif ele=='d':
#        file_emotion.append('disgust')
#    elif ele=='f':
#        file_emotion.append('fear')
#    elif ele=='h':
#        file_emotion.append('happy')
#    elif ele=='n':
#        file_emotion.append('neutral')
#    elif ele=='sa':
#        file_emotion.append('sad')
#    else:
#        file_emotion.append('surprise')
#        
## dataframe for emotion of files
#emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
#
## dataframe for path of files.
#path_df = pd.DataFrame(file_path, columns=['Path'])
#Savee_df = pd.concat([emotion_df, path_df], axis=1)
#print(Savee_df.head())
#
###  main dataframe
#data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
data_path = pd.concat([Ravdess_df, Tess_df], axis = 0)
data_path.to_csv("data_path.csv",index=False)
print(data_path.head())
print("########### FINISHED PREPPIN DATA #############")

#sns.countplot(x='Emotions', data=data_path, palette='Set3')  # Specify 'Set3' palette for different colors
#plt.title('Count of Emotions', size=16)
#plt.xlabel('Emotions', size=12)
#plt.ylabel('Count', size=12)
#sns.despine(top=True, right=True, left=False, bottom=False)
#
#plt.xticks(rotation=45)  # Rotate x-axis labels if needed
#plt.tight_layout()  # Adjust layout to prevent clipping of labels
#plt.show()


##################################################################### augmentation and feature extraction functions
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data,rate = rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data,sample_rate):
    result = np.array([])
    zcr = librosa.feature.zero_crossing_rate(y=data).T
    stft = np.abs(librosa.stft(data))
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate).T
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T
    rms = librosa.feature.rms(y=data).T
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate).T
    result = np.hstack((result,
        get_stats(zcr),
        get_stats(chroma_stft),
        get_stats(mfcc),
        get_stats(rms),
        get_stats(mel)
        ))
    return result

def get_stats(feature):
    result = np.array([])
    mean = np.mean(feature, axis=0)
    median = np.median(feature, axis=0)
    std = np.std(feature, axis=0)
    energy = np.sum(np.square(feature), axis=0)
    ptp = np.ptp(feature, axis=0)
    #skews = skew(feature, axis=0)
    #kurt = kurtosis(feature,axis=0)
    #result = np.array([mean,median, std, energy, ptp, skews, kurt])
    result=np.hstack((result, mean)) # stacking horizontally
    result=np.hstack((result, median)) # stacking horizontally
    result=np.hstack((result, std)) # stacking horizontally
    result=np.hstack((result, energy)) # stacking horizontally
    result=np.hstack((result, ptp)) # stacking horizontally
    #result=np.hstack((result, skews)) # stacking horizontally
    #result=np.hstack((result, kurt)) # stacking horizontally
    return result

    

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2, offset=0.6, sr=16000)
    
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

def get_clean_features(path):
    data, sample_rate = librosa.load(path, duration=2, offset=0.6, sr=16000)
    
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    return result

def split_audio(data, sr, seg_duration):
    ##############       NO OVERLAP            #############
    #seg_len = seg_duration * sr  # making sure segment is indeed 2 secs
    #num_seg = int(np.ceil(len(data)/seg_len))
    #segments = []
    #for i in range(num_seg):
    #    start = i*seg_len
    #    end = min((i+1) * seg_len, len(data))
    #    segments.append(data[start:end])
    #return segments
    #############        WITH OVERLAP          ############
    overlap_duration = 0.5
    segment_length = seg_duration * sr
    overlap_length = round(overlap_duration * sr)
    step_size = segment_length - overlap_length
    
    segments = []
    start = 0
    
    while start < len(data):
        end = min(start + segment_length, len(data))
        segments.append(data[start:end])
        start += step_size
    
    return segments
def get_features_segments(segments, sr): ##need to change feature func(takes path need data)
    features = []
    for seg in segments:
        seg_features = np.array(extract_features(seg, sr))
        features.append(seg_features)
    return np.array(features)
def classify_segments(seg_features, model):
    seg_features = np.expand_dims(seg_features, axis=2)
    predictions = model.predict(seg_features)
    return predictions
def combine_predictions(predictions):
    predictions = predictions.flatten()
    unique_classes, counts = np.unique(predictions, return_counts=True)
    most_frequent_class = unique_classes[np.argmax(counts)]
    return most_frequent_class


#X_train, Y_train = [], []
#X_test, Y_test = [], []
#
## Split the data into training and testing sets
#train_paths, test_paths, train_emotions, test_emotions = train_test_split(data_path.Path, data_path.Emotions, random_state=0, shuffle=True)
## Iterate over training data paths and emotions
#for path, emotion in zip(train_paths, train_emotions):
#    # Extract features from the original audio sample
#    features = get_features(path)
#    for ele in features:
#        X_train.append(ele)
#        Y_train.append(emotion)
#    #features = get_clean_features(path)
#    #X_train.append(features)
#    #Y_train.append(emotion)
## Iterate over testing data paths and emotions
#for path, emotion in zip(test_paths, test_emotions):
#    # Extract features from the original audio sample
#    features = get_clean_features(path)
#    X_test.append(features)
#    Y_test.append(emotion)
#    
#
#    
#TrainFeatures = pd.DataFrame(X_train)
#TrainFeatures['labels'] = Y_train
#TrainFeatures.to_csv('train_features.csv', index=False)
#
#TestFeatures = pd.DataFrame(X_test)
#TestFeatures['labels'] = Y_test
#TestFeatures.to_csv('test_features.csv', index=False)

TrainFeatures = pd.read_csv('train_features.csv')
TestFeatures = pd.read_csv('test_features.csv')
print("#\nTHIS IS PRE PREPARED DATA FROM CSV, CHECK CODE\n#")

X_train = TrainFeatures.iloc[: ,:-1].values
Y_train = TrainFeatures['labels'].values
X_test = TestFeatures.iloc[: ,:-1].values
Y_test = TestFeatures['labels'].values


encoder = OneHotEncoder()
y_train = encoder.fit_transform(np.array(Y_train).reshape(-1,1)).toarray()
y_test = encoder.fit_transform(np.array(Y_test).reshape(-1,1)).toarray()

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)
# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)


# Creating Validation data set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0, shuffle=True)


#model=Sequential()
#model.add(Conv1D(32, kernel_size=5, strides=1, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
#
#model.add(Conv1D(32, kernel_size=5, strides=1, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
#
#model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
#
#model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
#model.add(Dropout(0.2))
#
#model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
#model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
#
#model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
#
#model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
#
#model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
#
#model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
#
#model.add(LSTM(256, return_sequences=True))
#model.add(LSTM(128))
##model.add(LSTM(128))
##model.add(LSTM(64))
#
#
##model.add(Flatten())
#
#
#
#model.add(Dense(units=256, activation='relu'))
#
#model.add(Dense(units=128, activation='relu'))
#
#model.add(Dense(units=64, activation='relu'))
#
#model.add(Dense(units=32, activation='relu'))
#model.add(Dropout(0.3))
#
#model.add(Dense(units=8, activation='softmax'))
#model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
#
#model.summary()

model=Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=8, activation='softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()


rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history=model.fit(x_train, y_train, batch_size=256, epochs=50, validation_data=(x_val, y_val), callbacks=[rlrp])
#history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=25, shuffle=True)


print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")


pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)
y_test = encoder.inverse_transform(y_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()
print(classification_report(y_test, y_pred))

long_angry = "DataSets/mad.mp3"
data, sample_rate = librosa.load(long_angry)
segs = split_audio(data, sample_rate, 2)
seg_features = get_features_segments(segs, sample_rate)
print("Check segmentation prediction:\n")
predictions = classify_segments(seg_features, model)
combined_prediction = combine_predictions(encoder.inverse_transform(predictions))
print("Predictions ang are: " + str(encoder.inverse_transform(predictions)))
print("Combined prediction is: " + str(combined_prediction))



