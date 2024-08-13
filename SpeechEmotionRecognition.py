import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import save_model

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras import regularizers

import keras  # High-level neural networks API
#from tensorflow.keras.utils import to_categorical  # Utility for one-hot encoding
from keras.models import Sequential  # Sequential model for stacking layers
from keras.layers import *  # Different layers for building neural networks

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
        if int(part[2]) == 2:
            continue
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
#Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df.Emotions.replace({1: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)

print(Ravdess_df)

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

##################################     SAVEE
savee_directory_list = os.listdir(Savee)

file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele=='a':
        file_emotion.append('angry')
    elif ele=='d':
        file_emotion.append('disgust')
    elif ele=='f':
        file_emotion.append('fear')
    elif ele=='h':
        file_emotion.append('happy')
    elif ele=='n':
        file_emotion.append('neutral')
    elif ele=='sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)
#print(Savee_df.head())
#
###  main dataframe
#data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
data_path = pd.concat([Ravdess_df,Savee_df ,Tess_df], axis = 0)
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
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)
    mel = librosa.power_to_db(mel_spec, ref=np.max)

    mfcc = np.resize(mfcc, (128, mfcc.shape[1]))
    mfcc = np.expand_dims(mfcc, axis=-1)
    mel = np.expand_dims(mel, axis=-1)
    # Stack the features along the channel dimension
    combined_features = np.concatenate((mfcc, mel), axis=-1)

    return combined_features

def padFeature(features, shape = None): #padding function the make shape equal
    if shape is None:
        max_length = max(feature.shape[1] for feature in features)
    else:
        max_length = shape
    for i, feature in enumerate(features):
        current_length = feature.shape[1]
        if current_length < max_length:
            temp = np.zeros((feature.shape[0], max_length, feature.shape[2]))
            temp[:, :current_length,:] = feature
            features[i] = temp

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=3, offset=0.6, sr=16000)
    
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    ress = []
    ress.append(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    ress.append(res2)

    
    # data with stretching and pitching
    #new_data = stretch(data)
    data_stretch_pitch = pitch(data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    ress.append(res3)
    
    return ress

def get_clean_features(path):
    data, sample_rate = librosa.load(path, duration=3, offset=0.6, sr=16000)
    
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    return result

def applyScaler(data):
    mfcc_features = np.array(data)[..., 0]
    mel_features = np.array(data)[..., 1]
    # Reshape for scaling
    mfcc_features_2d = mfcc_features.reshape(-1, 128 * mfcc_features.shape[2])
    mel_features_2d = mel_features.reshape(-1, 128 * mel_features.shape[2])

    # Scale each channel separately
    scaler_mfcc = StandardScaler()
    scaler_mel = StandardScaler()

    mfcc_scaled = scaler_mfcc.fit_transform(mfcc_features_2d)
    mel_scaled = scaler_mel.fit_transform(mel_features_2d)

    # Reshape back to 3D
    mfcc_scaled = mfcc_scaled.reshape(-1, 128, mfcc_features.shape[2])
    mel_scaled = mel_scaled.reshape(-1, 128, mel_features.shape[2])

    # Combine the channels back into a multi-channel format
    data_scaled = np.stack((mfcc_scaled, mel_scaled), axis=-1)

    return data_scaled

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
    segment_length = round(seg_duration * sr)
    overlap_length = round(overlap_duration * sr)
    step_size = segment_length - overlap_length
    
    segments = []
    start = 0
    
    while start < len(data):
        end = min(start + segment_length, len(data))
        segments.append(data[start:end])
        start += step_size
    
    return segments
def get_features_segments(segments, sr): 
    features = []
    for seg in segments:
        seg_features = np.array(extract_features(seg, sr))
        features.append(seg_features)
    #padFeature(features)
    return features
def combine_predictions(predictions):
    predictions = predictions.flatten()
    unique_classes, counts = np.unique(predictions, return_counts=True)
    most_frequent_class = unique_classes[np.argmax(counts)]
    return most_frequent_class

if __name__ == "__main__":
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
    ## Iterate over testing data paths and emotions
    #for path, emotion in zip(test_paths, test_emotions):
    #    # Extract features from the original audio sample
    #    features = get_clean_features(path)
    #    X_test.append(features)
    #    Y_test.append(emotion)
    #    
    #
    #padFeature(X_train)
    #padFeature(X_test)
    #    
    #np.savez('features.npz', X_train=X_train, X_test=X_test, Y_train = Y_train, Y_test = Y_test)
    
    
    # To load it back
    print("########## THIS IS PRELOADED DATA, IF DATA WAS CHANGED NEED TO RUN FEATURE EXTRACTION AGAIN ##########")
    X_train = np.load('features.npz')['X_train']
    X_test = np.load('features.npz')['X_test']
    Y_train = np.load('features.npz')['Y_train']
    Y_test = np.load('features.npz')['Y_test']
    
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(np.array(Y_train).reshape(-1,1)).toarray()
    y_test = encoder.fit_transform(np.array(Y_test).reshape(-1,1)).toarray()
    
    
    
    X_train_scaled = applyScaler(X_train)
    X_test_scaled = applyScaler(X_test)    
    
    
    # Creating Validation data set
    x_train, x_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=0, shuffle=False)
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Second Cional Block
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    ## Third Coonal Block
    #model.add(Conv2D(128, (3, 3), activation='relu'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.25))
    
    # Fully Co Block
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    #model.add(Dense(32, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    
    # Output L
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    
    
    model.summary()
    
    
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=1, patience=2, min_lr=0.0000001)
    history=model.fit(x_train, y_train, batch_size=128, epochs=15, validation_data=(x_val, y_val), callbacks=[rlrp])
    #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=25, shuffle=True)
    
    
    print("Accuracy of our model on test data : " , model.evaluate(X_test_scaled,y_test)[1]*100 , "%")
    
    
    pred_test = model.predict(X_test_scaled)
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
    
    ##saving the model
    #model.save('SER Model.keras')
    
    long_angry = "DataSets/mad.mp3"
    data, sample_rate = librosa.load(long_angry,offset=0.6, sr=16000)
    segs = split_audio(data, sample_rate, 3)
    seg_features = get_features_segments(segs, sample_rate)
    padFeature(seg_features, x_train.shape[2])
    seg_features = applyScaler(seg_features)
    print("Check segmentation prediction:\n")
    predictions = model.predict(np.array(seg_features))
    combined_prediction = combine_predictions(encoder.inverse_transform(predictions))
    print("Predictions ang are: " + str(encoder.inverse_transform(predictions)))
    print("Combined prediction is: " + str(combined_prediction))
    


