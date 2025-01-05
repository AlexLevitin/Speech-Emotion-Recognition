import pandas as pd
import numpy as np
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
from sklearn.utils.class_weight import compute_class_weight


# to play the audio files

from keras.models import Sequential, load_model
from keras.regularizers import l2, l1

from keras.models import Sequential  # Sequential model for stacking layers
from keras.layers import *  # Different layers for building neural networks

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

##############################################################################################################################################################

print("SUCCESS - loading libraries")


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

def extract_features(data, sample_rate):
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)
    mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    #mfcc = np.resize(mfcc, (128, mfcc.shape[1]))
    mfcc = np.pad(mfcc, ((0, 88), (0, 0)), mode='constant')
    mfcc = np.expand_dims(mfcc, axis=-1)
    mel = np.expand_dims(mel, axis=-1)
    combined_features = np.concatenate((mfcc, mel), axis=1)
    return combined_features

def extract_features_multi(data,sample_rate):
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)
    mel = librosa.power_to_db(mel_spec, ref=np.max)

    #mfcc = np.resize(mfcc, (128, mfcc.shape[1]))
    mfcc = np.pad(mfcc, ((0, 88), (0, 0)), mode='constant')
    mfcc = np.expand_dims(mfcc, axis=-1)
    mel = np.expand_dims(mel, axis=-1)
    # stack the features along the channel dimension
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
        else:
            features[i] = feature[:, :max_length, :]

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2, offset=0.6, sr=16000)
    
    # without augmentation
    res1 = extract_features_multi(data,sample_rate)
    result = np.array(res1)
    ress = []
    ress.append(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features_multi(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    ress.append(res2)

    
    # data with stretching and pitching
    #new_data = stretch(data)
    data_stretch_pitch = pitch(data, sample_rate)
    res3 = extract_features_multi(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    ress.append(res3)
    
    return ress

def get_clean_features(path):
    data, sample_rate = librosa.load(path, duration=2, offset=0.6, sr=16000)
    
    # without augmentation
    res1 = extract_features_multi(data,sample_rate)
    result = np.array(res1)
    return result

def applyScalerSingle(data):
    data = np.array(data)
    data_2d = data.reshape(-1, data.shape[1] * data.shape[2])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_2d)
    data_scaled = data_scaled.reshape(-1, data.shape[1], data.shape[2])

    return data_scaled

def applyScaler(data):
    mfcc_features = np.array(data)[..., 0]
    mel_features = np.array(data)[..., 1]
    mfcc_features_2d = mfcc_features.reshape(-1, 128 * mfcc_features.shape[2])
    mel_features_2d = mel_features.reshape(-1, 128 * mel_features.shape[2])

    scaler_mfcc = StandardScaler()
    scaler_mel = StandardScaler()

    mfcc_scaled = scaler_mfcc.fit_transform(mfcc_features_2d)
    mel_scaled = scaler_mel.fit_transform(mel_features_2d)

    mfcc_scaled = mfcc_scaled.reshape(-1, 128, mfcc_features.shape[2])
    mel_scaled = mel_scaled.reshape(-1, 128, mel_features.shape[2])

    data_scaled = np.stack((mfcc_scaled, mel_scaled), axis=-1)

    return data_scaled

def split_audio(data, sr, seg_duration, overlap_duration):
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
        seg_features = np.array(extract_features_multi(seg, sr))
        features.append(seg_features)
    #padFeature(features)
    return features
def combine_predictions(predictions):
    predictions = predictions.flatten()
    unique_classes, counts = np.unique(predictions, return_counts=True)
    most_frequent_class = unique_classes[np.argmax(counts)]
    return most_frequent_class
def show_graphs(test, pred):
        cm = confusion_matrix(test, pred)
        plt.figure(figsize = (12, 10))
        cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
        sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
        plt.title('Confusion Matrix', size=20)
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('Actual Labels', size=14)
        plt.show()
        print(classification_report(test, pred))
        
        loss_values = history.history['loss']
        val_loss_values = history.history['val_loss']
        accuracy_values = history.history['accuracy']
        val_accuracy_values = history.history['val_accuracy']
        max_val_accuracy = max(val_accuracy_values)
    
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(loss_values, label='Training Loss')
        plt.plot(val_loss_values, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
        plt.subplot(2,1,2)
        plt.plot(accuracy_values, label='Training Accuracy')
        plt.plot(val_accuracy_values, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.annotate(f'Max Val Acc: {max_val_accuracy:.4f}', 
                 xy=(len(val_accuracy_values)-1, max_val_accuracy),
                 xytext=(len(val_accuracy_values)-1, max_val_accuracy - 0.1),
                 arrowprops=dict(facecolor='red', arrowstyle="->"))
    
        plt.show()

if __name__ == "__main__":
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    X_val, Y_val = [], []
    data_path = pd.read_csv('data_path.csv')
    print("\n#####Current data is read from CSV, if changes were made update the csv file#####\n\n")
    # Split the data into training and testing sets
    train_paths, test_paths, train_emotions, test_emotions = train_test_split(data_path.Path, data_path.Emotions, random_state=0, shuffle=True)
    train_paths, x_val_paths, train_emotions, y_val_emotions = train_test_split(train_paths, train_emotions, test_size=0.2, random_state=0, shuffle=True)
    for path, emotion in zip(train_paths, train_emotions):
        features = get_features(path)
        for ele in features:
            X_train.append(ele)
            Y_train.append(emotion)
    for path, emotion in zip(test_paths, test_emotions):
        features = get_clean_features(path)
        X_test.append(features)
        Y_test.append(emotion)
    for path, emotion in zip(x_val_paths, y_val_emotions):
        features = get_features(path)
        for ele in features:
            X_val.append(ele)
            Y_val.append(emotion)
        
    
    padFeature(X_train)
    padFeature(X_test)
    padFeature(X_val)
    #    
    #np.savez('features.npz', X_train=X_train, X_test=X_test, X_val = X_val, Y_train = Y_train, Y_test = Y_test, Y_val = Y_val)
    #
    
   #print("########## THIS IS PRELOADED DATA, IF DATA WAS CHANGED NEED TO RUN FEATURE EXTRACTION AGAIN ##########")
   #X_train = np.load('features.npz')['X_train']
   #X_test = np.load('features.npz')['X_test']
   #Y_train = np.load('features.npz')['Y_train']
   #Y_test = np.load('features.npz')['Y_test']
   #X_val = np.load('features.npz')['X_val']
   #Y_val = np.load('features.npz')['Y_val']

    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(np.array(Y_train).reshape(-1,1)).toarray()
    y_test = encoder.fit_transform(np.array(Y_test).reshape(-1,1)).toarray()
    y_val = encoder.fit_transform(np.array(Y_val).reshape(-1,1)).toarray()
    
    
    
    x_train = applyScaler(X_train)
    x_test= applyScaler(X_test)    
    x_val = applyScaler(X_val)
    
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #x_val = np.expand_dims(x_val, -1)

    
    model = Sequential()
    input_shape = x_train.shape[1:]
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='leaky_relu',input_shape = x_train.shape[1:], padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='leaky_relu',kernel_regularizer=l1(0.003), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='leaky_relu',kernel_regularizer=l2(0.003), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='leaky_relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='leaky_relu',kernel_regularizer=l2(0.003), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='leaky_relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='leaky_relu'))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights_dict = dict(enumerate(class_weights))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    model.summary()
    
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights_dict = dict(enumerate(class_weights))
    history=model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val),class_weight=class_weights_dict)
    
    print("Accuracy of our model on VALIDATION data : " , model.evaluate(x_val,y_val)[1]*100 , "%")
    #print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")
    
    pred_val = model.predict(x_val)
    y_pred = encoder.inverse_transform(pred_val)
    y_val = encoder.inverse_transform(y_val)
    #pred_test = model.predict(x_test)
    #y_pred = encoder.inverse_transform(pred_test)
    #y_test = encoder.inverse_transform(y_test)
    show_graphs(y_val, y_pred)
    #
    ###saving the model
    ##model.save('SER Model.keras')
    #model = load_model('SER Model.keras')
  

