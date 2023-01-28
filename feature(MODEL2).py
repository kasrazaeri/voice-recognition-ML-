import os
import pickle
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import csv

df = pd.read_csv("C:/Users/vcc/Desktop/project ml/dataV_revised4.csv")

train_percent =0.8
val_percent=0.1

_df = df.sample(frac=1)
train = _df.iloc[:int(_df.shape[0]*train_percent), :]
test = _df.iloc[int(_df.shape[0]*train_percent):, :]

int(train.shape[0]*val_percent)

val = train.iloc[:int(train.shape[0]*val_percent), :]
train =  train.iloc[int(train.shape[0]*val_percent):, :]

def train_val_test_split(df, train_percent=0.8, val_percent=0.1):
    _df = df.sample(frac=1)
    
    train = _df.iloc[:int(_df.shape[0]*train_percent), :]
    test = _df.iloc[int(_df.shape[0]*train_percent):, :]
    
    val = train.iloc[:int(train.shape[0]*val_percent), :]
    train =  train.iloc[int(train.shape[0]*val_percent):, :]
    
    return train, val, test

train, val, test = train_val_test_split(df, train_percent=0.8, val_percent=0.1)

train.to_csv("C:/Users/vcc/Desktop/project ml/Train2.csv")

val.to_csv("C:/Users/vcc/Desktop/project ml/Validation2.csv")

test.to_csv("C:/Users/vcc/Desktop/project ml/Test2.csv")

#########################################################################################################

def feature_extract(file):
    """
    Define function that takes in a file an returns features in an array
    """
    
    #get wave representation
    y, sr = librosa.load(file)
        
    #determine if instruemnt is harmonic or percussive by comparing means
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    if np.mean(y_harmonic)>np.mean(y_percussive):
        harmonic=1
    else:
        harmonic=0
        
    #Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    #temporal averaging
    mfcc=np.mean(mfcc,axis=1)
    
    #spectral_centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    #temporal averaging
    spectral_centroid=np.mean(spectral_centroid,axis=1)
    
    
    #get the mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)  
    #temporally average spectrogram
    spectrogram = np.mean(spectrogram, axis = 1)
    
    
    #compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast, axis= 1)
    
    #compute spectral_bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth = np.mean(spectral_bandwidth, axis= 1)
    
    
    #compute spectral_rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff = np.mean(spectral_rolloff, axis= 1)
    
    
    #compute flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness = np.mean(spectral_flatness, axis= 1)
    
    
    #compute poly_features
    poly_features = librosa.feature.poly_features(y=y)
    poly_features = np.mean(poly_features, axis= 1)
    
    
    #compute tempogram
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    tempogram = np.mean(tempogram, axis= 1)
    
    
    #compute tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)        
    tonnetz = np.mean(tonnetz,axis=1)
    
    return [spectral_bandwidth, mfcc, spectral_centroid, spectrogram, contrast, spectral_rolloff, spectral_flatness, poly_features, tempogram, tonnetz]


##############################################


def category_id(file_name):
    df[ df["address"]==file_name]["emotionID"].iloc[0]


def get_df_features(_df, saving_path=None):
    _dict = dict()
    
    print("Extracting Features...")
    
    with tqdm(total=_df.shape[0]) as pbar:
        for filename in _df["address"]:
            features = feature_extract(filename) #specify directory and .wav
            _dict[filename] = features
            pbar.update(1)

            
    features = pd.DataFrame.from_dict(_dict, orient='index',
                                           columns=[ 'spectral_bandwidth', 'mfcc', 'spectro', 'contrast', 'spectral_centroid', 'spectral_rolloff', 'spectral_flatness', 'poly_features', 'tempogram', 'tonnetz'])

    features.head()
    
    print("Extracting MFCCs, Spectro, Chroma, Contrast and Concatenating them...")
    
    #extract mfccs
    mfcc = pd.DataFrame(features.mfcc.values.tolist(),index=features.index)
    mfcc = mfcc.add_prefix('mfcc_')

    #extract spectro
    spectro = pd.DataFrame(features.spectro.values.tolist(),index=features.index)
    spectro = spectro.add_prefix('spectro_')


    #extract spectral_bandwidth
    spectral_bandwidth = pd.DataFrame(features.spectral_bandwidth.values.tolist(),index=features.index)
    spectral_bandwidth = spectral_bandwidth.add_prefix('spectral_bandwidth_')


    #extract contrast
    contrast = pd.DataFrame(features.contrast.values.tolist(),index=features.index)
    contrast = contrast.add_prefix('contrast_')
    
    #extract spectral_centroid
    spectral_centroid = pd.DataFrame(features.spectral_centroid.values.tolist(),index=features.index)
    spectral_centroid = spectral_centroid.add_prefix('spectral_centroid_')
    
    
    #extract spectral_rolloff
    spectral_rolloff = pd.DataFrame(features.spectral_rolloff.values.tolist(),index=features.index)
    spectral_rolloff = spectral_rolloff.add_prefix('spectral_rolloff_')
    
    
    #extract spectral_flatness
    spectral_flatness = pd.DataFrame(features.spectral_flatness.values.tolist(),index=features.index)
    spectral_flatness = spectral_flatness.add_prefix('spectral_flatness_')


    #extract poly_features
    poly_features = pd.DataFrame(features.poly_features.values.tolist(),index=features.index)
    poly_features = poly_features.add_prefix('poly_features_')
    
    
    
    #extract tempogram
    tempogram = pd.DataFrame(features.tempogram.values.tolist(),index=features.index)
    tempogram = tempogram.add_prefix('tempogram_')
    
    
        
    #extract tonnetz
    tonnetz = pd.DataFrame(features.tonnetz.values.tolist(),index=features.index)
    tonnetz = tonnetz.add_prefix('tonnetz_')
    
    
    #drop the old columns
    features = features.drop(labels=['mfcc', 'spectro', 'spectral_bandwidth', 'contrast', 'spectral_centroid', 'spectral_rolloff', 'spectral_flatness', 'poly_features', 'tempogram', 'tonnetz'], axis=1)

    #concatenate
    df_features=pd.concat([features, mfcc, spectro, spectral_bandwidth, contrast, spectral_centroid, spectral_rolloff, spectral_flatness, poly_features, tempogram, tonnetz], axis=1, join='inner')
    df_features.head()
    
    targets = list()
    
    for file_name in df_features.index.tolist():
    #   import ipdb; ipdb.set_trace()
        targets.append( category_id(file_name) )
    
    df_features["target"] = targets
    
    if saving_path is not None:
        print("Saving Features Dataframe...")
        
        with open(saving_path, 'wb') as f:
            pickle.dump(df_features, f)
            
    return df_features
    
df_features_test = get_df_features(test, saving_path="C:/Users/vcc/Desktop/project ml/Test2.pickle")

df_features_train = get_df_features(train, saving_path="C:/Users/vcc/Desktop/project ml/Train2.pickle")

df_features_val = get_df_features(val, saving_path="C:/Users/vcc/Desktop/project ml/Val2.pickle")
    
