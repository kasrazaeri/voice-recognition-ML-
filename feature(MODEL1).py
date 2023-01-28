import os
import pickle
from statistics import harmonic_mean
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

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

train.to_csv("C:/Users/vcc/Desktop/project ml/Train.csv")

val.to_csv("C:/Users/vcc/Desktop/project ml/Validation.csv")

test.to_csv("C:/Users/vcc/Desktop/project ml/Test.csv")

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
        
    
    #compute chroma energy
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    #temporally average chroma
    chroma = np.mean(chroma, axis = 1)
    
    #compute rms
    rms = librosa.feature.rms(y=y)
    rms = np.mean(rms, axis = 1)

    #compute zero_crossing_rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate = np.mean(zero_crossing_rate, axis = 1)
    
    #compute chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft = np.mean(chroma_stft, axis = 1)

    #compute tempogram
    tempogram=librosa.feature.tempogram(y=y, sr=sr)        
    tempogram = np.mean(tempogram,axis=1)
    
    #compute tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)        
    tonnetz = np.mean(tonnetz,axis=1)
    
    return [harmonic, chroma, rms, zero_crossing_rate, chroma_stft, tempogram, tonnetz]


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
                                           columns=['harmonic', 'rms', 'zero_crossing_rate', 'chroma', 'chroma_stft', 'tempogram', 'tonnetz'])

    features.head()
    
    print("Extracting rms, zero_crossing_rate, chroma, chroma_stft, tempogram, tonnetz and Concatenating them...")
    
    #extract harmonic
    harmonic = pd.DataFrame(features.harmonic.values.tolist(),index=features.index)
    harmonic = harmonic.add_prefix('harmonic_')

    #extract rms
    rms = pd.DataFrame(features.rms.values.tolist(),index=features.index)
    rms = rms.add_prefix('rms_')


    #extract chroma
    chroma = pd.DataFrame(features.chroma.values.tolist(),index=features.index)
    chroma = chroma.add_prefix('chroma_')


    #extract zero_crossing_rate
    zero_crossing_rate = pd.DataFrame(features.zero_crossing_rate.values.tolist(),index=features.index)
    zero_crossing_rate = zero_crossing_rate.add_prefix('zero_crossing_rate_')

    
    #extract chroma_stft
    chroma_stft = pd.DataFrame(features.chroma_stft.values.tolist(),index=features.index)
    chroma_stft = chroma_stft.add_prefix('chroma_stft_') 
    
    
    #extract tempogram
    tempogram = pd.DataFrame(features.tempogram.values.tolist(),index=features.index)
    tempogram = tempogram.add_prefix('tempogram_') 
     
    
    #extract tonnetz
    tonnetz = pd.DataFrame(features.tonnetz.values.tolist(),index=features.index)
    tonnetz = tonnetz.add_prefix('tonnetz_') 
      
     
    #drop the old columns
    features = features.drop(labels=['harmonic', 'rms', 'chroma', 'zero_crossing_rate', 'chroma_stft','tempogram', 'tonnetz'], axis=1)


    #concatenate
    df_features=pd.concat([features, harmonic, rms, chroma, zero_crossing_rate, chroma_stft, tempogram, tonnetz], axis=1, join='inner')
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
    
df_features_test = get_df_features(test, saving_path="C:/Users/vcc/Desktop/project ml/Test.pickle")
    
df_features_train = get_df_features(train, saving_path="C:/Users/vcc/Desktop/project ml/Train.pickle")

df_features_val = get_df_features(val, saving_path="C:/Users/vcc/Desktop/project ml/Val.pickle")
    