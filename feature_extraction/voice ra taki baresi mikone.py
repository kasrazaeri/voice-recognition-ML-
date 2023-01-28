import warnings
warnings.filterwarnings("ignore")
import librosa
import os
from tqdm.auto import tqdm
import numpy as np

def extract_feature(y, sr): 
    fn_list_i = [
        librosa.feature.chroma_stft, librosa.feature.spectral_centroid,
        librosa.feature.spectral_bandwidth, librosa.feature.spectral_rolloff
    ]
    fn_list_ii = [ librosa.feature.zero_crossing_rate]
    feat_vect_i = [ np.mean(funct(y,sr)) for funct in fn_list_i]
    feat_vect_ii = [ np.mean(funct(y)) for funct in fn_list_ii] 
    mfccs = list(np.mean(librosa.feature.mfcc(y, sr=sr),axis=1))
    feature_vector = feat_vect_i + feat_vect_ii + mfccs
    feature_names = ["chroma_stft", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate"]+["mfccs_"+str(i) for i in range (0,len(mfccs))]
    return feature_vector, feature_names

y, sr = librosa.load('C:/Users/vcc/Desktop/project ml/voice/2474.wav')######voice bedin
feature_vector = extract_feature(y, sr)
print(feature_vector)


