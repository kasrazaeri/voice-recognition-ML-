import pandas as pd
import os
import pickle
import numpy as np

with open(os.path.join("model_1", "Val.pickle"), "rb") as bfile:
    val1_data = pickle.load(bfile)

with open(os.path.join("model_1", "Train.pickle"), "rb") as bfile:
    train1_data = pickle.load(bfile)

with open(os.path.join("model_1", "Test.pickle"), "rb") as bfile:
    test1_data = pickle.load(bfile)

val1_data = val1_data.drop(columns=["target"])
train1_data = train1_data.drop(columns=["target"])
test1_data = test1_data.drop(columns=["target"])

val1_label = pd.read_csv(os.path.join("model_1", "Validation.csv"))
train1_label = pd.read_csv(os.path.join("model_1", "Train.csv"))
test1_label = pd.read_csv(os.path.join("model_1", "Test.csv"))

val1_label = val1_label.drop(columns=["Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "Unnamed: 7",
                                    "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14"])

train1_label = train1_label.drop(columns=["Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "Unnamed: 7",
                                    "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14"])

test1_label = test1_label.drop(columns=["Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "Unnamed: 7",
                                    "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14"])


sex_val1 = val1_label["sex"]
sex_val1 = list(map(lambda x : 0 if x == "f" else 1, sex_val1))

sex_train1 = train1_label["sex"]
sex_train1 = list(map(lambda x : 0 if x == "f" else 1, sex_train1))

sex_test1 = test1_label["sex"]
sex_test1 = list(map(lambda x : 0 if x == "f" else 1, sex_test1))

emotion_val1 = val1_label["emotionID"]
emotion_train1 = train1_label["emotionID"]
emotion_test1 = test1_label["emotionID"]

data1 = pd.concat([train1_data, val1_data, test1_data])
sex_data1 = np.concatenate([sex_train1, sex_val1, sex_test1])
emotion_data1 = np.concatenate([emotion_train1, emotion_val1, emotion_test1])


with open(os.path.join("model_2", "Val2.pickle"), "rb") as bfile:
    val2_data = pickle.load(bfile)

with open(os.path.join("model_2", "Train2.pickle"), "rb") as bfile:
    train2_data = pickle.load(bfile)

with open(os.path.join("model_2", "Test2.pickle"), "rb") as bfile:
    test2_data = pickle.load(bfile)

val2_data = val2_data.drop(columns=["target"])
train2_data = train2_data.drop(columns=["target"])
test2_data = test2_data.drop(columns=["target"])

val2_label = pd.read_csv(os.path.join("model_2", "Validation2.csv"))
train2_label = pd.read_csv(os.path.join("model_2", "Train2.csv"))
test2_label = pd.read_csv(os.path.join("model_2", "Test2.csv"))

val2_label = val2_label.drop(columns=["Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "Unnamed: 7",
                                    "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14"])

train2_label = train2_label.drop(columns=["Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "Unnamed: 7",
                                    "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14"])

test2_label = test2_label.drop(columns=["Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "Unnamed: 7",
                                    "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14"])


sex_val2 = val2_label["sex"]
sex_val2 = list(map(lambda x : 0 if x == "f" else 1, sex_val2))

sex_train2 = train2_label["sex"]
sex_train2 = list(map(lambda x : 0 if x == "f" else 1, sex_train2))

sex_test2 = test2_label["sex"]
sex_test2 = list(map(lambda x : 0 if x == "f" else 1, sex_test2))

emotion_val2 = val2_label["emotionID"]
emotion_train2 = train2_label["emotionID"]
emotion_test2 = test2_label["emotionID"]

data2 = pd.concat([train2_data, val2_data, test2_data])
sex_data2 = np.concatenate([sex_train2, sex_val2, sex_test2])
emotion_data2 = np.concatenate([emotion_train2, emotion_val2, emotion_test2])