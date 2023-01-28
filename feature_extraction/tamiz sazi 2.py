import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df2=pd.read_csv('C:/Users/vcc/Desktop/project ml/dataV_revised4.csv')

#df.loc[df.duplicated(subset='voice id')]
df2=df2.drop_duplicates(subset='voice id',keep='first',inplace=False)          
df2 = df2[df2['emotionID'] != 10]
df2 = df2[df2['emotionID'] != 9]
df2 = df2[df2['emotionID'] != 8]
df2 = df2[df2['emotionID'] != 7]
df2 = df2[df2['emotionID'] != 6]
df2 = df2[df2['emotionID'] != 5]




df2.to_csv("C:/Users/vcc/Desktop/project ml/dataV_revised5.csv",)

