import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy 
from sklearn import metrics 
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import pickle

from tqdm import tqdm
from pathlib import Path  
from collections import defaultdict 
import warnings
warnings.filterwarnings("ignore")

feature_names=[ 'mfcc_simple_mean_new',  'chroma_cqt_simple_mean_new']
Main_folder = "./ExtractedFeatures/"
prefix = "spcup_2022_training_part1"
extracted_feature_folders = [
    Main_folder + prefix + "/",
    Main_folder + prefix + "_noise_added/",
    Main_folder + prefix + "_reverb_added/",
    Main_folder + prefix + "_compressed/"
]

mfccs = pd.read_csv(extracted_feature_folders[0] + 'mfcc_features.csv', index_col= 0)
chroma = pd.read_csv(extracted_feature_folders[0] + 'chroma_cqt_features.csv', index_col= 0)
labels = pd.read_csv("./data/spcup_2022_training_part1/labels.csv").algorithm
df1 = pd.concat([mfccs, chroma], axis = 1)
df1['class'] = labels

mfccs = pd.read_csv(extracted_feature_folders[1] + 'mfcc_features.csv', index_col= 0)
chroma = pd.read_csv(extracted_feature_folders[1] + 'chroma_cqt_features.csv', index_col= 0)
labels = pd.read_csv("./data/spcup_2022_training_part1_noise_added/labels.csv").algorithm
df2 = pd.concat([mfccs, chroma], axis = 1)
df2['class'] = labels

mfccs = pd.read_csv(extracted_feature_folders[2] + 'mfcc_features.csv', index_col= 0)
chroma = pd.read_csv(extracted_feature_folders[2] + 'chroma_cqt_features.csv', index_col= 0)
labels = pd.read_csv("./data/spcup_2022_training_part1_reverb_added/labels.csv").algorithm
df3 = pd.concat([mfccs, chroma], axis = 1)
df3['class'] = labels

mfccs = pd.read_csv(extracted_feature_folders[3] + 'mfcc_features.csv', index_col= 0)
chroma = pd.read_csv(extracted_feature_folders[3] + 'chroma_cqt_features.csv', index_col= 0)
labels = pd.read_csv("./data/spcup_2022_training_part1_compressed/labels.csv").algorithm
df4 = pd.concat([mfccs, chroma], axis = 1)
df4['class'] = labels

df = pd.concat([df1, df2, df3, df4], axis = 0)

print(f"{df.shape}: Features queried")



