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
from sklearn.metrics import confusion_matrix
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

y_singleNode = np.array(df['class'])
X = np.array(df.drop(['class'],axis=1))
print(X.shape)
print(y_singleNode.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y_singleNode, test_size=0.25, random_state=42, stratify=y_singleNode)
model = svm.SVC(kernel = 'linear', C = 4, probability = True).fit(X, y_singleNode)
results = model.score(X_val, y_val)
print(results) 

num_classes=6
y_val_predict = model.predict(X_val)
Confusion_Matrix = confusion_matrix(y_val, y_val_predict, labels = [0,1,2,3,4,5])
print(Confusion_Matrix)

Model_Name=str(model).replace(",","_").replace(" ", "_").replace(")", "_").replace("(", "_").replace("'", "").replace("=","_")
print(Model_Name)
MODEL_SAVE_PATH = './Results/svm_on_mfcc_and_chroma_cqt_' + Model_Name + "/"
print(MODEL_SAVE_PATH)

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

pickle.dump(model, open(MODEL_SAVE_PATH + "SavedModel.sav", 'wb'))