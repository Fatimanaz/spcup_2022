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

# Load the saved model
Model_Name = "SVC_C_4__kernel_linear__probability_True_"
MODEL_SAVE_PATH = './Results/svm_on_mfcc_and_chroma_cqt_' + Model_Name + "/"
checkpoint_path = MODEL_SAVE_PATH +"SavedModel.sav"
loaded_model = pickle.load(open(checkpoint_path, 'rb'))


# evaluating results for eval1
df_mfcc_features = pd.read_csv('./ExtractedFeatures/spcup_2022_eval_part1/mfcc_features.csv', index_col = 0)
df_chroma_features = pd.read_csv('./ExtractedFeatures/spcup_2022_eval_part1/chroma_cqt_features.csv', index_col = 0)
labels = pd.read_csv('./data/spcup_2022_eval_part1/labels.csv' ,index_col=None)
df = pd.concat([df_mfcc_features, df_chroma_features], axis = 1)
print(df.shape, ": features queried from eval part 1 dataset")

out  = loaded_model.predict(df)
labels['algorithm'] = out
labels.to_csv('./Results/eval_part1_answers.txt')
print("Evaluation for part 1 complete")



# evaluating results for eval2
df_mfcc_features = pd.read_csv('./ExtractedFeatures/spcup_2022_eval_part2/mfcc_features.csv', index_col = 0)
df_chroma_features = pd.read_csv('./ExtractedFeatures/spcup_2022_eval_part2/chroma_cqt_features.csv', index_col = 0)
labels = pd.read_csv('./data/spcup_2022_eval_part2/labels.csv' ,index_col=None)
df = pd.concat([df_mfcc_features, df_chroma_features], axis = 1)
print(df.shape, ": features queried from eval part 2 dataset")

out = loaded_model.predict(df)
labels['algorithm'] = out
labels.to_csv('./Results/eval_part2_answers.txt')
print("Evaluation for part 2 complete")