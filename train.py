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
import pickle

from tqdm import tqdm
from pathlib import Path  
from collections import defaultdict 
import warnings
warnings.filterwarnings("ignore")

# CONCATENATING FEATURES FOR TRAINING FROM EXTRACTED_FEATURES/TRAIN_PART1 FOLDER
# features are stored in the main folder extracted features: 
# ExtractedFeatures
#       train_part1
#           chroma_cqt_features_train_part1.csv 
#           mfcc_features_train_part1.csv
#           create: train_features_concatenated.csv
#       eval_part1 (later)
#       eval_part2 (later)


feature_names=[ 'mfcc_simple_mean_new',  'chroma_cqt_simple_mean_new']#  'melspectogram_mean_new_eval'     ]
Main_folder = "./"

feature_storage_folder = Main_folder+'ExtractedFeatures/train_part1/'
if not os.path.exists(feature_storage_folder):
  os.makedirs(feature_storage_folder)
cnt = 1
mfcc_df = pd.read_csv(feature_storage_folder + 'mfcc_features_train_part1.csv', index_col = 0)
print(mfcc_df.head())
chroma_cqt_df = pd.read_csv(feature_storage_folder + 'chroma_cqt_features_train_part1.csv', index_col = 0)
print(chroma_cqt_df.head())
# for fn in feature_names:
#   print(fn)
#   print(len(Features[fn]))

#   extracted_features_df=pd.DataFrame(Features[fn],columns=['feature'])



#   feature_list = extracted_features_df['feature'].tolist()
  
#   df = pd.DataFrame(feature_list)
 
#   if cnt==1:
#     print("H1")
#     FeatureConcatDF=df.copy();
#   elif cnt< len(feature_names):
#     print("H2")
#     FeatureConcatDF=pd.concat([FeatureConcatDF, df.reindex(FeatureConcatDF.index)], axis=1).copy()
#   else:
#     print("H3")
#     FeatureConcatDF=pd.concat([FeatureConcatDF, df.reindex(FeatureConcatDF.index)], axis=1).copy()
#     feature_filepath=Path(feature_storage_folder+"concatenated_features_eval_2.csv")  
#     feature_filepath.parent.mkdir(parents=True, exist_ok=True)  
#     FeatureConcatDF.to_csv(feature_filepath) 
#     print(fn , " csv writing complete")

#   cnt=cnt+1
 

