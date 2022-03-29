import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path  
from collections import defaultdict 
import warnings
warnings.filterwarnings("ignore")

Main_folder = "./"
train_data_path = Main_folder+"data/spcup_2022_training_part1/"
eval_part1_path = Main_folder+"data/spcup_2022_eval_part1/"
eval_part2_path = Main_folder+"data/spcup_2022_eval_part2/"
dataset_folders = [train_data_path, eval_part1_path, eval_part2_path]

#chroma_cqt
def chroma_cqt_simple_mean_feature(audio, sample_rate):
    C=librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    C_mean = np.mean(C.T,axis=0)
    return C_mean

#MFCC
def mfcc_simple_mean_feature(audio, sample_rate ):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features


metadata=pd.read_csv(train_data_path+"seen_unseen_labels.csv")
file_list = metadata.track
print("Number of files listed: ",len(file_list))

feature_names=[ 'mfcc_simple_mean_new',"chroma_cqt_simple_mean_new"]#,  "chroma_cqt_simple_mean_new",  "melspectogram_mean_new"     ]

feature_storage_folder = Main_folder+'ExtractedFeatures/'
if not os.path.exists(feature_storage_folder):
    os.makedirs(feature_storage_folder)

def read_audio_clip(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    return audio, sample_rate

# for every folder now and then
feature_storage_subfolders = ['train_part1', 'eval_part1', 'eval_part2']
for subfolder_name in feature_storage_subfolders:
    Features = defaultdict(list)  

    for i in tqdm(range(len(file_list))):
        file_name = train_data_path + file_list[i]
        audio, sr = read_audio_clip(file_name) 

        Features['mfcc_simple_mean_new'].append([ mfcc_simple_mean_feature(audio, sr )])
        Features["chroma_cqt_simple_mean_new"].append([chroma_cqt_simple_mean_feature(audio, sr) ])
    

    # storing the extracted feaures into a folder
    subfolder = feature_storage_folder + subfolder_name+ "/"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    extracted_features_mfcc_df=pd.DataFrame(Features['mfcc_simple_mean_new'],columns=['feature'])
    feature_list = extracted_features_mfcc_df['feature'].tolist()
    df = pd.DataFrame(feature_list)
    feature_filepath = Path(subfolder + f"mfcc_features_{subfolder_name}.csv")  
    # feature_filepath.parent.mkdir(parents=True, exist_ok=True) 
    df.to_csv(feature_filepath)  

    extracted_features_chroma_df = pd.DataFrame(Features['chroma_cqt_simple_mean_new'], columns = ['feature'])
    feature_list = extracted_features_chroma_df['feature'].tolist()
    df = pd.DataFrame(feature_list)
    feature_filepath = Path(subfolder + f"chroma_cqt_features_{subfolder_name}.csv")
    df.to_csv(feature_filepath)