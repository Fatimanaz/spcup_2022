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

def read_audio_clip(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    return audio, sample_rate


metadata = pd.read_csv(train_data_path + 'labels.csv')
file_list = metadata.track 
labels = metadata.algorithm 

noise_labels = defaultdict(list)
reverb_labels = defaultdict(list)
compressed_labels = defaultdict(list)

for i in tqdm(range(len(file_list))):
    file_name_prefix = file_list[i].split('.')[0]
    noise_file_name = file_name_prefix + '_noise.wav'
    noise_labels['track'].append(noise_file_name)
    noise_labels['algorithm'].append(labels[i])
    reverb_file_name = file_name_prefix + '_reverb.wav'
    reverb_labels['track'].append(reverb_file_name)
    reverb_labels['algorithm'].append(labels[i])
    compressed_file_name = file_name_prefix + '_compressed.wav'
    compressed_labels['track'].append(compressed_file_name)
    compressed_labels['algorithm'].append(labels[i])


df = pd.DataFrame(noise_labels)
print(df.head())
noise_labels_filepath=Path( "./data/spcup_2022_training_part1_noise_added/"+"labels.csv")  
noise_labels_filepath.parent.mkdir(parents=True, exist_ok=True) 
df.to_csv(noise_labels_filepath)  
df = pd.DataFrame(reverb_labels)
print(df.head())
noise_labels_filepath=Path("./data/spcup_2022_training_part1_reverb_added/" +"labels.csv")  
noise_labels_filepath.parent.mkdir(parents=True, exist_ok=True) 
df.to_csv(noise_labels_filepath) 
df = pd.DataFrame(compressed_labels)
print(df.head())
noise_labels_filepath=Path("./data/spcup_2022_training_part1_compressed/" +"labels.csv")  
noise_labels_filepath.parent.mkdir(parents=True, exist_ok=True) 
df.to_csv(noise_labels_filepath) 


feature_storage_folder = Main_folder+'ExtractedFeatures/'
if not os.path.exists(feature_storage_folder):
    os.makedirs(feature_storage_folder)

data_folders = os.listdir('./data')
for folder in data_folders:

    metadata = pd.read_csv(f'./data/{folder}/labels.csv')
    print("read data successfully")
    file_list = metadata.track
    print(len(file_list))

    feature_names=[ 'mfcc_simple_mean_new',"chroma_cqt_simple_mean_new"]


    Features = defaultdict(list)  

    for i in tqdm(range(len(file_list))):
        file_name = f'./data/{folder}/' + file_list[i]
        audio, sr = read_audio_clip(file_name) 

        Features['mfcc_simple_mean_new'].append([ mfcc_simple_mean_feature(audio, sr )])
        Features["chroma_cqt_simple_mean_new"].append([chroma_cqt_simple_mean_feature(audio, sr) ])
    

    # storing the extracted feaures into a folder
    subfolder = feature_storage_folder + folder+ "/"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    extracted_features_mfcc_df=pd.DataFrame(Features['mfcc_simple_mean_new'],columns=['feature'])
    feature_list = extracted_features_mfcc_df['feature'].tolist()
    df = pd.DataFrame(feature_list)
    feature_filepath = Path(subfolder + f"mfcc_features.csv")  
    # feature_filepath.parent.mkdir(parents=True, exist_ok=True) 
    df.to_csv(feature_filepath, index = False)  
    print(df.head())

    extracted_features_chroma_df = pd.DataFrame(Features['chroma_cqt_simple_mean_new'], columns = ['feature'])
    feature_list = extracted_features_chroma_df['feature'].tolist()
    df = pd.DataFrame(feature_list)
    feature_filepath = Path(subfolder + f"chroma_cqt_features.csv")
    df.to_csv(feature_filepath, index = False)
    print(df.head())