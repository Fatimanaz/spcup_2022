# spcup_2022
>Matlab version used: R2021b  
>Python version used: 3.8.10
```
git clone https://github.com/Fatimanaz/spcup_2022.git 
cd spcup_2022
```
Now create a virtual environment (optional)
```
pip install -r requirements.txt
```

**Directory Structure**  
1. **data**:  
    folder consists of raw audio files and their corresponding labels.csv files. Both the training dataset as well as the evaluation dataset folders are available in the data folder.
2. **MATLAB_SCRIPTS**:  
    folder contains the matlab scripts required to augment the original dataset. The Scripts produce three new folders in the **data** folder: noise_added, reverb_added, and compressed.
3. **Results**:  
    consists of the evaluated results for eval_dataset_part1 and eval_dataset_part2 and trained models.
4. **feature_extraction.py**:  
    python script that processes the raw audio files provided in **data** folder and produces handcrafted-features mfcc, and chroma_cqt features. 
5. **ExtractedFeatures**:  
    consists of mfcc_features and chroma_cqt_features extracted using feature_extraction python script
6. **train.py**: python script to train the model. Takes input in form of handcrafted features stored in ExtractedFeatures folder
7. **eval.py**: python script to produce labels on evaluation dataset



## **Using the trained model to evaluate results from already Extracted Features**
We have already provided the extracted features (mfccs and croma cqt) csv files for all the files in the _ExtractedFeatures_ folder.
```
python3 eval.py
```
This produces the required csv files in the following format:  
||track|algorithm|
|-----|-----|---------|
|0|audio_file_name|label(int)|

## **Training the entire model from scratch**

### Structure of the data folder 
You can look at the detailed directory structure in directory_structure.txt


* To train the entire model from scratch, raw audio files need to be augmented, processed, converted into handcrafted features and then stored in the ExtractedFeatures folder.
* Keep all the audio files in a folder. Rename the folder to _spcup_2022_training_part1_. Place this folder inside the data folder.
* Run adapted_script.mlx using matlab. All relative paths have already been set inside the script.
* Now just run the provided python scripts, that will generate features, train the model and store it, and evaluate the results on the evaulation dataset. 
```
python3 feature_extraction.py
python3 train.py
python3 eval.py
```
 
***
