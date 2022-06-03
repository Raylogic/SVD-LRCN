# coding=utf-8
import os
import numpy as np
import joblib as jl
import math
import librosa

from matplotlib import pyplot as plt
from tqdm import tqdm
from Feature import AudioFeat
from UNet import split_vocal

# ======================================================================================================================

# Replace a NaN value in the features
def replace_nan(dataX):

    # Get feature values in Float32 type
    dataX = np.float32(dataX)

    # List of final feature values
    finalX = []

    # Traverse each feature and get its value
    for feature in dataX:
        for number in feature:

            # If the value is Infinite or None, define it as 0
            if math.isinf(number) or math.isnan(number):
                feature = [0] * len(feature)

            # Append the new value to the list of final feature values
            finalX.append(feature)
    
    # Return the new features values in a Float32 Array
    return np.array(finalX, dtype='float32')

# ======================================================================================================================

def get_vocals(mp3_path):

    # SVS process
    vocal_path = mp3_path.replace('.wav', '.unet.Vocal.wav')
    if not os.path.isfile(vocal_path):
        split_vocal(mp3_path)

    # Feature extraction
    feat_path = mp3_path.replace('audio', 'joblibs').replace('.mp3', '.joblib')

    # If .joblib file doesn't exists, perform the extraction
    if not os.path.exists(feat_path):
        
        # Open .joblib file
        open(feat_path, 'w')

        # Load .joblib file with librosa
        audio_data , sr = librosa.load(mp3_path)

        # Extract audio features
        dataX = AudioFeat().get_audio_features(audio_data, 16000)

        # Dump audio features in .joblib file
        jl.dump(dataX, feat_path)

    # Load .joblib file
    feat_data = jl.load(feat_path)

    # Replace None values
    feat_data = np.vstack(feat_data)
    feat_data = replace_nan(feat_data)
    feat_data = np.nan_to_num(feat_data)

# ======================================================================================================================

# Extract Jamendo song .lab file tags
def jamendo_electrobyte_labtags(lab_path):

    # List of labels
    label_list = []

    # Open song .lab file
    with open(lab_path, 'r') as lab:
        labels = lab.readlines()

    # Read each label
    for line in labels:
        
        # Store the label values in a list
        label = line.strip('\n').split(' ')

        # Assuming the label has 3 values, traverse the time range at a 10ms pace
        if len(label) == 3:
            for time_item in range(int(float(label[0])*1000), int(float(label[1])*1000), 10):

                # If the label indicates that there is sing, hot-encode it to 1 and add it to the list of labels
                # If the label indicates that there is no-sing, hot-encode it to 0 and add it to the list of labels
                if label[2] == 'sing':
                    label_list.append(1)
                else:
                    label_list.append(0)

    return label_list

# ======================================================================================================================

# Extract the Jamendo and Electrobyte song labels and features
def jamendo_electrobyte_api(mp3_path):

    get_vocals(mp3_path)

    # Check whether the song belongs to the train, valid or test set
    # Replace the song path to redirect to its .lab file
    if 'audio\\train' in mp3_path:
        lab_path = mp3_path.replace('audio', 'labels').replace('\\train', '').replace('.mp3', '.lab')
    elif 'audio\\test' in mp3_path:
        lab_path = mp3_path.replace('audio', 'labels').replace('\\test', '').replace('.mp3', '.lab')
    elif 'audio\\valid' in mp3_path:
        lab_path = mp3_path.replace('audio', 'labels').replace('\\valid', '').replace('.mp3', '.lab')
    else:
        lab_path = ''
        raise ('error in .mp3 path')

    # Extract the label values of the song
    label_list = jamendo_electrobyte_labtags(lab_path)
    return 0

# ======================================================================================================================

# Extract the labels and features from Jamendo Corpus dataset
def jamendo_electrobyte(wav_dir):

    # Find the Jamendo Corpus OS path
    for root, dirs, names in os.walk(wav_dir):

        # Traverse the Jamendo Corpus and Electrobyte songs
        for name in tqdm(names):

            # Get the song OS path
            mp3_path = os.path.join(root, name)

            # Run the API to extract the song features and labels 
            if '.mp3' in mp3_path and '.unet.Vocal' not in mp3_path:
                jamendo_electrobyte_api(mp3_path)

    return 0

# ======================================================================================================================

# Main code
if __name__ == '__main__':
    # jamendo_electrobyte('.\Datasets\Jamendo')
    # jamendo_electrobyte('.\Datasets\Electrobyte')
    pass
