import os
import h5py
import joblib
import numpy as np
import soundfile
from matplotlib import pyplot as plt
from pydub import AudioSegment

from Setup import mirik
from Feature import AudioFeat
from UNet import split_vocal


# Process MIR1K tag file
def dataset_mir1k():

    # Access the song labels path
    pitch_label_dir = mirik + "\PitchLabel"

    print(pitch_label_dir)

    # Explore each tag file
    for tag_item in os.listdir(pitch_label_dir):

        # Concatenate the tag file to the path
        tag_path = os.path.join(pitch_label_dir, tag_item)

        # Read the tag file
        # Store each tag in a list
        with open(tag_path, 'r')as tag_f:
            tag_count = tag_f.readlines()
        
        # List of label values
        start_end_label = []

        # Start time
        start = None

        # Label value (Sing/No sing)
        label = None

        print(tag_count)

        # Explore each tag in the list
        for line_item in tag_count:

            print(line_item.strip('\t\n').split('\t'))

            # Get the current time and its pitch
            now_time, pitch = map(float, line_item.strip('\t\n').split('\t'))

            # Get the current label value
            if pitch > 0:
                now_label = 'sing\n'
            else:
                now_label = 'nosing\n'

            # If it is the first tag, set start time to 0, get the label value, and set end time as current time
            if start == None:
                start = 0
                end = now_time
                label = now_label
            else:
                # If current label value differs with the last one, set end time as current time
                # Generate a new label instance in the list of label values
                # Update the start time and label value for the next tag identification
                if label != now_label:
                    end = now_time
                    start_end_label.append('%.3f %.3f %s' % (start, end, label))
                    start = now_time
                    label = now_label

                # If current label value not differs with the last one, set end time as current time
                # Keep current label value for the next tag identification
                else:
                    end = now_time
                    label = now_label
        
        # Generate a new label instance in the list of label values
        start_end_label.append('%.3f %.3f %s' % (start, end, label))

        # Convert the tag .pv file into a .lab file
        transfer_tag_path = tag_path.replace('/PitchLabel', '/tags').replace('.pv', '.lab')

        # Move to the .lab file location
        transfer_tag_dir = os.path.dirname(transfer_tag_path)

        # If .lab file location doesn't exists, create it
        if not os.path.isdir(transfer_tag_dir):
            os.makedirs(transfer_tag_dir)

        # Write the list of label values into the .lab file
        with open(transfer_tag_path, 'w') as ttF:
            ttF.writelines(start_end_label)

    return 0

# ===================================================================================================================================

class AudioPreprocess:

    # Load Dataset from H5File
    def load_Dataset_from_h5file(self, h5file_path):

        # Read the H5File
        h5_file = h5py.File(h5file_path, 'r')

        # Separate the file values into train, valid and test set
        trainX = h5_file['trainX'][:]
        trainY = h5_file['trainY'][:]
        testX = h5_file['testX'][:]
        testY = h5_file['testY'][:]
        validX = h5_file['validX'][:]
        validY = h5_file['validY'][:]

        # Close the H5File
        h5_file.close()

        # Return the train, valid and test set
        return trainX, trainY, testX, testY, validX, validY

    # ======================================================================================================================
    
    # Write Dataset to H5File
    def write_h5file(self, data_set_dir, data_ext_str, vocal_only):

        # Convert the Joblib into H5 File, regardless is the original or SVS Vocal File
        if vocal_only:
            h5name = os.path.join(data_set_dir, data_ext_str.replace('.joblib', '_vocal.h5'))
            data_ext_str = '.unet.Vocal' + data_ext_str
        else:
            h5name = os.path.join(data_set_dir, data_ext_str.replace('.joblib', '_normal.h5'))
        
        # Finish if H5 File already exists
        if os.path.isfile(h5name):
            return

        # Train, Valid, Test sets
        trainX = []
        trainY = []
        testX = []
        testY = []
        validX = []
        validY = []

        # Explore the dataset
        for root, dirs, names in os.walk(data_set_dir):

            # Explore each song
            for item in names:
                if data_ext_str in item:

                    # Skip if SVS Vocal File already exists
                    if not vocal_only:
                        if '.unet.Vocal' in item:
                            continue

                    # Get the song path
                    file_path = os.path.join(root, item)

                    # Load the song file
                    data_lable_dict = joblib.load(file_path)

                    # Split the song data and labels
                    data = data_lable_dict['data']
                    label = data_lable_dict['lable']

                    # Generate train, valid and test set
                    if 'train' in file_path:
                        trainX.extend(data)
                        trainY.extend(label)
                    elif 'test' in file_path:
                        testX.extend(data)
                        testY.extend(label)
                    elif 'valid' in file_path:
                        validX.extend(data)
                        validY.extend(label)
                    else:
                        raise ('error train_test_valid')

        # Stack the data vertically
        trainX = np.vstack(trainX)
        testX = np.vstack(testX)
        validX = np.vstack(validX)
        trainY = np.array(trainY)
        testY = np.array(testY)
        validY = np.array(validY)

        # Create the dataset file
        file = h5py.File(h5name, 'w')

        # Add the data to the dataset
        file.create_dataset('trainX', data=trainX)
        file.create_dataset('trainY', data=trainY)
        file.create_dataset('testX', data=testX)
        file.create_dataset('testY', data=testY)
        file.create_dataset('validX', data=validX)
        file.create_dataset('validY', data=validY)

        # Close the file
        file.close()
        return 0
    
    # ======================================================================================================================
    
    # View the label distribution of a dataset
    def view_label_resolution(self, data_dir):

        # Dictionary of sing/nosing labels
        sing_nosing_resolution = {'sing': [], 'nosing': []}

        # Path to labels on the dataset
        labels_dir = '%s/tags' % data_dir

        # Counters
        sing_num = 0
        nosing_num = 0
        sing_total = 0
        nosing_total = 0

        # Traverse each .lab file
        for item in os.listdir(labels_dir):
            
            # Get .lab file path
            lab_path = os.path.join(labels_dir, item)

            # Open song .lab file
            with open(lab_path, 'r') as lab:
                labels = lab.readlines()
            
            # Read each label
            for line in labels:

                # Store the label values in a list
                start_end = line.split(' ')

                # Assuming the label has 3 values, extract them
                if len(start_end) == 3:
                    
                    # Start time, end time, tag value
                    start = float(start_end[0])
                    end = float(start_end[1])
                    tag = start_end[2].strip('\n')

                    # Append label values into dictionary of sing/nosing labels
                    sing_nosing_resolution[tag].append(end - start)

                    # If label is sing, add one to total counter and sing counter
                    # If label is nosing, add one to total counter and nosingsing counter
                    if tag == 'sing':
                        sing_total += 1
                        if (end - start) < 0.3:
                            sing_num += 1
                    elif tag == 'nosing':
                        nosing_total += 1
                        if (end - start) < 0.3:
                            nosing_num += 1

        # Plot sing/nosing distribution
        plt.figure()
        ax0 = plt.subplot(211)
        plt.gca().set_title('sing |<0.3s %d of %d' % (sing_num, sing_total))
        plt.axis([0, 0.5, 0, 100])
        plt.hist(sing_nosing_resolution['sing'], bins=2000)
        plt.subplot(212, sharex=ax0)
        plt.gca().set_title('nosing|<0.3s %d  of %d' % (nosing_num, nosing_total))
        plt.axis([0, 0.5, 0, 100])
        plt.hist(sing_nosing_resolution['nosing'], bins=2000)
        plt.subplots_adjust(hspace=1.0)
        plt.savefig('svg_plots/%s_sing_nosing_time_dur.svg' % data_dir.split('/')[-1], format='svg')
        return sing_nosing_resolution
    
    # ======================================================================================================================
    # Perform SVS preprocessing on the dataset
    def svs_split(self, data_dir):

        # Find the dataset OS path
        for root, dirs, names in os.walk(data_dir):

            # Traverse the dataset songs
            for name in names:

                # Perform SVS on the songs in .wav format
                if '.wav' in name:
                    wav_path = os.path.join(root, name)
                    split_vocal(wav_path)
        return 0

# ===================================================================================================================================

class Pre_procession_wav:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.DataX = []
        self.LabelY = []

    # Format files from Datasets
    def format_wav(self):

        # Find the dataset OS path
        for root, dirs, names in os.walk(self.dir_path):

            # Traverse the dataset songs
            for name in names:

                # Get the song OS path
                audio_path = os.path.join(root, name)

                # Remove the song OS path if the dir starts with .
                if os.path.basename(audio_path).startswith('.'):
                    os.remove(audio_path)
                    continue

                # Split the dir information
                format_audio = audio_path.split('.')[-1]

                # Skip tag files
                if format_audio == 'lab' or format_audio == 'joblib':  
                    continue

                if 'audio' in audio_path:
                    # Set the song values for preprocessing
                    song = AudioSegment.from_file(audio_path, format_audio)
                    song = song.set_channels(1)
                    song = song.set_frame_rate(16000)
                    song = song.set_sample_width(2)

                    # Change the song format name to .wav
                    wav_path = audio_path.replace('.' + format_audio, '.wav')

                    # Remove the dir path
                    os.remove(audio_path)

                    # Export the song in .wav format
                    song.export(wav_path, 'wav')
    
    # ======================================================================================================================
    
    # SVS Preprocessing
    def svs_get_vocal(self):

         # Find the dataset OS path
        for root, dirs, names in os.walk(self.dir_path):

            # Traverse the dataset songs
            for name in names:

                # Get song path
                wav_path = os.path.join(root, name)

                # Check if SVS has not been done into the song
                if '.wav' in name and '.unet.Vocal.wav' not in name:

                    # Get SVS file path
                    vocal_path = wav_path.replace('.wav', '.unet.Vocal.wav')
                    
                    # Perform SVS on the song
                    if not os.path.isfile(vocal_path):
                        split_vocal(wav_path)

                    # os.remove(wav_path)

    # ======================================================================================================================
    # Feature extraction
    def extract_feat_from_wav(self, win_size, hop_size, vocal_only):

        # Find the dataset OS path
        for root, dirs, names in os.walk(self.dir_path):

            # Traverse the dataset songs
            for name in names:
                
                # Feature vector
                DataX = []

                # Label list
                LabelY = []

                # Get the song OS path
                wav_path = os.path.join(root, name)
                
                # Skip SVS Vocal Files
                if vocal_only:
                    if '.unet.Vocal.' not in wav_path:
                        continue

                # Split the path
                wav_ext_str = '.' + wav_path.split('.')[-1]

                # Skip label file
                if wav_ext_str in '.lab.joblib':
                    continue

                # Change the song format name to .joblib
                feat_path = wav_path.replace('.wav', '_%.2f.joblib' % win_size)

                # Skip already extracted features file
                if os.path.isfile(feat_path):
                    continue
                
                # Read the song file
                signal, samplerate = soundfile.read(wav_path)

                # Calculate the window and hop size
                win_size_num = int(win_size * samplerate) 
                hop_size_num = int(hop_size * samplerate)

                # Change the song format of all sets name to .lab 
                if 'audio\\train' in wav_path:
                    lab_path = wav_path.replace('audio', 'labels').replace('\\train', '').replace(wav_ext_str, '.lab')
                elif 'audio\\test' in wav_path:
                    lab_path = wav_path.replace('audio', 'labels').replace('\\test', '').replace(wav_ext_str, '.lab')
                elif 'audio\\valid' in wav_path:
                    lab_path = wav_path.replace('audio', 'labels').replace('\\valid', '').replace(wav_ext_str, '.lab')

                # Change the song format of SVS files to .lab
                if 'unet.Vocal' in lab_path:
                    lab_path = lab_path.replace('.unet.Vocal', '')
                if '_MIX.lab' in lab_path:
                    lab_path = lab_path.replace('_MIX.lab', '.lab')

                # Read .lab files
                lab_file = open(lab_path)
                labels = lab_file.readlines()
                lab_file.close()

                # Extract the label values (start time, end time, sing/nosing label)
                for line in labels:

                    # Store the label values in a list
                    start_end = line.split(' ')

                    # Start time, end time, tag value
                    start = float(start_end[0])
                    end = float(start_end[1])
                    tag = start_end[2][:-1]

                    # Get the part signal from the label time range
                    part_signal = signal[int(start * samplerate):int(end * samplerate)]

                    # Declare the Feature AudioFeat class instance 
                    audio_feater = AudioFeat(win_size_n=win_size_num, hop_size_n=hop_size_num)

                    # Extract the song part signal features
                    part_signal_feat = audio_feater.get_audio_features(part_signal, samplerate)

                    # Add the extracted features to the feature vector
                    DataX.extend(part_signal_feat)

                    # Get the part signal label
                    f = lambda x: 1 if x == 'sing' else 0

                    # Add the label to the label list
                    LabelY.extend([f(tag)] * len(part_signal_feat))
                
                # Dump the extracted features and labels into the .joblib file
                joblib.dump({'data': DataX, 'lable': LabelY}, feat_path, compress=3)

# ======================================================================================================================

if __name__ == '__main__':
    action = '12'

    # Extract features and labels into .joblib file
    if '1' in action: 
        for dataset in ['Jamendo', 'Electrobyte']:
            dataset_dir_path = '.\Datasets\%s' % dataset
            preprocess_wav = Pre_procession_wav(dataset_dir_path)
            preprocess_wav.format_wav()
            preprocess_wav.svs_get_vocal();
            preprocess_wav.extract_feat_from_wav(1, 0.04, False) # False

    # Convert .joblib to .h5 file
    if '2' in action:  
        for data_dir in ['Jamendo', 'Electrobyte']: 
            for ext_str in [1]:  # , 0.10, 0.20, 0.30, 0.50, 0.80, 1, 2]:
                data_set_dir = '.\Datasets\%s' % data_dir
                data_ext_str = '_%.2f.joblib' % ext_str
                for vocal_only in [True, False]:
                    dataset_saver = AudioPreprocess()
                    dataset_saver.write_h5file(data_set_dir, data_ext_str, vocal_only)
    
