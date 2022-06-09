# coding=utf-8
import csv
import os
import numpy

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Flatten, Dropout, Reshape, ConvLSTM2D, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
# from keras.utils import plot_model
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import StandardScaler

from Evaluation import Evaluator
from Setup import VERBOSE
from Feature import AudioFeat
from Preprocessing import AudioPreprocess

# LRCN model
def LRCN(input_size, time_steps):
    model = Sequential()
    drop_rate = 0.2
    model.add(Reshape((1, 1, time_steps, input_size), input_shape=(time_steps, input_size)))
    model.add(ConvLSTM2D(13, 1, 4, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(drop_rate))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1, activation='sigmoid'))
    adamOP = Adam()
    model.compile(loss='binary_crossentropy', optimizer=adamOP, metrics=['accuracy'])
    return model

# ========================================================================================================================================

# Apply the window size to the data instances
def get_step_data(dataX, dataY, step, array_dim_num=3):
    feature_dimension = numpy.shape(dataX)[-1]
    finalX = []
    finalY = []
    for item_y in range(0, len(dataY), step):
        if list(dataY[item_y:item_y + step]).count(1) == step:
            finalX.append(dataX[item_y:item_y + step])
            finalY.append(1)
        if list(dataY[item_y:item_y + step]).count(0) == step:
            finalX.append(dataX[item_y:item_y + step])
            finalY.append(0)
    finalX = numpy.array(finalX)
    finalY = numpy.array(finalY)
    if array_dim_num == 4:
        finalX = numpy.reshape(finalX, (-1, 1, 1, step, feature_dimension))
    return finalX, finalY

# ========================================================================================================================================

class LRCN_Model:

    # Constructor
    def __init__(self, case, dataset_path, feature_list=None):

        # Case of use
        self.case = case

        # List for features to extract
        if feature_list is None:
            feature_list = ['feat_plp', 'feat_chroma', 'feat_lpc', 'feat_mfcc', 'feat_MFCC_delta', 'feat_MFCC_delta_delta', 'feat_spec_all']
        
        # Audio Preprocessing Instance
        ADP = AudioPreprocess()

        # Get train, valid and test set
        self.trainX, self.trainY, self.testX, self.testY, self.validX, self.validY = ADP.load_Dataset_from_h5file(dataset_path)

        # Feature Extraction Instance
        AFE = AudioFeat()

        # Extract the audio features from train, valid and test set
        self.trainX = AFE.specify_feature_list_index(self.trainX, feature_list)
        self.testX = AFE.specify_feature_list_index(self.testX, feature_list)
        self.validX = AFE.specify_feature_list_index(self.validX, feature_list)

    # =========================================================================================================================

    # Train LRCN and perform the sing/nosing classification
    def train_load(self, case, use_model='lrcn', load=True, scale_normal=True, step_size=20):

        # Call LRCN Model
        model = LRCN(input_size=self.trainX.shape[1], time_steps=step_size)

        # Scale the values through standarization
        if scale_normal:
            scaler = StandardScaler()
            scaler.fit(self.validX)
            self.trainX = scaler.transform(self.trainX)
            self.validX = scaler.transform(self.validX)

        # Get step data in train and valid sets
        train_X, train_Y = get_step_data(self.trainX, self.trainY, step=step_size)
        valid_X, valid_Y = get_step_data(self.validX, self.validY, step=step_size)
        
        # If model is loaded
        if load:
            model = load_model('models/%s.model' % use_model)
            model.load_weights('models/%s.weights' % use_model)
        else:
            # Load checkpoints of model during training
            checkpoint = ModelCheckpoint('models/weights_best.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')

            # Stop the training when the model stops improving
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', )  # baseline=None, restore_best_weights=True)

            # Fit training data into model history
            history = model.fit(train_X, train_Y,
                                epochs=10000, verbose=VERBOSE,  # 10000
                                shuffle=True,
                                validation_data=(valid_X, valid_Y),
                                callbacks=[checkpoint, early_stop, TensorBoard(log_dir='logs'), ]) #early_stop

            # Plot Model Accuracy
            plt.figure()
            plt.plot()
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('plots/%s_accuracy_%s.svg' % (use_model, case), format='svg')

            # Plot Model Loss
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('plots/%s_loss_%s.svg' % (use_model, case), format='svg')

            # Save model and weights
            model.save('models/%s.model' % use_model)
            model.save_weights('models/%s.weights' % use_model)
        
        # print('Predicting...')

        # Predict the class from Valid set
        predict_Y = model.predict(valid_X)
        predict_Y = numpy.where(predict_Y > 0.5, 1, 0)

        # Call Evaluator instance
        evaluator = Evaluator(predict_Y, valid_Y)
        
        # Get metrics from predicting valid set
        accuracy, precision, recall, f1measure = evaluator.evaluate()

        # Return metrics
        return accuracy, precision, recall, f1measure

# ========================================================================================================================================

# Compare features across datasets
def compare_features(dataset_h5_path):

    # List of features to compare
    list_features = ['feat_plp', 'feat_chroma', 'feat_lpc', 'feat_mfcc', 'feat_MFCC_delta', 'feat_MFCC_delta_delta', 'feat_spec_all']

    # Explore each feature from the list
    for feature in list_features:
        print('Used Feature:\t%s' % feature)

        # Train the LRCN with the feature
        LRCN_Build = LRCN_Model('compare_features', dataset_h5_path, feature_list=[feature])

        # Extract the model metrics
        accuracy, precision, recall, f1measure = LRCN_Build.train_load('lrcn', 'compare_features', load=False)

        # Open a CSV file
        res_csv = open('.\plots\compare_features.csv', 'a')

        # Create a file writer
        writer = csv.writer(res_csv)

        # Write the metrics in the CSV file
        writer.writerow([dataset_h5_path, feature, accuracy, precision, recall, f1measure])

        # Close the CSV file
        res_csv.close()

    return 0

# ========================================================================================================================================

# Get best features for LRCN
def select_features(dataset_h5_path):

    # List of features to add
    feature_list = ['feat_spec_all', 'feat_mfcc', 'feat_MFCC_delta', 'feat_MFCC_delta_delta', 'feat_lpc', 'feat_plp', 'feat_chroma']

    # Open a CSV file
    res_csv = open('.\plots\combine_features.csv', 'a')

    # Create a file writer
    writer = csv.writer(res_csv)

    # List of best added features
    # Add the first feature to the list
    combine_features = [feature_list[0]]

    # Train the LRCN with the initial feature
    LRCN_Build = LRCN_Model('combine_features', dataset_h5_path, feature_list=combine_features)

    # Extract the model metrics
    accuracy, precision, recall, curr_f1measure = LRCN_Build.train_load('lrcn', 'combine_features', load=False)

    # Write the metrics in the CSV file
    writer.writerow([dataset_h5_path,','.join(combine_features), accuracy, precision, recall, curr_f1measure])

    # Counter
    cont = 1

    # Iterate over the remaining feature list
    for item in feature_list[1:]:
        # Add one to counter
        cont += 1

        # Train the LRCN with the added feature
        LRCN_Build = LRCN_Model('combine_features', dataset_h5_path, feature_list=combine_features + [item])

        # Extract the model metrics
        accuracy, precision, recall, added_f1measure = LRCN_Build.train_load('lrcn', 'combine_features', load=False)

        # Write the metrics in the CSV file
        writer.writerow([dataset_h5_path,','.join(combine_features+[item]), accuracy, precision, recall, added_f1measure])

        # If the new F1 Measure is better than the current one, update it and add the bew feature to the list 
        if added_f1measure > curr_f1measure:
            curr_f1measure = added_f1measure
            combine_features.append(item)
            print('++++%d %s' % (cont, item))
        else:
            print('----%d %s' % (cont, item))

    # Print the features that give the best F1-Measure
    print('Best fused features = {}'.format(combine_features))

    # Close the CSV
    res_csv.close()

    # Return the list of best features
    return combine_features

# ========================================================================================================================================

# Get best features for LRCN
def best_features_window_size(dataset_dir):

    # List of features to add
    feat_list = ['feat_spec_all', 'feat_mfcc', 'feat_MFCC_delta', 'feat_MFCC_delta_delta', 'feat_lpc', 'feat_plp']

    # Best window step
    best_win = ''

    # Best F1-Measure
    best_f1 = 0

    # Open a CSV file
    res_csv = open('.\plots\get_best_features.csv', 'a')

    # Create a file writer
    writer = csv.writer(res_csv)
    
    # Traverse each dataset to evaluate after being preprocessed into an H5 File 
    for h5_file_item in os.listdir(dataset_dir):
        if '.h5' in h5_file_item:

            # Get the H5 Dataset path
            dataset_h5_path = os.path.join(dataset_dir, h5_file_item)

            # Train the LRCN with the current dataset
            LRCN_Build = LRCN_Model('best_features', dataset_h5_path, feature_list=feat_list)

            # Extract the model metrics
            accuracy, precision, recall, f1measure = LRCN_Build.train_load('lrcn', 'best_features', load=False)

            # If the F1 Measure is better than the current best one, update it and the window step used
            if f1measure > best_f1:
                best_f1 = f1measure
                best_win = h5_file_item.split('_')[-2]
                print('Window Size = \t%s\t%.3f\t%.3f\t%.3f\t%.3f' % (h5_file_item.split('_')[-2], accuracy, precision, recall, f1measure))
            
            # Write the metrics in the CSV file
            writer.writerow([dataset_h5_path, h5_file_item.split('_')[-2], accuracy, precision, recall, f1measure])

    # Close the CSV
    res_csv.close()

    # Return the best window size
    return best_win

# ========================================================================================================================================

# Compare results between using or not using SVS preprocessing
def LRCN_Standard(dataset_h5_path):

    # Open a CSV file
    res_csv = open('.\plots\LRCN_standard.csv', 'a')

    # Create a file writer
    writer = csv.writer(res_csv)

    # List of features to add
    feat_list = ['feat_spec_all', 'feat_mfcc', 'feat_MFCC_delta', 'feat_MFCC_delta_delta', 'feat_lpc', 'feat_plp']

    # Train the LRCN with the current dataset
    LRCN_Build = LRCN_Model('standard', dataset_h5_path, feature_list=feat_list)

    # Extract the model metrics
    accuracy, precision, recall, f1measure = LRCN_Build.train_load('lrcn', 'standard', load=False)

    # Print the model metrics
    print('%s\t %.3f\t%.3f\t%.3f\t%.3f\n' % (dataset_h5_path, accuracy, precision, recall, f1measure))

    # Write the metrics in the CSV file
    writer.writerow([dataset_h5_path, accuracy, precision, recall, f1measure])

    return 0

# ==========================================================================================================================

# Compare results between different frame window sizes
def LRCN_Window_Size(dataset_h5_path, st_size):

    # Open a CSV file
    res_csv = open('.\plots\LRCN_window.csv', 'a')

    # Create a file writer
    writer = csv.writer(res_csv)

    # List of features to add
    feat_list = ['feat_spec_all', 'feat_mfcc', 'feat_MFCC_delta', 'feat_MFCC_delta_delta', 'feat_lpc', 'feat_plp']

    # Train the LRCN with the current dataset
    LRCN_Build = LRCN_Model('window_size', dataset_h5_path, feature_list=feat_list)

    # Extract the model metrics
    accuracy, precision, recall, f1measure = LRCN_Build.train_load('lrcn', 'window_size', load=False, step_size=st_size)

    # Print the model metrics
    print('%s\t %.3f\t%.3f\t%.3f\t%.3f\n' % (dataset_h5_path, accuracy, precision, recall, f1measure))

    # Write the metrics in the CSV file
    writer.writerow([dataset_h5_path, accuracy, precision, recall, f1measure])

    return 0

# ==========================================================================================================================

# Compare final resulta across datasets
def LRCN_Datasets(dataset_h5_path):

    # Open a CSV file
    res_csv = open('.\plots\LRCN_datasets.csv', 'a')

    # Create a file writer
    writer = csv.writer(res_csv)

    # List of features to add
    feat_list = ['feat_spec_all', 'feat_mfcc', 'feat_MFCC_delta', 'feat_MFCC_delta_delta', 'feat_lpc', 'feat_plp']

    # Train the LRCN with the current dataset
    LRCN_Build = LRCN_Model('datasets', dataset_h5_path, feature_list=feat_list)

    # Extract the model metrics
    accuracy, precision, recall, f1measure = LRCN_Build.train_load('lrcn', 'datasets', load=False)

    # Print the model metrics
    print('%s\t %.3f\t%.3f\t%.3f\t%.3f\n' % (dataset_h5_path, accuracy, precision, recall, f1measure))

    # Write the metrics in the CSV file
    writer.writerow([dataset_h5_path, accuracy, precision, recall, f1measure])

    return 0

if __name__ == '__main__':
    pass
