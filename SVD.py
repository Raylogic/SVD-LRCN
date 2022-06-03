from LRCN import compare_features, select_features, best_features_window_size, LRCN_Standard, LRCN_Window_Size, LRCN_Datasets

def SVD():
    action = '6'
    # '.\Datasets\Jamendo\_1.00_vocal.h5', '.\Datasets\Electrobyte\_1.00_vocal.h5'
    if '1' in action:  
        for dataset_h5_path in ['.\Datasets\Jamendo\_1.00_vocal.h5', '.\Datasets\Jamendo\_1.00_normal.h5',
                                '.\Datasets\Electrobyte\_1.00_vocal.h5', '.\Datasets\Electrobyte\_1.00_normal.h5',
                                '.\Datasets\Fusion\_1.00_vocal.h5', '.\Datasets\Fusion\_1.00_normal.h5']:
            LRCN_Standard(dataset_h5_path)
    if '2' in action:  
        for dataset_h5_path in ['.\Datasets\Jamendo\_1.00_vocal.h5', 
                                '.\Datasets\Electrobyte\_1.00_vocal.h5', 
                                '.\Datasets\Fusion\_1.00_vocal.h5']: 
            compare_features(dataset_h5_path)
    if '3' in action:
        for dataset_h5_path in ['.\Datasets\Jamendo\_1.00_vocal.h5', 
                                '.\Datasets\Electrobyte\_1.00_vocal.h5', 
                                '.\Datasets\Fusion\_1.00_vocal.h5']: 
            select_features(dataset_h5_path)
    if '4' in action:
        for data_dir_item in ['.\Datasets\Jamendo', '.\Datasets\Electrobyte', '.\Datasets\Fusion']:
            best_features_window_size(data_dir_item)
    if '5' in action:
        for data_dir_item in ['.\Datasets\Jamendo', '.\Datasets\Electrobyte', '.\Datasets\Fusion']:
            for step_size in range(5, 30):
                LRCN_Window_Size(data_dir_item + '\_1.00_vocal.h5', step_size)
    if '6' in action:
        for dataset_h5_path in ['.\Datasets\Jamendo\_1.00_vocal.h5', '.\Datasets\Jamendo\_1.00_normal.h5',
                                '.\Datasets\Electrobyte\_1.00_vocal.h5', '.\Datasets\Electrobyte\_1.00_normal.h5',
                                '.\Datasets\Fusion\_1.00_vocal.h5', '.\Datasets\Fusion\_1.00_normal.h5']:
            LRCN_Datasets(dataset_h5_path)

if __name__ == '__main__':
    for item in range(1):
        SVD()
