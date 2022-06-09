# SVD-LRCN
Evaluating Singing Voice Detection in Electronic Music with a Long-Term Recurrent Convolutional Network

## Content description
- Config.py = Feature vector and LRCN parameter settings
- Api.py = Test the feature extraction and label identification
- Feature.py = Perform the feature extraction
- Preprocessing.py = Dataset preprocessing (Audio formatting, Singing Voice Separation, Feature extraction, Data generation)
- LRCN.py = LRCN model implementation and experimentation
- SVD.py = Main program to use the LRCN
- Evaluation.py = Evaluation metrics calculation
- Box.py = Plot experiments results

Additionals:
- Network.py = U-Net model implementation
- UNet.py = Singing Voice Separation task
- Line.py = Alternative to plot experiments results

## Experiments
1. Feature combination
2. Block size comparison [5-29]
3. Comparison of Singing Voice Separation (SVS)
4. Evaluated datasets: [Jamendo, Electrobyte]

## How to run the program
1. Install the necessary Python libraries: `pip install requirements.txt`
2. Download the Jamendo and Electrobyte datasets
3. Run the Preprocessing.py script
4. On the SVD.py, select the experiment to test and run the script
