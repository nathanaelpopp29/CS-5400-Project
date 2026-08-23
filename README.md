# CS5400_semester_project

## Getting Started
To run this program, you must first install the dependencies found in "requirements.txt", then run the code with "python Run_all.py --data ./data --experiment Experiment_1" and "python Run_all.py --data ./data --experiment Experiment_1" for both datasets.


## Dataset Description 

This dataset includes data from two separate experiments, namely Experiment 1 and Experiment 2, featuring participants' behavioral and physiological responses. It covers the behavioral data from n-back tasks, i.e., 'n_back_responses' file, and data collected from Empatica wristbands and a Muse headband. Each participant's data is arranged in designated folders, and the dataset also accounts for excluded participants. 

Their reaction times and correct incorrect responses are stored in the 'n_back_responses.csv' file. EEG signals are stored in the file 'EEG_recordings.csv'. Other physiological data collected from Empatica wristbands from Left and Right hand are stored in separate CSV files such as 'EDA.csv', 'TEMP.csv', 'ACC.csv', 'BVP.csv', 'HR.csv', 'IBI.csv', and 'tags.csv'. Each file provides specific details about the measurements and the recording times. 

• n_back_responses.csv: Data collected from Chronos.
• EEG_recordings.csv: Data recorded using the muse headband.
• Left_ACC.csv and Right_ACC.csv: Accelerometer data.
• Left_BVP.csv and Right_BVP.csv: Blood Volume Pulse data.
• Left_EDA.csv and Right_EDA.csv: Electrodermal Activity data.
• Left_HR.csv and Right_HR.csv: Heart Rate data.
• Left_IBI.csv and Right_IBI.csv: Inter-Beat Interval data.
• Left_TEMP.csv and Right_TEMP.csv: Temperature data.
• Left_tags.csv and Right_tags.csv: Tags files.

Due to the inadvertent receipt of certain tags during experiments, the correct and unified tags are also included as 'tags.csv'.


## Original Paper and Database
https://physionet.org/content/brain-wearable-monitoring/1.0.0/#files-panel
