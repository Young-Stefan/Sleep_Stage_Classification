Author: Shao-Po (Shawn) Huang
Contact: shawnh871@gmail.com
Team Members: Bryce Carr, Ajay Gadwal, Ethan Muyskens, Christian Schonhoeft
Institution: University of Texas at Austin, Department of Biomedical Engineering
Date Last Modified: 05/11/20

Instructions for the Machine Learning (ML) Software:

Included in ML folder are the following files:

1) edf_to_csv.py
2) Train_Validate_Test.py
3) Untrained_CNN.py
4) Untrained_kNN.py
5) Untrained_SVM.py
6) test_csv_file1.csv
7) SleepLabels1.csv
8) "kNN_6category"
9) "SVM_6category"
10) "CNN_6category"
11) "kNN_3category"
12) "SVM_3category"
13) "CNN_3category"
14) "LDA_6category"
15) "LDA_3category"

The "edf_to_csv.py" program will convert data stored in .edf files to .csv files. 

Data was obtained from the following database:
https://www.physionet.org/content/sleep-edfx/1.0.0/

The database separates patient data into two files: a PSG with the raw signal data and a hypnogram with the sleep stages.
After converting the PSG files, remove all baseline values. The hypnogram cannot be converted into a .csv with the "edf_to_csv.py" program.
An online EDF viewer can be used instead: https://bilalzonjy.github.io/EDFViewer/EDFViewer.html
The values can then be transferred and reformatted manually in Excel and saved as a .csv.
Sample PSG and Hypnogram data (after processing) have been provided in the ML folder as "test_csv_file1.csv" and "SleepLabels1.csv", respectively, for reference.

After initial processing, the data can then be loaded into the "Train_Validate_Test.py" program.
There are sections within the program that have been denoted as "kNN/SVM only" vs. "CNN only", "6-category" vs. "3-category", and "Training/Validation" vs. "Testing".
Only use sections that pertain to the model that you wish to use. Comment out all other sections.

Trained models have been provided in the ML folder.
6-category kNN: "kNN_6category"
6-category SVM: "SVM_6category"
6-category CNN: "CNN_6category"
3-category kNN: "kNN_3category"
3-category SVM: "SVM_3category"
3-category CNN: "CNN_3category"
6-category LDA: "LDA_6category"
3-category LDA: "LDA_3category"

If you would like to retrain your own model, please use the untrained models:
1) Untrained_kNN.py
2) Untrained_SVM.py
3) Untrained_CNN.py

At the top of each program, the libraries/packages required are listed. Please download all of them before using the programs.