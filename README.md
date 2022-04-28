# PUCCH-Format-0-ML

- Run Dataset/datagen_main.m
- Run single_snr_train_multiple_snr_test.py for training and validation
    - Loads datasets from mat file
    - Compiles and fits NN model
    - Saves weights to h5 file
    - Plots train and val accuracy and loss, saves to Plots directory

- Then run analysis_single_snr_train_multiple_snr_test.py for testing results
    - Builds NN model by calling the neural_net_model function
    - Loads weights from h5 file
    - Recompiles  the model
    - 3 separate scenarios of testing are supported
        - Combined HW
            - Calls the test_comb_snr_hw function
                - Loads single mat file with hardware captures of all SNRs
                - Tests the NN performance, displays test accuracy, plots and saves confusion matrix
        - Separate Sim
            - Calls the test_sep_snr_sim function
                - Loads MATLAB simulated test data from mat files, separately for each SNR
                - Tests the NN performance, displays test accuracy, plots and saves confusion matrices for each SNR
        - Separate HW
            - Calls the test_sep_snr_hw function
                - Loads hardware capture mat files, saparately for each SNR
                - Tests the NN performance, displays test accuracy, plots and saves confusion matrices for each SNR

