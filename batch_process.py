import settle_prediction_steps_main
import pandas as pd
import os

input_dir = 'data'
output_dir = 'output'
input_files = []

df = pd.DataFrame(columns=['File', 'Data_usage',
                           'RMSE_hyper_original',
                           'RMSE_hyper_nonlinear',
                           'RMSE_step',
                           'Final_error_hyper_original',
                           'Final_error_hyper_nonlinear',
                           'Final_error_step'])

for (root, directories, files) in os.walk(input_dir):
    for file in files:
        file_path = os.path.join(root, file)
        input_files.append(file_path)

for input_file in input_files:
    for i in range(20, 100, 20):
        ERROR = settle_prediction_steps.run_settle_prediction(input_file,
                                                              output_dir, i, 100, False, False)

        df.loc[len(df.index)] = [input_file, i, ERROR[0], ERROR[1], ERROR[2],
                                 ERROR[3], ERROR[4], ERROR[5]]


df.to_csv('Error.csv')