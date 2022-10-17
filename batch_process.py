import settle_prediction_steps_main
import pandas as pd
import os

input_dir = 'data_1'
output_dir = 'output_1'
input_files = []

df_overall = pd.DataFrame(columns=['File', 'Data_usage',
                           'RMSE_hyper_original',
                           'RMSE_hyper_nonlinear',
                           'Final_error_hyper_original',
                           'Final_error_hyper_nonlinear'])

df_multi_step = pd.DataFrame(columns=['File', 'Data_usage',
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
    for i in range(20, 100, 10):

        RETURN_VALUES = settle_prediction_steps_main.\
            run_settle_prediction(input_file, output_dir, i, 100, False, False)

        df_overall.loc[len(df_overall.index)] = [input_file, i,
                                                 RETURN_VALUES[0],
                                                 RETURN_VALUES[1],
                                                 RETURN_VALUES[3],
                                                 RETURN_VALUES[4]]

        if RETURN_VALUES[6]:
            df_multi_step.loc[len(df_overall.index)] = [input_file, i,
                                                        RETURN_VALUES[0],
                                                        RETURN_VALUES[1],
                                                        RETURN_VALUES[2],
                                                        RETURN_VALUES[3],
                                                        RETURN_VALUES[4],
                                                        RETURN_VALUES[5]]

# 에러 파일 출력
df_overall.to_csv('Error_overall.csv')
df_multi_step.to_csv('Error_multi_step.csv')
