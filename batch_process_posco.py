import pandas as pd
import os
import settle_prediction_main_posco

# 입력 파일이 있는 폴더명 지정: 사용자 직접 지정 필요
input_dir = 'data_posco'

# 침하 예측 결과 파일을 저장할 폴더명 지정: 사용자 직접 지정 필요
output_dir = 'output_posco'

# 에러 분석 결과 파일을 저장할 폴더명 지정: 사용자 직접 지정 필요
output_error = 'error'

# 입력 파일의 이름을 저장할 리스트 초기화
input_files = []

# 입력 파일 저장 폴더에서 입력 파일의 이름을 파악하여 배열에 저장
for (root, directories, files) in os.walk(input_dir):  # 입력 파일 안의 모든 파일에 대해서
    for file in files:  # 모든 파일 중 하나의 파일에 대해서
        file_path = os.path.join(root, file)  # 파일 경로를 포함한 파일명 설정
        input_files.append(file_path)  # 파일명을 배열에 저장

# 입력 파일명 저장소의 파일 하나에 대해서 예측을 수행하고, 결과값으로 잔차값을 받아서 저장
# 데이터 사용 구간 = 60 + 30 * i where i = 0, 1, 2, 3, ....
# RMSE 산정 구간 = 140-160, 280-300, 420-440, 560-600, 700-720

for input_file in input_files:

    settle_prediction_main_posco.run_settle_prediction_from_file(input_file=input_file,
                                                                 output_dir=output_dir,
                                                                 final_step_predict_percent=100,
                                                                 additional_predict_days=600,
                                                                 plot_show=True,
                                                                 print_values=True)