import pandas as pd
import os
import settle_prediction_main

# 입력 파일이 있는 폴더명 지정: 사용자 직접 지정 필요
input_dir = 'data'

# 침하 예측 결과 파일을 저장할 폴더명 지정: 사용자 직접 지정 필요
output_dir = 'output'

# 에러 분석 결과 파일을 저장할 폴더명 지정: 사용자 직접 지정 필요
output_error = 'error'

# 침하 계측값의 단위 지정: 응동 m, 서컨 cm
settle_unit = 'cm'

# 입력 파일의 이름을 저장할 리스트 초기화
input_files = []

# 일단 및 다단 성토를 포함한 예측의 에러를 저장할 데이터프레임 초기화
df_overall = pd.DataFrame(columns=['File',
                                   'Data_usage',
                                   'RMSE_hyper_original',
                                   'RMSE_hyper_nonlinear',
                                   'RMSE_hyper_weighted_nonlinear',
                                   'Final_error_hyper_original',
                                   'Final_error_hyper_nonlinear',
                                   'Final_error_hyper_weighted_nonlinear'])

# 입력 파일 저장 폴더에서 입력 파일의 이름을 파악하여 배열에 저장
for (root, directories, files) in os.walk(input_dir):  # 입력 파일 안의 모든 파일에 대해서
    for file in files:  # 모든 파일 중 하나의 파일에 대해서
        file_path = os.path.join(root, file)  # 파일 경로를 포함한 파일명 설정
        input_files.append(file_path)  # 파일명을 배열에 저장

# 입력 파일명 저장소의 파일 하나에 대해서 예측을 수행하고, 결과값으로 잔차값을 받아서 저장
for input_file in input_files:

    # 최종 성토 이후 데이터 사용 영역에 대해서 [20 30 40 50 60 70 80 90]
    for i in range(20, 100, 10):
        # 침하 예측을 수행하고 반환값 저장
        return_values = settle_prediction_main.run_settle_prediction_from_file(input_file=input_file)

        # 데이터프레임에 일단 및 다단 성토를 포함한 예측의 에러를 저장
        df_overall.loc[len(df_overall.index)] = [input_file,  # 파일명
                                                 i,  # 데이터 사용 영역
                                                 return_values[6], return_values[7], return_values[8],  # RMSE
                                                 return_values[9], return_values[10], return_values[11]]  # 최종 침하량 에러

# 에러 파일을 출력
df_overall.to_csv('error_single.csv')
