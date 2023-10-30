import pandas as pd
import os
import settle_prediction_main2

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
                                   'Data_date',
                                   'RMSE_date',
                                   'Final_date',
                                   'RMSE_hyper_original',
                                   'RMSE_hyper_nonlinear',
                                   'RMSE_hyper_weighted_nonlinear'])

# 입력 파일 저장 폴더에서 입력 파일의 이름을 파악하여 배열에 저장
for (root, directories, files) in os.walk(input_dir):  # 입력 파일 안의 모든 파일에 대해서
    for file in files:  # 모든 파일 중 하나의 파일에 대해서
        file_path = os.path.join(root, file)  # 파일 경로를 포함한 파일명 설정
        input_files.append(file_path)  # 파일명을 배열에 저장

# 입력 파일명 저장소의 파일 하나에 대해서 예측을 수행하고, 결과값으로 잔차값을 받아서 저장
# 데이터 사용 구간 = 60 + 30 * i where i = 0, 1, 2, 3, ....
# RMSE 산정 구간 = 140-160, 280-300, 420-440, 560-600, 700-720

for input_file in input_files:

    # RMSE 산정 구간 = 140-160, 280-300, 420-440, 560-600, 700-720
    for j in range(140, 840, 140):

        # 침하 예측 구간 설정 60, 90, 120, ... , j - 30 까지 수행
        for i in range(60, j, 30):

            # 침하 예측 수행
            return_values = settle_prediction_main2.run_settle_prediction_from_file(input_file=input_file,
                                                                                    output_dir='output',
                                                                                    data_usage=i,
                                                                                    is_data_usage_percent=False,
                                                                                    rmse_start=j,
                                                                                    rmse_range=20,
                                                                                    is_rmse_usage_percent=False,
                                                                                    additional_predict_percent=100,
                                                                                    plot_show=True,
                                                                                    print_values=True)

            # 반환값이 존재할 경우 (침하예측이 가능할 경우)
            if return_values is not None:
                df_overall.loc[len(df_overall.index)] = [input_file,  # 파일명
                                                         i,  # 데이터 사용 영역
                                                         j,  # RMSE 산정 영역
                                                         return_values[11], # 전체 데이터 영역
                                                         return_values[6],  # RMSE 1
                                                         return_values[7],  # RMSE 2
                                                         return_values[8]]  # RMSE 3

# 에러 파일을 출력
df_overall.to_csv('error_single.csv')