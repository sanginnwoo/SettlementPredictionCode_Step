
# =================
# Import 섹션
# =================

import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

# =================
# Function 섹션
# =================

# 주어진 계수를 이용하여 아사오카법 기반 시간-침하 곡선 반환
def generate_data_asaoka(px, pt, dt):
    return (px[1] / (1 - px[0])) * (1 - (px[0] ** (pt / dt)))

# 아사오카법 목표 함수
def fun_asaoka(px, ps_b, ps_a):
    return px[0] * ps_b + px[1] - ps_a


# ====================
# 파일 읽기, 데이터 설정
# ====================

# CSV 파일 읽기
data = pd.read_csv("data/2-6_J-01.csv")

# 시간, 침하량, 성토고 배열 생성
time = data['Time'].to_numpy()
settle = data['Settlement'].to_numpy()
surcharge = data['Surcharge'].to_numpy()

# 만일 침하량의 단위가 m일 경우, 조정
settle = settle * 100

# 데이터 닫기

# 마지막 계측 데이터 index + 1 파악
final_index = time.size

# =================
# 성토 단계 구분
# =================

# 성토 단계 시작 index 리스트 초기화
step_start_index = [0]

# 성토 단계 끝 index 리스트 초기화
step_end_index = []

# 현재 성토고 설정
current_surcharge = surcharge[0]

# 단계 시작 시점 초기화
step_start_date = 0

# 모든 시간-성토고 데이터에서 순차적으로 확인
for index in range(len(surcharge)):

    # 만일 성토고의 변화가 있을 경우,
    if surcharge[index] != current_surcharge:
        step_end_index.append(index)
        step_start_index.append(index)
        current_surcharge = surcharge[index]

# 마지막 성토 단계 끝 index 추가
step_end_index.append(len(surcharge) - 1)

# =================
# 성토 단계 조정
# =================
# 성토고 유지 기간이 매우 짧을 경우, 해석 단계에서 제외

# 조정 성토 시작 및 끝 인덱스 리스트 초기화
step_start_index_adjust = []
step_end_index_adjust = []

# 각 성토 단계 별로 분석
for i in range(0, len(step_start_index)):

    # 현 단계 성토 시작일 / 끝일 파악
    step_start_date = time[step_start_index[i]]
    step_end_date = time[step_end_index[i]]

    # 현 성토고 유지 일수 및 데이터 개수 파악
    step_span = step_end_date - step_start_date
    step_data_num = step_end_index[i] - step_start_index[i] + 1

    # 성토고 유지일 및 데이터 개수 기준 적용
    if step_span > 30 and step_data_num > 5:
        step_start_index_adjust.append((step_start_index[i]))
        step_end_index_adjust.append((step_end_index[i]))

#  성토 시작 및 끝 인덱스 리스트 업데이트
step_start_index = step_start_index_adjust
step_end_index = step_end_index_adjust

# 침하 예측을 수행할 단계 설정 (현재 끝에서 2단계 이용)
step_start_index = step_start_index[-2:]
step_end_index = step_end_index[-2:]

# 성토 단계 횟수 파악 및 저장
num_steps = len(step_start_index)

# ===========================
# 최종 단계 데이터 사용 범위 조정
# ===========================

# 데이터 사용 퍼센트에 해당하는 기간 계산
final_step_end_date = time[-1]
final_step_start_date = time[step_start_index[num_steps - 1]]
final_step_period = final_step_end_date - final_step_start_date
final_step_predict_end_date = final_step_start_date + final_step_period * 50 / 100

# 데이터 사용 끝 시점 인덱스 초기화
final_step_predict_end_index = -1

# 데이터 사용 끝 시점 인덱스 검색
count = 0
for day in time:
    count = count + 1
    if day > final_step_predict_end_date:
        final_step_predict_end_index = count - 1
        break

# 마지막 성토 단계, 마지막 계측 시점 인덱스 업데이트
final_step_monitor_end_index = step_end_index[num_steps - 1]
step_end_index[num_steps - 1] = final_step_predict_end_index

# =================
# 추가 예측 구간 반영
# =================

# 추가 예측 일 입력 (현재 전체 계측일 * 계수)
add_days = time[-1]

# 마지막 성토고 및 마지막 계측일 저장
final_surcharge = surcharge[final_index - 1]
final_time = time[final_index - 1]

# 추가 시간 및 성토고 배열 설정 (100개의 시점 설정)
time_add = np.linspace(final_time + 1, final_time + add_days, 100)
surcharge_add = np.ones(100) * final_surcharge

# 기존 시간 및 성토고 배열에 붙이기
time = np.append(time, time_add)
surcharge = np.append(surcharge, surcharge_add)

# 마지막 인덱스값 재조정
final_index = time.size

# ===============================
# Settlement prediction (Asaoka)
# ===============================

# 성토 마지막 데이터 추출
tm_asaoka = time[step_start_index[num_steps - 1]:step_end_index[num_steps - 1]]
sm_asaoka = settle[step_start_index[num_steps - 1]:step_end_index[num_steps - 1]]

# 초기 시점 및 침하량 산정
t0_asaoka = tm_asaoka[0]
s0_asaoka = sm_asaoka[0]

# 초기 시점에 대한 시간 조정
tm_asaoka = tm_asaoka - t0_asaoka

# 초기 침하량에 대한 침하량 조정
sm_asaoka = sm_asaoka - s0_asaoka

# Interpolation 함수 설정 (3차 보간)
inter_fn = interp1d(tm_asaoka, sm_asaoka, kind='cubic')

# 데이터 구축 간격 및 그에 해당하는 데이터 포인트 개수 설정
interval = 10
num_data = int(tm_asaoka[-1]/3)

# 등간격 시간 및 침하량 데이터 설정
tm_asaoka_inter = np.linspace(0, tm_asaoka[-1], num=num_data, endpoint=True)
sm_asaoka_inter = inter_fn(tm_asaoka_inter)

# 이전 이후 등간격 침하량 배열 구축
sm_asaoka_before = sm_asaoka_inter[0:-2]
sm_asaoka_after = sm_asaoka_inter[1:-1]

# Least square 변수 초기화
x0 = np.ones(2)

# Least square 분석을 통한 침하 곡선 계수 결정
res_lsq_asaoka = least_squares(fun_asaoka, x0, args=(sm_asaoka_before, sm_asaoka_after))

# 계측 및 예측 침하량 표시
plt.scatter(sm_asaoka_before, sm_asaoka_after, s=50,
            facecolors='white', edgecolors='black', label='measured data')

# 그래프 표시
plt.show()


