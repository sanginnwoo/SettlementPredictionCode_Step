"""
Title: Soft ground settlement prediction considering the step loading
Main Developer: Sang Inn Woo, Ph.D. @ Incheon National University
Starting Date: 2022-08-11
Abstract:
This main objective of this code is to predict
time vs. (consolidation) settlement curves of soft clay ground
under step loading conditions.
The methodologies used are 1) superposition of time-settlement curves
and 2) nonlinear regression for hyperbolic curves.
"""

# =================
# Import 섹션
# =================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares



# =================
# Function 섹션
# =================

# 주어진 계수를 이용하여 쌍곡선 시간-침하 곡선 반환
def generate_data_hyper(px, pt):
    return pt / (px[0] * pt + px[1])

# 회귀식과 측정치와의 잔차 반환 (비선형 쌍곡선)
def fun_hyper_nonlinear(px, pt, py):
    return pt / (px[0] * pt + px[1]) - py

# 회귀식과 측정치와의 잔차 반환 (기존 쌍곡선)
def fun_hyper_original(px, pt, py):
    return px[0] * pt + px[1] - pt / py

# RMSE 산정
def fun_rmse(py1, py2):
    mse = np.square(np.subtract(py1, py2)).mean()
    return np.sqrt(mse)



# =================
# 파일 설정 / 입력값
# =================

# 파일명 설정 : 임시
#filename = "1_S-12.csv"
#filename = "1_SP-11.csv"
#filename = "1_SP-17.csv"
#filename = "1_SP-23.csv"
#filename = "3_SP3-65.csv"
#filename = "3_SP3-68.csv"
#filename = "4_S-11.csv"
filename = "west_test_2_5_No_54.csv"

# 최종 성토 단계의 데이터 사용 퍼센트 설정 : 사용자 입력값
final_step_predict_percent = 20

# 추가 계측 구간 퍼센트 설정 : 사용자 입력값
additional_predict_percent = 100

# 성토 단계 시작 index 리스트 초기화 : 사용자 입력값
step_start_index = []

# 성토 단계 끝 index + 1 리스트 초기화
step_end_index = []

# 파일명에 따라서, 성토 단계 index 설정
if filename == "1_S-12.csv":
    step_start_index = [0, 56]
    step_end_index = [56, 143]
elif filename == "1_SP-11.csv":
    step_start_index = [0, 10, 37, 79]
    step_end_index = [10, 37, 79, 124]
elif filename == "1_SP-17.csv":
    step_start_index = [0, 122]
    step_end_index = [122, 163]
elif filename == "1_SP-23.csv":
    step_start_index = [0, 18, 40, 90]
    step_end_index = [18, 40, 90, 124]
elif filename == "3_SP3-65.csv":
    step_start_index = [0, 94, 136]
    step_end_index = [ 94, 136, 182]
elif filename == "3_SP3-68.csv":
    step_start_index = [0, 9, 48, 88]
    step_end_index = [9, 48, 88, 127]
elif filename == "4_S-11.csv":
    step_start_index = [0, 10, 46, 51, 120]
    step_end_index = [10, 46, 51, 120, 157]
elif filename == "west_test_2_5_No_54.csv":
    step_start_index = [111, 269]
    step_end_index = [269, 409]

# 성토 단계 횟수 파악 및 저장
num_steps = len(step_start_index)



# ====================
# 파일 읽기, 데이터 설정
# ====================

# CSV 파일 읽기
data = pd.read_csv(filename)

# 시간, 침하량, 성토고 배열 생성
time = data['Time'].to_numpy()
settle = data['Settle'].to_numpy()
surcharge = data['Surcharge'].to_numpy()

# 마지막 계측 데이터 index + 1 파악
final_index = time.size



# =================
# 성토 단계 구분
# =================

# todo: 성토고 데이터를 분석하여, 각 단계 계측 시작 및 끝일에 해당하는 인덱스 파악 필요
# 꼭 이전 단계 마지막 인덱스와 현재 단계 처음 인덱스가 이어질 필요는 없음
# (각 단계별 시간, 침하를 초기화 한후 예측을 수행하므로...)



# ===========================
# 최종 단계 데이터 사용 범위 조정
# ===========================

# 데이터 사용 퍼센트에 해당하는 기간 계산
final_step_end_date = time[-1]
final_step_start_date = time[step_start_index[num_steps - 1]]
final_step_period = final_step_end_date - final_step_start_date
final_step_predict_end_date = final_step_start_date + final_step_period * final_step_predict_percent / 100

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
add_days = (additional_predict_percent / 100) * time[-1]

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



# =============================
# Settlement Prediction (Step)
# =============================

# 예측 침하량 초기화
sp_step = np.zeros(time.size)

# 만일 계수 중에 하나가 음수가 나오면 에러 출력
error_step = 0

# 각 단계별로 진행
for i in range(0, num_steps):

    # 각 단계별 계측 시점과 계측 침하량 배열 생성
    tm_this_step = time[step_start_index[i]:step_end_index[i]]
    sm_this_step = settle[step_start_index[i]:step_end_index[i]]

    # 이전 단계까지 예측 침하량 중 현재 단계에 해당하는 부분 추출
    sp_this_step = sp_step[step_start_index[i]:step_end_index[i]]

    # 현재 단계 시작 부터 끝까지 시간 데이터 추출
    tm_to_end = time[step_start_index[i]:final_index]

    # 기존 예측 침하량에 대한 보정
    sm_this_step = sm_this_step - sp_this_step

    # 초기 시점 및 침하량 산정
    t0_this_step = tm_this_step[0]
    s0_this_step = sm_this_step[0]

    # 초기 시점에 대한 시간 조정
    tm_this_step = tm_this_step - t0_this_step
    tm_to_end = tm_to_end - t0_this_step

    # 초기 침하량에 대한 침하량 조정
    sm_this_step = sm_this_step - s0_this_step

    # 침하 곡선 계수 초기화
    x0 = np.ones(2)

    # 회귀분석 시행
    res_lsq_hyper_nonlinear \
        = least_squares(fun_hyper_nonlinear, x0, args=(tm_this_step, sm_this_step))

    # 쌍곡선 계수 저장 및 출력
    x_step = res_lsq_hyper_nonlinear.x
    print(x_step)

    # 만일 계수 중에 하나가 음수일 경우, 에러 메세지 출력하고 Break
    #if x_step[0] < 0 or x_step[0] < 0 :
    #    print("More than one parameter is negative!")
    #    error_step = 1
    #    break

    # 현재 단계 예측 침하량 산정 (침하 예측 끝까지)
    sp_to_end_update = generate_data_hyper(x_step, tm_to_end)

    # 예측 침하량 업데이트
    sp_step[step_start_index[i]:final_index] = \
        sp_step[step_start_index[i]:final_index] + sp_to_end_update + s0_this_step



# =========================================================
# Settlement prediction (nonliner and original hyperbolic)
# =========================================================

# 성토 마지막 데이터 추출
tm_hyper = time[step_start_index[num_steps-1]:step_end_index[num_steps-1]]
sm_hyper = settle[step_start_index[num_steps-1]:step_end_index[num_steps-1]]

# 현재 단계 시작 부터 끝까지 시간 데이터 추출
time_hyper = time[step_start_index[num_steps-1]:final_index]

# 초기 시점 및 침하량 산정
t0_hyper = tm_hyper[0]
s0_hyper = sm_hyper[0]

# 초기 시점에 대한 시간 조정
tm_hyper = tm_hyper - t0_hyper
time_hyper = time_hyper - t0_hyper

# 초기 침하량에 대한 침하량 조정
sm_hyper = sm_hyper - s0_hyper

# 회귀분석 시행 (비선형 쌍곡선)
x0 = np.ones(2)
res_lsq_hyper_nonlinear = least_squares(fun_hyper_nonlinear, x0,
                                        args=(tm_hyper, sm_hyper))
# 비선형 쌍곡선 법 계수 저장 및 출력
x_hyper_nonlinear = res_lsq_hyper_nonlinear.x
print(x_hyper_nonlinear)

# 회귀분석 시행 (기존 쌍곡선법) - (0, 0)에 해당하는 초기 데이터를 제외하고 회귀분석 실시
x0 = np.ones(2)
res_lsq_hyper_original = least_squares(fun_hyper_original, x0,
                                       args=(tm_hyper[1:], sm_hyper[1:]))
# 기존 쌍곡선 법 계수 저장 및 출력
x_hyper_original = res_lsq_hyper_original.x
print(x_hyper_original)

# 현재 단계 예측 침하량 산정 (침하 예측 끝까지)
sp_hyper_nonlinear = generate_data_hyper(x_hyper_nonlinear, time_hyper)
sp_hyper_original = generate_data_hyper(x_hyper_original, time_hyper)

# 예측 침하량 산정
sp_hyper_nonlinear = sp_hyper_nonlinear + s0_hyper
sp_hyper_original = sp_hyper_original + s0_hyper
time_hyper = time_hyper + t0_hyper



# ==========
# 에러 산정
# ==========

# RMSE 계산 데이터 구간 설정 (계측)
sm_rmse = settle[final_step_predict_end_index:final_step_monitor_end_index]

# RMSE 계산 데이터 구간 설정 (단계)
sp_step_rmse = sp_step[final_step_predict_end_index:final_step_monitor_end_index]

# RMSE 계산 데이터 구간 설정 (쌍곡선)
sp_hyper_nonlinear_rmse = sp_hyper_nonlinear[final_step_predict_end_index - step_start_index[num_steps - 1]:
                                             final_step_predict_end_index - step_start_index[num_steps - 1] +
                                             final_step_monitor_end_index - final_step_predict_end_index]
sp_hyper_original_rmse = sp_hyper_original[final_step_predict_end_index - step_start_index[num_steps - 1]:
                                           final_step_predict_end_index - step_start_index[num_steps - 1] +
                                           final_step_monitor_end_index - final_step_predict_end_index]

# RMSE 산정  (단계, 비선형 쌍곡선, 기존 쌍곡선)
RMSE_step = fun_rmse(sm_rmse, sp_step_rmse)
RMSE_hyper_nonlinear = fun_rmse(sm_rmse, sp_hyper_nonlinear_rmse)
RMSE_hyper_original = fun_rmse(sm_rmse, sp_hyper_original_rmse)

# RMSE 출력 (단계, 비선형 쌍곡선, 기존 쌍곡선)
print("RMSE (Nonlinear Hyper + Step): %0.3f" %RMSE_step)
print("RMSE (Nonlinear Hyperbolic): %0.3f" %RMSE_hyper_nonlinear)
print("RMSE (Original Hyperbolic): %0.3f" %RMSE_hyper_original)

# (최종 계측 침하량 - 예측 침하량) 계산
final_error_step = settle[-1] - sp_step_rmse[-1]
final_error_hyper_nonlinear = settle[-1] - sp_hyper_nonlinear_rmse[-1]
final_error_hyper_original = settle[-1] - sp_hyper_original_rmse[-1]

# (최종 계측 침하량 - 예측 침하량) 출력 (단계, 비선형 쌍곡선, 기존 쌍곡선)
print("Error in Final Settlement (Nonlinear Hyper + Step): %0.3f" %final_error_step)
print("Error in Final Settlement (Nonlinear Hyperbolic): %0.3f" %final_error_hyper_nonlinear)
print("Error in Final Settlement (Original Hyperbolic): %0.3f" %final_error_hyper_original)



# =====================
# Post-Processing
# =====================

# 그래프 크기, 서브 그래프 개수 및 비율 설정
fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                         gridspec_kw={'height_ratios':[1,3]})

# 성토고 그래프 표시
axes[0].plot(time, surcharge, color='black', label='surcharge height')

# 성토고 그래프 설정
axes[0].set_ylabel("Surcharge height (m)", fontsize=15)
axes[0].set_xlim(left=0)
axes[0].grid(color="gray", alpha=.5, linestyle='--')
axes[0].tick_params(direction='in')

# 계측 및 예측 침하량 표시
axes[1].scatter(time[0:settle.size], -settle, s=50, facecolors='white', edgecolors='black', label='measured data')
axes[1].plot(time, -sp_step, linestyle='-', color='blue', label='Nonlinear + Step Loading')
axes[1].plot(time_hyper, -sp_hyper_nonlinear,
             linestyle='--', color='green', label='Nonlinear Hyperbolic')
axes[1].plot(time_hyper, -sp_hyper_original,
             linestyle='--', color='red', label='Original Hyperbolic')

# 침하량 그래프 설정
axes[1].set_xlabel("Time (day)", fontsize=15)
axes[1].set_ylabel("Settlement (cm)", fontsize=15)
axes[1].set_ylim(top=0)
axes[1].set_ylim(bottom=-1.5 * settle.max())
axes[1].set_xlim(left=0)
axes[1].grid(color="gray", alpha=.5, linestyle='--')
axes[1].tick_params(direction='in')

# 범례 표시
axes[1].legend(loc=1, ncol=2, frameon=True, fontsize=12)

# 예측 데이터 사용 범위 음영 처리 - 단계성토
plt.axvspan(0, final_step_predict_end_date,
            alpha=0.1, color='grey', hatch='//')

# 예측 데이터 사용 범위 음영 처리 - 기존 및 비선형 쌍곡선
plt.axvspan(final_step_start_date, final_step_predict_end_date,
            alpha=0.1, color='grey', hatch='\\')

# 예측 데이터 사용 범위 표시 화살표 세로 위치 설정
arrow1_y_loc = 1.3 * min(-settle)
arrow2_y_loc = 1.4 * min(-settle)

# 화살표 크기 설정
arrow_head_width = 0.03 * max(settle)
arrow_head_length = 0.01 * max(time)

# 예측 데이터 사용 범위 화살표 처리 - 단계성토
axes[1].arrow(0, arrow1_y_loc, final_step_predict_end_date, 0,
              head_width=arrow_head_width, head_length=arrow_head_length,
              color='black', length_includes_head='True')
axes[1].arrow(final_step_predict_end_date, arrow1_y_loc, -final_step_predict_end_date, 0,
              head_width=arrow_head_width, head_length=arrow_head_length,
              color='black', length_includes_head='True')

# 예측 데이터 사용 범위 화살표 처리 - 기존 및 비선형 쌍곡선
axes[1].arrow(final_step_start_date, arrow2_y_loc,
              final_step_predict_end_date - final_step_start_date, 0,
              head_width=arrow_head_width, head_length=arrow_head_length,
              color='black', length_includes_head='True')
axes[1].arrow(final_step_predict_end_date, arrow2_y_loc,
              final_step_start_date - final_step_predict_end_date, 0,
              head_width=arrow_head_width, head_length=arrow_head_length,
              color='black', length_includes_head='True')

# Annotation 표시용 공간 설정
space = max(time) * 0.01

# 예측 데이터 사용 범위 범례 표시 - 단계성토
plt.annotate('Data Range Used (Nonlinear + Step Loading)', xy=(final_step_predict_end_date, arrow1_y_loc),
             xytext=(final_step_predict_end_date + space, arrow1_y_loc),
             horizontalalignment='left', verticalalignment='center')

# 예측 데이터 사용 범위 범례 표시 - 기존 및 비선형 쌍곡선
plt.annotate('Data Range Used (Nonlinear and Original Hyperbolic)', xy=(final_step_predict_end_date, arrow1_y_loc),
             xytext=(final_step_predict_end_date + space, arrow2_y_loc),
             horizontalalignment='left', verticalalignment='center')

# RMSE 산정 범위 표시 화살표 세로 위치 설정
arrow3_y_loc = 0.55 * min(-settle)

# RMSE 산정 범위 화살표 표시
axes[1].arrow(final_step_predict_end_date, arrow3_y_loc,
              final_step_end_date - final_step_predict_end_date, 0,
              head_width=arrow_head_width, head_length=arrow_head_length,
              color='black', length_includes_head='True')
axes[1].arrow(final_step_end_date, arrow3_y_loc,
              final_step_predict_end_date - final_step_end_date, 0,
              head_width=arrow_head_width, head_length=arrow_head_length,
              color='black', length_includes_head='True')

# RMSE 산정 범위 세로선 설정
axes[1].axvline(x=final_step_end_date, color='silver', linestyle=':')

# RMSE 산정 범위 범례 표시
plt.annotate('RMSE Estimation Section', xy=(final_step_end_date, arrow3_y_loc),
             xytext=(final_step_end_date + space, arrow3_y_loc),
             horizontalalignment='left', verticalalignment='center')

# RMSE 출력
mybox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round', 'alpha': 0.2}
plt.text(max(time), 0.25 * min(-settle),
         "Root Mean Squared Error"
         + "\n" + "Nonlinear + Step Loading: %0.3f" % RMSE_step
         + "\n" + "Nonlinear Hyperbolic: %0.3f" % RMSE_hyper_nonlinear
         + "\n" + "Original Hyperbolic: %0.3f" % RMSE_hyper_original,
         color='r', horizontalalignment='right',
         verticalalignment='top', fontsize='12', bbox=mybox)

# (최종 계측 침하량 - 예측값) 출력
plt.text(max(time), 0.65 * min(-settle),
         "Error in Final Monitored Settlement"
         + "\n" + "Nonlinear + Step Loading: %0.3f" % final_error_step
         + "\n" + "Nonlinear Hyperbolic: %0.3f" % final_error_hyper_nonlinear
         + "\n" + "Original Hyperbolic: %0.3f" % final_error_hyper_original,
         color='r', horizontalalignment='right',
         verticalalignment='top', fontsize='12', bbox=mybox)

# 그래프 제목 표시
plt.title(filename + ": up to %i%% data used in the final step" % final_step_predict_percent)

# 그래프 저장 (SVG 및 PNG)
plt.savefig(filename +' %i percent (SVG).svg' %final_step_predict_percent, bbox_inches='tight')
plt.savefig(filename +' %i percent (PNG).png' %final_step_predict_percent, bbox_inches='tight')

# 그래프 출력
plt.show()