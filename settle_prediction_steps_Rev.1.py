# =================
# Import 섹션
# =================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
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

# RMSE 계산
def fun_rmse(py1, py2):
    mse = np.square(np.subtract(py1, py2)). mean()
    return np.sqrt(mse)


# =================
# 입력값 설정
# =================

# CSV 파일 읽기
data = pd.read_csv("4. S-11.csv")

# 시간, 침하량, 성토고 배열 생성
time = data['Time'].to_numpy()
settle = data['Settle'].to_numpy()
surcharge = data['Surcharge'].to_numpy()


# =================
# 성토 단계 구분
# =================

step_start_index = [0, 10, 46, 51, 120]  # 성토 단계 시작 index
step_end_index = [10, 46, 51, 120, 139]   # 실제 최종 성토 종료 index : 157
final_index = time.size         # 마지만 계측 데이터 index + 1
num_steps = 5                   # 성토 단계 횟수


# =================
# 추가 예측 구간 반영
# =================

# 추가 예측 일 입력
add_days = 500

# 마지막 성토고 및 마지막 계측일 저장
final_surcharge = surcharge[final_index - 1]
final_time = time[final_index -1]

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
sp = np.zeros(time.size)

# 각 단계별로 진행
for i in range(0, num_steps):

    # 각 단계별 계측 시점과 계측 침하량 배열 생성
    tm_this_step = time[step_start_index[i]:step_end_index[i]]
    sm_this_step = settle[step_start_index[i]:step_end_index[i]]

    # 이전 단계까지 예측 침하량 중 현재 단계에 해당하는 부분 추출
    sp_this_step = sp[step_start_index[i]:step_end_index[i]]

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

    # 현재 단계 예측 침하량 산정 (침하 예측 끝까지)
    sp_to_end_update = generate_data_hyper(x_step, tm_to_end)

    # 예측 침하량 업데이트
    sp[step_start_index[i]:final_index] = \
        sp[step_start_index[i]:final_index] + sp_to_end_update + s0_this_step

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

# 각 방법에 대한 RMSE 계산
RMSE_hyper_nonlinear_Step = fun_rmse(settle[step_start_index[-1]:step_end_index[-1]], sp_this_step)

RMSE_hyper_original = fun_rmse(settle[step_start_index[-1]:step_end_index[-1]], sp_hyper_original[0:(step_end_index[-1]-step_start_index[-1])])

RMSE_hyper_nonlinear = fun_rmse(settle[step_start_index[-1]:step_end_index[-1]], sp_hyper_nonlinear[0:(step_end_index[-1]-step_start_index[-1])])

# =====================
# Post-Processing
# =====================

# 그래프 크기, 서브 그래프 개수 및 비율 설정
fig, axes = plt.subplots(2, 1, figsize=(10, 10),
                         gridspec_kw={'height_ratios':[1,2]})

# 성토고 그래프 표시
axes[0].plot(time, surcharge, color='black', label='surcharge height')

# 성토고 그래프 설정
axes[0].set_ylabel("Surcharge height (m)", fontsize=17)
axes[0].set_xlim(left=0)
axes[0].grid(color="gray", alpha=.5, linestyle='--')
axes[0].tick_params(direction='in')

# 계측 및 예측 침하량 표시
axes[1].scatter(time[0:settle.size], -settle, s=50, facecolors='white', edgecolors='black', label='measured data')
axes[1].plot(time, -sp, linestyle='-', color='blue', label='Nonlinear + Step Loading')
axes[1].plot(time_hyper, -sp_hyper_nonlinear,
             linestyle='--', color='green', label='Nonlinear Hyperbolic')
axes[1].plot(time_hyper, -sp_hyper_original,
             linestyle='--', color='red', label='Original Hyperbolic')

# 침하량 그래프 설정
axes[1].set_xlabel("Time (day)", fontsize=15)
axes[1].set_ylabel("Settlement (mm)", fontsize=15)
axes[1].set_ylim(top=0)
axes[1].set_ylim(bottom=-1.5 * settle.max())
axes[1].set_xlim(left=0)
axes[1].grid(color="gray", alpha=.5, linestyle='--')
axes[1].tick_params(direction='in')

# 범례 표시
axes[1].legend(loc=1, ncol=2, frameon=True, fontsize=12)

# 침하예측 활용 구간 표시
axes[1].axvspan(time[step_start_index[-1]], time[step_end_index[-1]], alpha = 0.2, color = 'gray', hatch = '///')
axes[1].annotate('Date range used', xy=(time[step_end_index[-1]], min(-settle) * 0.5),
             xytext=(time[step_end_index[-1]] + 20, min(-settle) * 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='left', verticalalignment='bottom')


# RMSE 박스 생성
mybox = {'facecolor':'red', 'edgecolor':'black', 'boxstyle':'round', 'alpha':0.4}
axes[1].text(0.015 * max(time), -1.4 * max(settle),
         " RMSE(Hyperbolic(Nonlinear_Step)) = %0.3f " % RMSE_hyper_nonlinear_Step
         + "\n" + " RMSE(Hyperbolic(original)) = %0.3f " % RMSE_hyper_original
         + "\n" + " RMSE(Hyperbolic(Nonlinear)) = %0.3f " % RMSE_hyper_nonlinear,
         color = 'r', horizontalalignment='left', verticalalignment='bottom',
         fontsize='14', bbox=mybox)

# 그래프 저장
plt.savefig('4_S-11.svg')

# 그래프 출력
plt.show()