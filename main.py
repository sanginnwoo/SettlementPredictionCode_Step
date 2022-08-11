# 주요 수정 사항
# 주석 철저히: 한글로 작성해도 괜찮아요
# 입력 1: 시간-침하 데이터, 시간-성토고 데이터 --> 파일로 부터 읽는 것
# 입력 2: 사용자가 단계 지정 ---> 간 단계별 처음과 끝 INDEX
# 입력 3: 전체 성토 단계 횟수

# 라이브러리 import
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

# Functions

# generate a time-settlement curve for hyperbolic method
def generate_data_hyper(px, pt):
    return pt / (px[0] * pt + px[1])

# error between regression and measurement
def fun_hyper_nonlinear(px, pt, py):
    return pt / (px[0] * pt + px[1]) - py

# i단계 보정 침하량 산정
def fun_step_measured_correction(m, p):
    return m - p

# i단계 t-ti 산정
def fun_step_time_correction(t, ti):
    return t - ti

# i단계 침하곡선 작성
def settlement_prediction_curve(m1, p1):
    return m1 + p1

# i단계 보정 예측 침하량 산정
def fun_step_prediction_correction(m2, p2):
    return p2 + (m2[0] - p2[0])



# 파일 읽기, 리스트 설정

# Read .csv file using pandas
data = pd.read_csv("1_SP-11.csv")

# Set arrays for time and settlement
time = data['Time'].to_numpy()
settle = data['Settle'].to_numpy()
surcharge = data['Surcharge'].to_numpy()

#
# 성토 단계 시작, 끝 인덱스 입력 / 전체 성토 단계 입력
# 예: 1단계: (0, 9), 2단계: (10, 37), 3단계: (38, 80)
# 예: 전체 성토 단계: 3


#
# 각 단계별 예측을 반복문으로 처리
#

step_start_index = [0, 10, 38]
step_end_index = [9, 37, 80]
x0 = np.ones(2)

for i in range(0,3):

    if i == 0 : # 1단계

        # 1단계 실측 기간 및 침하량
        globals()['tm_{}'.format(i)] = time[step_start_index[i]:step_end_index[i]]
        globals()['ym_{}'.format(i)] = settle[step_start_index[i]:step_end_index[i]]

        res_lsq_hyper_nonlinear_0 = least_squares(fun_hyper_nonlinear, x0, args=(tm_0, ym_0))
        print(res_lsq_hyper_nonlinear_0.x)

        globals()['settle_predicted_{}'.format(i)] = generate_data_hyper(res_lsq_hyper_nonlinear_0.x, time)

    elif 0 < i < 2 : # 최종단계 3단계 이므로 2를 넘지 않도록 설정

        # i단계 실측 기간 및 침하량
        globals()['tm_{}'.format(i)] = time[step_start_index[i]:step_end_index[i]]
        globals()['ym_{}'.format(i)] = settle[step_start_index[i]:step_end_index[i]]

        # i단계~최종 실측 기간 및 침하량
        globals()['tmm_{}'.format(i)] = time[step_start_index[i]:step_end_index[i+1]]
        globals()['ymm_{}'.format(i)] = settle[step_start_index[i]:step_end_index[i+1]]

        # i-1 단계 예측 침하량 (i단계에 해당하는)
        globals()['yp_{}'.format(i)] = settle_predicted_0[step_start_index[i]:step_end_index[i]]
        # i-1 단계 예측 침하량 (i단계~최종)
        globals()['ypp_{}'.format(i)] = settle_predicted_0[step_start_index[i]:step_end_index[i + 1]]

        # i단계 실측 보정 침하량 산정
        globals()['step_{}_measured_correction'.format(i)] = fun_step_measured_correction(ym_1, yp_1)

        # i단계 t-ti 산정
        globals()['step_{}_time_correction'.format(i)] = fun_step_time_correction(tmm_1, tm_1[0])

        # i 단계 보정 침하량에 대한 예측 침하량 산정
        globals()['res_lsq_hyper_nonlinear_{}'.format(i)] = least_squares(fun_hyper_nonlinear, x0,
                                          args=(step_1_time_correction[0:(step_end_index[i]-step_start_index[i])], step_1_measured_correction))
        print(res_lsq_hyper_nonlinear_1.x)

        globals()['settle_hyper_nonlinear_{}'.format(i)] = generate_data_hyper(res_lsq_hyper_nonlinear_1.x, step_1_time_correction)

        # i단계 침하곡선 작성
        globals()['step_{}_prediction_curve'.format(i)] = settlement_prediction_curve(settle_hyper_nonlinear_1, ypp_1)

        # i단계 보정 예측 침하량 산정
        globals()['settle_predicted_{}'.format(i)] = fun_step_prediction_correction(ymm_1, step_1_prediction_curve)

    else: # 최종 성토 단계

        # 최종 단계 실측 기간 및 침하량
        globals()['tm_{}'.format(i)] = time[step_start_index[i]:step_end_index[i]]
        globals()['ym_{}'.format(i)] = settle[step_start_index[i]:step_end_index[i]]

        # i-1 단계 예측 침하량 (최종 단계에 해당하는)
        globals()['yp_{}'.format(i)] = settle_predicted_1[(step_start_index[i]-step_start_index[i-1]):step_end_index[i]]

        # 최종 단계 실측 보정 침하량 산정
        globals()['step_{}_measured_correction'.format(i)] = fun_step_measured_correction(ym_2, yp_2)

        # 최종 단계 t-ti 산정
        globals()['step_{}_time_correction'.format(i)] = fun_step_time_correction(tm_2, tm_2[0])

        # 최종 단계 보정 침하량에 대한 예측 침하량 산정
        globals()['res_lsq_hyper_nonlinear_{}'.format(i)] = least_squares(fun_hyper_nonlinear, x0,
                                                        args=(step_2_time_correction, step_2_measured_correction))
        print(res_lsq_hyper_nonlinear_2.x)

        globals()['settle_hyper_nonlinear_{}'.format(i)] = generate_data_hyper(res_lsq_hyper_nonlinear_2.x,
                                                                               step_2_time_correction)

        # 최종 단계 침하곡선 작성
        globals()['step_{}_prediction_curve'.format(i)] = settlement_prediction_curve(settle_hyper_nonlinear_2, yp_2)

        # i단계 보정 예측 침하량 산정
        globals()['settle_predicted_{}'.format(i)] = fun_step_prediction_correction(ym_2, step_2_prediction_curve)
        break

'''
나중에: 그래프 작성
'''

# Set parameters for plotting
rcParams['figure.figsize'] = (10, 10)

# Subplot
f, axes = plt.subplots(2,1)
plt.subplots_adjust(hspace = 0.1)

# draw surcharge data
axes[0].plot(time, surcharge, color='black', label='surcharge height')

axes[0].set_ylabel("Surcharge height (m)", fontsize = 17)
axes[0].set_xlim(left = 0)

# draw measured data
axes[1].scatter(time, -settle, s = 50, facecolors='white', edgecolors='black', label = 'measured data')

# draw predicted data
axes[1].plot(time, -settle_predicted_0, linestyle='--', color='red', label='Predicted Curve_Step 1')
axes[1].plot(tmm_1, -settle_predicted_1, linestyle='--', color='blue', label='Predicted Curve_Step 2')
axes[1].plot(tm_2, -settle_predicted_2, linestyle='--', color='green', label='Predicted Curve_Step 3')

# Set axes title
axes[1].set_xlabel("Time (day)", fontsize = 17)
axes[1].set_ylabel("Settlement (mm)", fontsize = 17)

# Set min values of x and y axes
axes[1].set_ylim(top = 0)
axes[1].set_ylim(bottom = -1.5 * settle.max())
axes[1].set_xlim(left = 0)

# Set legend
axes[1].legend(bbox_to_anchor = (0, 0, 1, 0), loc =4, ncol = 3, mode="expand",
           borderaxespad = 0,  frameon = False, fontsize = 12)


plt.savefig('main_Rev.1.png', dpi=300)
plt.show()