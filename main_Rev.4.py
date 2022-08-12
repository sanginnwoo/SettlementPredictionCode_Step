# =================
# Import 섹션
# =================

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

# =================
# Function 섹션
# =================

# 주어진 계수를 이용하여 쌍곡선 시간-침하 곡선 반환
def generate_data_hyper(px, pt):
    return pt / (px[0] * pt + px[1])

# 회귀식과 측정치와의 잔차 반환 (비선형 쌍곡선)
def fun_hyper_nonlinear(px, pt, py):
    return pt / (px[0] * pt + px[1]) - py

# =================
# Step별 활용 Function
# =================

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



# =================
# 입력값 설정
# =================

# CSV 파일 읽기
data = pd.read_csv("3_SP-68_Test.csv")

# 시간, 침하량, 성토고 배열 생성
time = data['Time'].to_numpy()
settle = data['Settle'].to_numpy()
surcharge = data['Surcharge'].to_numpy()

# =================
# 성토 단계 구분
# =================

step_start_index = [0, 9, 49, 90] # 단계별 성토 시작 지점 입력(4단계 이므로 4개)
step_end_index = [8, 48, 89, 129] # 단계별 성토 종료 지점 입력(4단계 이므로 4개)
x0 = np.ones(2)
num_step = 4

for i in range(0, num_step): # 성토 단계에 따라 수정(4단계 이므로 0~4)

    # i단계 실측 기간 및 침하량
    globals()['tm_{}'.format(i)] = time[step_start_index[i]:step_end_index[i]]
    globals()['ym_{}'.format(i)] = settle[step_start_index[i]:step_end_index[i]]

    if i == 0 : # 1단계

        res_lsq_hyper_nonlinear_0 = least_squares(fun_hyper_nonlinear, x0, args=(tm_0, ym_0))
        print(res_lsq_hyper_nonlinear_0.x)

        globals()['settle_predicted_{}'.format(i)] = generate_data_hyper(res_lsq_hyper_nonlinear_0.x, time)

    elif 0 < i < (num_step - 1):

        # i단계~최종 실측 기간 및 침하량
        globals()['tmm_{}'.format(i)] = time[step_start_index[i]:step_end_index[-1]]
        globals()['ymm_{}'.format(i)] = settle[step_start_index[i]:step_end_index[-1]]

        # i-1단계 예측 침하량(i단계 기간에 해당하는)
        globals()['yp_{}'.format(i)] = globals()['settle_predicted_{}'.format(i - 1)][(step_start_index[i]-step_start_index[i-1]):(step_end_index[i]-step_start_index[i-1])]
        # i-1 단계 예측 침하량 (i단계~최종)
        globals()['ypp_{}'.format(i)] = globals()['settle_predicted_{}'.format(i - 1)][(step_start_index[i]-step_start_index[i-1]):(step_end_index[-1]-step_start_index[i-1])]

        # i단계 실측 보정 침하량 산정
        globals()['step_{}_measured_correction'.format(i)] = fun_step_measured_correction(globals()['ym_{}'.format(i)],globals()['yp_{}'.format(i)])
        # i단계 t-ti 산정
        globals()['step_{}_time_correction'.format(i)] = fun_step_time_correction(globals()['tmm_{}'.format(i)],
                                                                                  globals()['tm_{}'.format(i)][0])

        # i 단계 보정 침하량에 대한 예측 침하량 산정
        globals()['res_lsq_hyper_nonlinear_{}'.format(i)] = least_squares(fun_hyper_nonlinear, x0,
                                          args=(globals()['step_{}_time_correction'.format(i)][0:(step_end_index[i]-step_start_index[i])],
                                                globals()['step_{}_measured_correction'.format(i)]))


        print(globals()['res_lsq_hyper_nonlinear_{}'.format(i)].x)

        globals()['settle_hyper_nonlinear_{}'.format(i)] = generate_data_hyper(globals()['res_lsq_hyper_nonlinear_{}'.format(i)].x,
                                                                               globals()['step_{}_time_correction'.format(i)])

        # i단계 침하곡선 작성
        globals()['step_{}_prediction_curve'.format(i)] = settlement_prediction_curve(globals()['settle_hyper_nonlinear_{}'.format(i)],
                                                                                      globals()['ypp_{}'.format(i)])

        # i단계 보정 예측 침하량 산정
        globals()['settle_predicted_{}'.format(i)] = fun_step_prediction_correction(globals()['ymm_{}'.format(i)],
                                                                                    globals()['step_{}_prediction_curve'.format(i)])


    else: # 최종 성토 단계

        # i-1 단계 예측 침하량 (최종 단계에 해당하는)
        globals()['yp_{}'.format(i)] = globals()['settle_predicted_{}'.format(i - 1)][(step_start_index[i]-step_start_index[i-1]):step_end_index[i]]

        # 최종 단계 실측 보정 침하량 산정
        globals()['step_{}_measured_correction'.format(i)] = fun_step_measured_correction(globals()['ym_{}'.format(i)],
                                                                                          globals()['yp_{}'.format(i)])

        # 최종 단계 t-ti 산정
        globals()['step_{}_time_correction'.format(i)] = fun_step_time_correction(globals()['tm_{}'.format(i)],
                                                                                  globals()['tm_{}'.format(i)][0])

        # 최종 단계 보정 침하량에 대한 예측 침하량 산정
        globals()['res_lsq_hyper_nonlinear_{}'.format(i)] = least_squares(fun_hyper_nonlinear, x0,
                                                             args=(globals()['step_{}_time_correction'.format(i)],
                                                              globals()['step_{}_measured_correction'.format(i)]))

        print(globals()['res_lsq_hyper_nonlinear_{}'.format(i)].x)

        globals()['settle_hyper_nonlinear_{}'.format(i)] = generate_data_hyper(globals()['res_lsq_hyper_nonlinear_{}'.format(i)].x,
                                                                               globals()['step_{}_time_correction'.format(i)])

        # 최종 단계 침하곡선 작성
        globals()['step_{}_prediction_curve'.format(i)] = settlement_prediction_curve(globals()['settle_hyper_nonlinear_{}'.format(i)],
                                                                                      globals()['yp_{}'.format(i)])

        # 최종단계 보정 예측 침하량 산정
        globals()['settle_predicted_{}'.format(i)] = fun_step_prediction_correction(globals()['ym_{}'.format(i)],
                                                                                    globals()['step_{}_prediction_curve'.format(i)])

'''
나중에: 그래프 작성
'''

# 그래프 크기, 서브 그래프 개수 및 비율 설정
f, axes = plt.subplots(2,1, figsize=(10, 10),
                         gridspec_kw={'height_ratios':[1,2]})

# 성토고 그래프 표시
axes[0].plot(time, surcharge, color='black', label='surcharge height')

axes[0].set_ylabel("Surcharge height (m)", fontsize = 17)
axes[0].set_xlim(left = 0)
axes[0].grid(color="gray", alpha=.5, linestyle='--')
axes[0].tick_params(direction='in')

# 계측 침하량 표시
axes[1].scatter(time, -settle, s = 50, facecolors='white', edgecolors='black', label = 'measured data')

# 예측 침하량 표시
axes[1].plot(time, -settle_predicted_0, linestyle='--', color='red', label='Predicted Curve_Step 1')
axes[1].plot(tmm_1, -settle_predicted_1, linestyle='--', color='blue', label='Predicted Curve_Step 2')
axes[1].plot(tmm_2, -settle_predicted_2, linestyle='--', color='green', label='Predicted Curve_Step 3')
axes[1].plot(tm_3, -settle_predicted_3, linestyle='--', color='orange', label='Predicted Curve_Step 4')

# 예측 침하량 그래프 설정
axes[1].set_xlabel("Time (day)", fontsize = 17)
axes[1].set_ylabel("Settlement (mm)", fontsize = 17)
axes[1].set_ylim(top = 0)
axes[1].set_ylim(bottom = -1.5 * settle.max())
axes[1].set_xlim(left = 0)

# 범례 표시
axes[1].legend(loc=1, ncol=2, frameon=True, fontsize=12)

# 그래프 저장 및 출력
plt.savefig('3_SP-68_Rev.4_Test.svg', dpi=300)
plt.show()