import numpy as np
from scipy.optimize import least_squares

# python library for visualization
import matplotlib.pyplot as plt
from matplotlib import rcParams

# python library for data management
import pandas as pd

# generate a time-settlement curve for hyperbolic method
def generate_data_hyper(px, pt):
    return pt / (px[0] * pt + px[1])

# error between regression and measurement
def fun_hyper_linear(px, pt, py):
    return pt / (px[0] * pt + px[1]) - py

# Read .csv file using pandas
data = pd.read_csv("1_SP-11.csv")

# Set arrays for time and settlement
time = data['Time'].to_numpy()
settle = data['Settle'].to_numpy()
surcharge = data['Surcharge'].to_numpy()

# Set data range (in%) to use in the prediction
start = 0
end = 100

# Find the data range (in data) to use in the prediction
end_date = time[-1]
pred_start_date = int(end_date * start / 100) # prediction start date
pred_end_date = int(end_date * end / 100) # prediction end data

# initialize the indices for start and end date
start_index = -1
end_index = -1

# Find the index of the initial data
count = 0
for day in time: # time = [0, 1, 2, 3, ...]
    count = count + 1
    if day > pred_start_date:
        start_index = count - 1
        break

# Find the index of the final data
count = 0
for day in time: # time = [... 100, 101, 104, ...]
    count = count + 1
    if day > pred_end_date:
        end_index = count - 1
        break

# Set data for the prediction

'''
1단계 성토고 침하 예측
'''

#1단계에 해당하는 실측 데이터 범위 지정
tm_1 = time[start:10]
ym_1 = settle[start:10]

# Set a list for the coefficient; here a and b
x0 = np.ones(2)

# declare a least square object
res_lsq_hyper_linear_1 = least_squares(fun_hyper_linear, x0, args=(tm_1, ym_1))

# Print the calculated coefficient
print(res_lsq_hyper_linear_1.x)

# Generate predicted settlement data
settle_hyper_linear_1 = generate_data_hyper(res_lsq_hyper_linear_1.x, time)

'''
2단계 성토고 침하 예측
'''

# 2단계 실측 침하량 (2단계~최종)
tm_2 = time[10:80]
ym_2 = settle[10:80]
# 2단계 실측 침하량 (2단계 구간만)
tm_22 = time[10:37]
ym_22 = settle[10:37]
# 1단계 예측 침하량 (2단계 구간만)
yp_2 = settle_hyper_linear_1[10:37]
# 1단계 예측 침하량 (2단계 ~ 최종)
yp_22 = settle_hyper_linear_1[10:80]

# 2단계 보정 침하량 산정
def fun_step_measured_correction(m, p):
    return m - p

step2_measured_correction = fun_step_measured_correction(ym_22, yp_2)

# 2단계 t-ti 산정
def fun_step_time_correction(t, ti):
    return t - ti

step2_time_correction = fun_step_time_correction(tm_2[0:69],tm_2[0])

# 2단계 보정 침하량에 대한 예측 침하량 산정

# declare a least square object
# step2_time_correction[0:27]는 회귀분석 적용할 2단계 범위만 추출한 것
res_lsq_hyper_linear_2 = least_squares(fun_hyper_linear, x0, args=(step2_time_correction[0:27], step2_measured_correction))

# Print the calculated coefficient
print(res_lsq_hyper_linear_2.x)

# Generate predicted settlement data
settle_hyper_linear_2 = generate_data_hyper(res_lsq_hyper_linear_2.x, step2_time_correction)

# 2단계 침하곡선 작성
def settlement_prediction_curve(m1, p1):
    return m1 + p1

step2_prediction_curve = settlement_prediction_curve(settle_hyper_linear_2, yp_22)

# 2단계 보정 예측 침하량 산정
def fun_step_prediction_correction(m2, p2):
    return p2 + (m2[0] - p2[0])

step2_prediction_correction = fun_step_prediction_correction(ym_2, step2_prediction_curve)

'''
3단계 성토고 침하 예측
'''

# 3단계 실측 침하량 (3단계~최종)
tm_3 = time[37:end]
ym_3 = settle[37:end]
# 2단계 예측 침하량 (3단계 구간만)
yp_3 = step2_prediction_correction[27:end]


# 3단계 보정 침하량 산정
step3_measured_correction = fun_step_measured_correction(ym_3, yp_3)

# 3단계 t-ti 산정

step3_time_correction = fun_step_time_correction(tm_3,tm_3[0])

# 3단계 보정 침하량에 대한 예측 침하량 산정

res_lsq_hyper_linear_3 = least_squares(fun_hyper_linear, x0, args=(step3_time_correction, step3_measured_correction))

print(res_lsq_hyper_linear_3.x)

settle_hyper_linear_3 = generate_data_hyper(res_lsq_hyper_linear_3.x, step3_time_correction)

# 3단계 침하곡선 작성

step3_prediction_curve = settlement_prediction_curve(settle_hyper_linear_3, yp_3)

# 3단계 보정 예측 침하량 산정

step3_prediction_correction = fun_step_prediction_correction(ym_3, step3_prediction_curve)

'''
그래프 작성
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
axes[1].plot(time, -settle_hyper_linear_1, linestyle='--', color='red', label='original Hyperbolic_1')
axes[1].plot(tm_2, -step2_prediction_correction, linestyle='--', color='blue', label='original Hyperbolic_2')
axes[1].plot(tm_3, -step3_prediction_correction, linestyle='--', color='green', label='original Hyperbolic_3')

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


plt.savefig('1_SP_11_Rev.2.png', dpi=300)
plt.show()