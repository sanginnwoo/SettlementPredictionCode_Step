"""
Title: Error analysis of the settlement prediction curves
Developer:
Sang Inn Woo, Ph.D. @ Incheon National University
Starting Date: 2022-11-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
데이터 구조: error_single.csv
['File', 'Data_usage', 
 'RMSE_hyper_original', 'RMSE_hyper_nonlinear', 'RMSE_hyper_weighted_nonlinear',
 'Final_error_hyper_original', 'Final_error_hyper_nonlinear', 'Final_error_hyper_weighted_nonlinear'])
'''

# CSV 파일 읽기 (일단, 다단 성토 포함)
df_single = pd.read_csv('error_single.csv', encoding='euc-kr')

# 통계량 저장소 statistics 초기화
statistic = []

# 통계량 저장소 statistics 데이터 구조
#            mean    /   median  /   std    / 90% percentile
# RMSE (O)
# RMSE (NL)
# RMSE (WNL)

# 카운트 초기화
count = 0

#
num_bars = 5
rmse_min = 0
rmse_max = 50
probability_min = 0
probability_max = 0.5
data_usage_min = 30
data_usage_max = 180
is_data_usage_percent = False

data_usages = range(60, 180, 30)

# 최종 성토 단계에서 각 침하 데이터 사용 영역에 대해서 다음을 수행
for data_usage in data_usages:

    # 전체 Error 분석을 위한 Dataframe 설정
    df_single_sel = df_single.loc[df_single['Data_usage'] == data_usage]

    # RMSE 및 FE를 불러서 메모리에 저장
    RMSE_hyper_original = df_single_sel['RMSE_hyper_original'].to_numpy()
    RMSE_hyper_nonlinear = df_single_sel['RMSE_hyper_nonlinear'].to_numpy()
    RMSE_hyper_weighted_nonlinear = df_single_sel['RMSE_hyper_weighted_nonlinear'].to_numpy()

    # 평균, 중앙값, 표준편차 산정 및 저장
    statistic.append([np.mean(RMSE_hyper_original), np.median(RMSE_hyper_original),
                      np.std(RMSE_hyper_original), np.percentile(RMSE_hyper_original, 90)])
    statistic.append([np.mean(RMSE_hyper_nonlinear), np.median(RMSE_hyper_nonlinear),
                      np.std(RMSE_hyper_nonlinear), np.percentile(RMSE_hyper_nonlinear, 90)])
    statistic.append([np.mean(RMSE_hyper_weighted_nonlinear), np.median(RMSE_hyper_weighted_nonlinear),
                      np.std(RMSE_hyper_weighted_nonlinear), np.percentile(RMSE_hyper_weighted_nonlinear, 90)])

    # 그래프 설정 (2 by 3)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # 그래프 제목 설정
    if is_data_usage_percent:
        fig.suptitle('Histograms: ' + str(data_usage) + '% of Settlement Data Used')
    else:
        fig.suptitle('Histograms: ' + str(data_usage) + ' days of Settlement Data Used')

    # 각 Subplot의 제목 설정
    ax1.set_xlabel('RMSE (Original Hyperbolic) (cm)')
    ax2.set_xlabel('RMSE (Nonlinear Hyperbolic) (cm)')
    ax3.set_xlabel('RMSE (Weighted Nonlinear Hyperbolic) (cm)')

    # 각 subplot에 히스토그램 작성
    ax1.hist(RMSE_hyper_original, num_bars, density=True, facecolor='r', edgecolor='k', alpha=0.75)
    ax2.hist(RMSE_hyper_nonlinear, num_bars, density=True, facecolor='g', edgecolor='k', alpha=0.75)
    ax3.hist(RMSE_hyper_weighted_nonlinear, num_bars, density=True, facecolor='b', edgecolor='k', alpha=0.75)

    # 각 subplot을 포함한 리스트 설정
    axes = [ax1, ax2, ax3]

    # 공통 사항 적용
    for i in range(len(axes)):
        text_loc_x = rmse_min + (rmse_max - rmse_min) * 0.2
        text_loc_y = probability_min + (probability_max - probability_min) * 0.7
        ax = axes[i]
        ax.text(text_loc_x, text_loc_y,
                'Mean = ' + "{:0.2f}".format(statistic[count * 3 + i][0]) + '\n' +
                'Median = ' + "{:0.2f}".format(statistic[count * 3 + i][1]) + '\n' +
                'Standard Deviation = ' + "{:0.2f}".format(statistic[count * 3 + i][2]) + '\n' +
                '90% Percentile = ' + "{:0.2f}".format(statistic[count * 3 + i][3]))
        ax.set_ylabel("Probability")
        ax.grid(color="gray", alpha=.5, linestyle='--')
        ax.tick_params(direction='in')
        ax.set_xlim(rmse_min, rmse_max)
        ax.set_ylim(probability_min, probability_max)

    # 그래프 저장 (SVG 및 PNG)
    if is_data_usage_percent:
        plt.savefig('error/error_single(%i percent).png' % data_usage, bbox_inches='tight')
    else:
        plt.savefig('error/error_single(%i days).png' % data_usage, bbox_inches='tight')

    # 카운트 증가
    count = count + 1


# 통계량 배열을 numpy 배열로 전환
statistic = np.array(statistic)

# 그래프 설정 (총 4개의 그래프)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

# 그래프 전체 제목 설정
fig.suptitle("RMSE Analysis of Original, Nonlinear, and Weighted Nonlinear Hyperbolic")

# 데이터 사용량 대비 RMSE 평균값 변화 도시
ax1.set_ylabel('Mean(RMSE) (cm)')
ax1.plot(data_usages, statistic[0::3, 0], color='r', label='Original Hyperbolic')
ax1.plot(data_usages, statistic[1::3, 0], color='g', label='Nonlinear Hyperbolic')
ax1.plot(data_usages, statistic[2::3, 0], color='b', label='Weighted Nonlinear Hyperbolic')
ax1.legend()

# 데이터 사용량 대비 RMSE 중앙값 변화 도시
ax2.set_ylabel('Median(RMSE) (cm)')
ax2.plot(data_usages, statistic[0::3, 1], color='r')
ax2.plot(data_usages, statistic[1::3, 1], color='g')
ax2.plot(data_usages, statistic[2::3, 1], color='b')

# 데이터 사용량 대비 RMSE 표준편차 변화 도시
ax3.set_ylabel('Standard Deviation of RMSE (cm)')
ax3.plot(data_usages, statistic[0::3, 2], color='r')
ax3.plot(data_usages, statistic[1::3, 2], color='g')
ax3.plot(data_usages, statistic[2::3, 2], color='b')

# 데이터 사용량 대비 RMSE 중앙값 변화 도시
ax4.set_ylabel('90% Percentile of RMSE (cm)')
ax4.plot(data_usages, statistic[0::3, 3], color='r')
ax4.plot(data_usages, statistic[1::3, 3], color='g')
ax4.plot(data_usages, statistic[2::3, 3], color='b')

# 각 subplot을 포함한 리스트 설정
axes = [ax1, ax2, ax3, ax4]

# 공통 사항 적용
for ax in axes:

    if is_data_usage_percent:
        ax.set_xlabel("Data Usage (%)")
    else:
        ax.set_xlabel("Data Usage (days)")

    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')
    ax.set_xlim(data_usage_min, data_usage_max)
    ax.set_ylim(rmse_min, rmse_max)

# 그래프 저장 (SVG 및 PNG)
plt.savefig('error/error_overall.png', bbox_inches='tight')
