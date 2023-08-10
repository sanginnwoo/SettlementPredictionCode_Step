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
statistic =[]

# 통계량 저장소 statistics 데이터 구조
#            mean    /   median  /   std
# RMSE (O)
# RMSE (NL)
# RMSE (WNL)
# FE (O)
# FE (NL)
# FE (WNL)

# 카운트 초기화
count = 0

# 최종 성토 단계에서 각 침하 데이터 사용 영역에 대해서 다음을 수행
for data_usage in range(20, 100, 10):

    # 전체 Error 분석을 위한 Dataframe 설정
    df_single_sel = df_single.loc[df_single['Data_usage'] == data_usage]

    # RMSE 및 FE를 불러서 메모리에 저장
    RMSE_hyper_original = df_single_sel['RMSE_hyper_original'].to_numpy()
    RMSE_hyper_nonlinear = df_single_sel['RMSE_hyper_nonlinear'].to_numpy()
    RMSE_hyper_weighted_nonlinear = df_single_sel['RMSE_hyper_weighted_nonlinear'].to_numpy()
    FE_hyper_original = df_single_sel['Final_error_hyper_original'].to_numpy()
    FE_hyper_nonlinear = df_single_sel['Final_error_hyper_nonlinear'].to_numpy()
    FE_hyper_weighted_nonlinear = df_single_sel['Final_error_hyper_weighted_nonlinear'].to_numpy()

    # 평균, 중앙값, 표준편차 산정 및 저장
    statistic.append([np.mean(RMSE_hyper_original), np.median(RMSE_hyper_original),
                      np.std(RMSE_hyper_original)])
    statistic.append([np.mean(RMSE_hyper_nonlinear), np.median(RMSE_hyper_nonlinear),
                      np.std(RMSE_hyper_nonlinear)])
    statistic.append([np.mean(RMSE_hyper_weighted_nonlinear), np.median(RMSE_hyper_weighted_nonlinear),
                      np.std(RMSE_hyper_weighted_nonlinear)])
    statistic.append([np.mean(FE_hyper_original), np.median(FE_hyper_original),
                      np.std(FE_hyper_original)])
    statistic.append([np.mean(FE_hyper_nonlinear), np.median(FE_hyper_nonlinear),
                      np.std(FE_hyper_nonlinear)])
    statistic.append([np.mean(FE_hyper_weighted_nonlinear), np.median(FE_hyper_weighted_nonlinear),
                      np.std(FE_hyper_weighted_nonlinear)])

    # 그래프 설정 (2 by 3)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))

    # 그래프 제목 설정
    fig.suptitle('Histograms: ' + str(data_usage) + '% of Settlement Data Used in the Final Step')

    # 각 Subplot의 제목 설정
    ax1.set_xlabel('RMSE (Original Hyperbolic) (cm)')
    ax2.set_xlabel('RMSE (Nonlinear Hyperbolic) (cm)')
    ax3.set_xlabel('RMSE (Weighted Nonlinear Hyperbolic) (cm)')
    ax4.set_xlabel('FE (Original Hyperbolic) (cm)')
    ax5.set_xlabel('FE (Nonliner Hyperbolic) (cm)')
    ax6.set_xlabel('FE (Weighted Nonliner Hyperbolic) (cm)')

    # 각 subplot에 히스토그램 작성
    ax1.hist(RMSE_hyper_original, 5, density=True, facecolor='r', edgecolor='k', alpha=0.75)
    ax2.hist(RMSE_hyper_nonlinear, 5, density=True, facecolor='g', edgecolor='k', alpha=0.75)
    ax3.hist(RMSE_hyper_weighted_nonlinear, 5, density=True, facecolor='b', edgecolor='k', alpha=0.75)
    ax4.hist(FE_hyper_original, 5, density=True, facecolor='r', edgecolor='k', alpha=0.75)
    ax5.hist(FE_hyper_nonlinear, 5, density=True, facecolor='g', edgecolor='k', alpha=0.75)
    ax6.hist(FE_hyper_weighted_nonlinear, 5, density=True, facecolor='b', edgecolor='k', alpha=0.75)

    # 각 subplot을 포함한 리스트 설정
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # 공통 사항 적용
    for i in range(len(axes)):
        ax = axes[i]
        ax.text(10, 0.4,
                'Mean = ' + "{:0.2f}".format(statistic[count * 6 + i][0]) + '\n' +
                'Median = ' + "{:0.2f}".format(statistic[count * 6 + i][1]) + '\n' +
                'Standard Deviation = ' + "{:0.2f}".format(statistic[count * 6 + i][2]))
        ax.set_ylabel("Probability")
        ax.grid(color="gray", alpha=.5, linestyle='--')
        ax.tick_params(direction='in')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 0.5)

    # 그래프 저장 (SVG 및 PNG)
    plt.savefig('error/error_nonstep(%i percent).png' % data_usage, bbox_inches='tight')

    # 카운트 증가
    count = count + 1

# 데이터 사용 영역 설정
data_usages = range(20, 100, 10)

# 통계량 배열을 numpy 배열로 전환
statistic = np.array(statistic)

# 그래프 설정 (총 4개의 그래프)
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (8, 12))

# 그래프 전체 제목 설정
fig.suptitle("Original Hyperbolic vs. Nonlinear Hyperbolic")

# 데이터 사용량 대비 RMSE 평균값 변화 도시
ax1.set_ylabel('Mean(RMSE) (cm)')
ax1.plot(data_usages, statistic[0::4, 0], label = 'Original Hyperbolic')
ax1.plot(data_usages, statistic[1::4, 0], label = 'Nonlinear Hyperbolic')
ax1.legend()

# 데이터 사용량 대비 FE 평균값 변화 도시
ax2.set_ylabel('Mean(FE) (cm)')
ax2.plot(data_usages, statistic[2::4, 0], label = 'Original Hyperbolic')
ax2.plot(data_usages, statistic[3::4, 0], label = 'Nonlinear Hyperbolic')

# 데이터 사용량 대비 RMSE 중앙값 변화 도시
ax3.set_ylabel('Median(RMSE) (cm)')
ax3.plot(data_usages, statistic[0::4, 1])
ax3.plot(data_usages, statistic[1::4, 1])

# 데이터 사용량 대비 FE 중앙값 변화 도시
ax4.set_ylabel('Median(FE) (cm)')
ax4.plot(data_usages, statistic[2::4, 1])
ax4.plot(data_usages, statistic[3::4, 1])

# 데이터 사용량 대비 RMSE의 90% Percentile 변화 도시
ax5.set_ylabel('90% Percentile(RMSE) (cm)')
ax5.plot(data_usages, statistic[0::4, 2])
ax5.plot(data_usages, statistic[1::4, 2])

# 데이터 사용량 대비 FE의 90% Percentile 변화 도시
ax6.set_ylabel('90% Percentile(FE) (cm)')
ax6.plot(data_usages, statistic[2::4, 2])
ax6.plot(data_usages, statistic[3::4, 2])

# 각 subplot을 포함한 리스트 설정
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

# 공통 사항 적용
for ax in axes:
    ax.set_xlabel("Data Usage (%)")
    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

# 그래프 저장 (SVG 및 PNG)
plt.savefig('error/error_overall.png', bbox_inches='tight')