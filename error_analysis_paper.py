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
데이터 구조: rmse_analysis.csv
['No', 'Site', 'File', 'Data_date', 'RMSE_date', 'Final_date' 
 'RMSE_hyper_original', 'RMSE_hyper_nonlinear', 'RMSE_hyper_weighted_nonlinear',
 'Div_hyper_original', 'Div_hyper_nonlinear', 'Div_hyper_weighted_nonlinear'])
'''

# CSV 파일 읽기 (일단, 다단 성토 포함)
df_single = pd.read_csv('rmse_analysis.csv')#, encoding='euc-kr')

# 통계량 저장소 statistics 초기화
statistic = []

# 통계량 저장소 statistics 데이터 구조
#               mean    /   median  /   std   / 90% percentile
# RMSE (OH)
# RMSE (NH)
# RMSE (WNH)

# 카운트 초기화
count = 0

#
#num_bars = 10
rmse_min = 0
rmse_max = 40
probability_min = 0
probability_max = 1.0
data_usage_min = 30
data_usage_max = 180
is_data_usage_percent = False

# RMSE 산정 구간 = 140-160, 280-300, 420-440, 560-600, 700-720
for rmse_date in range(140, 840, 140):

    # 침하 예측 구간 설정 60, 90, 120, ... , j - 30 까지 수행
    for data_date in range(60, rmse_date, 30):

        # 전체 Error 분석을 위한 Dataframe 설정
        df_single_sel = df_single.loc[df_single['RMSE_date'] == rmse_date]
        df_single_sel = df_single_sel.loc[df_single['Data_date'] == data_date]

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
        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        # 히스토그램 작성
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # 그래프 제목 설정
        #if is_data_usage_percent:
        #    ax.title('Histograms: ' + str(data_date) + '% of Data Used')
        #else:
        #    ax.title('RMSE: ' + str(rmse_date) + ' ~ ' + str(rmse_date + 20) + ' days, ' +
        #              'Data used in Prediction: 0 ~ ' + str(data_date) + ' days')

        # 각 Subplot의 제목 설정
        #ax1.set_xlabel('RMSE (Original Hyperbolic) (cm)')
        #ax2.set_xlabel('RMSE (Nonlinear Hyperbolic) (cm)')
        #ax3.set_xlabel('RMSE (Weighted Nonlinear Hyperbolic) (cm)')


        ax.set_ylim(0, 0.20)
        ax.set_xlim(0, 50)
        ax.set_yticks(np.arange(0, 0.25, 0.05))
        ax.set_xticks(np.arange(0, 55, 5))
        ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
        ax.grid(axis='both', color='lightgray', ls='-', lw=0.5)
        ax.set_xlabel("RMSE (cm)")
        ax.set_ylabel("Relative Frequency / $\\Delta$")

        ax.text(48, 0.15,
                'RMSE estimation in ' + str(rmse_date) + '~' + str(rmse_date + 20) + ' days \n' +
                'Prediction using settlement data in 0 ~ ' + str(data_date) + ' days \n' +
                '$\\Delta$ = 5 days',
                ha='right', va='top', size=10)

        n, bins, patches = ax.hist([RMSE_hyper_original, RMSE_hyper_nonlinear, RMSE_hyper_weighted_nonlinear],
                                   label=['Original Hyperbolic', 'Nonlinear Hyperbolic', 'Weighed Nonlinear Hyperbolic'],
                                   bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                                   density=True, stacked=False, edgecolor='black', alpha=0.75,
                                   color =['dimgray', 'darkgray', 'black'])

        ax.legend(loc = 1, fontsize=10)

        # 그래프 저장 (SVG 및 PNG)
        if is_data_usage_percent:
            plt.savefig('error/error_single(%i percent).png' % data_date, bbox_inches='tight')
        else:
            plt.savefig('error/error_single(rmse %i data %i).png' % (rmse_date, data_date), bbox_inches='tight')

        plt.close()

        # 카운트 증가
        count = count + 1


    # 통계량 배열을 numpy 배열로 전환
    #statistic = np.array(statistic)

    # 그래프 설정 (총 4개의 그래프)
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

    # 그래프 전체 제목 설정
    #fig.suptitle("RMSE Analysis of Original, Nonlinear, and Weighted Nonlinear Hyperbolic")

    # 데이터 사용량 대비 RMSE 평균값 변화 도시
    #ax1.set_ylabel('Mean(RMSE) (cm)')
    ##ax1.plot(data_usages, statistic[0::3, 0], color='r', label='Original Hyperbolic')
    #ax1.plot(data_usages, statistic[1::3, 0], color='g', label='Nonlinear Hyperbolic')
    #ax1.plot(data_usages, statistic[2::3, 0], color='b', label='Weighted Nonlinear Hyperbolic')
    #ax1.legend()

    # 데이터 사용량 대비 RMSE 중앙값 변화 도시
    #ax2.set_ylabel('Median(RMSE) (cm)')
    #ax2.plot(data_usages, statistic[0::3, 1], color='r')
    #ax2.plot(data_usages, statistic[1::3, 1], color='g')
    #ax2.plot(data_usages, statistic[2::3, 1], color='b')

    # 데이터 사용량 대비 RMSE 표준편차 변화 도시
    #ax3.set_ylabel('Standard Deviation of RMSE (cm)')
    #ax3.plot(data_usages, statistic[0::3, 2], color='r')
    #ax3.plot(data_usages, statistic[1::3, 2], color='g')
    #ax3.plot(data_usages, statistic[2::3, 2], color='b')

    # 데이터 사용량 대비 RMSE 중앙값 변화 도시
    #ax4.set_ylabel('90% Percentile of RMSE (cm)')
    #ax4.plot(data_usages, statistic[0::3, 3], color='r')
    #ax4.plot(data_usages, statistic[1::3, 3], color='g')
    #ax4.plot(data_usages, statistic[2::3, 3], color='b')

    # 각 subplot을 포함한 리스트 설정
    #axes = [ax1, ax2, ax3, ax4]

    # 공통 사항 적용
    #for ax in axes:

        #    if is_data_usage_percent:
        #    ax.set_xlabel("Data Usage (%)")
        #else:
        #    ax.set_xlabel("Data Usage (days)")

        #ax.grid(color="gray", alpha=.5, linestyle='--')
        ##ax.tick_params(direction='in')
        #ax.set_xlim(data_usage_min, data_usage_max)
        #ax.set_ylim(rmse_min, rmse_max)

    # 그래프 저장 (SVG 및 PNG)
    #plt.savefig('error/error_overall.png', bbox_inches='tight')
