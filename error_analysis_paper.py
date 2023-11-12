"""
Title: Error analysis of the settlement prediction curves
Developer:
Sang Inn Woo, Ph.D. @ Incheon National University
Starting Date: 2022-11-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

'''
데이터 구조: rmse_analysis.csv
['No', 'Site', 'File', 'Data_date', 'RMSE_date', 'Final_date' 
 'RMSE_hyper_original', 'RMSE_hyper_nonlinear', 'RMSE_hyper_weighted_nonlinear',
 'Div_hyper_original', 'Div_hyper_nonlinear', 'Div_hyper_weighted_nonlinear'])
'''

# CSV 파일 읽기 (일단, 다단 성토 포함)
df_single = pd.read_csv('rmse_analysis.csv')  # , encoding='euc-kr')

# 통계량 저장소 statistics 초기화
statistic = []

# 통계량 저장소 statistics 데이터 구조
#               mean    /   median  /   std   / 90% percentile
# RMSE (OH)
# RMSE (NH)
# RMSE (WNH)

df_stats = pd.DataFrame(columns=['Data_date',
                                 'RMSE_date',
                                 'mean_hyper_original',
                                 'mean_hyper_nonlinear',
                                 'mean_hyper_weighted_nonlinear',
                                 'median_hyper_original',
                                 'median_hyper_nonlinear',
                                 'median_hyper_weighted_nonlinear',
                                 'std_hyper_original',
                                 'std_hyper_nonlinear',
                                 'std_hyper_weighted_nonlinear',
                                 'q90_hyper_original',
                                 'q90_hyper_nonlinear',
                                 'q90_hyper_weighted_nonlinear',
                                 'gamma_shape_hyper_original',
                                 'gamma_shape_hyper_nonlinear',
                                 'gamma_shape_hyper_weighted_nonlinear',
                                 'gamma_scale_hyper_original',
                                 'gamma_scale_hyper_nonlinear',
                                 'gamma_scale_hyper_weighted_nonlinear'])

# 카운트 초기화
count = 0

#
# num_bars = 10
probability_min = 0
probability_max = 1.0
is_data_usage_percent = False

rmse_dates = range(140, 500, 140)

# RMSE 산정 구간 = 140-160, 280-300, 420-440, 560-600, 700-720
for rmse_date in rmse_dates:

    data_dates = range(60, rmse_date, 30)
    statistic = []

    # 침하 예측 구간 설정 60, 90, 120, ... , j - 30 까지 수행
    for data_date in data_dates:

        # 전체 Error 분석을 위한 Dataframe 설정
        df_single_sel = df_single.loc[df_single['RMSE_date'] == rmse_date]
        df_single_sel = df_single_sel.loc[df_single['Data_date'] == data_date]

        # RMSE 및 FE를 불러서 메모리에 저장
        RMSE_hyper_original = df_single_sel['RMSE_hyper_original'].to_numpy()
        RMSE_hyper_nonlinear = df_single_sel['RMSE_hyper_nonlinear'].to_numpy()
        RMSE_hyper_weighted_nonlinear = df_single_sel['RMSE_hyper_weighted_nonlinear'].to_numpy()

        # Gamma 분포 산정을 위한 개체 선언
        gamma = stats.gamma
        x = np.linspace(0, rmse_date, 1000)

        # Gamma 분포 매개변수 산정
        gamma_param_original = gamma.fit(RMSE_hyper_original, floc=0)
        gamma_param_nonlinear = gamma.fit(RMSE_hyper_nonlinear, floc=0)
        gamma_param_weighted_nonlinear = gamma.fit(RMSE_hyper_weighted_nonlinear, floc=0)

        # Gamma 분포 PDF 구축
        gamma_pdf_original = gamma.pdf(x, *gamma_param_original)
        gamma_pdf_nonlinear = gamma.pdf(x, *gamma_param_nonlinear)
        gamma_pdf_weighted_nonlinear = gamma.pdf(x, *gamma_param_weighted_nonlinear)

        # 평균, 중앙값, 표준편차, Q90 산정 및 저장 - 그래프 도시용
        statistic.append([np.mean(RMSE_hyper_original), np.median(RMSE_hyper_original),
                          np.std(RMSE_hyper_original), np.percentile(RMSE_hyper_original, 90),
                          gamma_param_original[0], gamma_param_original[2]])
        statistic.append([np.mean(RMSE_hyper_nonlinear), np.median(RMSE_hyper_nonlinear),
                          np.std(RMSE_hyper_nonlinear), np.percentile(RMSE_hyper_nonlinear, 90),
                          gamma_param_nonlinear[0], gamma_param_nonlinear[2]])
        statistic.append([np.mean(RMSE_hyper_weighted_nonlinear), np.median(RMSE_hyper_weighted_nonlinear),
                          np.std(RMSE_hyper_weighted_nonlinear), np.percentile(RMSE_hyper_weighted_nonlinear, 90),
                          gamma_param_weighted_nonlinear[0], gamma_param_weighted_nonlinear[2]])

        df_stats.loc[len(df_stats.index)] = [rmse_date, data_date,
                                             np.mean(RMSE_hyper_original),
                                             np.mean(RMSE_hyper_nonlinear),
                                             np.mean(RMSE_hyper_weighted_nonlinear),
                                             np.median(RMSE_hyper_original),
                                             np.median(RMSE_hyper_nonlinear),
                                             np.median(RMSE_hyper_weighted_nonlinear),
                                             np.std(RMSE_hyper_original),
                                             np.std(RMSE_hyper_nonlinear),
                                             np.std(RMSE_hyper_weighted_nonlinear),
                                             np.percentile(RMSE_hyper_original, 90),
                                             np.percentile(RMSE_hyper_nonlinear, 90),
                                             np.percentile(RMSE_hyper_weighted_nonlinear, 90),
                                             gamma_param_original[0],
                                             gamma_param_nonlinear[0],
                                             gamma_param_weighted_nonlinear[0],
                                             gamma_param_original[2],
                                             gamma_param_nonlinear[2],
                                             gamma_param_weighted_nonlinear[2]]

        # 히스토그램 작성
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.set_ylim(0, 0.30)
        ax.set_xlim(0, 50)
        ax.set_yticks(np.arange(0, 0.35, 0.05))
        ax.set_xticks(np.arange(0, 55, 5))
        ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
        ax.grid(axis='both', color='lightgray', ls='-', lw=0.5)
        ax.set_xlabel("RMSE (cm)")
        ax.set_ylabel("Relative Frequency / $\\Delta$")
        ax.text(48, 0.20,
                'PDF is fitted using the Gamma distribution \n' +
                'RMSE estimation in ' + str(rmse_date) + ' - ' + str(rmse_date + 20) + ' days \n' +
                'Prediction using settlement data in 0 - ' + str(data_date) + ' days \n' +
                '$\\Delta$ = 5 cm',
                ha='right', va='top', size=10)

        ax.hist([RMSE_hyper_original, RMSE_hyper_nonlinear, RMSE_hyper_weighted_nonlinear],
                label=['Original Hyperbolic', 'Nonlinear Hyperbolic', 'Weighed Nonlinear Hyperbolic'],
                bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                density=True, stacked=False, edgecolor='black', alpha=0.75,
                color=['dimgray', 'darkgray', 'black'])

        # Gamma 분포 PDF 도시
        ax.plot(x, gamma_pdf_original, label='PDF (Original Hyperbolic)',
                color='dimgray', linestyle='--', linewidth=1.0)
        ax.plot(x, gamma_pdf_nonlinear, label='PDF (Nonlinear Hyperbolic)',
                color='darkgray', linestyle='--', linewidth=1.0)
        ax.plot(x, gamma_pdf_weighted_nonlinear, label='PDF (Weighted Nonlinear Hyperbolic)',
                color='black', linestyle='--', linewidth=1.0)

        # 범례 삽입
        ax.legend(loc=1, fontsize=10)

        # 그래프 저장 (SVG 및 PNG)
        if is_data_usage_percent:
            plt.savefig('error/error_single(%i percent).png' % data_date, bbox_inches='tight')
        else:
            plt.savefig('error/error_single(rmse %i data %i).png' % (rmse_date, data_date), bbox_inches='tight')

        plt.close()

        # 카운트 증가
        count = count + 1

    rmse_min = 0
    rmse_max = max(max(statistic))
    rmse_max = math.ceil(rmse_max / 10) * 10

    data_usage_min = min(data_dates) - 30
    data_usage_max = max(data_dates) + 30

    # 통계량 배열을 numpy 배열로 전환
    statistic = np.array(statistic)

    # 데이터 사용량 대비 RMSE 평균값 변화 도시
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_ylabel('Mean of RMSEs (cm)')
    ax.set_xlabel("Data Used (days)")

    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')
    ax.set_xticks(np.arange(30, data_usage_max + 30, 30))

    ax.set_xlim(data_usage_min, data_usage_max)
    ax.set_ylim(0, rmse_max)

    ax.text(data_usage_max * 0.97, rmse_max * 0.80,
            'RMSE estimation in ' + str(rmse_date) + ' - ' + str(rmse_date + 20) + ' days',
            ha='right', va='top', size=10)

    ax.plot(data_dates, statistic[0::3, 0], color='dimgray',
            marker='o', markerfacecolor='dimgray', markeredgecolor='black', label='Original Hyperbolic')
    ax.plot(data_dates, statistic[1::3, 0], color='darkgray',
            marker='o', markerfacecolor='darkgray', markeredgecolor='black', label='Nonlinear Hyperbolic')
    ax.plot(data_dates, statistic[2::3, 0], color='black',
            marker='o', markerfacecolor='black', markeredgecolor='black', label='Weighted Nonlinear Hyperbolic')

    ax.legend()
    plt.savefig("error/evolution_rmse(rmse date %i mean).png" % rmse_date, bbox_inches='tight')
    plt.close()

    # 데이터 사용량 대비 RMSE 중앙값 변화 도시
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_ylabel('Median of RMSEs (cm)')
    ax.set_xlabel("Data Used (days)")

    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')
    ax.set_xticks(np.arange(30, data_usage_max + 30, 30))

    ax.set_xlim(data_usage_min, data_usage_max)
    ax.set_ylim(0, rmse_max)

    ax.text(data_usage_max * 0.97, rmse_max * 0.80,
            'RMSE estimation in ' + str(rmse_date) + ' - ' + str(rmse_date + 20) + ' days',
            ha='right', va='top', size=10)

    ax.plot(data_dates, statistic[0::3, 1], color='dimgray',
            marker='o', markerfacecolor='dimgray', markeredgecolor='black', label='Original Hyperbolic')
    ax.plot(data_dates, statistic[1::3, 1], color='darkgray',
            marker='o', markerfacecolor='darkgray', markeredgecolor='black', label='Nonlinear Hyperbolic')
    ax.plot(data_dates, statistic[2::3, 1], color='black',
            marker='o', markerfacecolor='black', markeredgecolor='black', label='Weighted Nonlinear Hyperbolic')

    ax.legend()
    plt.savefig("error/evolution_rmse(rmse date %i median).png" % rmse_date, bbox_inches='tight')
    plt.close()

    # 데이터 사용량 대비 RMSE 표준편차 변화 도시
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_ylabel('Standard Deviation of RMSEs (cm)')
    ax.set_xlabel("Data Used (days)")

    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')
    ax.set_xticks(np.arange(30, data_usage_max + 30, 30))

    ax.set_xlim(data_usage_min, data_usage_max)
    ax.set_ylim(0, rmse_max)

    ax.text(data_usage_max * 0.97, rmse_max * 0.80,
            'RMSE estimation in ' + str(rmse_date) + ' - ' + str(rmse_date + 20) + ' days',
            ha='right', va='top', size=10)

    ax.plot(data_dates, statistic[0::3, 2], color='dimgray',
            marker='o', markerfacecolor='dimgray', markeredgecolor='black', label='Original Hyperbolic')
    ax.plot(data_dates, statistic[1::3, 2], color='darkgray',
            marker='o', markerfacecolor='darkgray', markeredgecolor='black', label='Nonlinear Hyperbolic')
    ax.plot(data_dates, statistic[2::3, 2], color='black',
            marker='o', markerfacecolor='black', markeredgecolor='black', label='Weighted Nonlinear Hyperbolic')

    ax.legend()
    plt.savefig("error/evolution_rmse(rmse date %i std).png" % rmse_date, bbox_inches='tight')
    plt.close()

    # 데이터 사용량 대비 RMSE 90% 변위수 변화 도시
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_ylabel('90% Percentile of RMSEs (cm)')
    ax.set_xlabel("Data Used (days)")

    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')
    ax.set_xticks(np.arange(30, data_usage_max + 30, 30))

    ax.set_xlim(data_usage_min, data_usage_max)
    ax.set_ylim(0, rmse_max)

    ax.text(data_usage_max * 0.97, rmse_max * 0.80,
            'RMSE estimation in ' + str(rmse_date) + ' - ' + str(rmse_date + 20) + ' days',
            ha='right', va='top', size=10)

    ax.plot(data_dates, statistic[0::3, 3], color='dimgray',
            marker='o', markerfacecolor='dimgray', markeredgecolor='black', label='Original Hyperbolic')
    ax.plot(data_dates, statistic[1::3, 3], color='darkgray',
            marker='o', markerfacecolor='darkgray', markeredgecolor='black', label='Nonlinear Hyperbolic')
    ax.plot(data_dates, statistic[2::3, 3], color='black',
            marker='o', markerfacecolor='black', markeredgecolor='black', label='Weighted Nonlinear Hyperbolic')

    ax.legend()
    plt.savefig("error/evolution_rmse(rmse date %i 90q).png" % rmse_date, bbox_inches='tight')
    plt.close()

    # 데이터 사용량 대비 RMSE 90% 변위수 변화 도시
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_ylabel('Probability')
    ax.set_xlabel("RMSE (cm)")
    ax.set_ylim(0, 0.55)
    ax.set_xlim(0, 50)
    ax.set_yticks(np.arange(0, 0.55, 0.05))
    ax.set_xticks(np.arange(0, 55, 5))
    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')

    for i in range(len(data_dates)):

        gamma_param_original = [statistic[i * 3, 4], 0, statistic[i * 3, 5]]
        gamma_param_nonlinear = [statistic[1 + i * 3, 4], 0, statistic[1 + i * 3, 5]]
        gamma_param_weighted_nonlinear = [statistic[2 + i * 3, 4], 0, statistic[2 + i * 3, 5]]

        x = np.linspace(0, rmse_date, 1000)

        gamma_pdf_original = gamma.pdf(x, *gamma_param_original)
        gamma_pdf_nonlinear = gamma.pdf(x, *gamma_param_nonlinear)
        gamma_pdf_weighted_nonlinear = gamma.pdf(x, *gamma_param_weighted_nonlinear)

        # Gamma 분포 PDF 도시
        ax.plot(x, gamma_pdf_original, color='dimgray', linestyle='--', linewidth=1.0)

    ax.legend()
    plt.savefig("error/gamma_dist_original(rmse date %i).png" % rmse_date, bbox_inches='tight')
    plt.close()

    # 데이터 사용량 대비 RMSE 90% 변위수 변화 도시
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_ylabel('Probability')
    ax.set_xlabel("RMSE (cm)")
    ax.set_ylim(0, 0.55)
    ax.set_xlim(0, 50)
    ax.set_yticks(np.arange(0, 0.55, 0.05))
    ax.set_xticks(np.arange(0, 55, 5))
    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')

    for i in range(len(data_dates)):
        gamma_param_original = [statistic[i * 3, 4], 0, statistic[i * 3, 5]]
        gamma_param_nonlinear = [statistic[1 + i * 3, 4], 0, statistic[1 + i * 3, 5]]
        gamma_param_weighted_nonlinear = [statistic[2 + i * 3, 4], 0, statistic[2 + i * 3, 5]]

        x = np.linspace(0, rmse_date, 1000)

        gamma_pdf_original = gamma.pdf(x, *gamma_param_original)
        gamma_pdf_nonlinear = gamma.pdf(x, *gamma_param_nonlinear)
        gamma_pdf_weighted_nonlinear = gamma.pdf(x, *gamma_param_weighted_nonlinear)

        # Gamma 분포 PDF 도시
        ax.plot(x, gamma_pdf_nonlinear, color='darkgray', linestyle='--', linewidth=1.0)

    ax.legend()
    plt.savefig("error/gamma_dist_nonlinear(rmse date %i).png" % rmse_date, bbox_inches='tight')
    plt.close()

    # 데이터 사용량 대비 RMSE 90% 변위수 변화 도시
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_ylabel('Probability')
    ax.set_xlabel("RMSE (cm)")
    ax.set_ylim(0, 0.55)
    ax.set_xlim(0, 50)
    ax.set_yticks(np.arange(0, 0.55, 0.05))
    ax.set_xticks(np.arange(0, 55, 5))
    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')

    for i in range(len(data_dates)):
        gamma_param_original = [statistic[i * 3, 4], 0, statistic[i * 3, 5]]
        gamma_param_nonlinear = [statistic[1 + i * 3, 4], 0, statistic[1 + i * 3, 5]]
        gamma_param_weighted_nonlinear = [statistic[2 + i * 3, 4], 0, statistic[2 + i * 3, 5]]

        x = np.linspace(0, rmse_date, 1000)

        gamma_pdf_original = gamma.pdf(x, *gamma_param_original)
        gamma_pdf_nonlinear = gamma.pdf(x, *gamma_param_nonlinear)
        gamma_pdf_weighted_nonlinear = gamma.pdf(x, *gamma_param_weighted_nonlinear)

        # Gamma 분포 PDF 도시
        ax.plot(x, gamma_pdf_weighted_nonlinear, color='black', linestyle='--', linewidth=1.0)

    ax.legend()
    plt.savefig("error/gamma_dist_weighted_nonlinear(rmse date %i).png" % rmse_date, bbox_inches='tight')
    plt.close()



# 에러 파일을 출력
df_stats.to_csv('error/error_stats.csv')
