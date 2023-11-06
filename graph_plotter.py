"""
Title: Drawing a graph with a given csv file
Developer:
Sang Inn Woo, Ph.D. @ Incheon National University
Starting Date: 2023-11-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

# CSV 파일 읽기
data = pd.read_csv('data.csv', encoding='euc-kr')

# x, y, 축 설정
x = data['x'].to_numpy()
y = data['y'].to_numpy()

x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

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