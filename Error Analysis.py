import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
데이터 구조: Error_overall.csv
'File', 'Data_usage',
'RMSE_hyper_original', 'RMSE_hyper_nonlinear',
'Final_error_hyper_original', 'Final_error_hyper_nonlinear']
'''

# CSV 파일 읽기
df_overall = pd.read_csv('Error_overall.csv', encoding='euc-kr')

# 통계량 저장소
# 열 mean / median / percentile
# 행 RMSE (O) / RMSE (NL) / FE (O) / FE (NL)
statistic =[]

count = 0

# 최종 성토 단계에서 각 침하 데이터 사용 영역에 대해서 다음을 수행
for data_usage in range(20, 100, 10):

    # 전체 Error 분석을 위한 Dataframe 설정
    df_overall_sel = df_overall.loc[df_overall['Data_usage'] == data_usage]

    # RMSE 및 FE를 불러서 메모리에 저장
    RMSE_hyper_original = df_overall_sel['RMSE_hyper_original'].to_numpy()
    RMSE_hyper_nonlinear = df_overall_sel['RMSE_hyper_nonlinear'].to_numpy()
    FE_hyper_original = df_overall_sel['Final_error_hyper_original'].to_numpy()
    FE_hyper_nonlinear = df_overall_sel['Final_error_hyper_nonlinear'].to_numpy()

    # 중앙값, 평균, 90% percentile 산정 및 저장
    statistic.append([np.mean(RMSE_hyper_original),
                      np.median(RMSE_hyper_original),
                      np.percentile(RMSE_hyper_original, 90)])
    statistic.append([np.mean(RMSE_hyper_nonlinear),
                      np.median(RMSE_hyper_nonlinear),
                      np.percentile(RMSE_hyper_nonlinear, 90)])
    statistic.append([np.mean(FE_hyper_original),
                      np.median(FE_hyper_original),
                      np.percentile(FE_hyper_original, 90)])
    statistic.append([np.mean(FE_hyper_nonlinear),
                      np.median(FE_hyper_nonlinear),
                      np.percentile(FE_hyper_nonlinear, 90)])

    # 그래프 설정 (2 by 2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (8, 8))

    # 그래프 제목 설정
    fig.suptitle('Histograms: ' + str(data_usage) +
                 '% of Settlement Data Used in the Final Step')

    # 각 Subplot의 제목 설정
    ax1.set_xlabel('RMSE (Original Hyperbolic) (cm)')
    ax2.set_xlabel('RMSE (Nonlinear Hyperbolic) (cm)')
    ax3.set_xlabel('FE (Original Hyperbolic) (cm)')
    ax4.set_xlabel('FE (Nonliner Hyperbolic) (cm)')

    # 각 subplot에 히스토그램 작성
    ax1.hist(RMSE_hyper_original, 5, density=True, facecolor='r', edgecolor='k', alpha=0.75)
    ax2.hist(RMSE_hyper_nonlinear, 5, density=True, facecolor='b', edgecolor='k', alpha=0.75)
    ax3.hist(FE_hyper_original, 5, density=True, facecolor='r', edgecolor='k', alpha=0.75)
    ax4.hist(FE_hyper_nonlinear, 5, density=True, facecolor='b', edgecolor='k', alpha=0.75)

    # 각 subplot을 포함한 리스트 설정
    axes = [ax1, ax2, ax3, ax4]

    # 공통 사항 적용
    for i in range(len(axes)):
        ax = axes[i]
        ax.text(10, 0.4, 'Mean = ' + "{:0.2f}".format(statistic[count * 4 + i][0]) + '\n' +
                'Median = ' + "{:0.2f}".format(statistic[count * 4 + i][1]) + '\n' +
                '90% Percentile = ' + "{:0.2f}".format(statistic[count * 4 + i][2]))

        ax.set_ylabel("Probability")
        ax.grid(color="gray", alpha=.5, linestyle='--')
        ax.tick_params(direction='in')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 0.5)

    count = count + 1

    # 그래프 저장 (SVG 및 PNG)
    plt.savefig('error_analysis/error_nonstep(%i percent).png' % data_usage,
                bbox_inches='tight')

data_usages = range(20, 100, 10)
statistic = np.array(statistic)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (8, 12))

fig.suptitle("Original Hyperbolic vs. Nonlinear Hyperbolic")

# mean rmse
ax1.set_ylabel('Mean(RMSE) (cm)')
ax1.plot(data_usages, statistic[0::4, 0], label = 'Original Hyperbolic')
ax1.plot(data_usages, statistic[1::4, 0], label = 'Nonlinear Hyperbolic')
ax1.legend()

# mean fe
ax2.set_ylabel('Mean(FE) (cm)')
ax2.plot(data_usages, statistic[2::4, 0])
ax2.plot(data_usages, statistic[3::4, 0])

# median rmse
ax3.set_ylabel('Median(RMSE) (cm)')
ax3.plot(data_usages, statistic[0::4, 1])
ax3.plot(data_usages, statistic[1::4, 1])

# median fe
ax4.set_ylabel('Median(FE) (cm)')
ax4.plot(data_usages, statistic[2::4, 1])
ax4.plot(data_usages, statistic[3::4, 1])

# percentile rmse
ax5.set_ylabel('90% Percentile(RMSE) (cm)')
ax5.plot(data_usages, statistic[0::4, 2])
ax5.plot(data_usages, statistic[1::4, 2])

# percentile fe
ax6.set_ylabel('90% Percentile(FE) (cm)')
ax6.plot(data_usages, statistic[2::4, 2])
ax6.plot(data_usages, statistic[3::4, 2])

axes = [ax1, ax2, ax3, ax4, ax5, ax6]

# 공통 사항 적용
for ax in axes:
    ax.set_xlabel("Data Usage (%)")
    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

# 그래프 저장 (SVG 및 PNG)
plt.savefig('error_analysis/error_overall.png', bbox_inches='tight')


'''
데이터 구조: Error_multi_step.csv
'File', 'Data_usage',
'RMSE_hyper_original', 'RMSE_hyper_nonlinear', 'RMSE_step',
'Final_error_hyper_original', 'Final_error_hyper_nonlinear', 'Final_error_step'
'''

df_multi_step = pd.read_csv('Error_multi_step.csv', encoding='euc-kr')

# 통계량 저장소
# 열 mean / median / percentile
# 행 RMSE (O) / RMSE (NL) / RMSE(S) / FE (O) / FE (NL) / FE (S)
statistic2 =[]
count = 0

# 최종 성토 단계에서 각 침하 데이터 사용 영역에 대해서 다음을 수행
for data_usage in range(20, 100, 10):

    # 전체 Error 분석을 위한 Dataframe 설정
    df_multi_step_sel = df_multi_step.loc[df_multi_step['Data_usage'] == data_usage]

    # RMSE 및 FE를 불러서 메모리에 저장
    RMSE_hyper_original = df_multi_step_sel['RMSE_hyper_original'].to_numpy()
    RMSE_hyper_nonlinear = df_multi_step_sel['RMSE_hyper_nonlinear'].to_numpy()
    RMSE_step = df_multi_step_sel['RMSE_step'].to_numpy()
    FE_hyper_original = df_multi_step_sel['Final_error_hyper_original'].to_numpy()
    FE_hyper_nonlinear = df_multi_step_sel['Final_error_hyper_nonlinear'].to_numpy()
    FE_step = df_multi_step_sel['Final_error_step'].to_numpy()

    # 중앙값, 평균, 90% percentile 산정 및 저장
    statistic2.append([np.mean(RMSE_hyper_original),
                       np.median(RMSE_hyper_original),
                       np.percentile(RMSE_hyper_original, 90)])
    statistic2.append([np.mean(RMSE_hyper_nonlinear),
                       np.median(RMSE_hyper_nonlinear),
                       np.percentile(RMSE_hyper_nonlinear, 90)])
    statistic2.append([np.mean(RMSE_step),
                       np.median(RMSE_step),
                       np.percentile(RMSE_step, 90)])
    statistic2.append([np.mean(FE_hyper_original),
                       np.median(FE_hyper_original),
                       np.percentile(FE_hyper_original, 90)])
    statistic2.append([np.mean(FE_hyper_nonlinear),
                       np.median(FE_hyper_nonlinear),
                       np.percentile(FE_hyper_nonlinear, 90)])
    statistic2.append([np.mean(FE_step),
                       np.median(FE_step),
                       np.percentile(FE_step, 90)])

    # 그래프 설정 (2 by 2)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (12, 8))

    # 그래프 제목 설정
    fig.suptitle('Histograms: ' + str(data_usage) +
                 '% of Settlement Data Used in the Final Step')

    # 각 Subplot의 제목 설정
    ax1.set_xlabel('RMSE (Original Hyperbolic) (cm)')
    ax2.set_xlabel('RMSE (Nonlinear Hyperbolic) (cm)')
    ax3.set_xlabel('RMSE (Step) (cm)')
    ax4.set_xlabel('FE (Original Hyperbolic) (cm)')
    ax5.set_xlabel('FE (Nonliner Hyperbolic) (cm)')
    ax6.set_xlabel('FE (Step) (cm)')

    # 각 subplot에 히스토그램 작성
    ax1.hist(RMSE_hyper_original, 5, density=True, facecolor='r', edgecolor='k', alpha=0.75)
    ax2.hist(RMSE_hyper_nonlinear, 5, density=True, facecolor='b', edgecolor='k', alpha=0.75)
    ax3.hist(RMSE_step, 5, density=True, facecolor='g', edgecolor='k', alpha=0.75)
    ax4.hist(FE_hyper_original, 5, density=True, facecolor='r', edgecolor='k', alpha=0.75)
    ax5.hist(FE_hyper_nonlinear, 5, density=True, facecolor='b', edgecolor='k', alpha=0.75)
    ax6.hist(FE_step, 5, density=True, facecolor='g', edgecolor='k', alpha=0.75)

    # 각 subplot을 포함한 리스트 설정
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # 공통 사항 적용
    for i in range(len(axes)):
        ax = axes[i]
        ax.text(10, 0.4, 'Mean = ' + "{:0.2f}".format(statistic2[count * 6 + i][0]) + '\n' +
                'Median = ' + "{:0.2f}".format(statistic2[count * 6 + i][1]) + '\n' +
                '90% Percentile = ' + "{:0.2f}".format(statistic2[count * 6 + i][2]))
        ax.set_ylabel("Probability")
        ax.grid(color="gray", alpha=.5, linestyle='--')
        ax.tick_params(direction='in')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 0.5)

    count = count + 1

    # 그래프 저장 (SVG 및 PNG)
    plt.savefig('error_analysis/error_step(%i percent).png' % data_usage,
                bbox_inches='tight')


data_usages = range(20, 100, 10)
statistic2 = np.array(statistic2)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (8, 12))

fig.suptitle("Hyperbolic vs. Step loading")

# mean rmse
ax1.set_ylabel('Mean(RMSE) (cm)')
ax1.plot(data_usages, statistic2[0::6, 0], label = 'Original Hyperbolic')
ax1.plot(data_usages, statistic2[1::6, 0], label = 'Nonlinear Hyperbolic')
ax1.plot(data_usages, statistic2[2::6, 0], color='k', label = 'Step Loading')
ax1.legend()

# mean fe
ax2.set_ylabel('Mean(FE) (cm)')
ax2.plot(data_usages, statistic2[3::6, 0])
ax2.plot(data_usages, statistic2[4::6, 0])
ax2.plot(data_usages, statistic2[5::6, 0], color='k')

# median rmse
ax3.set_ylabel('Median(RMSE) (cm)')
ax3.plot(data_usages, statistic2[0::6, 1])
ax3.plot(data_usages, statistic2[1::6, 1])
ax3.plot(data_usages, statistic2[2::6, 1], color='k')

# median fe
ax4.set_ylabel('Median(FE) (cm)')
ax4.plot(data_usages, statistic2[3::6, 1])
ax4.plot(data_usages, statistic2[4::6, 1])
ax4.plot(data_usages, statistic2[5::6, 1], color='k')

# percentile rmse
ax5.set_ylabel('90% Percentile(RMSE) (cm)')
ax5.plot(data_usages, statistic2[0::6, 2])
ax5.plot(data_usages, statistic2[1::6, 2])
ax5.plot(data_usages, statistic2[2::6, 2], color='k')

# percentile fe
ax6.set_ylabel('90% Percentile(FE) (cm)')
ax6.plot(data_usages, statistic2[3::6, 2])
ax6.plot(data_usages, statistic2[4::6, 2])
ax6.plot(data_usages, statistic2[5::6, 2], color='k')

axes = [ax1, ax2, ax3, ax4, ax5, ax6]

# 공통 사항 적용
for ax in axes:
    ax.set_xlabel("Data Usage (%)")
    ax.grid(color="gray", alpha=.5, linestyle='--')
    ax.tick_params(direction='in')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

# 그래프 저장 (SVG 및 PNG)
plt.savefig('error_analysis/error_step.png', bbox_inches='tight')