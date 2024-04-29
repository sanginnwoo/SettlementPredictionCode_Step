"""
Title: Soft ground settlement prediction
Developer:
Sang Inn Woo, Ph.D. @ Incheon National University
Kwak Taeyoung, Ph.D. @ KICT

Starting Date: 2022-08-11
Abstract:
This main objective of this code is to predict
time vs. (consolidation) settlement curves of soft clay ground.
"""

# =================
# Import 섹션
# =================

import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import least_squares
from datetime import datetime, timedelta


# 한글 폰트 설정
font_path = 'C:\\Windows\\Fonts\\malgun.ttf'
font = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font)


# =================
# Function 섹션
# =================

# 주어진 계수를 이용하여 쌍곡선 시간-침하 곡선 반환
def generate_data_hyper(px, pt):
    return pt / (px[0] * pt + px[1])


# 주어진 계수를 이용하여 아사오카 시간-침하 곡선 반환
def generate_data_asaoka(px, pt, dt):
    return (px[1] / (1 - px[0])) * (1 - (px[0] ** (pt / dt)))


# 회귀식과 측정치와의 잔차 반환 (비선형 쌍곡선)
def fun_hyper_nonlinear(px, pt, py):
    return pt / (px[0] * pt + px[1]) - py


# 회귀식과 측정치와의 잔차 반환 (가중 비선형 쌍곡선)
def fun_hyper_weight_nonlinear(px, pt, py, pw):
    return (pt / (px[0] * pt + px[1]) - py) * pw


# 회귀식과 측정치와의 잔차 반환 (기존 쌍곡선)
def fun_hyper_original(px, pt, py):
    return px[0] * pt + px[1] - pt / py


# 회귀식과 측정치와의 잔차 반환 (아사오카)
def fun_asaoka(px, ps_b, ps_a):
    return px[0] * ps_b + px[1] - ps_a


# RMSE 산정
def fun_rmse(py1, py2):
    mse = np.square(np.subtract(py1, py2)).mean()
    return np.sqrt(mse)


def run_settle_prediction_from_file(input_file, output_dir,
                                    final_step_predict_percent,
                                    additional_predict_days,
                                    plot_show, print_values):

    # 현재 파일 이름 출력
    print("Working on " + input_file)

    # 파일 이름만 추출
    filename = os.path.basename(input_file)
    filename = os.path.splitext(filename)

    # CSV 파일 읽기 - 목표 침하량 및 목표 성토고
    data_all = pd.read_csv("final_values_posco.csv", encoding='euc-kr')
    index = data_all.index[data_all['Point'] == filename[0]].tolist()
    settle_goal = data_all['settle_goal'][index[0]]
    surcharge_goal = data_all['surcharge_goal'][index[0]]

    # CSV 파일 읽기 - 계측데이터
    data = pd.read_csv(input_file, encoding='euc-kr')

    time = []
    date = []
    settle = []
    surcharge = []

    # 시간 배열 생성
    if 'Time' in data.columns:
        time = data['Time'].to_numpy()
    elif 'time' in data.columns:
        time = data['time'].to_numpy()
    elif 'Day' in data.columns:
        time = data['Day'].to_numpy()

    # 날짜 배열 생성
    if 'Date' in data.columns:
        date = pd.to_datetime(data['Date'], format="%Y-%m-%d").dt.date

    # 침하량 배열 생성
    if 'Settle' in data.columns:
        settle = data['Settle'].to_numpy()
    elif 'settle' in data.columns:
        settle = data['settle'].to_numpy()
    elif 'Settlement' in data.columns:
        settle = data['Settlement'].to_numpy()
    elif 'settlement' in data.columns:
        settle = data['settlement'].to_numpy()

    # 성토고 배열 생성
    if 'Surcharge' in data.columns:
        surcharge = data['Surcharge'].to_numpy()
    elif 'surcharge' in data.columns:
        surcharge = data['surcharge'].to_numpy()

    # 침하 예측 수행
    run_settle_prediction(point_name=input_file, output_dir=output_dir,
                          np_time=time, np_date=date,
                          np_surcharge=surcharge,
                          np_settlement=settle,
                          final_step_predict_percent=final_step_predict_percent,
                          additional_predict_days=additional_predict_days,
                          plot_show=plot_show,
                          print_values=print_values,
                          settle_goal=settle_goal,
                          surcharge_goal=surcharge_goal)


def run_settle_prediction(point_name, output_dir,
                          np_time, np_date,
                          np_surcharge,
                          np_settlement,
                          final_step_predict_percent,
                          additional_predict_days,
                          plot_show,
                          print_values,
                          settle_goal = None,
                          surcharge_goal = None):

    # ====================
    # 파일 읽기, 데이터 설정
    # ====================

    # 시간, 침하량, 성토고 배열 생성
    time = np_time
    date = np_date
    settle = np_settlement
    surcharge = np_surcharge

    # 마지막 계측 데이터 index + 1 파악
    final_index = time.size

    # =================
    # 성토 단계 구분
    # =================

    # 성토 단계 시작 index 리스트 초기화
    step_start_index = [0]

    # 성토 단계 끝 index 리스트 초기화
    step_end_index = []

    # 현재 성토고 설정
    current_surcharge = surcharge[0]

    # 모든 시간-성토고 데이터에서 순차적으로 확인
    for index in range(len(surcharge)):

        # 만일 성토고의 변화가 있을 경우,
        if (surcharge[index] > current_surcharge * 1.05 or
                surcharge[index] < current_surcharge * 0.95):
            step_end_index.append(index)
            step_start_index.append(index)
            current_surcharge = surcharge[index]

    # 마지막 성토 단계 끝 index 추가
    step_end_index.append(len(surcharge) - 1)

    # =================
    # 성토 단계 조정
    # =================
    # 성토고 유지 기간이 매우 짧을 경우, 해석 단계에서 제외

    # 조정 성토 시작 및 끝 인덱스 리스트 초기화
    step_start_index_adjust = []
    step_end_index_adjust = []

    # 각 성토 단계 별로 분석
    for i in range(0, len(step_start_index)):

        # 현 단계 성토 시작일 / 끝일 파악
        step_start_date = time[step_start_index[i]]
        step_end_date = time[step_end_index[i]]

        # 현 성토고 유지 일수 및 데이터 개수 파악
        step_span = step_end_date - step_start_date
        step_data_num = step_end_index[i] - step_start_index[i] + 1

        # 성토고 유지일 및 데이터 개수 기준 적용
        if step_span > 3 and step_data_num > 2:
            step_start_index_adjust.append((step_start_index[i]))
            step_end_index_adjust.append((step_end_index[i]))

    #  성토 시작 및 끝 인덱스 리스트 업데이트
    step_start_index = step_start_index_adjust
    step_end_index = step_end_index_adjust

    # 성토 단계 횟수 파악 및 저장
    num_steps = len(step_start_index)

    # ===========================
    # 최종 단계 데이터 사용 범위 조정
    # ===========================

    # 데이터 사용 퍼센트에 해당하는 기간 계산
    final_step_end_date = time[-1]
    final_step_start_date = time[step_start_index[num_steps - 1]]
    final_step_period = final_step_end_date - final_step_start_date

    final_step_predict_end_date = (final_step_start_date + final_step_period * final_step_predict_percent / 100)

    # =================
    # 추가 예측 구간 반영
    # =================

    # 추가 예측 일 입력 (현재 전체 계측일 * 계수)
    add_days = additional_predict_days #(additional_predict_percent / 100) * time[-1]

    # 마지막 성토고 및 마지막 계측일 저장
    final_surcharge = surcharge[final_index - 1]
    final_time = time[final_index - 1]
    final_date = date[final_index - 1]

    # 추가 시간 및 성토고 배열 설정 (100개의 시점 설정)
    time_add = np.linspace(final_time + 1, final_time + add_days, 100)
    surcharge_add = np.ones(100) * final_surcharge
    date_add = pd.date_range(start=final_date + timedelta(days=1),
                             end=final_date + timedelta(days=add_days),
                             periods=100)
    date_add = [date.date() for date in date_add]


    # 기존 시간 및 성토고 배열에 붙이기
    time = np.append(time, time_add)
    surcharge = np.append(surcharge, surcharge_add)
    date = np.append(date, date_add)

    # 마지막 인덱스값 재조정
    final_index = time.size

    # =========================================================
    # Settlement prediction (nonliner, weighted nonlinear and original hyperbolic)
    # =========================================================

    # 성토 마지막 데이터 추출
    tm_hyper = time[step_start_index[num_steps - 1]:step_end_index[num_steps - 1]]
    sm_hyper = settle[step_start_index[num_steps - 1]:step_end_index[num_steps - 1]]

    # 현재 단계 시작 부터 끝까지 시간 데이터 추출
    time_hyper = time[step_start_index[num_steps - 1]:final_index]

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
    if print_values:
        print(x_hyper_nonlinear)

    # 가중 비선형 쌍곡선 가중치 산정
    weight = tm_hyper / np.sum(tm_hyper)

    # 회귀분석 시행 (가중 비선형 쌍곡선)
    x0 = np.ones(2)
    res_lsq_hyper_weight_nonlinear = least_squares(fun_hyper_weight_nonlinear, x0,
                                                   args=(tm_hyper, sm_hyper, weight))
    # 비선형 쌍곡선 법 계수 저장 및 출력
    x_hyper_weight_nonlinear = res_lsq_hyper_weight_nonlinear.x
    if print_values:
        print(x_hyper_weight_nonlinear)

    # 회귀분석 시행 (기존 쌍곡선법) - (0, 0)에 해당하는 초기 데이터를 제외하고 회귀분석 실시
    x0 = np.ones(2)
    start_index_oh = 0
    for i in range(0, len(sm_hyper)):
        if sm_hyper[i] != 0:
            start_index_oh = i
            break
    res_lsq_hyper_original = least_squares(fun_hyper_original, x0,
                                               args=(tm_hyper[start_index_oh:], sm_hyper[start_index_oh:]))

    # 기존 쌍곡선 법 계수 저장 및 출력
    x_hyper_original = res_lsq_hyper_original.x
    if print_values:
        print(x_hyper_original)

    # 현재 단계 예측 침하량 산정 (침하 예측 끝까지)
    sp_hyper_nonlinear = generate_data_hyper(x_hyper_nonlinear, time_hyper)
    sp_hyper_weight_nonlinear = generate_data_hyper(x_hyper_weight_nonlinear, time_hyper)
    sp_hyper_original = generate_data_hyper(x_hyper_original, time_hyper)

    # 예측 침하량 산정
    sp_hyper_nonlinear = sp_hyper_nonlinear + s0_hyper
    sp_hyper_weight_nonlinear = sp_hyper_weight_nonlinear + s0_hyper
    sp_hyper_original = sp_hyper_original + s0_hyper
    time_hyper = time_hyper + t0_hyper

    # ==============================
    # Post-Processing #1 : 에러 산정
    # ==============================

    # RMSE 계산 데이터 구간 설정 (계측)
    sm_rmse = settle[step_start_index[-1]:step_end_index[-1]]

    # RMSE 계산 데이터 구간 설정 (쌍곡선)
    sp_hyper_nonlinear_rmse = sp_hyper_nonlinear[0: step_end_index[- 1] - step_start_index[- 1]]
    sp_hyper_weight_nonlinear_rmse = sp_hyper_weight_nonlinear[0: step_end_index[- 1] - step_start_index[- 1]]
    sp_hyper_original_rmse = sp_hyper_original[0: step_end_index[- 1] - step_start_index[- 1]]

    # RMSE 산정  (단계, 비선형 쌍곡선, 기존 쌍곡선)
    rmse_hyper_nonlinear = fun_rmse(sm_rmse, sp_hyper_nonlinear_rmse)
    rmse_hyper_weight_nonlinear = fun_rmse(sm_rmse, sp_hyper_weight_nonlinear_rmse)
    rmse_hyper_original = fun_rmse(sm_rmse, sp_hyper_original_rmse)

    # 최종 침하량 산정
    sf_hyper_original = 1.0 / x_hyper_original[0] + s0_hyper
    sf_hyper_nonlinear = 1.0 / x_hyper_nonlinear[0] + s0_hyper
    sf_hyper_weight_nonliner = 1.0 / x_hyper_weight_nonlinear[0] + s0_hyper


    # RMSE 출력 (단계, 비선형 쌍곡선, 기존 쌍곡선)
    if print_values:
        print("RMSE (비선형 쌍곡선): %0.3f" % rmse_hyper_nonlinear)
        print("RMSE (가중 비선형 쌍곡선): %0.3f" % rmse_hyper_weight_nonlinear)
        print("RMSE (기존 쌍곡선): %0.3f" % rmse_hyper_original)

    # ==========================================
    # Post-Processing #2 : 그래프 작성
    # ==========================================

    # 만약 그래프 도시가 필요할 경우,
    if plot_show:

        # 그래프 크기, 서브 그래프 개수 및 비율 설정
        fig, axes = plt.subplots(3, 1, figsize=(9, 9),
                                 gridspec_kw={'height_ratios': [5, 15, 15]})

        # 성토고 그래프 표시
        axes[0].plot(time, surcharge, color='black', label='surcharge height')

        # 그래프 시간 x 축 최대 최소값 설정
        time_min = 0
        time_max = np.ceil(time.max()/200) * 200
        days = np.arange(time_min, time_max, 1)
        dates = pd.date_range(date[0], periods=time_max).strftime('%Y-%m-%d')

        # 성토고 그래프 설정
        axes[0].set_ylabel("성토고 (m)", fontsize=15)
        axes[0].set_xlim(left=time_min)
        axes[0].set_xlim(right=time_max)
        axes[0].set_ylim(top=np.ceil(np.max(surcharge)))
        axes[0].grid(color="gray", alpha=.5, linestyle='--')
        axes[0].tick_params(direction='in')

        ax1_2 = axes[0].twiny()
        ax1_2.set_xlim(axes[0].get_xlim())
        ax1_2.set_xlabel('일자')
        ax1_2.set_xticks(days[::200])
        ax1_2.set_xticklabels(dates[::200], ha='left')

        # 계측 및 예측 침하량 표시
        axes[1].scatter(time[0:settle.size], settle, s=50,
                        facecolors='white', edgecolors='black', label='계측 침하량')
        axes[1].plot(time_hyper, sp_hyper_original,
                     linestyle='--', color='red', label='기존 쌍곡선')
        axes[1].plot(time_hyper, sp_hyper_nonlinear,
                     linestyle='--', color='orange', label='비선형 쌍곡선')
        axes[1].plot(time_hyper, sp_hyper_weight_nonlinear,
                     linestyle='--', color='blue', label='가중 비선형 쌍곡선')
        axes[1].plot([0, time_max], [settle_goal,settle_goal],
                     linestyle='--', color='black', label='목표 침하량')

        # 계측 및 예측 침하량 표시
        axes[2].scatter(time[0:settle.size], settle, s=50,
                        facecolors='white', edgecolors='black', label='계측 침하량')
        axes[2].plot(time_hyper, sp_hyper_original,
                     linestyle='--', color='red', label='기존 쌍곡선')
        axes[2].plot(time_hyper, sp_hyper_nonlinear,
                     linestyle='--', color='orange', label='비선형 쌍곡선')
        axes[2].plot(time_hyper, sp_hyper_weight_nonlinear,
                     linestyle='--', color='blue', label='가중 비선형 쌍곡선')
        axes[2].plot([0, time_max], [settle_goal, settle_goal],
                     linestyle='--', color='black', label='목표 침하량')

        # 그래프 침하량 y 축 최대 최소값 설정
        settle_min_1 = 0
        settle_min_2 = np.floor((sm_rmse[0])/5) * 5
        settle_max = np.max([sp_hyper_nonlinear.max(),
                             sp_hyper_weight_nonlinear.max(),
                             sp_hyper_original.max()])
        settle_max = np.min([settle_max, sm_rmse.max()*1.2])
        settle_max = np.max([settle_max, settle_goal*1.1])
        settle_max_1 = (np.ceil((1.10*settle_max)/5) * 5)
        settle_max_2 = (np.ceil((1.05*settle_max)/5) * 5)

        # 침하량 그래프 설정
        axes[1].set_ylabel("침하량 (cm)", fontsize=15)
        axes[1].set_ylim(top=settle_min_1)
        axes[1].set_ylim(bottom=settle_max_1)
        axes[1].set_xlim(left=time_min)
        axes[1].set_xlim(right=time_max)
        axes[1].grid(color="gray", alpha=.5, linestyle='--')
        axes[1].tick_params(direction='in')

        # 침하량 그래프 설정
        axes[2].set_xlabel("시간 (일)", fontsize=15)
        axes[2].set_ylabel("침하량 (cm)", fontsize=15)
        axes[2].set_ylim(top=settle_min_2)
        axes[2].set_ylim(bottom=settle_max_2)
        axes[2].set_xlim(left=time_min)
        axes[2].set_xlim(right=time_max)
        axes[2].grid(color="gray", alpha=.5, linestyle='--')
        axes[2].tick_params(direction='in')

        # 범례 표시
        axes[1].legend(loc=1, ncol=2, frameon=True, fontsize=9)
        axes[2].legend(loc=1, ncol=2, frameon=True, fontsize=9)

        # 예측 데이터 사용 범위 음영 처리 - 기존 및 비선형 쌍곡선
        axes[1].axvspan(final_step_start_date, final_step_predict_end_date,
                        alpha=0.1, color='grey', hatch='\\')
        axes[2].axvspan(final_step_start_date, final_step_predict_end_date,
                        alpha=0.1, color='grey', hatch='\\')

        # 예측 데이터 사용 범위 표시 화살표 세로 위치 설정
        arrow1_y_loc = 0.15 * settle_max_1
        arrow2_y_loc = 0.15 * settle_max_1

        # 화살표 크기 설정
        arrow_head_width = 0.03 * (settle_max_1 - settle_min_1)
        arrow_head_length = 0.01 * time_max

        # 예측 데이터 사용 범위 화살표 처리 - 기존 및 비선형 쌍곡선
        axes[1].arrow(final_step_start_date, arrow2_y_loc,
                      final_step_predict_end_date - final_step_start_date, 0,
                      head_width=arrow_head_width, head_length=arrow_head_length,
                      color='black', length_includes_head='True')
        axes[1].arrow(final_step_predict_end_date, arrow2_y_loc,
                      final_step_start_date - final_step_predict_end_date, 0,
                      head_width=arrow_head_width, head_length=arrow_head_length,
                      color='black', length_includes_head='True')

        # Annotation 표시용 공간 설정
        space = time_max * 0.01

        # 예측 데이터 사용 범위 범례 표시 - 기존 및 비선형 쌍곡선
        axes[1].annotate('데이터 사용 영역', xy=(final_step_predict_end_date, arrow1_y_loc),
                     xytext=(final_step_predict_end_date + space, arrow2_y_loc),
                     horizontalalignment='left', verticalalignment='center')

        # 목표 침하량 표시
        axes[1].annotate('목표침하량 = ' +  str(settle_goal) + ' cm', xy=(time_max * 0.01, settle_goal),
                         xytext=(time_max * 0.01, settle_goal),
                         horizontalalignment='left', verticalalignment='bottom')

        # RMSE 출력
        mybox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round', 'alpha': 0.2}
        axes[1].text(time_max * 0.75, 0.35 * settle_max_1,
                     r"$\bf{RMSE (cm)}$"
                     + "\n" + "기존 쌍곡선: %0.3f" % rmse_hyper_original
                     + "\n" + "비선형 쌍곡선: %0.3f" % rmse_hyper_nonlinear
                     + "\n" + "가중 비선형 쌍곡선: %0.3f" % rmse_hyper_weight_nonlinear,
                     color='r', horizontalalignment='left',
                     verticalalignment='top', fontsize='9', bbox=mybox)

        # 예측 데이터 사용 범위 표시 화살표 세로 위치 설정
        arrow1_y_loc = settle_min_2 + 0.15 * (settle_max_2 - settle_min_2)
        arrow2_y_loc = settle_min_2 + 0.15 * (settle_max_2 - settle_min_2)

        # 화살표 크기 설정
        arrow_head_width = 0.03 * (settle_max_2 - settle_min_2)
        arrow_head_length = 0.01 * time_max

        # 예측 데이터 사용 범위 화살표 처리 - 기존 및 비선형 쌍곡선
        axes[2].arrow(final_step_start_date, arrow2_y_loc,
                      final_step_predict_end_date - final_step_start_date, 0,
                      head_width=arrow_head_width, head_length=arrow_head_length,
                      color='black', length_includes_head='True')
        axes[2].arrow(final_step_predict_end_date, arrow2_y_loc,
                      final_step_start_date - final_step_predict_end_date, 0,
                      head_width=arrow_head_width, head_length=arrow_head_length,
                      color='black', length_includes_head='True')

        # 예측 데이터 사용 범위 범례 표시 - 기존 및 비선형 쌍곡선
        axes[2].annotate('데이터 사용 영역', xy=(final_step_predict_end_date, arrow1_y_loc),
                         xytext=(final_step_predict_end_date + space, arrow2_y_loc),
                         horizontalalignment='left', verticalalignment='center')

        # 목표 침하량 표시
        axes[2].annotate('목표침하량 = ' + str(settle_goal) + ' cm', xy=(time_max * 0.01, settle_goal),
                         xytext=(time_max * 0.01, settle_goal),
                         horizontalalignment='left', verticalalignment='bottom')

        # 최종침하량 출력
        mybox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round', 'alpha': 0.2}
        axes[2].text(time_max * 0.75, settle_min_2 + 0.35 * (settle_max_2 - settle_min_2),
                     "최종침하량 (cm)"
                     + "\n" + "기존 쌍곡선: %0.3f" %  sf_hyper_original
                     + "\n" + "비선형 쌍곡선: %0.3f" % sf_hyper_nonlinear
                     + "\n" + "가중 비선형 쌍곡선: %0.3f" % sf_hyper_weight_nonliner,
                     color='r', horizontalalignment='left',
                     verticalalignment='top', fontsize='9', bbox=mybox)

        # 파일 이름만 추출
        filename = os.path.basename(point_name)
        filename = os.path.splitext(filename)[0]

        # 그래프 제목 표시
        plt.title("계측지점: " + filename)

        # 레이아웃 설정
        plt.tight_layout()

        # 그래프 저장 (SVG 및 PNG)
        plt.savefig(output_dir + '/' + filename + '.svg', bbox_inches='tight')

        # 그래프 출력
        #plt.show()

        # 그래프 닫기 (메모리 소모 방지)
        # plt.close()

        # 예측 완료 표시
        print("Settlement prediction is done for " + filename +
              " with " + str(final_step_predict_percent) + "% data usage")

    # 반환
    return [time_hyper, sp_hyper_original,
            time_hyper, sp_hyper_nonlinear,
            time_hyper, sp_hyper_weight_nonlinear,
            rmse_hyper_original,
            rmse_hyper_nonlinear,
            rmse_hyper_weight_nonlinear]


run_settle_prediction_from_file(input_file='data_posco/SP-R1-1.csv',
                                output_dir='output_posco',
                                final_step_predict_percent=100,
                                additional_predict_days=600,
                                plot_show=True,
                                print_values=True)
