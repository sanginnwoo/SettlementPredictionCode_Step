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
from scipy.optimize import least_squares
from scipy.interpolate import interp1d


# =================
# Function 섹션
# =================

# 주어진 계수를 이용하여 쌍곡선 시간-침하 곡선 반환
def generate_data_hyper(px, pt):
    return pt / (px[0] * pt + px[1])


# 회귀식과 측정치와의 잔차 반환 (비선형 쌍곡선)
def fun_hyper_nonlinear(px, pt, py):
    return pt / (px[0] * pt + px[1]) - py


# 회귀식과 측정치와의 잔차 반환 (가중 비선형 쌍곡선)
def fun_hyper_weight_nonlinear(px, pt, py, pw):
    return (pt / (px[0] * pt + px[1]) - py) * pw


# 회귀식과 측정치와의 잔차 반환 (기존 쌍곡선)
def fun_hyper_original(px, pt, py):
    return px[0] * pt + px[1] - pt / py


# RMSE 산정
def fun_rmse(py1, py2):
    mse = np.square(np.subtract(py1, py2)).mean()
    return np.sqrt(mse)


def run_settle_prediction_from_file(input_file,
                                    output_dir,
                                    data_usage,
                                    is_data_usage_percent,
                                    rmse_usage,
                                    is_rmse_usage_percent,
                                    additional_predict_percent,
                                    plot_show,
                                    print_values):
    # 현재 파일 이름 출력
    print("Working on " + input_file)

    # CSV 파일 읽기
    data = pd.read_csv(input_file, encoding='euc-kr')

    # 시간 배열 생성
    if 'Time' in data.columns:
        time = data['Time'].to_numpy()
    elif 'time' in data.columns:
        time = data['time'].to_numpy()

    # 침하량 배열 생성
    if 'Settle' in data.columns:
        settle = data['Settle'].to_numpy()
    elif 'settle' in data.columns:
        settle = data['settle'].to_numpy()
    elif 'Settlement' in data.columns:
        settle = data['Settlement'].to_numpy()
    elif 'settlement' in data.columns:
        settle = data['settlement'].to_numpy()

    return run_settle_prediction(point_name=input_file,
                                 output_dir=output_dir,
                                 np_time=time,
                                 np_settlement=settle,
                                 data_usage=data_usage,
                                 is_data_usage_percent=is_data_usage_percent,
                                 rmse_usage=rmse_usage,
                                 is_rmse_usage_percent=is_rmse_usage_percent,
                                 additional_predict_percent=additional_predict_percent,
                                 plot_show=plot_show,
                                 print_values=print_values)


def run_settle_prediction(point_name,
                          output_dir,
                          np_time,
                          np_settlement,
                          data_usage,
                          is_data_usage_percent,
                          rmse_usage,
                          is_rmse_usage_percent,
                          additional_predict_percent,
                          plot_show,
                          print_values):
    # ====================
    # 파일 읽기, 데이터 설정
    # ====================

    # 시간, 침하량, 성토고 배열 생성
    time = np_time
    settle = np_settlement

    # 마지막 계측 데이터 index + 1 파악
    final_index = time.size

    # ===========================
    # 데이터 사용 범위 설정
    # ===========================

    # 끝일, 시작일, 전체 기간 산정
    final_date = time[-1]
    start_date = time[0]
    total_period = final_date - start_date

    # 예측에 사용될 데이터 끝일 산정
    predict_end_date = start_date + data_usage
    if is_data_usage_percent:
        predict_end_date = start_date + total_period * data_usage / 100

    # 데이터 사용 끝 시점 인덱스 검색
    monitor_end_index = time.size
    predict_end_index = 0
    for day in time:
        predict_end_index = predict_end_index + 1
        if day > predict_end_date:
            break


    # =================
    # 추가 예측 구간 반영
    # =================

    # 추가 예측 일 입력 (현재 전체 계측일 * 계수)
    add_days = (additional_predict_percent / 100) * time[-1]

    # 추가 시간 및 성토고 배열 설정 (100개의 시점 설정)
    time_add = np.linspace(final_date + 1, final_date + add_days, 100)

    # 기존 시간 및 성토고 배열에 붙이기
    time = np.append(time, time_add)

    # =========================================================
    # Settlement prediction (nonliner, weighted nonlinear and original hyperbolic)
    # =========================================================

    # 성토 마지막 데이터 추출
    tm_hyper = time[:predict_end_index]
    sm_hyper = settle[:predict_end_index]

    # 현재 단계 시작 부터 끝까지 시간 데이터 추출
    time_hyper = time[0:]

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
    res_lsq_hyper_original = least_squares(fun_hyper_original, x0,
                                           args=(tm_hyper[1:], sm_hyper[1:]))
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

    #need to find rmse_start_index

    # 데이터 사용 끝 시점 인덱스 검색

    rmse_date = rmse_usage;
    if is_rmse_usage_percent:
        rmse_date = (rmse_usage / 100.0) * total_period

    rmse_start_index = 0
    for day in time:
        rmse_start_index = rmse_start_index + 1
        if day > monitor_end_index - rmse_date:
            break



    # RMSE 계산 데이터 구간 설정 (계측)
    sm_rmse = settle[rmse_start_index:monitor_end_index]

    # RMSE 계산 데이터 구간 설정 (비선형, 가중 비선형, 기존 쌍곡선)
    sp_hyper_nonlinear_rmse = sp_hyper_nonlinear[rmse_start_index:monitor_end_index]
    sp_hyper_weight_nonlinear_rmse = sp_hyper_weight_nonlinear[rmse_start_index:monitor_end_index]
    sp_hyper_original_rmse = sp_hyper_original[rmse_start_index:monitor_end_index]

    # RMSE 산정  (단계, 비선형 쌍곡선, 기존 쌍곡선)
    rmse_hyper_nonlinear = fun_rmse(sm_rmse, sp_hyper_nonlinear_rmse)
    rmse_hyper_weight_nonlinear = fun_rmse(sm_rmse, sp_hyper_weight_nonlinear_rmse)
    rmse_hyper_original = fun_rmse(sm_rmse, sp_hyper_original_rmse)

    # RMSE 출력 (단계, 비선형 쌍곡선, 기존 쌍곡선)
    if print_values:
        print("RMSE (Nonlinear Hyperbolic): %0.3f" % rmse_hyper_nonlinear)
        print("RMSE (Weighted Nonlinear Hyperbolic): %0.3f" % rmse_hyper_weight_nonlinear)
        print("RMSE (Original Hyperbolic): %0.3f" % rmse_hyper_original)

    # ==========================================
    # Post-Processing #2 : 그래프 작성
    # ==========================================

    # 만약 그래프 도시가 필요할 경우,
    if plot_show:

        # 그래프 크기, 서브 그래프 개수 및 비율 설정
        fig, axes = plt.subplots(figsize=(9, 6))

        # 계측 및 예측 침하량 표시
        axes.scatter(time[0:settle.size], -settle, s=50,
                        facecolors='white', edgecolors='black', label='measured data')
        axes.plot(time_hyper, -sp_hyper_original,
                     linestyle='--', color='red', label='Original Hyperbolic')
        axes.plot(time_hyper, -sp_hyper_nonlinear,
                     linestyle='--', color='green', label='Nonlinear Hyperbolic')
        axes.plot(time_hyper, -sp_hyper_weight_nonlinear,
                     linestyle='--', color='blue', label='Nonlinear Hyperbolic (Weighted)')

        # 침하량 그래프 설정
        axes.set_xlabel("Time (day)", fontsize=15)
        axes.set_ylabel("Settlement (cm)", fontsize=15)
        axes.set_ylim(top=0)
        axes.set_ylim(bottom=-1.5 * settle.max())
        axes.set_xlim(left=0)
        axes.grid(color="gray", alpha=.5, linestyle='--')
        axes.tick_params(direction='in')

        # 범례 표시
        axes.legend(loc=1, ncol=3, frameon=True, fontsize=10)

        # 예측 데이터 사용 범위 음영 처리 - 기존 및 비선형 쌍곡선
        plt.axvspan(start_date, predict_end_date,
                    alpha=0.1, color='grey', hatch='\\')

        # 예측 데이터 사용 범위 표시 화살표 세로 위치 설정
        arrow1_y_loc = 1.3 * min(-settle)
        arrow2_y_loc = 1.4 * min(-settle)

        # 화살표 크기 설정
        arrow_head_width = 0.03 * max(settle)
        arrow_head_length = 0.01 * max(time)

        # 예측 데이터 사용 범위 화살표 처리 - 기존 및 비선형 쌍곡선
        axes.arrow(start_date, arrow2_y_loc,
                      predict_end_date - start_date, 0,
                      head_width=arrow_head_width, head_length=arrow_head_length,
                      color='black', length_includes_head='True')
        axes.arrow(predict_end_date, arrow2_y_loc,
                      start_date - predict_end_date, 0,
                      head_width=arrow_head_width, head_length=arrow_head_length,
                      color='black', length_includes_head='True')

        # Annotation 표시용 공간 설정
        space = max(time) * 0.01

        # 예측 데이터 사용 범위 범례 표시 - 기존 및 비선형 쌍곡선
        plt.annotate('Data Range Used', xy=(predict_end_date, arrow1_y_loc),
                     xytext=(predict_end_date + space, arrow2_y_loc),
                     horizontalalignment='left', verticalalignment='center')

        # RMSE 산정 범위 표시 화살표 세로 위치 설정
        arrow3_y_loc = 0.55 * min(-settle)

        # RMSE 산정 범위 화살표 표시
        axes.arrow(final_date - rmse_date, arrow3_y_loc,
                      rmse_date, 0,
                      head_width=arrow_head_width, head_length=arrow_head_length,
                      color='black', length_includes_head='True')
        axes.arrow(final_date, arrow3_y_loc,
                      -rmse_date, 0,
                      head_width=arrow_head_width, head_length=arrow_head_length,
                      color='black', length_includes_head='True')

        # RMSE 산정 범위 세로선 설정
        axes.axvline(x=final_date - rmse_date, color='silver', linestyle=':')
        axes.axvline(x=final_date, color='silver', linestyle=':')

        # RMSE 산정 범위 범례 표시
        plt.annotate('RMSE Estimation Section', xy=(final_date, arrow3_y_loc),
                     xytext=(final_date + space, arrow3_y_loc),
                     horizontalalignment='left', verticalalignment='center')

        # RMSE 출력
        mybox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round', 'alpha': 0.2}
        plt.text(max(time) * 1.04, 0.25 * min(-settle),
                 r"$\bf{Root\ Mean\ Squared\ Error}$"
                 + "\n" + "Original Hyperbolic: %0.3f" % rmse_hyper_original
                 + "\n" + "Nonlinear Hyperbolic: %0.3f" % rmse_hyper_nonlinear
                 + "\n" + "Nonlinear Hyperbolic (Weighted): %0.3f" % rmse_hyper_weight_nonlinear,
                 color='r', horizontalalignment='right',
                 verticalalignment='top', fontsize='10', bbox=mybox)


        # 파일 이름만 추출
        filename = os.path.basename(point_name)

        # 그래프 제목 표시
        if is_data_usage_percent:
            plt.title(filename + ": initial %i%% data used" % data_usage)
        else:
            plt.title(filename + ": initial %i days data used" % data_usage)

        # 그래프 출력
        if plot_show:
            if is_data_usage_percent:
                plt.savefig(output_dir + '/' + filename +' %i percent (PNG).png' % data_usage, bbox_inches='tight')
                #plt.savefig(output_dir + '/' + filename +' %i percent (PNG).svg' % data_usage, bbox_inches='tight')
            else:
                plt.savefig(output_dir + '/' + filename + ' %i days (PNG).png' % data_usage, bbox_inches='tight')
                #plt.savefig(output_dir + '/' + filename + ' %i days (PNG).svg' % data_usage, bbox_inches='tight')

        # 그래프 닫기 (메모리 소모 방지)
        plt.close()

        # 예측 완료 표시
        print("Settlement prediction is done for " + filename +
              " with " + str(data_usage) + "% data usage")

        # 반환
        axes.plot(time_hyper, -sp_hyper_original,
                     linestyle='--', color='red', label='Original Hyperbolic')
        axes.plot(time_hyper, -sp_hyper_nonlinear,
                     linestyle='--', color='green', label='Nonlinear Hyperbolic')
        axes.plot(time_hyper, -sp_hyper_weight_nonlinear,
                     linestyle='--', color='blue', label='Nonlinear Hyperbolic (Weighted)')

    return [time_hyper, sp_hyper_original,
            time_hyper, sp_hyper_nonlinear,
            time_hyper, sp_hyper_weight_nonlinear,
            rmse_hyper_original,
            rmse_hyper_nonlinear,
            rmse_hyper_weight_nonlinear]
