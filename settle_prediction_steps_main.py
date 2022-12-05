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
                                    final_step_predict_percent, additional_predict_percent,
                                    plot_show,
                                    print_values,
                                    run_original_hyperbolic='True',
                                    run_nonlinear_hyperbolic='True',
                                    run_weighted_nonlinear_hyperbolic='False',
                                    run_asaoka='True',
                                    run_step_prediction='True',
                                    asaoka_interval=3):

    # 현재 파일 이름 출력
    print("Working on " + input_file)

    # CSV 파일 읽기
    data = pd.read_csv(input_file, encoding='euc-kr')

    # 시간, 침하량, 성토고 배열 생성
    time = data['Time'].to_numpy()
    settle = data['Settlement'].to_numpy()
    surcharge = data['Surcharge'].to_numpy()

    run_settle_prediction(np_time=time,
                          np_surcharge=surcharge,
                          np_settlement=settle,
                          final_step_predict_percent=final_step_predict_percent,
                          additional_predict_percent=additional_predict_percent,
                          plot_show=plot_show,
                          print_values=print_values,
                          run_original_hyperbolic=run_original_hyperbolic,
                          run_nonlinear_hyperbolic=run_nonlinear_hyperbolic,
                          run_weighted_nonlinear_hyperbolic='False',
                          run_asaoka=run_asaoka,
                          run_step_prediction=run_step_prediction,
                          asaoka_interval=asaoka_interval)

def run_settle_prediction(np_time, np_surcharge, np_settlement,
                          final_step_predict_percent, additional_predict_percent,
                          plot_show,
                          print_values,
                          run_original_hyperbolic='True',
                          run_nonlinear_hyperbolic='True',
                          run_weighted_nonlinear_hyperbolic='False',
                          run_asaoka = 'True',
                          run_step_prediction='True',
                          asaoka_interval = 3):

    # ====================
    # 파일 읽기, 데이터 설정
    # ====================

    # 시간, 침하량, 성토고 배열 생성
    time = np_time
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

    # 단계 시작 시점 초기화
    step_start_date = 0

    # 모든 시간-성토고 데이터에서 순차적으로 확인
    for index in range(len(surcharge)):

        # 만일 성토고의 변화가 있을 경우,
        if surcharge[index] != current_surcharge:
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
        if step_span > 30 and step_data_num > 5:
            step_start_index_adjust.append((step_start_index[i]))
            step_end_index_adjust.append((step_end_index[i]))

    #  성토 시작 및 끝 인덱스 리스트 업데이트
    step_start_index = step_start_index_adjust
    step_end_index = step_end_index_adjust

    # 침하 예측을 수행할 단계 설정 (현재 끝에서 2단계 이용)
    step_start_index = step_start_index[-2:]
    step_end_index = step_end_index[-2:]

    # 성토 단계 횟수 파악 및 저장
    num_steps = len(step_start_index)

    # ===========================
    # 최종 단계 데이터 사용 범위 조정
    # ===========================

    # 데이터 사용 퍼센트에 해당하는 기간 계산
    final_step_end_date = time[-1]
    final_step_start_date = time[step_start_index[num_steps - 1]]
    final_step_period = final_step_end_date - final_step_start_date
    final_step_predict_end_date = final_step_start_date + final_step_period * final_step_predict_percent / 100

    # 데이터 사용 끝 시점 인덱스 초기화
    final_step_predict_end_index = -1

    # 데이터 사용 끝 시점 인덱스 검색
    count = 0
    for day in time:
        count = count + 1
        if day > final_step_predict_end_date:
            final_step_predict_end_index = count - 1
            break

    # 마지막 성토 단계, 마지막 계측 시점 인덱스 업데이트
    final_step_monitor_end_index = step_end_index[num_steps - 1]
    step_end_index[num_steps - 1] = final_step_predict_end_index

    # =================
    # 추가 예측 구간 반영
    # =================

    # 추가 예측 일 입력 (현재 전체 계측일 * 계수)
    add_days = (additional_predict_percent / 100) * time[-1]

    # 마지막 성토고 및 마지막 계측일 저장
    final_surcharge = surcharge[final_index - 1]
    final_time = time[final_index - 1]

    # 추가 시간 및 성토고 배열 설정 (100개의 시점 설정)
    time_add = np.linspace(final_time + 1, final_time + add_days, 100)
    surcharge_add = np.ones(100) * final_surcharge

    # 기존 시간 및 성토고 배열에 붙이기
    time = np.append(time, time_add)
    surcharge = np.append(surcharge, surcharge_add)

    # 마지막 인덱스값 재조정
    final_index = time.size









    # ==========================================
    # Settlement Prediction (Step + Hyperbolic)
    # ==========================================

    # 예측 침하량 초기화
    sp_step = np.zeros(time.size)

    # 각 단계별로 진행
    for i in range(0, num_steps):

        # 각 단계별 계측 시점과 계측 침하량 배열 생성
        tm_this_step = time[step_start_index[i]:step_end_index[i]]
        sm_this_step = settle[step_start_index[i]:step_end_index[i]]

        # 이전 단계까지 예측 침하량 중 현재 단계에 해당하는 부분 추출
        sp_this_step = sp_step[step_start_index[i]:step_end_index[i]]

        # 현재 단계 시작 부터 끝까지 시간 데이터 추출
        tm_to_end = time[step_start_index[i]:final_index]

        # 기존 예측 침하량에 대한 보정
        sm_this_step = sm_this_step - sp_this_step

        # 초기 시점 및 침하량 산정
        t0_this_step = tm_this_step[0]
        s0_this_step = sm_this_step[0]

        # 초기 시점에 대한 시간 조정
        tm_this_step = tm_this_step - t0_this_step
        tm_to_end = tm_to_end - t0_this_step

        # 초기 침하량에 대한 침하량 조정
        sm_this_step = sm_this_step - s0_this_step

        # 침하 곡선 계수 초기화
        x0 = np.ones(2)

        # 회귀분석 시행
        res_lsq_hyper_nonlinear \
            = least_squares(fun_hyper_nonlinear, x0,
                            args=(tm_this_step, sm_this_step))

        # 쌍곡선 계수 저장 및 출력
        x_step = res_lsq_hyper_nonlinear.x
        if print_values:
            print(x_step)

        # 현재 단계 예측 침하량 산정 (침하 예측 끝까지)
        sp_to_end_update = generate_data_hyper(x_step, tm_to_end)

        # 예측 침하량 업데이트
        sp_step[step_start_index[i]:final_index] = \
            sp_step[step_start_index[i]:final_index] + sp_to_end_update + s0_this_step




    '''
    # ======================================
    # Settlement Prediction (Step + Asaoka)
    # ======================================

    # TODO: Modify this

    # 예측 침하량 초기화
    sp_step_asaoka = np.zeros(time.size)

    # 각 단계별로 진행
    for i in range(0, num_steps):

        # 각 단계별 계측 시점과 계측 침하량 배열 생성
        tm_this_step = time[step_start_index[i]:step_end_index[i]]
        sm_this_step = settle[step_start_index[i]:step_end_index[i]]

        # 이전 단계 까지 예측 침하량 중 현재 단계에 해당 하는 부분 추출
        sp_this_step = sp_step[step_start_index[i]:step_end_index[i]]

        # 현재 단계 시작 부터 끝까지 시간 데이터 추출
        tm_to_end = time[step_start_index[i]:final_index]

        # 기존 예측 침하량에 대한 보정
        sm_this_step = sm_this_step - sp_this_step

        # 초기 시점 및 침하량 산정
        t0_this_step = tm_this_step[0]
        s0_this_step = sm_this_step[0]

        # 초기 시점에 대한 시간 조정
        tm_this_step = tm_this_step - t0_this_step
        tm_to_end = tm_to_end - t0_this_step

        # 초기 침하량에 대한 침하량 조정
        sm_this_step = sm_this_step - s0_this_step







        # 등간격 데이터 생성을 위한 Interpolation 함수 설정
        inter_fn = interp1d(tm_this_step, sm_this_step, kind='cubic')

        # 데이터 구축 간격 및 그에 해당하는 데이터 포인트 개수 설정
        num_data = int(tm_this_step[-1] / asaoka_interval)

        # 등간격 시간 및 침하량 데이터 설정
        tm_this_step_inter = np.linspace(0, tm_this_step[-1], num=num_data, endpoint=True)
        sm_this_step_inter = inter_fn(tm_this_step_inter)

        # 이전 이후 등간격 침하량 배열 구축
        sm_this_step_before = sm_this_step_inter[0:-2]
        sm_this_step_after = sm_this_step_inter[1:-1]




        # Least square 변수 초기화
        x0 = np.ones(2)

        # Least square 분석을 통한 침하 곡선 계수 결정
        res_lsq_asaoka = least_squares(fun_asaoka, x0, args=(sm_this_step_before, sm_this_step_after))

        # 기존 쌍곡선 법 계수 저장 및 출력
        x_step_asaoka = res_lsq_asaoka.x
        if print_values:
            print(x_step_asaoka)

        # 현재 단계 예측 침하량 산정 (침하 예측 끝까지)
        sp_to_end_update = generate_data_asaoka(x_step_asaoka, tm_to_end, asaoka_interval)


        # 예측 침하량 업데이트
        sp_step_asaoka[step_start_index[i]:final_index] = \
            sp_step_asaoka[step_start_index[i]:final_index] + sp_to_end_update + s0_this_step

    '''

    # =========================================================
    # Settlement prediction (nonliner and original hyperbolic)
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







    # ===============================
    # Settlement prediction (Asaoka)
    # ===============================

    # 성토 마지막 데이터 추출
    tm_asaoka = time[step_start_index[num_steps - 1]:step_end_index[num_steps - 1]]
    sm_asaoka = settle[step_start_index[num_steps - 1]:step_end_index[num_steps - 1]]

    # 현재 단계 시작 부터 끝까지 시간 데이터 추출
    time_asaoka = time[step_start_index[num_steps - 1]:final_index]

    # 초기 시점 및 침하량 산정
    t0_asaoka = tm_asaoka[0]
    s0_asaoka = sm_asaoka[0]

    # 초기 시점에 대한 시간 조정
    tm_asaoka = tm_asaoka - t0_asaoka
    time_asaoka = time_asaoka - t0_asaoka

    # 초기 침하량에 대한 침하량 조정
    sm_asaoka = sm_asaoka - s0_asaoka

    # 등간격 데이터 생성을 위한 Interpolation 함수 설정
    inter_fn = interp1d(tm_asaoka, sm_asaoka, kind='cubic')

    # 데이터 구축 간격 및 그에 해당하는 데이터 포인트 개수 설정
    num_data = int(tm_asaoka[-1] / asaoka_interval)

    # 등간격 시간 및 침하량 데이터 설정
    tm_asaoka_inter = np.linspace(0, tm_asaoka[-1], num=num_data, endpoint=True)
    sm_asaoka_inter = inter_fn(tm_asaoka_inter)

    # 이전 이후 등간격 침하량 배열 구축
    sm_asaoka_before = sm_asaoka_inter[0:-2]
    sm_asaoka_after = sm_asaoka_inter[1:-1]

    # Least square 변수 초기화
    x0 = np.ones(2)

    # Least square 분석을 통한 침하 곡선 계수 결정
    res_lsq_asaoka = least_squares(fun_asaoka, x0, args=(sm_asaoka_before, sm_asaoka_after))

    # 기존 쌍곡선 법 계수 저장 및 출력
    x_asaoka = res_lsq_asaoka.x
    if print_values:
        print(x_asaoka)

    # 현재 단계 예측 침하량 산정 (침하 예측 끝까지)
    sp_asaoka = generate_data_asaoka(x_asaoka, time_asaoka, asaoka_interval)

    # 예측 침하량 산정
    sp_asaoka = sp_asaoka + s0_asaoka
    time_asaoka = time_asaoka + t0_asaoka















    # ==========
    # 에러 산정
    # ==========

    # RMSE 계산 데이터 구간 설정 (계측)
    sm_rmse = settle[final_step_predict_end_index:final_step_monitor_end_index]

    # RMSE 계산 데이터 구간 설정 (단계)
    sp_step_rmse = sp_step[final_step_predict_end_index:final_step_monitor_end_index]

    # RMSE 계산 데이터 구간 설정 (쌍곡선)
    sp_hyper_nonlinear_rmse = sp_hyper_nonlinear[final_step_predict_end_index - step_start_index[num_steps - 1]:
                                                 final_step_predict_end_index - step_start_index[num_steps - 1] +
                                                 final_step_monitor_end_index - final_step_predict_end_index]
    sp_hyper_weight_nonlinear_rmse \
                        = sp_hyper_weight_nonlinear[final_step_predict_end_index - step_start_index[num_steps - 1]:
                                                    final_step_predict_end_index - step_start_index[num_steps - 1] +
                                                    final_step_monitor_end_index - final_step_predict_end_index]
    sp_hyper_original_rmse = sp_hyper_original[final_step_predict_end_index - step_start_index[num_steps - 1]:
                                               final_step_predict_end_index - step_start_index[num_steps - 1] +
                                               final_step_monitor_end_index - final_step_predict_end_index]

    # RMSE 계산 데이터 구간 설정 (아사오카)
    sp_asaoka_rmse = sp_asaoka[final_step_predict_end_index - step_start_index[num_steps - 1]:
                               final_step_predict_end_index - step_start_index[num_steps - 1] +
                               final_step_monitor_end_index - final_step_predict_end_index]

    # RMSE 산정  (단계, 비선형 쌍곡선, 기존 쌍곡선)
    rmse_step = fun_rmse(sm_rmse, sp_step_rmse)
    rmse_hyper_nonlinear = fun_rmse(sm_rmse, sp_hyper_nonlinear_rmse)
    rmse_hyper_weight_nonlinear = fun_rmse(sm_rmse, sp_hyper_weight_nonlinear_rmse)
    rmse_hyper_original = fun_rmse(sm_rmse, sp_hyper_original_rmse)
    rmse_asaoka = fun_rmse(sm_rmse, sp_asaoka_rmse)

    # RMSE 출력 (단계, 비선형 쌍곡선, 기존 쌍곡선)
    if print_values:
        print("RMSE (Nonlinear Hyper + Step): %0.3f" % rmse_step)
        print("RMSE (Nonlinear Hyperbolic): %0.3f" % rmse_hyper_nonlinear)
        print("RMSE (Weighted Nonlinear Hyperbolic): %0.3f" % rmse_hyper_weight_nonlinear)
        print("RMSE (Original Hyperbolic): %0.3f" % rmse_hyper_original)
        print("RMSE (Asaoka): %0.3f" % rmse_asaoka)

    # (최종 계측 침하량 - 예측 침하량) 계산
    final_error_step = np.abs(settle[-1] - sp_step_rmse[-1])
    final_error_hyper_nonlinear = np.abs(settle[-1] - sp_hyper_nonlinear_rmse[-1])
    final_error_hyper_weight_nonlinear = np.abs(settle[-1] - sp_hyper_weight_nonlinear_rmse[-1])
    final_error_hyper_original = np.abs(settle[-1] - sp_hyper_original_rmse[-1])
    final_error_asaoka = np.abs(settle[-1] - sp_asaoka_rmse[-1])

    # (최종 계측 침하량 - 예측 침하량) 출력 (단계, 비선형 쌍곡선, 기존 쌍곡선)
    if print_values:
        print("Error in Final Settlement (Nonlinear Hyper + Step): %0.3f" % final_error_step)
        print("Error in Final Settlement (Nonlinear Hyperbolic): %0.3f" % final_error_hyper_nonlinear)
        print("Error in Final Settlement (Weighted Nonlinear Hyperbolic): %0.3f" % final_error_hyper_weight_nonlinear)
        print("Error in Final Settlement (Original Hyperbolic): %0.3f" % final_error_hyper_original)
        print("Error in Final Settlement (Asaoka): %0.3f" % final_error_asaoka)






    # =====================
    # Post-Processing
    # =====================

    # 그래프 크기, 서브 그래프 개수 및 비율 설정
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [1, 3]})

    # 성토고 그래프 표시
    axes[0].plot(time, surcharge, color='black', label='surcharge height')

    # 성토고 그래프 설정
    axes[0].set_ylabel("Surcharge height (m)", fontsize=15)
    axes[0].set_xlim(left=0)
    axes[0].grid(color="gray", alpha=.5, linestyle='--')
    axes[0].tick_params(direction='in')

    # 계측 및 예측 침하량 표시
    axes[1].scatter(time[0:settle.size], -settle, s=50,
                    facecolors='white', edgecolors='black', label='measured data')
    axes[1].plot(time_hyper, -sp_hyper_original,
                 linestyle='--', color='red', label='Original Hyperbolic')
    axes[1].plot(time_hyper, -sp_hyper_nonlinear,
                 linestyle='--', color='green', label='Nonlinear Hyperbolic')
    axes[1].plot(time_hyper, -sp_hyper_weight_nonlinear,
                 linestyle='--', color='blue', label='Nonlinear Hyperbolic (Weighted)')
    axes[1].plot(time_asaoka, -sp_asaoka,
                 linestyle='--', color='orange', label='Asaoka')
    axes[1].plot(time[step_start_index[0]:], -sp_step[step_start_index[0]:],
                 linestyle='--', color='navy', label='Nonlinear + Step Loading')

    # 침하량 그래프 설정
    axes[1].set_xlabel("Time (day)", fontsize=15)
    axes[1].set_ylabel("Settlement (cm)", fontsize=15)
    axes[1].set_ylim(top=0)
    axes[1].set_ylim(bottom=-1.5 * settle.max())
    axes[1].set_xlim(left=0)
    axes[1].grid(color="gray", alpha=.5, linestyle='--')
    axes[1].tick_params(direction='in')

    # 범례 표시
    axes[1].legend(loc=1, ncol=3, frameon=True, fontsize=10)

    # 예측 데이터 사용 범위 음영 처리 - 단계성토
    plt.axvspan(time[step_start_index[0]], final_step_predict_end_date,
                alpha=0.1, color='grey', hatch='//')

    # 예측 데이터 사용 범위 음영 처리 - 기존 및 비선형 쌍곡선
    plt.axvspan(final_step_start_date, final_step_predict_end_date,
                alpha=0.1, color='grey', hatch='\\')

    # 예측 데이터 사용 범위 표시 화살표 세로 위치 설정
    arrow1_y_loc = 1.3 * min(-settle)
    arrow2_y_loc = 1.4 * min(-settle)

    # 화살표 크기 설정
    arrow_head_width = 0.03 * max(settle)
    arrow_head_length = 0.01 * max(time)

    # 예측 데이터 사용 범위 화살표 처리 - 단계성토
    axes[1].arrow(time[step_start_index[0]], arrow1_y_loc,
                  final_step_predict_end_date - time[step_start_index[0]], 0,
                  head_width=arrow_head_width, head_length=arrow_head_length,
                  color='black', length_includes_head='True')
    axes[1].arrow(final_step_predict_end_date, arrow1_y_loc,
                  time[step_start_index[0]] - final_step_predict_end_date, 0,
                  head_width=arrow_head_width, head_length=arrow_head_length,
                  color='black', length_includes_head='True')

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
    space = max(time) * 0.01

    # 예측 데이터 사용 범위 범례 표시 - 단계성토
    plt.annotate('Data Range Used (Nonlinear + Step Loading)', xy=(final_step_predict_end_date, arrow1_y_loc),
                 xytext=(final_step_predict_end_date + space, arrow1_y_loc),
                 horizontalalignment='left', verticalalignment='center')

    # 예측 데이터 사용 범위 범례 표시 - 기존 및 비선형 쌍곡선
    plt.annotate('Data Range Used (Hyperbolic and Asaoka)', xy=(final_step_predict_end_date, arrow1_y_loc),
                 xytext=(final_step_predict_end_date + space, arrow2_y_loc),
                 horizontalalignment='left', verticalalignment='center')

    # RMSE 산정 범위 표시 화살표 세로 위치 설정
    arrow3_y_loc = 0.55 * min(-settle)

    # RMSE 산정 범위 화살표 표시
    axes[1].arrow(final_step_predict_end_date, arrow3_y_loc,
                  final_step_end_date - final_step_predict_end_date, 0,
                  head_width=arrow_head_width, head_length=arrow_head_length,
                  color='black', length_includes_head='True')
    axes[1].arrow(final_step_end_date, arrow3_y_loc,
                  final_step_predict_end_date - final_step_end_date, 0,
                  head_width=arrow_head_width, head_length=arrow_head_length,
                  color='black', length_includes_head='True')

    # RMSE 산정 범위 세로선 설정
    axes[1].axvline(x=final_step_end_date, color='silver', linestyle=':')

    # RMSE 산정 범위 범례 표시
    plt.annotate('RMSE Estimation Section', xy=(final_step_end_date, arrow3_y_loc),
                 xytext=(final_step_end_date + space, arrow3_y_loc),
                 horizontalalignment='left', verticalalignment='center')

    # RMSE 출력
    mybox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round', 'alpha': 0.2}
    plt.text(max(time) * 1.04, 0.20 * min(-settle),
             r"$\bf{Root\ Mean\ Squared\ Error}$"
             + "\n" + "Original Hyperbolic: %0.3f" % rmse_hyper_original
             + "\n" + "Nonlinear Hyperbolic: %0.3f" % rmse_hyper_nonlinear
             + "\n" + "Nonlinear Hyperbolic (Weighted): %0.3f" % rmse_hyper_weight_nonlinear
             + "\n" + "Asaoka: %0.3f" % rmse_asaoka
             + "\n" + "Nonlinear + Step Loading: %0.3f" % rmse_step,
             color='r', horizontalalignment='right',
             verticalalignment='top', fontsize='10', bbox=mybox)

    # (최종 계측 침하량 - 예측값) 출력
    plt.text(max(time) * 1.04, 0.55 * min(-settle),
             r"$\bf{Error\ in\ Final\ Settlement}$"
             + "\n" + "Original Hyperbolic: %0.3f" % final_error_hyper_original
             + "\n" + "Nonlinear Hyperbolic: %0.3f" % final_error_hyper_nonlinear
             + "\n" + "Nonlinear Hyperbolic (Weighted): %0.3f" % final_error_hyper_weight_nonlinear
             + "\n" + "Asaoka: %0.3f" % final_error_asaoka
             + "\n" + "Nonlinear + Step Loading: %0.3f" % final_error_step,
             color='r', horizontalalignment='right',
             verticalalignment='top', fontsize='10', bbox=mybox)

    # 파일 이름만 추출
    filename = os.path.basename(input_file)

    # 그래프 제목 표시
    plt.title(filename + ": up to %i%% data used in the final step" % final_step_predict_percent)

    # 그래프 저장 (SVG 및 PNG)
    # plt.savefig(output_dir + '/' + filename +' %i percent (SVG).svg' %final_step_predict_percent, bbox_inches='tight')
    plt.savefig(output_dir + '/' + filename + ' %i percent (PNG).png' % final_step_predict_percent, bbox_inches='tight')

    # 그래프 출력
    if plot_show:
        plt.show()

    # 그래프 닫기 (메모리 소모 방지)
    plt.close()

    # 예측 완료 표시
    print("Settlement prediction is done for " + filename +
          " with " + str(final_step_predict_percent) + "% data usage")

    # 단계 성토 고려 여부 표시
    is_multi_step = True
    if len(step_start_index) == 1:
        is_multi_step = False

    # 반환
    return [rmse_hyper_original, rmse_hyper_nonlinear, rmse_hyper_weight_nonlinear, rmse_asaoka, rmse_step,
            final_error_hyper_original, final_error_hyper_nonlinear, final_error_hyper_weight_nonlinear,
            final_error_asaoka, final_error_step, is_multi_step]

'''
run_settle_prediction(input_file='data/2-5_No.39.csv',
                      output_dir='output',
                      final_step_predict_percent=50,
                      additional_predict_percent=100,
                      plot_show=True,
                      print_values=True,
                      run_original_hyperbolic=True,
                      run_nonlinear_hyperbolic=True,
                      run_weighted_nonlinear_hyperbolic=True,
                      run_asaoka=True,
                      run_step_prediction=True,
                      asaoka_interval=3,
                      settle_unit='cm')
'''