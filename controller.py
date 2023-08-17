"""
Title: Controller
Developer:
Sang Inn Woo, Ph.D. @ Incheon National University
Starting Date: 2022-11-10
"""
import psycopg2 as pg2
import sys
import numpy as np
import settle_prediction_main
import matplotlib.pyplot as plt


'''
apptb_surset01
cons_code: names of monitoring points

apptb_surset02
cons_code: names of monitoring points
amount_cum_sub: accumulated settlement
fill_height: height of surcharge fill
nod: number of date
'''

def settlement_prediction(business_code, cons_code):

    # connect the database
    connection = pg2.connect("host=localhost dbname=postgres user=postgres password=lab36981 port=5432") # local
    #connection = pg2.connect("host=192.168.0.13 dbname=sgis user=sgis password=sgis port=5432") # ICTWay internal

    # set cursor
    cursor = connection.cursor()

    # select monitoring data for the monitoring point
    postgres_select_query = """SELECT * FROM apptb_surset02 WHERE business_code='""" + business_code \
                            + """' and cons_code='""" + cons_code + """' ORDER BY nod ASC"""
    cursor.execute(postgres_select_query)
    monitoring_record = cursor.fetchall()

    # initialize time, surcharge, and settlement lists
    time = []
    surcharge = []
    settlement = []

    # fill lists
    for row in monitoring_record:
        settlement.append(float(row[6]))
        surcharge.append(float(row[8]))
        time.append(float(row[2]))

    # convert lists to np arrays
    settlement = np.array(settlement)
    surcharge = np.array(surcharge)
    time = np.array(time)

    # run the settlement prediction and get results
    results = settle_prediction_steps_main.run_settle_prediction(point_name=cons_code, np_time=time,
                                                                 np_surcharge=surcharge, np_settlement=settlement,
                                                                 data_usage=90,
                                                                 additional_predict_percent=300, plot_show=False,
                                                                 print_values=False, run_original_hyperbolic=True,
                                                                 run_nonlinear_hyperbolic=True,
                                                                 run_weighted_nonlinear_hyperbolic=True,
                                                                 run_asaoka=True, run_step_prediction=True,
                                                                 asaoka_interval=3)


    # prediction method code
    # 1: original hyperbolic method (쌍곡선법)
    # 2: nonlinear hyperbolic method (비선형 쌍곡선법)
    # 3: weighted nonlinear hyperbolic method (가중 비선형 쌍곡선법)
    # 4: Asaoka method (아사오카법)
    # 5: Step loading (단계성토 고려법)


    '''
    time_hyper, sp_hyper_original,
    time_hyper, sp_hyper_nonlinear,
    time_hyper, sp_hyper_weight_nonlinear,
    time_asaoka, sp_asaoka,
            time[step_start_index[0]:], -sp_step[step_start_index[0]:],
    '''

    for i in range(5):

        # if there are prediction data for the given data point, delete it first
        postgres_delete_query = """DELETE FROM apptb_pred02_no""" + str(i + 1) \
                                + """ WHERE business_code='""" + business_code \
                                + """' and cons_code='""" + cons_code + """'"""
        cursor.execute(postgres_delete_query)
        connection.commit()

        # get time and settlement arrays
        time = results[2 * i]
        predicted_settlement = results[2 * i + 1]

        # for each prediction time
        for j in range(len(time)):

            # construct insert query
            postgres_insert_query \
                = """INSERT INTO apptb_pred02_no""" + str(i + 1) + """ """ \
                  + """(business_code, cons_code, prediction_progress_days, predicted_settlement, prediction_method) """ \
                  + """VALUES (%s, %s, %s, %s, %s)"""

            # set data to insert
            record_to_insert = (business_code, cons_code, time[j], predicted_settlement[j], i + 1)

            # execute the insert query
            cursor.execute(postgres_insert_query, record_to_insert)

        # commit changes
        connection.commit()


def read_database_and_plot(business_code, cons_code):

    # connect the database
    connection = pg2.connect("host=localhost dbname=postgres user=postgres password=lab36981 port=5432")
    # connection = pg2.connect("host=192.168.0.13 dbname=sgis user=sgis password=sgis port=5432") # ICTWay internal

    # set cursor
    cursor = connection.cursor()

    # select monitoring data for the monitoring point
    postgres_select_query = """SELECT * FROM apptb_surset02 WHERE business_code='""" + business_code \
                            + """' and cons_code='""" + cons_code + """' ORDER BY nod ASC"""
    cursor.execute(postgres_select_query)
    monitoring_record = cursor.fetchall()

    # initialize time, surcharge, and settlement lists
    time_monitored = []
    surcharge_monitored = []
    settlement_monitored = []

    # fill lists
    for row in monitoring_record:
        time_monitored.append(float(row[2]))
        settlement_monitored.append(float(row[6]))
        surcharge_monitored.append(float(row[8]))

    # convert lists to np arrays
    settlement_monitored = np.array(settlement_monitored)
    surcharge_monitored = np.array(surcharge_monitored)
    time_monitored = np.array(time_monitored)

    # prediction method code
    # 0: original hyperbolic method
    # 1: nonlinear hyperbolic method
    # 2: weighted nonlinear hyperbolic method
    # 3: Asaoka method
    # 4: Step loading
    # 5: temp

    # temporarily set the prediction method as 0
    postgres_select_query = """SELECT prediction_progress_days, predicted_settlement """ \
                            + """FROM apptb_pred02_no""" + str(5) \
                            + """ WHERE business_code='""" + business_code \
                            + """' and cons_code='""" + cons_code \
                            + """' ORDER BY prediction_progress_days ASC"""

    # select predicted data for the monitoring point
    cursor.execute(postgres_select_query)
    prediction_record = cursor.fetchall()

    # initialize time, surcharge, and settlement lists
    time_predicted = []
    settlement_predicted = []

    # fill lists
    for row in prediction_record:
        time_predicted.append(float(row[0]))
        settlement_predicted.append(float(row[1]))

    # convert lists to np arrays
    settlement_predicted = np.array(settlement_predicted)
    time_predicted = np.array(time_predicted)

    # 그래프 크기, 서브 그래프 개수 및 비율 설정
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

    # 성토고 그래프 표시
    axes[0].plot(time_monitored, surcharge_monitored, color='black', label='surcharge height')

    # 성토고 그래프 설정
    axes[0].set_ylabel("Surcharge height (m)", fontsize=10)
    axes[0].set_xlim(left=0)
    axes[0].set_xlim(right=np.max(time_predicted))
    axes[0].grid(color="gray", alpha=.5, linestyle='--')
    axes[0].tick_params(direction='in')

    # 계측 및 예측 침하량 표시
    axes[1].scatter(time_monitored, -settlement_monitored, s=30,
                    facecolors='white', edgecolors='black', label='measured data')

    axes[1].plot(time_predicted, -settlement_predicted,
                 linestyle='--', color='red', label='Original Hyperbolic')

    axes[0].set_ylabel("Settlement (cm)", fontsize=10)
    axes[1].set_xlim(left=0)
    axes[1].set_xlim(right=np.max(time_predicted))


# script to call: python3 controller.py [business_code] [cons_code]
# for example: python3 controller.py 221222SA0003 CONS001
if __name__ == '__main__':

    args = sys.argv[1:]
    business_code = args[0]
    cons_code = args[1]

    settlement_prediction(business_code=business_code, cons_code=cons_code)
    print("The settlement prediction is over.")

    #read_database_and_plot(business_code=business_code, cons_code=cons_code)
    #print("Visualization is over.")