"""
Title: Controller
Developer:
Sang Inn Woo, Ph.D. @ Incheon National University
Starting Date: 2022-11-10
"""
import psycopg2 as pg2
import numpy as np
import settle_prediction_steps_main


'''
apptb_surset01
cons_code: names of monitoring points

apptb_surset02
cons_code: names of monitoring points
amount_cum_sub: accumulated settlement
fill_height: height of surcharge fill
nod: number of date
'''


# connect the database
connection = pg2.connect("host=localhost dbname=postgres user=postgres password=lab36981 port=5432")

# set cursor
cursor = connection.cursor()

# select all monitoring points
postgres_select_query = """SELECT * FROM apptb_surset01"""
cursor.execute(postgres_select_query)
point_record = cursor.fetchall()

# for a monitoring point, set name
point_name = point_record[0][3]

# select monitoring data for the monitoring point
postgres_select_query = """SELECT * FROM apptb_surset02 WHERE cons_code='""" \
                        + point_name + """' ORDER BY nod ASC"""
cursor.execute(postgres_select_query)
monitoring_record = cursor.fetchall()

# initialize time, surcharge, and settlement lists
time = []
surcharge = []
settlement = []

# fill list
for row in monitoring_record:
    settlement.append(float(row[6]))
    surcharge.append(float(row[8]))
    time.append(float(row[12]))

# convert lists to np arrays
settlement = np.array(settlement)
surcharge = np.array(surcharge)
time = np.array(time)

# run the settlement prediction and get results
results = settle_prediction_steps_main.run_settle_prediction(point_name=point_name,
                                                             np_time=time,
                                                             np_surcharge=surcharge,
                                                             np_settlement=settlement,
                                                             final_step_predict_percent=90,
                                                             additional_predict_percent=300,
                                                             plot_show=True,
                                                             print_values=True,
                                                             run_original_hyperbolic=True,
                                                             run_nonlinear_hyperbolic=True,
                                                             run_weighted_nonlinear_hyperbolic=True,
                                                             run_asaoka=True,
                                                             run_step_prediction=True,
                                                             asaoka_interval=3)

# if there are prediction data for the given data point, delete it first
postgres_delete_query = """DELETE FROM apptb_pred02 WHERE cons_code='""" + point_name + """'"""
cursor.execute(postgres_delete_query)
connection.commit()

time = results[0]
predicted_settlement = results[1]

for i in range(5):

    time = results[2 * i]
    predicted_settlement = results[2 * i + 1]

    for j in range(len(time)):

        postgres_insert_query \
            = """INSERT INTO apptb_pred02 (cons_code, prediction_progress_days, predicted_settlement, prediction_method) """\
              + """VALUES (%s, %s, %s, %s)"""

        record_to_insert = (point_name, time[j], predicted_settlement[j], i)
        cursor.execute(postgres_insert_query, record_to_insert)

connection.commit()


