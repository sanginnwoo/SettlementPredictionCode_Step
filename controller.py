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

print(settlement)
print(surcharge)
print(time)
