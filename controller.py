"""
Title: Controller
Developer:
Sang Inn Woo, Ph.D. @ Incheon National University
Starting Date: 2022-11-10
"""
import psycopg2 as pg2
import settle_prediction_steps_main

# connect the database
conn = pg2.connect("host=localhost dbname=postgres user=postgres password=lab36981 port=5432")

# set cursor
cur = conn.cursor()

# read data


# extract settlement prediction data using the prime key

# run settlement analysis

# save results in the database