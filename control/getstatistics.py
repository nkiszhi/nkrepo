import pandas as pd
import numpy
import sqlite3

con = sqlite3.connect("data4.db")
tracks = pd.read_sql_query("SELECT * FROM test1", con)
#tracks['FAMILY'].value_counts().to_csv("./final1.csv")
#tracks['PLATF'].value_counts().to_csv("./final2.csv")
#tracks['FTIME'].value_counts().to_csv("./final3.csv")
tracks['CATEG'].value_counts().to_csv("./final4.csv")

