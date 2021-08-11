## Test script for working out problem with calcium
import pandas as pd
import json

#In reality import the big flowsheet to use thing
flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_zscores.csv', parse_dates = ['taken_datetime'])

colums = {i:j for i, j in enumerate(flowsheet.columns)}

a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/files/All_flowsheet_columns.json", "w")
json.dump(colums, a_file)
a_file.close()