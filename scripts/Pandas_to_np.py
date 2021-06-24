#### Little script to convert big flowsheet file into np array for use
## Unfinished
import pandas as pd
import numpy as np
import re

#In reality import the big flowsheet to use thing
flowsheet = pd.read_csv('~/Project/Project_data/files/flowsheet_output.csv', parse_dates = ['taken_datetime'])

#Sort out the problem with MAP, make sure any lost go to BP etc 
mapped = list(map(type, flowsheet['MAP']))
slash = re.compile('[\D]+')
strings = [i for i, j in enumerate(mapped) if j == str]
flowsheet.loc[strings, 'MAP']

#Make a unique patients column
unique_patients 
for i in 

#Save as np array