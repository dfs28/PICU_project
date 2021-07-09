#### Little script to convert big flowsheet file into np array for use
## Unfinished
import pandas as pd
import numpy as np
import re

#In reality import the big flowsheet to use thing
flowsheet = pd.read_csv('~/Project/Project_data/files/flowsheet_output.csv', parse_dates = ['taken_datetime'])

#Sort out the problem with MAP, make sure any lost go to BP etc 
#Pull out where MAP is a **/** format
slash = re.compile('.*/+.*')
strings = [i for i, j in enumerate(flowsheet['MAP']) if not slash.match(str(j)) == None]
strings = np.array(strings)

#Find the location where sysbp and diabp are missing but map not
diaNAs = np.where(flowsheet['DiaBP'].isna())
sysNAs = np.where(flowsheet['SysBP'].isna())
BP_NAs = np.union1d(diaNAs, sysNAs)
MAPnotBP = np.intersect1d(BP_NAs, strings)
#Can see actually none here that are relevant

#Now just fix the problem for the MAPs that are strings
newBPs = flowsheet.loc[strings, 'MAP'].str.split("/", n = 1, expand = True)
newBPs = newBPs.astype('float')
flowsheet.loc[strings, 'MAP'] = newBPs.loc[:, 0]*(1/3) + newBPs.loc[:, 1]*(2/3)

#Make a unique patients column
unique_patients 
for i in 

#Save as np array