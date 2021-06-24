### PICU data wrangling
# Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
import os
from itertools import chain
import re

#Import the files - start with the massive flowsheet one
flowsheet = pd.read_csv('~/Project/Project_data/files/caboodle_patient_selected_flowsheetrows_main_pivot.csv', sep= ',', parse_dates=['taken_datetime'])
demographics = pd.read_csv('~/Project/Project_data/files/caboodle_patient_demographics.csv', sep = ',', parse_dates = ['birth_date', 'death_date'])








##### Make a whole load of helper functions to help with cleaning


## Some functions to help work out what variables are present/ needed

#Helper function for returning the column number as a name
def return_colname(original_column, sheet, give_inputs = False):
    """
    Function to give original column as string
    """
    
    #Get the unique values flexibly
    if type(original_column) == int:
        unique_inputs = sheet.loc[:, sheet.columns[original_column]].unique()
        
        #Make sure original column is a string
        original_column = sheet.columns[original_column]

    elif type(original_column) == str:
        unique_inputs = sheet.loc[:, (original_column)].unique()
    else:
        raise ValueError('original_column should either be a numbered column or the correct name of a column')

    if not give_inputs:
        return original_column
    else:
        return original_column, unique_inputs



#Print out all of the columns
def print_columns(sheet):
    for i in enumerate(sheet.columns):
        yield print(i)

def print_all(sheet):
    a = print_columns(sheet)
    length = len(sheet.columns)
    for i in range(length):
        next(a)



#Print out all of the unique parts of the columns
def print_uniquevars(sheet, column):
    
    column, unique_inputs = return_colname(column, sheet, True)

    for i in enumerate(unique_inputs):
        yield print(i)

def print_allunique(sheet, column, printAll = True):
    """
    Simple function that prints proportion of NaNs, absolute number of Nans and unique values
    Could also add in functionality to say what proportion of each exist
    """

    #Return colname
    column = return_colname(column, sheet)

    #Get number of Nas
    isnan = sheet.loc[:, column].isna() == False
    notNa = sum(isnan)
    propnotNa = notNa/sheet.shape[0]
    print('For {}'.format(column))
    print('{:2.2%} percent not Na'.format(propnotNa))
    print('{} of {} not Na'.format(notNa, sheet.shape[0]))

    #Print out the unique vals
    if printAll:
        a = print_uniquevars(sheet, column)
        length = len(sheet.loc[:, column].unique())
        for i in range(length):
            next(a)



## Bit of code to work out how many Nas there are per column

#Think about which variables are seen the most and the least
#Should help with what is worth going through
#Not run
"""
#This takes ages so I've saved it, only rerun if needed
if (os.path.exists('~/Project/Project_data/files/not_NA.csv') == False):
    colnames = flowsheet.columns
    numNotNa = np.zeros([len(colnames) - 3])

    #Make a progress bar because this takes ages
    bar = Bar('Processing', max=len(colnames))

    #Work  through columns and get numbers of Nans
    for i in range(3, len(colnames)):
        isnan = flowsheet.iloc[:, (i)].isna() == False
        numNotNa[i - 3] = sum(isnan)
        bar.next()

    bar.finish()
    
    np.savetxt('~/Project/Project_data/files/not_NA.csv', numNotNa, delimiter=',')
else: 
    numNotNa = np.genfromtxt('~/Project/Project_data/files/not_NA.csv', delimiter=',')

topcols = colnames[np.append(np.array([False, False, False]), numNotNa > 10000)]
for i in range(len(topcols)):
    print(topcols[i])
sortedNaNs = np.argsort(numNotNa)
"""

## Work out which columns overlap
def overlapping_columns(sheet):
    """ 
    Function that returns columns which have other subsets
    """
    columns = sheet.columns
    subsets = list()
    for i in columns:
        subset = list()
        for j in (k for k in columns if k != i):

            #Work out if col j is a subset of col i
            j_locs = (sheet[j].isna() == False)
            all_same = all(sheet.loc[j_locs, j] == sheet.loc[j_locs, i])
            if all_same:
                subset.append(j)
        subsets.append(subset)

    return subsets



## Helper function to redefine and merge columns

#Some helper functions for redefining columns
def make_new_columns(original_column, sheet, new_cols, inputIsoutput = False, **input2output):
    """
    Function to take some columns and then make new ones  \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New cols (need to be named strings) are the new columns to be made \n
    If inputIsoutput = True then it just takes the input and puts it in the new column \n
    Input2output passed as nested lists for linking  \n
    Will probably want to add some sort of missingness new column \n
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    #Allocate the new values
    if not inputIsoutput:
        i = 0
        
        #Work through the lists
        for key, value in input2output.items():
            new_column = new_cols[i]
            if not new_column in sheet.columns:

                #Should probably consider whether default should be zeros?
                #Could maybe consider this to be na most of the time, then subsequently interpolate
                #Fill with zeros if backfill limit reached on interpolation
                #Could do interpolation but also consider missingness?
                sheet[new_column] = np.nan
            
            #Work  through the items in this list
            for j, k in enumerate(value):

                #Only take values that are in this column
                if isinstance(k, list):
                    for l, m in enumerate(k):
                        original_values = value[j]
                        sheet.loc[sheet[original_column].isin(original_values), (new_column)] = j
                else:         
                    original_value = value[j]
                    sheet.loc[sheet[original_column] == original_value, (new_column)] = j

            i += 1
    
    else:
        if not new_cols[0] in sheet.columns:
            sheet[new_cols[0]] = np.nan

        #Find the values that are Na
        na_locs = (sheet.loc[:, original_column].isna() == False)
        new_values = sheet.loc[na_locs, original_column]

        #Now assign
        sheet.loc[na_locs, new_cols[0]] = new_values


# This is a helper function for the above helper function
# It returns all non-Na unique values so they can be converted into a categorical variable

def make_allnonNa(sheet, column, which_item, length):
    """
    Simple function to make all the non-na values into a list
    """

    column, unique_values = return_colname(column, sheet, True)
    if unique_values.dtype == np.dtype('O'):
        unique_values = pd.Series(unique_values, dtype = 'string')
        unique_nonan = unique_values[unique_values.isna() == False].tolist()
    else:
        unique_nonan = unique_values[np.isnan(unique_values) == False].tolist()
    l = [None] * length
    l[which_item -1 ] = unique_nonan
    return l


# Function to correct FiO2 and make it into new col

def cor_FiO2(original_column, sheet, new_col):
    """
    Function to take columns containing FiO2 and make a new one \n
    It corrects FiO2 into a number between 0 and 1 \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = np.nan

    #Find the values that are Na
    na_locs = (sheet.loc[:, original_column].isna() == False)
    new_values = sheet.loc[na_locs, original_column]

    #Work through the values to make them into a number between 0 and 1
    percent_FiO2 = 21 <= new_values
    new_values[percent_FiO2] /= 100

    dec_FiO2 = 1 <= new_values
    new_values[dec_FiO2] /= 10
    
    #Any that are less than 0.21 correct to 0.21
    lessthan21 = 0.21 >= new_values
    new_values[lessthan21] = 0.21

    #Now assign
    sheet.loc[na_locs, new_col] = new_values


#Helper function to search through the outcomes to help me
def get_relevant_cols(sheet, query):
    """
    Function to pull out relevant colnames
    Will have it spit out how many Nas in the matching ones, 
    Also a short list of the unique values
    """

    #Pull the colnames and then work through them with a query
    colnames = sheet.columns
    relevant_columns = list()
    
    #Make the query
    prog = re.compile(query, flags = re.IGNORECASE)
    for i, j in enumerate(colnames):

        #Search for the query in colnames, if present store it
        result = re.search(prog, j)
        try:
            start = result.start()
            relevant_columns.append(i)
        except:
            pass
    
    for i in relevant_columns:
        print(i)
        print_allunique(sheet, i, False)
        unique_vars = sheet.iloc[:, i].unique()
        length = len(unique_vars)
        if length > 10:
            length = 10
        print(unique_vars[0:length])
        print('\n')
        

#Can consider some sort of helper function to return appropriate vars?

def conv_BP(original_column, sheet, SysDia, *new_cols):
    """
    Function to take some columns and then make new ones  \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New cols (need to be named strings) are the new columns to be made \n
    SysDia needs to be true or false
    If inputIsoutput = True then it just takes the input and puts it in the new column \n
    Input2output passed as nested lists for linking  \n
    Will probably want to add some sort of missingness new column \n
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    #If both need to split into sys and dia
    if SysDia in ['Three', 'Both', 'both', 'three', 3]:
        new_sheet = sheet.loc[:, (original_column)].str.split("/", n = 1, expand = True)
        new_sheet = new_sheet.astype('float')
        new_sheet['MAP'] = new_sheet.loc[:, 0]*(1/3) + new_sheet.loc[:, 0]*(2/3)
    else:
        new_sheet = sheet.loc[:, (original_column)]

    #Allocate the new values
    i = 0
        
    #Work through the new columns
    for value in new_cols:
        new_column = value
        
        #Make a new column if its not alread there
        if not new_column in sheet.columns:
            sheet[new_column] = np.nan

        #Use only the ones from the original columns that aren't Na
        if SysDia in ['Three', 'Both', 'both', 'three', 3]:
            na_locs = (new_sheet.iloc[:, (i)].isna() == False)
            new_values = new_sheet.loc[na_locs, (new_sheet.columns[i])]
        else:
            na_locs = (new_sheet.isna() == False)
            new_values = new_sheet[na_locs]

        #Now assign
        sheet.loc[na_locs, (new_column)] = new_values

        i += 1


def conv_column(original_column, sheet, new_col, func):
    """
    Function to take columns and convert using a new function \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = np.nan

    #Find the values that are Na
    na_locs = (sheet.loc[:, original_column].isna() == False)
    new_values = sheet.loc[na_locs, original_column]

    #Apply the function
    new_values = func(new_values)

    #Now assign
    sheet.loc[na_locs, new_col] = new_values

def sum_inout(original_column, sheet, new_col):
    """
    Function to take a an existing column and add to it \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = np.nan

    #Find the values that are Na
    na_locs = (sheet.loc[:, original_column].isna() == False)
    new_values = sheet.loc[na_locs, original_column]

    #Now assign
    sheet.loc[na_locs, new_col] = sheet.loc[na_locs, new_col] + new_values

def bminusa(a, b):
    return b - a

def conv_2cols(original_column1, original_column2, sheet, new_col, func):
    """
    Function to take two existing columns and use function to apply them together to get a new col\n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column1 = return_colname(original_column1, sheet)
    original_column2 = return_colname(original_column2, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = np.nan

    #Find the values that are Na 1
    na_locs1 = (sheet.loc[:, original_column1].isna() == False)
    new_values1 = sheet[original_column1]

    #Find the values that are Na 2
    na_locs2 = (sheet.loc[:, original_column2].isna() == False)
    new_values2 = sheet[original_column2]

    #Find overlapping not nas
    both_nonan = na_locs1 & na_locs2

    #Calculate new values
    new_values = func(new_values1, new_values2)

    #Now assign
    sheet.loc[both_nonan, new_col] = new_values


### Get weights and heights for correction
def apply_wt_ht(original_column, sheet, new_column, scaling = 1):
    """
    Function to get wt or ht and apply them so there is an up to date weight or height \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    Scaling is a scaling factor so can convert to kg if necessary \n
    new_col is a string
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    #Split by patient and work through them
    unique_patients = sheet['project_id'].unique()
    for i in unique_patients:

        patient_locs = (sheet['project_id'] == i)
        
        #Interpolate weight linearly and fill in all missing values
        new_values = sheet.loc[patient_locs, original_column].interpolate('linear', limit = 100000, limit_direction = 'both')

        #Fill in new column with these weights, scaling
        sheet.loc[patient_locs, new_column] = new_values*scaling


def apply_demographic(original_column, demographic_sheet, flow_sheet, new_column, age = True):
    """
    Function to apply demographics to the new sheet \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    new_col is a string 
    """

    #Make the column
    flow_sheet[new_column] = np.nan

    original_column = return_colname(original_column, demographic_sheet)

    unique_patients = flow_sheet['project_id'].unique()
    for i in unique_patients:

        #Get demographic value
        demographic_loc = (demographic_sheet['project_id'] == i)
        value = demographic_sheet.loc[demographic_loc, original_column]
        
        #Get location on flowsheet
        flowsheet_locs = (flow_sheet['project_id'] == i)

        #Make sequence of values
        length = sum(flowsheet_locs)
        new_values = np.repeat(value, length)

        if age:
            #Get date of variable in current time
            dates = flow_sheet.loc[flowsheet_locs, 'taken_datetime']
            dates = dates.dt.round('D')

            #Make sequence of birth date
            birth_dates = new_values
            dates.index = birth_dates.index

            #Calculate age and convert to years
            ages = dates - birth_dates
            new_values = ages / np.timedelta64(1, 'Y')
            new_values = abs(new_values)    

        #Fill in new column with these weights, scaling
        new_values.index = flow_sheet.loc[flowsheet_locs, new_column].index
        flow_sheet.loc[flowsheet_locs, new_column] = new_values        


def fill_missing(original_column, sheet, new_col, missing = np.nan):
    """
    Function to take a an existing column and fill in missing values \n
    And make a new column  
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = 0

    #Find the values that are Na, put 1s here
    na_locs = sheet.loc[:, original_column].isna()
    sheet.loc[na_locs, new_col] = 1
    sheet.loc[na_locs, original_column] = missing

def interpolate_cols(column, sheet, method, how_far = 60, limit_direction = 'both', fill = 0, *args):
    """ 
    Function to interpolate some columns \n
    Method, how_far, limit_direction inherit from pd.interpolate \n
    Fill then fills in the rest of the columns with a filler \n
    Do you want to make a new column to interpolate into? - xgboost can handle NaNs
    """

    #Get the unique values flexibly
    column = return_colname(column, sheet)

    #Work through the patients
    unique_patients = sheet['project_id'].unique()
    
    for i in unique_patients:
        pt_locs = sheet['project_id'] == i
        
        #Now interpolate
        sheet.loc[pt_locs, column].interpolate(method = method,limit = how_far, limit_direction = limit_direction, inplace = True)

    #Allow fill to 
    na_locs = sheet[column].isna() == True
    if callable(fill):
        sheet.loc[na_locs, column] = fill(column, sheet.loc[na_locs, :], *args)
    sheet.loc[na_locs, column] = fill
        
        
    




#### Start by only including columns from within ITU admission



#### Now enforce a minute by minute time frame
#Frames for start and end dates
start_dates = pd.Series([], dtype = 'datetime64[ns]')
end_dates = pd.Series([], dtype = 'datetime64[ns]')

#Fill them
unique_patients = flowsheet['project_id'].unique()
for i in unique_patients:
    locs = flowsheet['project_id'] == i
    times = flowsheet.loc[locs, 'taken_datetime']
    sorted_times = np.argsort(times)
    start = sorted_times == min(sorted_times)
    finish = sorted_times == max(sorted_times)
    start_dates = start_dates.append(times[start])
    end_dates = end_dates.append(times[finish])

#Now make a new series to merge to which contains all of the possible time points
all_times = pd.DataFrame(pd.Series([], dtype = 'datetime64[ns]'), columns =['taken_datetime'])
all_times = all_times.append(pd.DataFrame([], columns = ['project_id']))
for i in range(len(unique_patients)):
    times = pd.date_range(start = start_dates[start_dates.index[i]], end = end_dates[end_dates.index[i]], freq = 'min')
    patient = pd.Series([unique_patients[i]]*len(times))
    times_df = pd.DataFrame({'taken_datetime': times, 'project_id': patient})
    all_times = all_times.append(times_df)

flowsheet = all_times.merge(flowsheet, how = 'outer', on = ['taken_datetime', 'project_id'])
flowsheet.to_csv('~/Project/Project_data/files/flowsheet_all_times.csv')

#### Now do some feature engineering



#Bodyweight
print_all(flowsheet)
get_relevant_cols(flowsheet, 'weight|kg|gram|wt')

def kg2g(new_values):
    return new_values*1000

#R DRUG CALCULATION WEIGHT_grams
make_new_columns(176, flowsheet, ['Weight(g)'], True)
#WEIGHT/SCALE_grams
make_new_columns(856, flowsheet, ['Weight(g)'], True)
#R GOSH POST TREATMENT WEIGHT_kg
conv_column(517, flowsheet, 'Weight(g)', kg2g)
#R GOSH HD PRE TREATMENT WEIGHT_kg
conv_column(530, flowsheet, 'Weight(g)', kg2g)

apply_wt_ht('Weight(g)', flowsheet, 'interpolated_wt_kg', scaling = 0.001)


### May need to estimate bodyweight for patients with no bodyweight
#Using formula from paper found
na_locs = flowsheet['interpolated_wt_kg'].isna()
over1s = na_locs & flowsheet['Age_yrs'] >= 1
under1s = na_locs & (flowsheet['Age_yrs'] < 1)

#Fill in weight estimations
flowsheet.loc[over1s, 'interpolated_wt_kg'] = flowsheet.loc[over1s, 'Age_yrs']*2 + 10
flowsheet.loc[under1s, 'interpolated_wt_kg'] = (flowsheet.loc[under1s, 'Age_yrs']*12 + 9)/2





#### Start with ventilation and O2
print_all(flowsheet)

"""
Here I'm making a ventilation column:
0 for no ventilation
1 for on oxygen
2 for NIV/Optiflow
3 for intubated and ventilated

High frequency oscillated ventilation - this is because I want to assume that anyone with a tube is ventilated
and possibly anyone with an EtCO2 value is also ventilated - its not clear from the variables which come from which
so I will be able to assume that more patients are ventilated without accidentally marking them as not on high frequency
oscillatory ventilation
True/False (1/0)

An intubated column - probably don't need this as above
0 for not
1 for intubated

An EtCO2 column:
Na for no value, rest just ported from any EtCO2 columns

An Oxygen flow column - this will then need to be corrected for bodyweight

An FiO2 column (not going to try to convert between but planning to have missingness columns)

A tracheostomy column (yes/no)

Could probably consider adding some further columns including:
PEEP/EPAP, IPAP, some other ventilation settings if they are available
"""
#'R GOSH ICU AIRWAY STATUS' 14 ##### Is HFO high frequency oscillation or high flow oxygen?
#Should I consider this as a separate var?
#print_allunique(flowsheet, 14)
ventilated14 = [['SVIA', 'NO'], 
                ['Low Flow NC', 'Face Mask', 'SVIA;Low Flow NC', 'Face Mask;Low Flow NC', 'SVIA;Face Mask', 
                'Low Flow NC;SVIA'], 
                ['High Flow NC', 'CPAP', 'SVIA;CPAP', 'SVIA;High Flow NC', 'Face Mask;CPAP', 'High Flow NC;SVIA', 'CPAP;SVIA'], 
                ['CMV', 'CMV;CPAP', 'SVIA;CMV', 'NO;CMV', 'CMV;NO', 'HFO', 'HFO;NO']]
make_new_columns('R GOSH ICU AIRWAY STATUS', flowsheet, ['Ventilation'], newcol1 = ventilated14)
HFO14 = [[], ['HFO', 'HFO;NO']]
make_new_columns(14, flowsheet, ['HFO'], newcol1 = HFO14)

#'R GOSH IP PANDA AIRWAY' 15
#print_allunique(flowsheet, 15)
ventilated15 = [[], [], [], ['Intubation']]
make_new_columns('R GOSH IP PANDA AIRWAY', flowsheet, ['Ventilation'], newcol1 = ventilated15)


#'R GOSH NICU AIRWAY STATUS' 16
#print_allunique(flowsheet, 16)
ventilated16 = [[], [], [], ['Conventional ventilation']]
make_new_columns('R GOSH NICU AIRWAY STATUS', flowsheet, ['Ventilation'], newcol1 = ventilated16)

#'R GOSH RESUS ABCDE AIRWAY' 17
#print_allunique(flowsheet, 17)
ventilated16 = [[],[], [], ['Intubated', 'Intubated;Opening manoeuvres', 'Intubated;Patent']]
make_new_columns('R GOSH RESUS ABCDE AIRWAY', flowsheet, ['Ventilation'], newcol1 = ventilated16)

#'R ED AIRWAY (WDL)' 18 - not using
#print_allunique(flowsheet, 18)

#'R AIRWAY LDA INSERTION ATTEMPTS' 19
#print_allunique(flowsheet, 19)
ventilated19 = [[], [], [], [1, 2, '1', '2', '3 or more']]
make_new_columns('R AIRWAY LDA INSERTION ATTEMPTS', flowsheet, ['Ventilation'], newcol1 = ventilated19)

#'R GOSH IP CATS AIRWAY STATUS' 20
#print_allunique(flowsheet, 20)
ventilated20 = [[], [], [],  ['Intubated']]
make_new_columns('R GOSH IP CATS AIRWAY STATUS', flowsheet, ['Ventilation'], newcol1 = ventilated20)

#R VENT ALARM APNOEA SECONDS_Seconds 28 - come back to this one - could be NIV or 
#print_allunique(flowsheet, 28)

#R IP VENT APNOEA RATE 30

#R IP VENT APNOEA VOLUME_mL 31

#34, 'R GOSH IP ARE THEY INTUBATED AND VENTILATED?'
#print_allunique(flowsheet, 34)
ventilated34 = [[], [], ['No - Non-invasive Ventilation', 'No - Optiflow'], ['Yes - Invasive ventilation']]
make_new_columns('R GOSH IP ARE THEY INTUBATED AND VENTILATED?', flowsheet, ['Ventilation'], newcol1 = ventilated34)

#127, 'R GOSH CPAP CMH2O_cmH2O'
#print_allunique(flowsheet, 127)
make_new_columns('R GOSH CPAP CMH2O_cmH2O', flowsheet, ['IPAP'], True)
make_new_columns('R GOSH CPAP CMH2O_cmH2O', flowsheet, ['EPAP'], True)

#128, 'R GOSH IP FF CPAP/BIPAP MASK'
#print_allunique(flowsheet, 128)

#129, 'R OPTIME RECOVERY CPAP/IPPV'
#print_allunique(flowsheet, 129)

#136, 'R GOSH TRACHE CUFF'
#print_allunique(flowsheet, 136)
trache136 = [[], ['Inflated', 'Deflated', 'Deflated;Inflated', 'Inflated;Deflated', 'Deflated;Inflated;Leak present', 
            'Leak present;Inflated', 'Inflated;Leak present', 'Leak present', 'Inflated;Deflated;Leak present', 
            'Other (Comment)', 'Cuffless', 'Cuffless;Leak present', 'Leak present;Cuffless', 'Deflated;Leak present', 
            'Inflated;Other (Comment)', 'Leak present;Other (Comment)', 'Deflated;Leak present;Other (Comment)', 
            'Deflated;Other (Comment)', 'Deflated;Leak present;Inflated', 'Leak present;Deflated', 
            'Inflated;Leak present;Deflated', 'Deflated;Inflated;Other (Comment)']]
make_new_columns('R GOSH TRACHE CUFF', flowsheet, ['Tracheostomy'], newcol1 = trache136)

#198, 'R GOSH ASTRAL EPAP_cm H2O') - Astral is NIV
#print_allunique(flowsheet, 198)
unique_vent198 = flowsheet.iloc[:, 'R GOSH ASTRAL EPAP_cm H2O'].unique()
vent198 = [[], [], unique_vent198[range(1, len(unique_vent198))].tolist(), []]
make_new_columns('R GOSH ASTRAL EPAP_cm H2O', flowsheet, ['Ventilation'], newcol1 = vent198)
make_new_columns('R GOSH ASTRAL EPAP_cm H2O', flowsheet, ['EPAP'], True)

#199, 'R GOSH EPAP'
#print_allunique(flowsheet, 'R GOSH EPAP')
make_new_columns('R GOSH EPAP', flowsheet, ['EPAP'], True)

#200, 'R GOSH NIPPY EPAP_cmH2O
#print_allunique(flowsheet, 'R GOSH NIPPY EPAP_cmH2O')
unique_vent200 = flowsheet.iloc[:, 'R GOSH NIPPY EPAP_cmH2O'].unique()
vent200 = [[], [], unique_vent200[range(1, len(unique_vent200))].tolist(), []]
make_new_columns('R GOSH NIPPY EPAP_cmH2O', flowsheet, ['Ventilation'], newcol1 = vent200)
make_new_columns(200, flowsheet, ['EPAP'], True)

#204, 'R GOSH ICU END TIDAL CO2_% Should work out if this means they are ventilated
#print_allunique(flowsheet, 204)
make_new_columns('R GOSH ICU END TIDAL CO2_%', flowsheet, ['ETCO2'], True)

#205, 'R VENT ETCO2_kPa Should work out if this means they are ventilated
#print_allunique(flowsheet, 205)
make_new_columns('R VENT ETCO2_kPa', flowsheet, ['ETCO2'], True)

#206, 'R IP VENT ALARM ETCO2/TCM MAX_mmHg
#print_allunique(flowsheet, 206)
make_new_columns('R IP VENT ALARM ETCO2/TCM MAX_mmHg', flowsheet, ['ETCO2'], True)

#207, 'R IP VENT ALARM ETCO2/TCM MIN_mmHg
#print_allunique(flowsheet, 207)

#208, 'R GOSH ICU TRANSFER ETT
#print_allunique(flowsheet, 208)

#209, 'R GOSH ICU TRANSFER ETT SECURE
#print_allunique(flowsheet, 209)

#210, 'R ETT TO LIP_cm
#print_allunique(flowsheet, 210)
unique_vent210 = flowsheet.iloc[:, 'R ETT TO LIP_cm'].unique()
vent210 = [[], [], [], unique_vent210[range(1, len(unique_vent210))].tolist()]
make_new_columns('R ETT TO LIP_cm', flowsheet, ['Ventilation'], newcol1 = vent210)

#211, 'R ETT TO NOSE_cm'
#print_allunique(flowsheet, 211)
unique_vent211 = flowsheet.iloc[:, 'R ETT TO NOSE_cm'].unique()
vent211 = [[], [], [], unique_vent211[range(1, len(unique_vent211))].tolist()]
make_new_columns('R ETT TO NOSE_cm', flowsheet, ['Ventilation'], newcol1 = vent211)

#213, 'R EVENT SPO2'
#print_allunique(flowsheet, 213)

#214, 'R AN AGENTS DESFLURANE EXPIRED_%'
#print_allunique(flowsheet, 214)
unique_vent214 = flowsheet.iloc[:, 'R AN AGENTS DESFLURANE EXPIRED_%'].unique()
vent214 = [[], [], [], unique_vent214[range(1, len(unique_vent214))].tolist()]
make_new_columns('R AN AGENTS DESFLURANE EXPIRED_%', flowsheet, ['Ventilation'], newcol1 = vent214)

#215, 'R AN AGENTS ISOFLURANE EXPIRED_%'
#print_allunique(flowsheet, 215)
unique_vent215 = flowsheet.iloc[:, 'R AN AGENTS ISOFLURANE EXPIRED_%'].unique()
vent215 = [[], [], [], unique_vent215[range(1, len(unique_vent215))].tolist()]
make_new_columns('R AN AGENTS ISOFLURANE EXPIRED_%', flowsheet, ['Ventilation'], newcol1 = vent215)

#216, 'R AN EXPIRED MINUTE VOLUME_L/min'
#print_allunique(flowsheet, 216)
make_new_columns('R AN EXPIRED MINUTE VOLUME_L/min', flowsheet, ['Ventilation'], 
                    newcol1 = make_allnonNa(flowsheet, 'R AN EXPIRED MINUTE VOLUME_L/min', 4, 4))

#217, 'R AN EXPIRED N2O_%'\\
#print_allunique(flowsheet, 217)
make_new_columns('R AN EXPIRED N2O_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R AN EXPIRED N2O_%', 4, 4))

#218, 'R AN AGENTS SEVOFLURANE EXPIRED_%'
#print_allunique(flowsheet, 218)
make_new_columns('R AN AGENTS SEVOFLURANE EXPIRED_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R AN AGENTS SEVOFLURANE EXPIRED_%', 4, 4))

#219, 'R GOSH EXP TV LITRES_L' - should consider expired tidal volume?
#print_allunique(flowsheet, 219)
make_new_columns('R GOSH EXP TV LITRES_L', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R GOSH EXP TV LITRES_L', 4, 4))

#220, 'R VENT EXP TIDAL VOLUME_mL'
#print_allunique(flowsheet, 220)
make_new_columns(220, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R VENT EXP TIDAL VOLUME_mL', 4, 4))

#222, 'R GOSH ICU WITHDRAWAL EXTUBATION'
#print_allunique(flowsheet, 222)

#234, 'R AN FIO2_%'
#print_allunique(flowsheet, 234)
cor_FiO2('R AN FIO2_%', flowsheet, 'FiO2')

#235, 'R PERF FIO2_%'
#print_allunique(flowsheet, 235)
cor_FiO2('R PERF FIO2_%', flowsheet, 'FiO2')

#236, 'R GOSH ICU SANDWICH FIO2'
#print_allunique(flowsheet, 236)

#237, 'R INSPIRED O2 FIO2 FOR NO_%' - FIO2, will need to correct
#print_allunique(flowsheet, 237)

#238, 'R GOSH IP FIO2 SETTING_%'
#print_allunique(flowsheet, 238)
cor_FiO2('R GOSH IP FIO2 SETTING_%', flowsheet, 'FiO2')

#239, 'R VENT INSP FLOWS_L/min' #Should probably do this inspired o2 corrected by bodyweight thing
#print_allunique(flowsheet, 239)
make_new_columns('R VENT INSP FLOWS_L/min', flowsheet, ['O2Flow'], True)

#240, 'R GOSH VENT FLOW ML/KG_mL/Kg'
#print_allunique(flowsheet, 240)
make_new_columns('R GOSH VENT FLOW ML/KG_mL/Kg', flowsheet, ['O2Flow/kg'], True)

#241, 'R IP VENT FLOW OBS_L/min'
#print_allunique(flowsheet, 241)
make_new_columns('R IP VENT FLOW OBS_L/min', flowsheet, ['O2Flow'], True)

#242, 'R GOSH AIRVO FLOW_L/min' Highflow o2
#print_allunique(flowsheet, 242)
make_new_columns('R GOSH AIRVO FLOW_L/min' , flowsheet, ['O2Flow'], True)
make_new_columns('R GOSH AIRVO FLOW_L/min' , flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R GOSH AIRVO FLOW_L/min' , 2, 4))

#R GOSH VENT FABIAN O2 THERAPY FLOW_L/min HFNC/ NIV
#print_allunique(flowsheet, 'R GOSH VENT FABIAN O2 THERAPY FLOW_L/min')
make_new_columns('R GOSH VENT FABIAN O2 THERAPY FLOW_L/min', flowsheet, ['O2Flow'], True)
make_new_columns('R GOSH VENT FABIAN O2 THERAPY FLOW_L/min', flowsheet, ['Ventilation'], 
                newcol1 = make_allnonNa(flowsheet, 'R GOSH VENT FABIAN O2 THERAPY FLOW_L/min', 2, 4))

#(284, 'R GOSH HFO BASE FLOW READING')
#print_allunique(flowsheet, 284)
make_new_columns('R GOSH HFO BASE FLOW READING', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH HFO BASE FLOW READING', 2, 2))
make_new_columns('R GOSH HFO BASE FLOW READING', flowsheet, ['O2Flow'], True)

#285, 'R GOSH HFO BASE FLOW SETTING')
#print_allunique(flowsheet, 285)
make_new_columns('R GOSH HFO BASE FLOW SETTING', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH HFO BASE FLOW SETTING', 2, 2))
make_new_columns('R GOSH HFO BASE FLOW SETTING', flowsheet, ['O2Flow'], True)

#287, 'R GOSH HFO FREQUENCY READING_Hz')
#print_allunique(flowsheet, 287)
make_new_columns('R GOSH HFO FREQUENCY READING_Hz', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH HFO FREQUENCY READING_Hz', 2, 2))

#(288, 'R GOSH IP HFV FREQUENCY SETTING_Hz')
#print_allunique(flowsheet, 288)
make_new_columns('R GOSH IP HFV FREQUENCY SETTING_Hz', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH IP HFV FREQUENCY SETTING_Hz', 2, 2))

#(289, 'R GOSH IP HFV I:E RATIO SETTING') - should probably consider calculating I:E manually
#print_allunique(flowsheet, 289)

#(290, 'R GOSH IP HFV TIDAL VOLUME_mL')HFV is high frequency ventilation
#print_allunique(flowsheet, 290)
make_new_columns('R GOSH IP HFV TIDAL VOLUME_mL', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH IP HFV TIDAL VOLUME_mL', 2, 2))

#(291, 'R GOSH HIGH FIO2_%') 
#print_allunique(flowsheet, 291)
cor_FiO2('R GOSH HIGH FIO2_%', flowsheet, 'FiO2')

#R IP VENT HUMIDIFIER TEMP (SET)')
#print_allunique(flowsheet, 'R IP VENT HUMIDIFIER TEMP (SET)')
make_new_columns("R IP VENT HUMIDIFIER TEMP (SET)", flowsheet, ['Ventilation'], 
                    newcol1 = make_allnonNa(flowsheet, 'R IP VENT HUMIDIFIER TEMP (SET)', 4, 4))

#(306, 'R VENT I:E RATIO') # Need to fix these
#print_allunique(flowsheet, 306)
#make_new_columns(306, flowsheet, ['Ventilation'], 
#                    newcol1 = make_allnonNa(flowsheet, 306, 4, 4))

#(307, 'R AN I:E RATIO')
#print_allunique(flowsheet, 307)
#make_new_columns(307, flowsheet, ['Ventilation'], 
#                    newcol1 = make_allnonNa(flowsheet, 307, 4, 4))

#(315, 'R IP VENT INSPIRATORY PRESSURE HIGH_cm H2O')
#print_allunique(flowsheet, 315)

#316, 'R IP VENT INSPIRATORY PRESSURE LOW_cm H2O')
#print_allunique(flowsheet, 316)

#(317, 'R GOSH INSPRIATORY PRESSURE')
#print_allunique(flowsheet, 317)
make_new_columns('R GOSH INSPRIATORY PRESSURE', flowsheet, ['IPAP'], True)

#(320, 'R VENT INSP TIME_Sec')
#print_allunique(flowsheet, 320)
make_new_columns('R VENT INSP TIME_Sec', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,'R VENT INSP TIME_Sec',  4, 5))

#(322, 'R AN AGENTS DESFLURANE INSPIRED_%')
#print_allunique(flowsheet, 322)
make_new_columns('R AN AGENTS DESFLURANE INSPIRED_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,'R AN AGENTS DESFLURANE INSPIRED_%',  4, 5))

#(323, 'R FIO2_%')
#print_allunique(flowsheet, 323)
make_new_columns('R FIO2_%', flowsheet, ['FiO2'], True)

#(324, 'R AN AGENTS ISOFLURANE INSPIRED_%')
#print_allunique(flowsheet, 324)
make_new_columns('R AN AGENTS ISOFLURANE INSPIRED_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN AGENTS ISOFLURANE INSPIRED_%',  4, 5))

#(325, 'R AN INSPIRED N2O_%')
#print_allunique(flowsheet, 325)
make_new_columns('R AN INSPIRED N2O_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN INSPIRED N2O_%',  4, 5))

#(326, 'R AN INSPIRED NO2_ppm')
#print_allunique(flowsheet, 326)
make_new_columns('R AN INSPIRED NO2_ppm', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN INSPIRED NO2_ppm',  4, 5))

#(327, 'R AN AGENTS NITRIC OXIDE_ppm')
#print_allunique(flowsheet, 327)
make_new_columns('R AN AGENTS NITRIC OXIDE_ppm', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN AGENTS NITRIC OXIDE_ppm',  4, 5))

#(328, 'R INSPIRED NO SETTING')
#(329, 'R AN INSPIRED O2 SETTING')
#(330, 'R AN AGENTS SEVOFLURANE INSPIRED_%')
#print_allunique(flowsheet, 330)
make_new_columns('R AN AGENTS SEVOFLURANE INSPIRED_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN AGENTS SEVOFLURANE INSPIRED_%',  4, 5))

#(331, 'R GOSH IP INSPIRED TIDAL VOLUME_mL')
#(333, 'R GOSH IPAP')
#print_allunique(flowsheet, 333)
make_new_columns('R GOSH IPAP', flowsheet, ['IPAP'], True)


#(332, 'R GOSH ASTRAL IPAP_cm H2O')
#print_allunique(flowsheet, 332)
make_new_columns('R GOSH ASTRAL IPAP_cm H2O', flowsheet, ['IPAP'], True)

#(334, 'R GOSH NIPPY IPAP_cmH2O')
#print_allunique(flowsheet, 334)
make_new_columns('R GOSH NIPPY IPAP_cmH2O', flowsheet, ['IPAP'], True)

#(335, 'R GOSH STELLA IPAP_cm H2O')
#print_allunique(flowsheet, 335)
make_new_columns('R GOSH STELLA IPAP_cm H2O', flowsheet, ['IPAP'], True)


"""(360, 'R GOSH ICU LEVEL WOB')
(367, 'R GOSH LOW FIO2_%')
(368, 'R GOSH IP VENT LOW FLOW ALARM_l/min')
(369, 'R GOSH LOW FLOW ALARM_L/min')
(370, 'R GOSH LOW MINUTE VENTILATION ALARM_L/min')
(371, 'R GOSH LOW MVE_L/min')
(372, 'R GOSH LOW PEEP')
(373, 'R GOSH IP VENT LOW RESP ALARM_bpm')
(374, 'R GOSH LOW RESPIRATORY RATE - BPM_bpm')
(376, 'R IP VENT MV LOW_L/min')
(377, 'R GOSH VENT RATE')"""
#print_allunique(flowsheet,'R IP VENT MV LOW_L/min')
make_new_columns('R IP VENT MV LOW_L/min', flowsheet, ['Ventilation_L_min'], True)

make_new_columns('R GOSH VENT RATE', flowsheet, ['Ventilation(ml)'], True)

"""
(378, 'R GOSH VENT RATE OBSERVED')
"""
#print_allunique(flowsheet, 378)
make_new_columns('R GOSH VENT RATE OBSERVED', flowsheet, ['Ventilation(ml)'], True)

"""
(379, 'R IP VENT VT MANDATORY (OBS)_mL')
"""

"""
(380, 'R MAP')
"""
#print_allunique(flowsheet, 380)
make_new_columns('R MAP', flowsheet, ['MeanAirwayPressure'], True)

"""
(381, 'R IP VENT MEAN AIRWAY PRESSURE HIGH_cm H2O')
(382, 'R IP VENT MEAN AIRWAY PRESSURE LOW_cm H2O')
"""

#(387, 'R VENT MAP_cm H2O')
#print_allunique(flowsheet, 387)
make_new_columns('R VENT MAP_cm H2O', flowsheet, ['MeanAirwayPressure'], True)


#(388, 'R IP VENT MEAN AIRWAY PRESSURE (SET)')
#print_allunique(flowsheet, 388)
make_new_columns('R IP VENT MEAN AIRWAY PRESSURE (SET)', flowsheet, ['MeanAirwayPressure'], True)

"""
(391, 'R VENT MINUTE VENTILATION_L/min')
(392, 'R GOSH MIN VOL LEAK_L/min')
(393, 'R GOSH IP MINUTE VOLUME SETTING_L/min')
"""

#(424, 'R AN AGENTS O2_L/min')
#print_allunique(flowsheet, 424)
make_new_columns('R AN AGENTS O2_L/min', flowsheet, ['O2Flow'], True)

#427 R OXYGEN DELIVERY METHOD)
#print_allunique(flowsheet, 427)
 
 #(428, 'R RT OXYGEN FLOW RATE_L/min')
#print_allunique(flowsheet, 428)
make_new_columns('R RT OXYGEN FLOW RATE_L/min', flowsheet, ['O2Flow'], True)

# 429, 'R OXYGEN FLOW RATE_L/min')
#print_allunique(flowsheet, 429)
make_new_columns('R OXYGEN FLOW RATE_L/min', flowsheet, ['O2Flow'], True)

#(442, 'R GOSH AIRVO OXYGEN %_%')
#print_allunique(flowsheet, 442)
cor_FiO2('R GOSH AIRVO OXYGEN %_%', flowsheet, 'FiO2')

#(446, 'R OXYGEN THERAPY')
#print_allunique(flowsheet, 446)

#(492, 'R GOSH ASTRAL PEEP_cm H2O')
#print_allunique(flowsheet, 492)
make_new_columns('R GOSH ASTRAL PEEP_cm H2O', flowsheet, ['PEEP'], True)

#(493, 'R GOSH PEEP_cmH2O')
#print_allunique(flowsheet, 493)
make_new_columns('R GOSH PEEP_cmH2O', flowsheet, ['PEEP'], True)

#(494, 'R VENT PEEP_cm H2O')
#print_allunique(flowsheet, 494)
make_new_columns('R VENT PEEP_cm H2O', flowsheet, ['PEEP'], True)

#(495, 'R GOSH ICU PEEP 8')
#print_allunique(flowsheet, 495)

#(496, 'R GOSH IP VENT PEEP SETTING_cm H2O')
#print_allunique(flowsheet, 496)
make_new_columns('R GOSH IP VENT PEEP SETTING_cm H2O', flowsheet, ['PEEP'], True)

#(497, 'R AN VENT PEEP_cm H20')
#print_allunique(flowsheet, 497)
make_new_columns('R AN VENT PEEP_cm H20', flowsheet, ['PEEP'], True)



#Some weight corrected values
O2nas = flowsheet['O2Flow/kg'].isna()
flowsheet.loc[O2nas, 'O2Flow/kg'] = flowsheet.loc[O2nas, 'O2Flow']/flowsheet['interpolated_wt_kg'] 




##Now fill missing values
fill_missing('Ventilation', flowsheet, 'Ventilation_missing')
fill_missing('O2Flow/kg', flowsheet, 'O2Flow/kg_missing')
fill_missing('IPAP', flowsheet, 'IPAP_missing')
fill_missing('EPAP', flowsheet, 'EPAP_missing')
fill_missing('FiO2', flowsheet, 'FiO2_missing')
fill_missing('PEEP', flowsheet, 'PEEP_missing')
fill_missing('HFO', flowsheet, 'HFO_missing')
fill_missing('Tracheostomy', flowsheet, 'Tracheostomy_missing')
fill_missing('Ventilation(ml)', flowsheet, 'Ventilation(ml)_missing')
fill_missing('MeanAirwayPressure', flowsheet, 'MeanAirwayPressure_missing')

#Now do interpolation
interpolate_cols('Ventilation', flowsheet, 'pad', how_far = 60, limit_direction = 'forward', fill = 0)
interpolate_cols('O2Flow/kg', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)
interpolate_cols('IPAP', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)
interpolate_cols('EPAP', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)
interpolate_cols('FiO2', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0.21)
interpolate_cols('PEEP', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)
interpolate_cols('HFO', flowsheet, 'pad', how_far = 60, limit_direction = 'forward', fill = 0)
interpolate_cols('Tracheostomy', flowsheet, 'pad', how_far = 60, limit_direction = 'forward', fill = 0)
interpolate_cols('Ventilation(ml)', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)
interpolate_cols('MeanAirwayPressure', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)

flowsheet.to_csv('~/Project/Project_data/files/flowsheet_ventilation_interpolated.csv')
#### Work on BP now

"""
This column is just going to be BP as pulled from all other columns containing what appears to be BP
I also wanted to include CVP but I'm unclear about these values - they appear to be in the 100s which would be 
an unreasonable value

For some reason there appears to be a problem with the MAP column - I if this is difficult to solve I will just 
calculate this from the SBP and DBP columns - should possibly change the function so it doesn't automatically produce
a MAP/ have the function calls not to produce MAP
"""

#Make a unified BP column
#4, 'R AN ARTERIAL BLOOD PRESSURE'
#print_allunique(flowsheet, 4)
conv_BP('R AN ARTERIAL BLOOD PRESSURE', flowsheet, 3, 'SysBP', 'DiaBP')

#5, 'R AN MEAN ARTERIAL BLOOD PRESSURE_mmHg'
#print_allunique(flowsheet, 5)
conv_BP('R AN MEAN ARTERIAL BLOOD PRESSURE_mmHg', flowsheet, False, 'MAP')

#26, 'R AN CENTRAL AORTIC PRESSURE (CAP)'
#print_allunique(flowsheet, 26)
conv_BP('R AN CENTRAL AORTIC PRESSURE (CAP)', flowsheet, 3, 'SysBP', 'DiaBP')

#27, 'R AN CENTRAL AORTIC PRESSURE (CAP) MEAN_mmHg'
#print_allunique(flowsheet, 27)
conv_BP('R AN CENTRAL AORTIC PRESSURE (CAP) MEAN_mmHg', flowsheet, False, 'MAP')

#(36, 'R ARTERIAL LINE WAVEFORM') - not used
#print_allunique(flowsheet, 36)

#37, 'R GOSH IP MEAN ARTERIAL BLOOD PRESSURE_mmHg')
#print_allunique(flowsheet, 37)
conv_BP('R GOSH IP MEAN ARTERIAL BLOOD PRESSURE_mmHg', flowsheet, False, 'MAP')

#39, 'R ARTERIAL LINE BLOOD PRESSURE')
#print_allunique(flowsheet, 39)
conv_BP('R ARTERIAL LINE BLOOD PRESSURE', flowsheet, 3, 'SysBP', 'DiaBP')

#(40, 'R MAP A-LINE_mmHg')
#print_allunique(flowsheet, 40)
conv_BP('R MAP A-LINE_mmHg', flowsheet, False, 'MAP')

#(68, 'R GOSH IP NEONATAL PAIN BLOOD PRESSURE') - not using
#print_allunique(flowsheet, 68)

#(69, 'R GOSH PEWS BP SCORE')  - PEWs score
#print_allunique(flowsheet, 69)

#(79, 'R PED BOYS DIASTOLIC BP PERCENTILE_%') - centile only
#print_allunique(flowsheet, 79)

#(80, 'R PED BOYS SYSTOLIC BP PERCENTILE_%')

#(81, 'BLOOD PRESSURE')
#print_allunique(flowsheet, 81)
conv_BP('BLOOD PRESSURE', flowsheet, 3, 'SysBP', 'DiaBP')

#(161, 'R PED DIASTOLIC BP PERCENTILE_%')
#print_allunique(flowsheet, 161)

#(162, 'R GOSH CV HAEMO DIASTOLE 1 NEW_mmHg')
#print_allunique(flowsheet, 162)

#(163, 'R GOSH CV HAEMO DIASTOLE 1_mmHg')
#print_allunique(flowsheet, 163)

#(164, 'R GOSH CV HAEMO DIASTOLE 2_mmHg')
#print_allunique(flowsheet, 164)

#(192, 'R GOSH CV HAEMO EDP 1 NEW_mmHg')
#print_allunique(flowsheet, 192)

#(193, 'R GOSH CV HAEMO EDP 1_mmHg')
#print_allunique(flowsheet, 193)

#(255, 'R PED GIRLS DIASTOLIC BP PERCENTILE_%')
#print_allunique(flowsheet, 255)

#(256, 'R PED GIRLS SYSTOLIC BP PERCENTILE_%')
#print_allunique(flowsheet, 256)

#(361, 'R BP LOCATION')
#print_allunique(flowsheet, 361)

#(385, 'R GOSH CV HAEMO MEAN 1_mmHg')
#print_allunique(flowsheet, 385)

#(386, 'R GOSH CV HAEMO MEAN 2_mmHg')
#print_allunique(flowsheet, 386)

#(411, 'R AN NIBP MEAN_mmHg')
#print_allunique(flowsheet, 411)
conv_BP('R AN NIBP MEAN_mmHg', flowsheet, False, 'MAP')



"""
(623, 'R GOSH CV HAEMO SYSTOLE 1 NEW_mmHg')
(624, 'R GOSH CV HAEMO SYSTOLE 1_mmHg')
(625, 'R GOSH CV HAEMO SYSTOLE 2_mmHg')
(621, 'R PED SYSTOLIC BP PERCENTILE_%')
(626, 'R IP DOPPLER BLOOD PRESSURE')
(627, 'R PIM SYSTOLIC BLOOD PRESSURE_mmHg')
(638, 'R GOSH ICU TARGET DIASTOLIC MAX_mmHg')
(639, 'R GOSH ICU TARGET DIASTOLIC MIN_mmHg')
"""


"""
(636, 'R GOSH ICU TARGET CVP MAX_mmHg')
(637, 'R GOSH ICU TARGET CVP MIN_mmHg')
"""
#(141, 'R AN CVP MEAN_mmHg')
#print_allunique(flowsheet, 141)

#(142, 'R DEVICE CVP MEAN_mmHg')
##print_allunique(flowsheet, 142)

fill_missing('SysBP', flowsheet, 'SysBP_missing')
fill_missing('DiaBP', flowsheet, 'DiaBP_missing')
fill_missing('MAP', flowsheet, 'MAP_missing')

#Possibly need a better solution for zero fill here? maybe expected normal for age?
def fill_median_BP(column, sheet):
    """Function to fill in median values where BP missing despite interpolation
    Not finished
    """

    column = return_colname(column, sheet)
    ages = np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 15, 25])
    SysBP_medians = np.array([])
    switcher = {}
    

interpolate_cols('SysBP', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)
interpolate_cols('DiaBP', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)
interpolate_cols('MAP', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)






#### Now do HR
get_relevant_cols(flowsheet, 'HR| rate')

#R GOSH IP HEART RATE ECG_beats per minute
make_new_columns('R GOSH IP HEART RATE ECG_beats per minute', flowsheet, ['HR'], True)
#R AN HEART RATE ECG_beats per minute
make_new_columns('R AN HEART RATE ECG_beats per minute', flowsheet, ['HR'], True)
#R AN SPO2 HR PULSE
make_new_columns('R AN SPO2 HR PULSE', flowsheet, ['HR'], True)
#R GOSH IP HEART RATE PLETHYSMOGRAM
make_new_columns('R GOSH IP HEART RATE PLETHYSMOGRAM', flowsheet, ['HR'], True)

fill_missing('HR', flowsheet, 'HR_missing')

#Should probably do something as above with median HR
interpolate_cols('HR', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 0)

#GCS/ AVPU
print_all(flowsheet)
get_relevant_cols(flowsheet, 'GCS|AVPU|Comfort|Consciousness|Alert|Pupils|GLASGOW COMA SCALE|PEWS')

#Comfort score
#R GOSH COMFORT ALERTNESS 22
make_new_columns('R GOSH COMFORT ALERTNESS', flowsheet, ['Comfort:Alertness'], True)
#R GOSH COMFORT BP 84
make_new_columns('R GOSH COMFORT BP', flowsheet, ['Comfort:BP'], True)
#R GOSH COMFORT CALMNESS
make_new_columns('R GOSH COMFORT CALMNESS', flowsheet, ['Comfort:Calmness'], True)
#R GOSH COMFORT SCORE
make_new_columns('R GOSH COMFORT SCORE', flowsheet, ['Comfort'], True)
#R GOSH IP COMFORT SCORE FOR GOSH GO
make_new_columns('R GOSH IP COMFORT SCORE FOR GOSH GO', flowsheet, ['Comfort'], True)
#R GOSH COMFORT HR
make_new_columns('R GOSH COMFORT HR', flowsheet, ['Comfort:HR'], True)
#R GOSH COMFORT RESP NO VENT - Consider this as surrogate for no vent/ vent below?
make_new_columns('R GOSH COMFORT RESP NO VENT', flowsheet, ['Comfort:Resp'], True)
#R GOSH COMFORT RESP VENT
make_new_columns('R GOSH COMFORT RESP VENT', flowsheet, ['Comfort:Resp'], True)
#Alertness
#make_new_columns(867, flowsheet, ['Comfort:Alertness'], True)

#R GOSH IP AVPU SCORE
make_new_columns('R GOSH IP AVPU SCORE', flowsheet, ['AVPU'], newcols1 = [['U'], ['P'], ['V'], ['A']])
#R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST AUD
make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST AUD', flowsheet, ['GCS_V'], True)
#R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST AU
make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST AU', flowsheet, ['GCS_V'], True)
#R NSR GLASGOW COMA SCALE BEST EYE RESPONSE
make_new_columns('R NSR GLASGOW COMA SCALE BEST EYE RESPONSE', flowsheet, ['GCS_E'], True)

#R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST MOT
motor60 = [[], ['1', 1], ['2', 2], [3.0, '3'], [4.0, '4', 'P'], ['5', 5.0], [6.0, '6']]
make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST MOT', flowsheet, ['GCS_M'], newcols1 = motor60)
#R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST MO
make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST MO', flowsheet, ['GCS_M'], newcols1 = motor60)
#R GLASGOW COMA SCALE BEST MOTOR RESPONSE
make_new_columns('R GLASGOW COMA SCALE BEST MOTOR RESPONSE', flowsheet, ['GCS_M'], newcols1 = motor60)
#R NSR GLASGOW COMA SCALE BEST MOTOR RESPONSE
make_new_columns('R NSR GLASGOW COMA SCALE BEST MOTOR RESPONSE', flowsheet, ['GCS_M'], newcols1 = motor60)

#R GLASGOW COMA SCALE BEST VERBAL RESPONSE
make_new_columns('R GLASGOW COMA SCALE BEST VERBAL RESPONSE', flowsheet, ['GCS_V'], newcols1 = motor60)

#R GLASGOW COMA SCALE EYE OPENING
eye223 = [[], ['1', 1], ['2', 2, 'P'], [3.0, '3'], [4.0, '4']]
make_new_columns('R GLASGOW COMA SCALE EYE OPENING', flowsheet, ['GCS_E'], newcols1 = eye223)

#R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) EYE OPEN
make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) EYE OPEN', flowsheet, ['GCS_E'], newcols1 = eye223)

#R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) EYE OPE
make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) EYE OPE', flowsheet, ['GCS_E'], newcols1 = eye223)

#R GLASGOW COMA SCALE SCORE
make_new_columns('R GLASGOW COMA SCALE SCORE', flowsheet, ['GCS'], True)

#R NSR GLASGOW COMA SCALE SCORE
make_new_columns('R NSR GLASGOW COMA SCALE SCORE', flowsheet, ['GCS'], True)
#R PEDSCPN GLASGOW COMA SCALE SCORE (AGE 6 MONTHS TO 2 YEARS)
make_new_columns('R PEDSCPN GLASGOW COMA SCALE SCORE (AGE 6 MONTHS TO 2 YEARS)', flowsheet, ['GCS'], True)
#R PEDSCPN GLASGOW COMA SCALE SCORE (AGE GREATER THAN 2 YEARS)
make_new_columns('R PEDSCPN GLASGOW COMA SCALE SCORE (AGE GREATER THAN 2 YEARS)', flowsheet, ['GCS'], True)


#Make GCS column sum of other columns where not filled
GCS_nas = flowsheet['GCS'].isna()
flowsheet.loc[GCS_nas, 'GCS'] = flowsheet.loc[GCS_nas, 'GCS_E'] + flowsheet.loc[GCS_nas, 'GCS_M'] + flowsheet.loc[GCS_nas, 'GCS_V']

fill_missing('GCS', flowsheet, 'GCS_missing')
fill_missing('AVPU', flowsheet, 'AVPU_missing')
interpolate_cols('GCS', flowsheet, 'pad', how_far = 60, limit_direction = 'forward', fill = 15)
interpolate_cols('AVPU', flowsheet, 'pad', how_far = 60, limit_direction = 'forward', fill = 3)

#CRT
print_all(flowsheet)
get_relevant_cols(flowsheet, 'Cap|Refill|CRT|Time')
#R CAPILLARY REFILL: GENERAL
crt98 = [[], [], ['Less than/equal to 2 seconds (All extremities)'], [], ['Greater than 2 seconds (All extremities)']]
make_new_columns('R CAPILLARY REFILL: GENERAL', flowsheet, ['CRT'], newcols1 = crt98)

#R CAPILLARY REFILL: ICU
crt99 = [[], [], ['<2', 'Brisk'], ['2 - 3'], ['3 - 5', 'Sluggish'], ['>5']]
make_new_columns('R CAPILLARY REFILL: ICU', flowsheet, ['CRT'], newcols1 = crt99)

#R CAPILLARY REFILL: SECONDS
crt100 = [[], [], ['Less than 2 seconds'], ['3 seconds'], ['4 seconds'], ['5 seconds'], ['Greater than 6 seconds']]
make_new_columns('R CAPILLARY REFILL: SECONDS', flowsheet, ['CRT'], newcols1 = crt100)

#R PVS CAPILLARY REFILL LLE
crt357 = [[], [1, '2'], [2, '2'], [3, '3'], [4, '4', '4+']]
make_new_columns('R PVS CAPILLARY REFILL LLE', flowsheet, ['CRT'], newcols1 = crt357)

#R PVS CAPILLARY REFILL RLE
make_new_columns('R PVS CAPILLARY REFILL RLE', flowsheet, ['CRT'], newcols1 = crt357)

fill_missing('CRT', flowsheet, 'CRT_missing')
interpolate_cols('CRT', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 2)



####Sats
print_all(flowsheet)
get_relevant_cols(flowsheet, 'sat|spo2|oxygen|\%|percent')
# R EVENT SPO2
make_new_columns('R EVENT SPO2', flowsheet, ['SpO2'], True)
#R PERF NORMALISED SAO2_%
make_new_columns('R PERF NORMALISED SAO2_%', flowsheet, ['SpO2'], True)
#R GOSH CV SATURATION NEW_%
make_new_columns('R GOSH CV SATURATION NEW_%', flowsheet, ['SpO2'], True)
#PERF PRIME BG POC SO2_%
make_new_columns('PERF PRIME BG POC SO2_%', flowsheet, ['SpO2'], True)
#ANAESTHESIA PULSE OXIMETRY_%
make_new_columns('ANAESTHESIA PULSE OXIMETRY_%', flowsheet, ['SpO2'], True)
#PULSE OXIMETRY_%
make_new_columns('PULSE OXIMETRY_%', flowsheet, ['SpO2'], True)
#R GOSH IP PULSE OXIMETRY ARTERIAL SPO2L_%
make_new_columns('R GOSH IP PULSE OXIMETRY ARTERIAL SPO2L_%', flowsheet, ['SpO2'], True)
#R GOSH IP PULSE OXIMETR ARTERIAL SPO2R_%
make_new_columns('R GOSH IP PULSE OXIMETR ARTERIAL SPO2R_%', flowsheet, ['SpO2'], True)
#R PULSE OXIMETRY PRE-DUCTAL_%
make_new_columns('R PULSE OXIMETRY PRE-DUCTAL_%', flowsheet, ['SpO2'], True)
#R CV SYSTEMIC ARTERIAL O2 SATURATION_%
make_new_columns('R CV SYSTEMIC ARTERIAL O2 SATURATION_%', flowsheet, ['SpO2'], True)

fill_missing('SpO2', flowsheet, 'SpO2_missing')
interpolate_cols('SpO2', flowsheet, 'linear', how_far = 60, limit_direction = 'both', fill = 95)



#Height
get_relevant_cols(flowsheet, 'height|ht|metres|cm')
#HEIGHT_centimetres
make_new_columns('HEIGHT_centimetres', flowsheet, ['Height(cm)'], True)
#R GOSH IP HEIGHT FOR WEIGHT
make_new_columns('R GOSH IP HEIGHT FOR WEIGHT', flowsheet, ['Height(cm)'], True)

apply_wt_ht('Height(cm)', flowsheet, 'interpolated_ht_m', scaling = 0.01)


#Input/ output
#Input
get_relevant_cols(flowsheet, 'input|intake|fluid|output|nappy')
get_relevant_cols(flowsheet, 'loss')
flowsheet['input'] = 0
flowhseet['output'] = 0

#R DRY NAPPY WEIGHT_g 182
#R NAPPY WITH STOOL WEIGHT_g
conv_2cols('R DRY NAPPY WEIGHT_g', 'R NAPPY WITH STOOL WEIGHT_g', flowsheet, 'Nappy_output', bminusa)
#R NAPPY WEIGHT WITH URINE_g
conv_2cols('R DRY NAPPY WEIGHT_g', 'R NAPPY WEIGHT WITH URINE_g', flowsheet, 'Nappy_output', bminusa)
#401 For R NAPPY WEIGHT URINE AND STOOL_g
conv_2cols('R DRY NAPPY WEIGHT_g', 'R NAPPY WEIGHT URINE AND STOOL_g', flowsheet, 'Nappy_output', bminusa)
sum_inout('Nappy_output', flowsheet, 'output')

#R GOSH IP DIET TOTAL FLUID
sum_inout('R GOSH IP DIET TOTAL FLUID', flowsheet, 'input')

#R ONC APHERESIS FLUID BALANCE
sum_inout('R ONC APHERESIS FLUID BALANCE', flowsheet, 'input')

#INTRAVENOUS INTAKE_mL
sum_inout('INTRAVENOUS INTAKE_mL', flowsheet, 'input')

#R AN NG/OG TUBE OUTPUT_mL
sum_inout('R AN NG/OG TUBE OUTPUT_mL', flowsheet, 'output')

#OTHER OUTPUT_mL
sum_inout('OTHER OUTPUT_mL', flowsheet, 'output')

#R CHEST TUBE OUTPUT_mL
sum_inout('R CHEST TUBE OUTPUT_mL', flowsheet, 'output')

#R GOSH LDA WOUND OUTPUT VOLUME_mL
sum_inout('R GOSH LDA WOUND OUTPUT VOLUME_mL', flowsheet, 'output')

#R DRAIN OUTPUT_mL
sum_inout('R DRAIN OUTPUT_mL', flowsheet, 'output')

#R GOSH URINE DRAIN OUTPUT_mL
sum_inout('R GOSH URINE DRAIN OUTPUT_m', flowsheet, 'output')

#R OSTOMY OUTPUT TYPE_mL
sum_inout('R OSTOMY OUTPUT TYPE_mL', flowsheet, 'output')

#R TUBE OUTPUT_mL
sum_inout('R TUBE OUTPUT_mL', flowsheet, 'output')

#R URINE OUTPUT_m
sum_inout('R URINE OUTPUT_m', flowsheet, 'output')

#R CRRT PROGRAMMED FLUID LOSS_mL/hr - need to check if this is every min or hour
sum_inout('R CRRT PROGRAMMED FLUID LOSS_mL/hr', flowsheet, 'output')

#R GOSH IP CRRT FLUID REMOVAL_mL
sum_inout('R GOSH IP CRRT FLUID REMOVAL_mL', flowsheet, 'output')

#URINE OUTPUT_mL
sum_inout('URINE OUTPUT_mL', flowsheet, 'output')

#R GOSH IP I/O ORAL INTAKE VOL ML_ml
sum_inout('R GOSH IP I/O ORAL INTAKE VOL ML_ml', flowsheet, 'input')

#(718, 'R GOSH IP CONT FEED TOTAL_mL')
#(719, 'R GOSH IP CONTINUED FEED TOTAL VOLUME_mL')
"""(720, 'R GOSH IP IV FLUID MAINTENANCE ACCUMULATIVE VOLUME_mL')
(721, 'R GOSH IP TPN ALL IN ONE INFUSION ACCUMULATIVE VOLUME_mL')
(722, 'R GOSH IP TPN AQUEOUS INFUSION ACCUMULATIVE VOLUME_mL')
(723, 'R GOSH IP TPN LIPIDS INFUSION ACCUMULATIVE VOLUME_mL')
(745, 'R GOSH TOTAL VOLUME TPN_mL')
(775, 'R CATHETER URINE RETURNED')
(773, 'R URINE AMOUNT')
(791, 'G TPN ALL IN ONE INFUSION VOLUME_mL')
(792, 'R CHEMO IV VOLUME')
(793, 'R GOSH IP I/O ORAL INTAKE VOL ML_ml')
(794, 'R GOSH IP IO CONT FEED VOL_mL')
(796, 'R IP BLOOD ADMINISTRATION TOTAL VOLUME_mL')
(797, 'R MAINTENANCE IV BOLUS VOLUME')
(798, 'R MAINTENANCE IV VOLUME_mL')
"""
#Will need to double check about duplicates here
#May need to normalise by bodyweight?

#Other organ support
print_all(flowsheet)
"""
(803, 'R GOSH ADRENALINE HOURLY VOLUME_mL')
(835, 'R GOSH MILRINONE HOURLY VOLUME_mL')
(837, 'R GOSH NORADRENALINE HOURLY VOLUME_mL')
(853, 'R GOSH VASOPRESSIN HOURLY VOLUME_mL')
(815, 'R DOPAMINE VOLUME_mL')
Adrenaline 1:1 with norad
Milrinone not sure yet
Vasopressin 0.4:1 norad
Possibly just assume equivalence across milrinone range? like bottom dose = bottom of norad?
Bottom dose of milrinone is 30/mcg/kg/hr, of norad is 20/ng/kg/min or 1.2mcg/kg/hr
Therefore milrinone to norad conversion would be 1.2/30 or 1/25
"""
flowsheet['Inotropes'] = 0
sum_inout('R GOSH ADRENALINE HOURLY VOLUME_mL', flowsheet, 'Inotropes')
sum_inout('R GOSH NORADRENALINE HOURLY VOLUME_mL', flowsheet, 'Inotropes')

def corAVP(input):
    return input*2.5
conv_column('R GOSH VASOPRESSIN HOURLY VOLUME_mL', flowsheet, 'EquivAVP', corAVP)
sum_inout('EquivAVP', flowsheet, 'Inotropes')

def corDopamine(input):
    return input/100
conv_column('R DOPAMINE VOLUME_mL', flowsheet, 'EquivDopamine', corDopamine)
sum_inout('EquivDopamine', flowsheet, 'Inotropes')

def corMilrinone(input):
    return input/25
conv_column('R GOSH MILRINONE HOURLY VOLUME_mL', flowsheet, 'EquivMilrinone', corMilrinone)
sum_inout('EquivMilrinone', flowsheet, 'Inotropes')

def aoverb(a, b):
    return a/b
conv_2cols('Inotropes', 'interpolated_wt_kg', flowsheet, 'Inotropes_kg', aoverb)
fill_missing('Inotropes', flowsheet, 'Inotropes_missing')



### Resp rate
get_relevant_cols(flowsheet, 'rate|RR|resp')

#378 R GOSH VENT RATE
#print_allunique(flowsheet, 378)

#567 ANAESTHESIA ECG RESPIRATION RATE
#print_allunique(flowsheet, 567)
make_new_columns('ANAESTHESIA ECG RESPIRATION RATE', flowsheet, ['RR'], True)

#566 ANAESTHESIA AIRWAY RESPIRATION RATE
#print_allunique(flowsheet, 566)
make_new_columns('ANAESTHESIA AIRWAY RESPIRATION RATE', flowsheet, ['RR'], True)

#559 RESPIRATIONS
#print_allunique(flowsheet, 559)
make_new_columns('RESPIRATIONS', flowsheet, ['RR'], True)

#379 For R GOSH VENT RATE OBSERVED
#print_allunique(flowsheet, 379)
fill_missing('RR', flowsheet, 'RR_missing')

#Need to do something about normals here too
interpolate_cols('RR', flowsheet, 'linear', how_far = 60, limit_direction = 'forward', fill = 15)
flowsheet.to_csv('~/Project/Project_data/files/flowsheet_nearly_interpolated.csv')

flowsheet = pd.read_csv('~/Project/Project_data/files/flowsheet_nearly_interpolated.csv', parse_dates = ['taken_datetime'])
### Get demographics
apply_demographic('birth_date', demographics, flowsheet, 'Age_yrs')
flowsheet['time_to_death'] = np.timedelta64(70, 'Y')
apply_demographic('death_date', demographics, flowsheet, 'time_to_death')
apply_demographic('deceased_flag', demographics, flowsheet, 'died', False)
apply_demographic('sex', demographics, flowsheet, 'sex', False)
apply_demographic('ethnicity_name', demographics, flowsheet, 'ethnicity', False)

flowsheet.to_csv('/local/data/public/2020/dfs28/PICU_data/flowsheet_demographics_applied.csv')






### Ecmo and dialysis


#Dialysis
get_relevant_cols(flowsheet, 'CRRT|dialysis|PD|HDF')

#231 R GOSH CRRT FILTER TYPE
#print_allunique(flowsheet, 231)
make_new_columns('R GOSH CRRT FILTER TYPE', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH CRRT FILTER TYPE', 2, 2))

#232 R GOSH CRRT FF_%
#print_allunique(flowsheet, 232)
make_new_columns('R GOSH CRRT FF_%', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH CRRT FF_%', 2, 2))

#576 R CRRT PROGRAMMED FLUID LOSS_mL/hr
#print_allunique(flowsheet, 576)
make_new_columns('R CRRT PROGRAMMED FLUID LOSS_mL/hr', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R CRRT PROGRAMMED FLUID LOSS_mL/hr', 2, 2))

#577 R GOSH IP CRRT FLUID REMOVAL_mL
#print_allunique(flowsheet, 577)
make_new_columns('R GOSH IP CRRT FLUID REMOVAL_mL', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH IP CRRT FLUID REMOVAL_mL', 2, 2))

#578 R GOSH CRRT END OF CYCLE VOL_mL
#print_allunique(flowsheet, 578)
make_new_columns('R GOSH CRRT END OF CYCLE VOL_mL', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH CRRT END OF CYCLE VOL_mL', 2, 2))

#733 R GOSH PD TOTAL VOLUME_mL
#print_allunique(flowsheet, 733)
make_new_columns('R GOSH PD TOTAL VOLUME_mL', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH PD TOTAL VOLUME_mL', 2, 2))

#766 R GOSH UF VOLUME HD/HDF
#print_allunique(flowsheet, 766)
make_new_columns('R GOSH UF VOLUME HD/HDF', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH UF VOLUME HD/HDF', 2, 2))

#769 R GOSH CRRT UF
#print_allunique(flowsheet, 769)
make_new_columns('R GOSH CRRT UF', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH CRRT UF', 2, 2))
fill_missing('dialysis', flowsheet, 'dialysis_missing')
interpolate_cols('dialysis', flowsheet, 'pad', how_far = 60, limit_direction = 'forward', fill = 0)

#ECMO
get_relevant_cols(flowsheet, 'ECMO')
make_new_columns('R ECMO  FLOW PROBE', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 147, 2, 2))
make_new_columns('R GOSH LDA ECMO LINE STATUS', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 185, 2, 2))
make_new_columns('R ECMO PUMP FLOW (L/MIN)_L/min', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 186, 2, 2))
make_new_columns('R ECMO CLAMPS CHECK', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 188, 2, 2))
make_new_columns('GOSH ICU ECMO HCT', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 190, 2, 2))
make_new_columns('R ECMO ON/OFF', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 279, 2, 2))
make_new_columns('R ECMO HEAT EXCHANGER WATER LEVEL', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 280, 2, 2))
make_new_columns('R HOURS ON ECMO', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 295, 2, 2))
make_new_columns('R GOSH ECMO NUMBER CIRCUITS', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 423, 2, 2))
fill_missing('ECMO', flowsheet, 'ECMO_missing')
interpolate_cols('ECMO', flowsheet, 'pad', how_far = 60, limit_direction = 'forward', fill = 0)

### Pull only patients which have enough data?


### Save the data so that can run childsds in R
flowsheet.to_csv('/local/data/public/2020/dfs28/PICU_data/flowsheet_final_output.csv')

