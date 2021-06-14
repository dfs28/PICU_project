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
flowsheet = pd.read_csv('Project/Project_data/files/caboodle_patient_selected_flowsheetrows_main_pivot.csv', sep= ',', parse_dates=['taken_datetime'])
demographics = pd.read_csv('Project/Project_data/files/caboodle_patient_demographics.csv', sep = ',', parse_dates = ['birth_date', 'death_date'])








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

#This takes ages so I've saved it, only rerun if needed
if (os.path.exists('Project/Project_data/files/not_NA.csv') == False):
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
    
    np.savetxt('Project/Project_data/files/not_NA.csv', numNotNa, delimiter=',')
else: 
    numNotNa = np.genfromtxt('Project/Project_data/files/not_NA.csv', delimiter=',')

topcols = colnames[np.append(np.array([False, False, False]), numNotNa > 10000)]
for i in range(len(topcols)):
    print(topcols[i])
sortedNaNs = np.argsort(numNotNa)




#Split up data hour by hour?


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











#### Now do some feature engineering

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
make_new_columns(14, flowsheet, ['Ventilation'], newcol1 = ventilated14)
HFO14 = [[], ['HFO', 'HFO;NO']]
make_new_columns(14, flowsheet, ['HFO'], newcol1 = HFO14)

#'R GOSH IP PANDA AIRWAY' 15
#print_allunique(flowsheet, 15)
ventilated15 = [[], [], [], ['Intubation']]
make_new_columns(15, flowsheet, ['Ventilation'], newcol1 = ventilated15)


#'R GOSH NICU AIRWAY STATUS' 16
#print_allunique(flowsheet, 16)
ventilated16 = [[], [], [], ['Conventional ventilation']]
make_new_columns(16, flowsheet, ['Ventilation'], newcol1 = ventilated16)

#'R GOSH RESUS ABCDE AIRWAY' 17
#print_allunique(flowsheet, 17)
ventilated16 = [[],[], [], ['Intubated', 'Intubated;Opening manoeuvres', 'Intubated;Patent']]
make_new_columns(17, flowsheet, ['Ventilation'], newcol1 = ventilated16)

#'R ED AIRWAY (WDL)' 18 - not using
#print_allunique(flowsheet, 18)

#'R AIRWAY LDA INSERTION ATTEMPTS' 19
#print_allunique(flowsheet, 19)
ventilated19 = [[], [], [], [1, 2, '1', '2', '3 or more']]
make_new_columns(19, flowsheet, ['Ventilation'], newcol1 = ventilated19)

#'R GOSH IP CATS AIRWAY STATUS' 20
#print_allunique(flowsheet, 20)
ventilated20 = [[], [], [],  ['Intubated']]
make_new_columns(20, flowsheet, ['Ventilation'], newcol1 = ventilated20)

#R VENT ALARM APNOEA SECONDS_Seconds 28 - come back to this one - could be NIV or 
#print_allunique(flowsheet, 28)

#R IP VENT APNOEA RATE 30

#R IP VENT APNOEA VOLUME_mL 31

#34, 'R GOSH IP ARE THEY INTUBATED AND VENTILATED?'
#print_allunique(flowsheet, 34)
ventilated34 = [[], [], ['No - Non-invasive Ventilation', 'No - Optiflow'], ['Yes - Invasive ventilation']]
make_new_columns(34, flowsheet, ['Ventilation'], newcol1 = ventilated34)

#127, 'R GOSH CPAP CMH2O_cmH2O'
#print_allunique(flowsheet, 127)
make_new_columns(127, flowsheet, ['IPAP'], True)
make_new_columns(127, flowsheet, ['EPAP'], True)

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
make_new_columns(136, flowsheet, ['Tracheostomy'], newcol1 = trache136)

#198, 'R GOSH ASTRAL EPAP_cm H2O') - Astral is NIV
#print_allunique(flowsheet, 198)
unique_vent198 = flowsheet.iloc[:, 198].unique()
vent198 = [[], [], unique_vent198[range(1, len(unique_vent198))].tolist(), []]
make_new_columns(198, flowsheet, ['Ventilation'], newcol1 = vent198)
make_new_columns(198, flowsheet, ['EPAP'], True)

#199, 'R GOSH EPAP'
#print_allunique(flowsheet, 199)
make_new_columns(199, flowsheet, ['EPAP'], True)

#200, 'R GOSH NIPPY EPAP_cmH2O
#print_allunique(flowsheet, 200)
unique_vent200 = flowsheet.iloc[:, 200].unique()
vent200 = [[], [], unique_vent200[range(1, len(unique_vent200))].tolist(), []]
make_new_columns(200, flowsheet, ['Ventilation'], newcol1 = vent200)
make_new_columns(200, flowsheet, ['EPAP'], True)

#204, 'R GOSH ICU END TIDAL CO2_% Should work out if this means they are ventilated
#print_allunique(flowsheet, 204)
make_new_columns(204, flowsheet, ['ETCO2'], True)

#205, 'R VENT ETCO2_kPa Should work out if this means they are ventilated
#print_allunique(flowsheet, 205)
make_new_columns(205, flowsheet, ['ETCO2'], True)

#206, 'R IP VENT ALARM ETCO2/TCM MAX_mmHg
#print_allunique(flowsheet, 206)
make_new_columns(206, flowsheet, ['ETCO2'], True)

#207, 'R IP VENT ALARM ETCO2/TCM MIN_mmHg
#print_allunique(flowsheet, 207)

#208, 'R GOSH ICU TRANSFER ETT
#print_allunique(flowsheet, 208)

#209, 'R GOSH ICU TRANSFER ETT SECURE
#print_allunique(flowsheet, 209)

#210, 'R ETT TO LIP_cm
#print_allunique(flowsheet, 210)
unique_vent210 = flowsheet.iloc[:, 210].unique()
vent210 = [[], [], [], unique_vent210[range(1, len(unique_vent210))].tolist()]
make_new_columns(210, flowsheet, ['Ventilation'], newcol1 = vent210)

#211, 'R ETT TO NOSE_cm'
#print_allunique(flowsheet, 211)
unique_vent211 = flowsheet.iloc[:, 211].unique()
vent211 = [[], [], [], unique_vent211[range(1, len(unique_vent211))].tolist()]
make_new_columns(211, flowsheet, ['Ventilation'], newcol1 = vent211)

#213, 'R EVENT SPO2'
#print_allunique(flowsheet, 213)

#214, 'R AN AGENTS DESFLURANE EXPIRED_%'
#print_allunique(flowsheet, 214)
unique_vent214 = flowsheet.iloc[:, 214].unique()
vent214 = [[], [], [], unique_vent214[range(1, len(unique_vent214))].tolist()]
make_new_columns(214, flowsheet, ['Ventilation'], newcol1 = vent214)

#215, 'R AN AGENTS ISOFLURANE EXPIRED_%'
#print_allunique(flowsheet, 215)
unique_vent215 = flowsheet.iloc[:, 215].unique()
vent215 = [[], [], [], unique_vent215[range(1, len(unique_vent215))].tolist()]
make_new_columns(215, flowsheet, ['Ventilation'], newcol1 = vent215)

#216, 'R AN EXPIRED MINUTE VOLUME_L/min'
#print_allunique(flowsheet, 216)
make_new_columns(216, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 216, 4, 4))

#217, 'R AN EXPIRED N2O_%'\\
#print_allunique(flowsheet, 217)
make_new_columns(217, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 217, 4, 4))

#218, 'R AN AGENTS SEVOFLURANE EXPIRED_%'
#print_allunique(flowsheet, 218)
make_new_columns(218, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 218, 4, 4))

#219, 'R GOSH EXP TV LITRES_L' - should consider expired tidal volume?
#print_allunique(flowsheet, 219)
make_new_columns(219, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 219, 4, 4))

#220, 'R VENT EXP TIDAL VOLUME_mL'
#print_allunique(flowsheet, 220)
make_new_columns(220, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 220, 4, 4))

#222, 'R GOSH ICU WITHDRAWAL EXTUBATION'
#print_allunique(flowsheet, 222)

#234, 'R AN FIO2_%'
#print_allunique(flowsheet, 234)
cor_FiO2(234, flowsheet, 'FiO2')

#235, 'R PERF FIO2_%'
#print_allunique(flowsheet, 235)
cor_FiO2(234, flowsheet, 'FiO2')

#236, 'R GOSH ICU SANDWICH FIO2'
#print_allunique(flowsheet, 236)

#237, 'R INSPIRED O2 FIO2 FOR NO_%' - FIO2, will need to correct
#print_allunique(flowsheet, 237)

#238, 'R GOSH IP FIO2 SETTING_%'
#print_allunique(flowsheet, 238)
cor_FiO2(238, flowsheet, 'FiO2')

#239, 'R VENT INSP FLOWS_L/min' #Should probably do this inspired o2 corrected by bodyweight thing
#print_allunique(flowsheet, 239)
make_new_columns(239, flowsheet, ['O2Flow'], True)

#240, 'R GOSH VENT FLOW ML/KG_mL/Kg'
#print_allunique(flowsheet, 240)
make_new_columns(240, flowsheet, ['O2Flow/kg'], True)

#241, 'R IP VENT FLOW OBS_L/min'
#print_allunique(flowsheet, 241)
make_new_columns(241, flowsheet, ['O2Flow'], True)

#242, 'R GOSH AIRVO FLOW_L/min' Highflow o2
#print_allunique(flowsheet, 242)
make_new_columns(241, flowsheet, ['O2Flow'], True)
make_new_columns(241, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 241, 2, 4))

#R GOSH VENT FABIAN O2 THERAPY FLOW_L/min HFNC/ NIV
#print_allunique(flowsheet, 'R GOSH VENT FABIAN O2 THERAPY FLOW_L/min')
make_new_columns('R GOSH VENT FABIAN O2 THERAPY FLOW_L/min', flowsheet, ['O2Flow'], True)
make_new_columns('R GOSH VENT FABIAN O2 THERAPY FLOW_L/min', flowsheet, ['Ventilation'], 
                newcol1 = make_allnonNa(flowsheet, 'R GOSH VENT FABIAN O2 THERAPY FLOW_L/min', 2, 4))

#(284, 'R GOSH HFO BASE FLOW READING')
#print_allunique(flowsheet, 284)
make_new_columns(284, flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 284, 2, 2))
make_new_columns(284, flowsheet, ['O2Flow'], True)

#285, 'R GOSH HFO BASE FLOW SETTING')
#print_allunique(flowsheet, 285)
make_new_columns(285, flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 285, 2, 2))
make_new_columns(285, flowsheet, ['O2Flow'], True)

#287, 'R GOSH HFO FREQUENCY READING_Hz')
#print_allunique(flowsheet, 287)
make_new_columns(287, flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 287, 2, 2))

#(288, 'R GOSH IP HFV FREQUENCY SETTING_Hz')
#print_allunique(flowsheet, 288)
make_new_columns(288, flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 288, 2, 2))

#(289, 'R GOSH IP HFV I:E RATIO SETTING') - should probably consider calculating I:E manually
#print_allunique(flowsheet, 289)

#(290, 'R GOSH IP HFV TIDAL VOLUME_mL')HFV is high frequency ventilation
#print_allunique(flowsheet, 290)
make_new_columns(290, flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 287, 2, 2))

#(291, 'R GOSH HIGH FIO2_%') 
#print_allunique(flowsheet, 291)
cor_FiO2(291, flowsheet, 'FiO2')

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
make_new_columns(317, flowsheet, ['IPAP'], True)

#(320, 'R VENT INSP TIME_Sec')
#print_allunique(flowsheet, 320)
make_new_columns(320, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,320,  4, 5))

#(322, 'R AN AGENTS DESFLURANE INSPIRED_%')
#print_allunique(flowsheet, 322)
make_new_columns(322, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,322,  4, 5))

#(323, 'R FIO2_%')
#print_allunique(flowsheet, 323)
make_new_columns(323, flowsheet, ['FiO2'], True)

#(324, 'R AN AGENTS ISOFLURANE INSPIRED_%')
#print_allunique(flowsheet, 324)
make_new_columns(324, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  324,  4, 5))

#(325, 'R AN INSPIRED N2O_%')
#print_allunique(flowsheet, 325)
make_new_columns(325, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  325,  4, 5))

#(326, 'R AN INSPIRED NO2_ppm')
#print_allunique(flowsheet, 326)
make_new_columns(326, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  326,  4, 5))

#(327, 'R AN AGENTS NITRIC OXIDE_ppm')
#print_allunique(flowsheet, 327)
make_new_columns(327, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  327,  4, 5))

#(328, 'R INSPIRED NO SETTING')
#(329, 'R AN INSPIRED O2 SETTING')
#(330, 'R AN AGENTS SEVOFLURANE INSPIRED_%')
#print_allunique(flowsheet, 330)
make_new_columns(330, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  330,  4, 5))

#(331, 'R GOSH IP INSPIRED TIDAL VOLUME_mL')
#(333, 'R GOSH IPAP')
#print_allunique(flowsheet, 333)
make_new_columns(333, flowsheet, ['IPAP'], True)


#(332, 'R GOSH ASTRAL IPAP_cm H2O')
#print_allunique(flowsheet, 332)
make_new_columns(332, flowsheet, ['IPAP'], True)

#(334, 'R GOSH NIPPY IPAP_cmH2O')
#print_allunique(flowsheet, 334)
make_new_columns(334, flowsheet, ['IPAP'], True)

#(335, 'R GOSH STELLA IPAP_cm H2O')
#print_allunique(flowsheet, 335)
make_new_columns(335, flowsheet, ['IPAP'], True)


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
#print_allunique(flowsheet, 377)
make_new_columns(377, flowsheet, ['Ventilation(ml)'], True)

"""
(378, 'R GOSH VENT RATE OBSERVED')
"""
#print_allunique(flowsheet, 378)
make_new_columns(378, flowsheet, ['Ventilation(ml)'], True)

"""
(379, 'R IP VENT VT MANDATORY (OBS)_mL')
"""

"""
(380, 'R MAP')
"""
#print_allunique(flowsheet, 380)
make_new_columns(380, flowsheet, ['MeanAirwayPressure'], True)

"""
(381, 'R IP VENT MEAN AIRWAY PRESSURE HIGH_cm H2O')
(382, 'R IP VENT MEAN AIRWAY PRESSURE LOW_cm H2O')
"""

#(387, 'R VENT MAP_cm H2O')
#print_allunique(flowsheet, 387)
make_new_columns(387, flowsheet, ['MeanAirwayPressure'], True)


#(388, 'R IP VENT MEAN AIRWAY PRESSURE (SET)')
#print_allunique(flowsheet, 388)
make_new_columns(388, flowsheet, ['MeanAirwayPressure'], True)

"""
(391, 'R VENT MINUTE VENTILATION_L/min')
(392, 'R GOSH MIN VOL LEAK_L/min')
(393, 'R GOSH IP MINUTE VOLUME SETTING_L/min')
"""

#(424, 'R AN AGENTS O2_L/min')
#print_allunique(flowsheet, 424)
make_new_columns(424, flowsheet, ['O2Flow'], True)

#427 R OXYGEN DELIVERY METHOD)
#print_allunique(flowsheet, 427)
 
 #(428, 'R RT OXYGEN FLOW RATE_L/min')
#print_allunique(flowsheet, 428)
make_new_columns(428, flowsheet, ['O2Flow'], True)

# 429, 'R OXYGEN FLOW RATE_L/min')
#print_allunique(flowsheet, 429)
make_new_columns(429, flowsheet, ['O2Flow'], True)

#(442, 'R GOSH AIRVO OXYGEN %_%')
#print_allunique(flowsheet, 442)
cor_FiO2(442, flowsheet, 'FiO2')

#(446, 'R OXYGEN THERAPY')
#print_allunique(flowsheet, 446)

#(492, 'R GOSH ASTRAL PEEP_cm H2O')
#print_allunique(flowsheet, 492)
make_new_columns(492, flowsheet, ['PEEP'], True)

#(493, 'R GOSH PEEP_cmH2O')
#print_allunique(flowsheet, 493)
make_new_columns(493, flowsheet, ['PEEP'], True)

#(494, 'R VENT PEEP_cm H2O')
#print_allunique(flowsheet, 494)
make_new_columns(494, flowsheet, ['PEEP'], True)

#(495, 'R GOSH ICU PEEP 8')
#print_allunique(flowsheet, 495)

#(496, 'R GOSH IP VENT PEEP SETTING_cm H2O')
#print_allunique(flowsheet, 496)
make_new_columns(496, flowsheet, ['PEEP'], True)

#(497, 'R AN VENT PEEP_cm H20')
#print_allunique(flowsheet, 497)
make_new_columns(497, flowsheet, ['PEEP'], True)


#Need to finish this

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
conv_BP(4, flowsheet, 3, 'SysBP', 'DiaBP')

#5, 'R AN MEAN ARTERIAL BLOOD PRESSURE_mmHg'
#print_allunique(flowsheet, 5)
conv_BP(5, flowsheet, False, 'MAP')

#26, 'R AN CENTRAL AORTIC PRESSURE (CAP)'
#print_allunique(flowsheet, 26)
conv_BP(26, flowsheet, 3, 'SysBP', 'DiaBP')

#27, 'R AN CENTRAL AORTIC PRESSURE (CAP) MEAN_mmHg'
#print_allunique(flowsheet, 27)
conv_BP(27, flowsheet, False, 'MAP')

#(36, 'R ARTERIAL LINE WAVEFORM') - not used
#print_allunique(flowsheet, 36)

#37, 'R GOSH IP MEAN ARTERIAL BLOOD PRESSURE_mmHg')
#print_allunique(flowsheet, 37)
conv_BP(37, flowsheet, False, 'MAP')

#39, 'R ARTERIAL LINE BLOOD PRESSURE')
#print_allunique(flowsheet, 39)
conv_BP(39, flowsheet, 3, 'SysBP', 'DiaBP')

#(40, 'R MAP A-LINE_mmHg')
#print_allunique(flowsheet, 40)
conv_BP(4, flowsheet, False, 'MAP')

#(68, 'R GOSH IP NEONATAL PAIN BLOOD PRESSURE') - not using
#print_allunique(flowsheet, 68)

#(69, 'R GOSH PEWS BP SCORE')  - PEWs score
#print_allunique(flowsheet, 69)

#(79, 'R PED BOYS DIASTOLIC BP PERCENTILE_%') - centile only
#print_allunique(flowsheet, 79)

#(80, 'R PED BOYS SYSTOLIC BP PERCENTILE_%')

#(81, 'BLOOD PRESSURE')
#print_allunique(flowsheet, 81)
conv_BP(39, flowsheet, 3, 'SysBP', 'DiaBP')

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
conv_BP(411, flowsheet, False, 'MAP')



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


#### Now do HR
get_relevant_cols(flowsheet, 'HR| rate')

#R GOSH IP HEART RATE ECG_beats per minute
make_new_columns(273, flowsheet, ['HR'], True)
#R AN HEART RATE ECG_beats per minute
make_new_columns(302, flowsheet, ['HR'], True)
#R AN SPO2 HR PULSE
make_new_columns(303, flowsheet, ['HR'], True)
#R GOSH IP HEART RATE PLETHYSMOGRAM
make_new_columns(533, flowsheet, ['HR'], True)

#GCS/ AVPU
print_all(flowsheet)
get_relevant_cols(flowsheet, 'GCS|AVPU|Comfort|Consciousness|Alert|Pupils|GLASGOW COMA SCALE|PEWS')

#Comfort score
#R GOSH COMFORT ALERTNESS
make_new_columns(22, flowsheet, ['Comfort:Alertness'], True)
#R GOSH COMFORT BP
make_new_columns(83, flowsheet, ['Comfort:BP'], True)
#R GOSH COMFORT CALMNESS
make_new_columns(97, flowsheet, ['Comfort:Calmness'], True)
#R GOSH COMFORT SCORE
make_new_columns(110, flowsheet, ['Comfort'], True)
#R GOSH IP COMFORT SCORE FOR GOSH GO
make_new_columns(111, flowsheet, ['Comfort'], True)
#R GOSH COMFORT HR
make_new_columns(304, flowsheet, ['Comfort:HR'], True)
#R GOSH COMFORT RESP NO VENT - Consider this as surrogate for no vent/ vent below?
make_new_columns(563, flowsheet, ['Comfort:Resp'], True)
#R GOSH COMFORT RESP VENT
make_new_columns(564, flowsheet, ['Comfort:Resp'], True)
#Alertness
make_new_columns(867, flowsheet, ['Comfort:Alertness'], True)

#R GOSH IP AVPU SCORE
make_new_columns(48, flowsheet, ['AVPU'], newcols1 = [['U'], ['P'], ['V'], ['A']])
#R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST AUD
make_new_columns(55, flowsheet, ['GCS_V'], True)
#R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST AU
make_new_columns(56, flowsheet, ['GCS_V'], True)
#R NSR GLASGOW COMA SCALE BEST EYE RESPONSE
make_new_columns(57, flowsheet, ['GCS_E'], True)

#R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST MOT
motor60 = [[], ['1', 1], ['2', 2], [3.0, '3'], [4.0, '4', 'P'], ['5', 5.0], [6.0, '6']]
make_new_columns(60, flowsheet, ['GCS_M'], newcols1 = motor60)
#R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST MO
make_new_columns(61, flowsheet, ['GCS_M'], newcols1 = motor60)
#R GLASGOW COMA SCALE BEST MOTOR RESPONSE
make_new_columns(58, flowsheet, ['GCS_M'], newcols1 = motor60)
#R NSR GLASGOW COMA SCALE BEST MOTOR RESPONSE
make_new_columns(59, flowsheet, ['GCS_M'], newcols1 = motor60)

#R GLASGOW COMA SCALE BEST VERBAL RESPONSE
make_new_columns(63, flowsheet, ['GCS_V'], newcols1 = motor60)

#R GLASGOW COMA SCALE EYE OPENING
eye223 = [[], ['1', 1], ['2', 2, 'P'], [3.0, '3'], [4.0, '4']]
make_new_columns(223, flowsheet, ['GCS_E'], newcols1 = eye223)

#R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) EYE OPEN
make_new_columns(224, flowsheet, ['GCS_E'], newcols1 = eye223)

#R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) EYE OPE
make_new_columns(225, flowsheet, ['GCS_E'], newcols1 = eye223)

#R GLASGOW COMA SCALE SCORE
make_new_columns(258, flowsheet, ['GCS'], True)

#R NSR GLASGOW COMA SCALE SCORE
make_new_columns(259, flowsheet, ['GCS'], True)
#R PEDSCPN GLASGOW COMA SCALE SCORE (AGE 6 MONTHS TO 2 YEARS)
make_new_columns(260, flowsheet, ['GCS'], True)
#R PEDSCPN GLASGOW COMA SCALE SCORE (AGE GREATER THAN 2 YEARS)
make_new_columns(251, flowsheet, ['GCS'], True)



#CRT
print_all(flowsheet)
get_relevant_cols(flowsheet, 'Cap|Refill|CRT|Time')
#R CAPILLARY REFILL: GENERAL
crt98 = [[], [], ['Less than/equal to 2 seconds (All extremities)'], [], ['Greater than 2 seconds (All extremities)']]
make_new_columns(98, flowsheet, ['CRT'], newcols1 = crt98)

#R CAPILLARY REFILL: ICU
crt99 = [[], [], ['<2', 'Brisk'], ['2 - 3'], ['3 - 5', 'Sluggish'], ['>5']]
make_new_columns(99, flowsheet, ['CRT'], newcols1 = crt99)

#R CAPILLARY REFILL: SECONDS
crt100 = [[], [], ['Less than 2 seconds'], ['3 seconds'], ['4 seconds'], ['5 seconds'], ['Greater than 6 seconds']]
make_new_columns(100, flowsheet, ['CRT'], newcols1 = crt100)

#R PVS CAPILLARY REFILL LLE
crt357 = [[], [1, '2'], [2, '2'], [3, '3'], [4, '4', '4+']]
make_new_columns(357, flowsheet, ['CRT'], newcols1 = crt357)

#R PVS CAPILLARY REFILL RLE
make_new_columns(573, flowsheet, ['CRT'], newcols1 = crt357)


####Sats
print_all(flowsheet)
get_relevant_cols(flowsheet, 'sat|spo2|oxygen|\%|percent')
# R EVENT SPO2
make_new_columns(213, flowsheet, ['SpO2'], True)
#R PERF NORMALISED SAO2_%
make_new_columns(419, flowsheet, ['SpO2'], True)
#R GOSH CV SATURATION NEW_%
make_new_columns(580, flowsheet, ['SpO2'], True)
#PERF PRIME BG POC SO2_%
make_new_columns(590, flowsheet, ['SpO2'], True)
#ANAESTHESIA PULSE OXIMETRY_%
make_new_columns(592, flowsheet, ['SpO2'], True)
#PULSE OXIMETRY_%
make_new_columns(593, flowsheet, ['SpO2'], True)
#R GOSH IP PULSE OXIMETRY ARTERIAL SPO2L_%
make_new_columns(595, flowsheet, ['SpO2'], True)
#R GOSH IP PULSE OXIMETR ARTERIAL SPO2R_%
make_new_columns(596, flowsheet, ['SpO2'], True)
#R PULSE OXIMETRY PRE-DUCTAL_%
make_new_columns(599, flowsheet, ['SpO2'], True)
#R CV SYSTEMIC ARTERIAL O2 SATURATION_%
make_new_columns(622, flowsheet, ['SpO2'], True)






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



#Height
get_relevant_cols(flowsheet, 'height|ht|metres|cm')
#HEIGHT_centimetres
make_new_columns(281, flowsheet, ['Height(cm)'], True)
#R GOSH IP HEIGHT FOR WEIGHT
make_new_columns(282, flowsheet, ['Height(cm)'], True)

apply_wt_ht('Height(cm)', flowsheet, 'interpolated_ht_m', scaling = 0.01)



#Input/ output
#Input
get_relevant_cols(flowsheet, 'input|intake|fluid|output|nappy')
get_relevant_cols(flowsheet, 'loss')
flowsheet['input'] = 0
flowhseet['output'] = 0

#R DRY NAPPY WEIGHT_g 182
#R NAPPY WITH STOOL WEIGHT_g
conv_2cols(182, 402, flowsheet, 'Nappy_output', bminusa)
#R NAPPY WEIGHT WITH URINE_g
conv_2cols(182, 403, flowsheet, 'Nappy_output', bminusa)
#401 For R NAPPY WEIGHT URINE AND STOOL_g
conv_2cols(182, 401, flowsheet, 'Nappy_output', bminusa)
sum_inout('Nappy_output', flowsheet, 'output')

#R GOSH IP DIET TOTAL FLUID
sum_inout(244, flowsheet, 'input')

#R ONC APHERESIS FLUID BALANCE
sum_inout(246, flowsheet, 'input')

#INTRAVENOUS INTAKE_mL
sum_inout(308, flowsheet, 'input')

#R AN NG/OG TUBE OUTPUT_mL
sum_inout(409, flowsheet, 'output')

#OTHER OUTPUT_mL
sum_inout(434, flowsheet, 'output')

#R CHEST TUBE OUTPUT_mL
sum_inout(435, flowsheet, 'output')

#R GOSH LDA WOUND OUTPUT VOLUME_mL
sum_inout(437, flowsheet, 'output')

#R DRAIN OUTPUT_mL
sum_inout(436, flowsheet, 'output')

#R GOSH URINE DRAIN OUTPUT_mL
sum_inout(438, flowsheet, 'output')

#R OSTOMY OUTPUT TYPE_mL
sum_inout(439, flowsheet, 'output')

#R TUBE OUTPUT_mL
sum_inout(440, flowsheet, 'output')

#R URINE OUTPUT_m
sum_inout(441, flowsheet, 'output')

#R CRRT PROGRAMMED FLUID LOSS_mL/hr - need to check if this is every min or hour
sum_inout(575, flowsheet, 'output')

#R GOSH IP CRRT FLUID REMOVAL_mL
sum_inout(576, flowsheet, 'output')

#URINE OUTPUT_mL
sum_inout(772, flowsheet, 'output')

#R GOSH IP I/O ORAL INTAKE VOL ML_ml
sum_inout(772, flowsheet, 'input')

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
sum_inout(803, flowsheet, 'Inotropes')
sum_inout(837, flowsheet, 'Inotropes')

def corAVP(input):
    return input*2.5
conv_column(853, flowsheet, 'EquivAVP', corAVP)
sum_inout('EquivAVP', flowsheet, 'Inotropes')

def corDopamine(input):
    return input/100
conv_column(815, flowsheet, 'EquivDopamine', corDopamine)
sum_inout('EquivDopamine', flowsheet, 'Inotropes')

def corMilrinone(input):
    return input/25
conv_column(835, flowsheet, 'EquivMilrinone', corMilrinone)
sum_inout('EquivMilrinone', flowsheet, 'Inotropes')



### Get demographics
apply_demographic('birth_date', demographics, flowsheet, 'Age_yrs')
flowsheet['time_to_death'] = 70
apply_demographic('death_date', demographics, flowsheet, 'time_to_death')
apply_demographic('deceased_flag', demographics, flowsheet, 'died', False)
apply_demographic('sex', demographics, flowsheet, 'sex', False)
apply_demographic('ethnicity_name', demographics, flowsheet, 'ethnicity', False)


### Save the data so that can run childsds in R
flowsheet.to_csv('Project/Project_data/files/flowsheet_output.csv')

##### Some splitting by patient

#Split up by patient
def split_by_pt(sheet, col):
    """
    Function to split up a sheet by patient \n
    Takes a pd.DataFrame, specify column as text
    """
    patient_dfs = list()
    unique_patients = sheet[col].unique()
    
    #Work through the the patients and split into new sheets
    for i, j in enumerate(unique_patients):
        temp_df = sheet.loc[sheet[col] == j, :]
        patient_dfs.append(temp_df) 

    return patient_dfs

#Start with only numeric values for now
def return_numeric(sheet):
    """
    Function to return only the numeric columns in a sheet
    """

    #Work out which columns are numeric
    numeric_dtypes = sheet.dtypes.unique()
    sheet_dtypes = sheet.dtypes.isin(numeric_dtypes[[1, 2, 3]])
    
    #Make sure you keep the patient ID
    sheet_dtypes[0] = True
    return sheet.loc[:, sheet_dtypes]

numeric_flowsheet = return_numeric(flowsheet)

flowsheet_pt_split = split_by_pt(numeric_flowsheet, 'project_id')


#Do some plotting
def plot_PICU_params(df, patient, title="", xlabel='Date/Time', ylabel='Value', dpi=500):
    ''' 
    Function to do some plotting of timeseries data, plots multiple parameters for same patient
    Takes parameters df (pandas dataframe), x and y values, plots them
    You have to choose a label to print
    Uses the input from the index
    Should probably change this after making more composite values so only plotting composites
    '''

    #Choose columns which have fewest NaNs
    pt_df = df[patient]
    colnames = pt_df.columns
    numNaNs = np.zeros([len(colnames) - 3])

    #Work  through columns and get numbers of Nans
    for i in range(3, len(colnames)):
        isnan = pt_df.iloc[:, (i)].isna() == False
        numNaNs[i - 3] = pt_df.loc[isnan, :].shape[0]

    #Sort the NaNs
    sortedNaNs = np.argsort(numNaNs)
    top25locs = sortedNaNs[range(-25, 0)]
    top25 = sortedNaNs[top25locs] + 3

    #Instantiate figure 
    fig, axs = plt.subplots(5, 5, figsize=(15,15), sharex=True)

    #Work through the different patients
    for i in range(5):
        for j in range(5):

            #Choose the correct column
            column = colnames[top25[5*i + j]]
            
            #First object in list for each pt is list of vars
            x = pt_df['taken_datetime']
            y = pt_df[column]
            axs[i, j].plot(x, y)
            axs[i, j].set_title(column)

    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='value')

    
    #Save it
    fig.savefig('Project/PICU_project/figs/PICU_plots' + str(patient) + '.png')

for i in range(15):
    plot_PICU_params(flowsheet_pt_split, i)