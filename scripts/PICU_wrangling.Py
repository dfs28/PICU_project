### PICU data wrangling
# Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
import os
from itertools import chain

#Import the files - start with the massive flowsheet one
flowsheet = pd.read_csv('Project/Project_data/files/caboodle_patient_selected_flowsheetrows_main_pivot.csv', sep= ',', parse_dates=['taken_datetime'])



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

def print_allunique(sheet, column):
    """
    Simple function that prints proportion of NaNs, absolute number of Nans and unique values
    Could also add in functionality to say what proportion of each exist
    """

    #Return colname
    column = return_colname(column, sheet)

    #Get number of Nas
    isnan = sheet.loc[:, column].isna()
    numNaNs = sheet.loc[isnan, :].shape[0]
    notNa = sheet.shape[0] - numNaNs
    propnotNa = notNa/sheet.shape[0]
    print('For {}'.format(column))
    print('{:2.2%} percent not Na'.format(propnotNa))
    print('{} of {} not Na'.format(notNa, sheet.shape[0]))

    #Print out the unique vals
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
    numNaNs = np.zeros([len(colnames) - 3])

    #Make a progress bar because this takes ages
    bar = Bar('Processing', max=len(colnames))

    #Work  through columns and get numbers of Nans
    for i in range(3, len(colnames)):
        isnan = flowsheet.iloc[:, (i)].isna()
        numNaNs[i - 3] = flowsheet.loc[isnan, :].shape[0]
        bar.next()

    bar.finish()
    
    #Sort the NaNs
    numNotNa = flowsheet.shape[0] - numNaNs
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
def get_relevant_cols(sheet):
    """
    Function to pull out relevant colnames
    Will have it spit out how many Nas in the matching ones, 
    Also a short list of the unique values
    """


#### Start with ventilation and O2
print_all(flowsheet)

"""
Here I'm making a ventilation column:
0 for no ventilation
1 for on oxygen
2 for NIV/Optiflow
3 for intubated and ventilated
5 for HFO

An intubated column
0 for not
1 for intubated

An EtCO2 column:
Na for no value, rest just ported from any EtCO2 columns

An Oxygen flow column - this will then need to be corrected for bodyweight

An FiO2 column (not going to try to convert between but planning to have missingness columns)

"""
#'R GOSH ICU AIRWAY STATUS' 14 ##### Is HFO high frequency oscillation or high flow oxygen?
#Should I consider this as a separate var?
#print_allunique(flowsheet, 14)
ventilated14 = [['SVIA', 'NO'], 
                ['Low Flow NC', 'Face Mask', 'SVIA;Low Flow NC', 'Face Mask;Low Flow NC', 'SVIA;Face Mask', 
                'Low Flow NC;SVIA'], 
                ['High Flow NC', 'CPAP', 'SVIA;CPAP', 'SVIA;High Flow NC', 'Face Mask;CPAP', 'High Flow NC;SVIA', 'CPAP;SVIA'], 
                ['CMV', 'CMV;CPAP', 'SVIA;CMV', 'NO;CMV', 'CMV;NO'], 
                ['HFO', 'HFO;NO']]
make_new_columns(14, flowsheet, ['Ventilation'], newcol1 = ventilated14)
intubated14 = list(chain.from_iterable(ventilated14[3:5]))
make_new_columns(14, flowsheet, ['Intubated'], newcol1 = [[], intubated14])

#'R GOSH IP PANDA AIRWAY' 15
#print_allunique(flowsheet, 15)
ventilated15 = [[], ['Intubation']]
make_new_columns(15, flowsheet, ['Intubated'], newcol1 = ventilated15)


#'R GOSH NICU AIRWAY STATUS' 16
#print_allunique(flowsheet, 16)
ventilated16 = [[], [], [], ['Conventional ventilation']]
make_new_columns(16, flowsheet, ['Ventilation'], newcol1 = ventilated16)
make_new_columns(16, flowsheet, ['Intubated'], newcol1 = ventilated16[2:5])

#'R GOSH RESUS ABCDE AIRWAY' 17
#print_allunique(flowsheet, 17)
ventilated16 = [[], ['Intubated', 'Intubated;Opening manoeuvres', 'Intubated;Patent']]
make_new_columns(17, flowsheet, ['Intubated'], newcol1 = ventilated16)

#'R ED AIRWAY (WDL)' 18 - not using
#print_allunique(flowsheet, 18)

#'R AIRWAY LDA INSERTION ATTEMPTS' 19
#print_allunique(flowsheet, 19)
ventilated19 = [[], [1, 2, '1', '2', '3 or more']]
make_new_columns(19, flowsheet, ['Intubated'], newcol1 = ventilated19)

#'R GOSH IP CATS AIRWAY STATUS' 20
#print_allunique(flowsheet, 20)
ventilated20 = [[], ['Intubated']]
make_new_columns(20, flowsheet, ['Intubated'], newcol1 = ventilated20)

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

#199, 'R GOSH EPAP'
#print_allunique(flowsheet, 199)

#200, 'R GOSH NIPPY EPAP_cmH2O
#print_allunique(flowsheet, 200)
unique_vent200 = flowsheet.iloc[:, 200].unique()
vent200 = [[], [], unique_vent200[range(1, len(unique_vent200))].tolist(), []]
make_new_columns(200, flowsheet, ['Ventilation'], newcol1 = vent200)

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
vent210 = [[], unique_vent210[range(1, len(unique_vent210))].tolist()]
make_new_columns(210, flowsheet, ['Intubated'], newcol1 = vent210)

#211, 'R ETT TO NOSE_cm'
#print_allunique(flowsheet, 211)
unique_vent211 = flowsheet.iloc[:, 211].unique()
vent211 = [[], unique_vent211[range(1, len(unique_vent211))].tolist()]
make_new_columns(211, flowsheet, ['Intubated'], newcol1 = vent211)

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
make_new_columns(284, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 284, 5, 5))
make_new_columns(284, flowsheet, ['O2Flow'], True)

#285, 'R GOSH HFO BASE FLOW SETTING')
#print_allunique(flowsheet, 285)
make_new_columns(285, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 285, 5, 5))
make_new_columns(285, flowsheet, ['O2Flow'], True)

#287, 'R GOSH HFO FREQUENCY READING_Hz')
#print_allunique(flowsheet, 287)
make_new_columns(287, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 287, 5, 5))

#(288, 'R GOSH IP HFV FREQUENCY SETTING_Hz')
#print_allunique(flowsheet, 288)
make_new_columns(288, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 288, 5, 5))

#(289, 'R GOSH IP HFV I:E RATIO SETTING') - should probably consider calculating I:E manually
#print_allunique(flowsheet, 289)

#(290, 'R GOSH IP HFV TIDAL VOLUME_mL')HFV is high frequency ventilation
#print_allunique(flowsheet, 290)
make_new_columns(290, flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 287, 5, 5))

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

"""(315, 'R IP VENT INSPIRATORY PRESSURE HIGH_cm H2O')
(316, 'R IP VENT INSPIRATORY PRESSURE LOW_cm H2O')
(317, 'R GOSH INSPRIATORY PRESSURE')
(318, 'R GOSH IP INSPIRATORY PRESSURE TRIGGER SETTING')
(319, 'R GOSH IP VENT RISE TIME_Seconds')
(320, 'R VENT INSP TIME_Sec')
(321, 'R VENT INSP TIME SETTING (SEC)_Sec')
(322, 'R AN AGENTS DESFLURANE INSPIRED_%')
(323, 'R FIO2_%')
(324, 'R AN AGENTS ISOFLURANE INSPIRED_%')
(325, 'R AN INSPIRED N2O_%')
(326, 'R AN INSPIRED NO2_ppm')
(327, 'R AN AGENTS NITRIC OXIDE_ppm')
(328, 'R INSPIRED NO SETTING')
(329, 'R AN INSPIRED O2 SETTING')
(330, 'R AN AGENTS SEVOFLURANE INSPIRED_%')
(331, 'R GOSH IP INSPIRED TIDAL VOLUME_mL')
(332, 'R GOSH ASTRAL IPAP_cm H2O')
(333, 'R GOSH IPAP')
(334, 'R GOSH NIPPY IPAP_cmH2O')
(335, 'R GOSH STELLA IPAP_cm H2O')
(336, 'R GOSH IPAP TRIGGER')"""

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
(377, 'R GOSH VENT RATE')
(378, 'R GOSH VENT RATE OBSERVED')
(379, 'R IP VENT VT MANDATORY (OBS)_mL')
(380, 'R MAP')
(381, 'R IP VENT MEAN AIRWAY PRESSURE HIGH_cm H2O')
(382, 'R IP VENT MEAN AIRWAY PRESSURE LOW_cm H2O')
(387, 'R VENT MAP_cm H2O')
(388, 'R IP VENT MEAN AIRWAY PRESSURE (SET)')
(391, 'R VENT MINUTE VENTILATION_L/min')
(392, 'R GOSH MIN VOL LEAK_L/min')
(393, 'R GOSH IP MINUTE VOLUME SETTING_L/min')
(424, 'R AN AGENTS O2_L/min')
"""

#Need to finish this

#### Work on BP now
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
    return_colname(original_column, sheet)

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


#Make a unified BP column
#4, 'R AN ARTERIAL BLOOD PRESSURE'
print_allunique(flowsheet, 4)
conv_BP(4, flowsheet, 3, 'SysBP', 'DiaBP', 'MAP')

#5, 'R AN MEAN ARTERIAL BLOOD PRESSURE_mmHg'
print_allunique(flowsheet, 5)
conv_BP(5, flowsheet, False, 'MAP')

#26, 'R AN CENTRAL AORTIC PRESSURE (CAP)'
print_allunique(flowsheet, 26)
conv_BP(26, flowsheet, 3, 'SysBP', 'DiaBP', 'MAP')

#27, 'R AN CENTRAL AORTIC PRESSURE (CAP) MEAN_mmHg'
print_allunique(flowsheet, 27)
conv_BP(27, flowsheet, False, 'MAP')

#(36, 'R ARTERIAL LINE WAVEFORM') - not used
print_allunique(flowsheet, 36)

#37, 'R GOSH IP MEAN ARTERIAL BLOOD PRESSURE_mmHg')
print_allunique(flowsheet, 37)
conv_BP(37, flowsheet, False, 'MAP')

#39, 'R ARTERIAL LINE BLOOD PRESSURE')
print_allunique(flowsheet, 39)
conv_BP(39, flowsheet, 3, 'SysBP', 'DiaBP', 'MAP')

#(40, 'R MAP A-LINE_mmHg')
print_allunique(flowsheet, 40)
conv_BP(4, flowsheet, False, 'MAP')

#(68, 'R GOSH IP NEONATAL PAIN BLOOD PRESSURE') - not using
print_allunique(flowsheet, 68)

#(69, 'R GOSH PEWS BP SCORE')  - PEWs score
print_allunique(flowsheet, 69)

#(79, 'R PED BOYS DIASTOLIC BP PERCENTILE_%') - centile only
print_allunique(flowsheet, 79)

#(80, 'R PED BOYS SYSTOLIC BP PERCENTILE_%')

#(81, 'BLOOD PRESSURE')
print_allunique(flowsheet, 81)
conv_BP(39, flowsheet, 3, 'SysBP', 'DiaBP', 'MAP')

#(161, 'R PED DIASTOLIC BP PERCENTILE_%')
print_allunique(flowsheet, 161)

#(162, 'R GOSH CV HAEMO DIASTOLE 1 NEW_mmHg')
print_allunique(flowsheet, 162)

#(163, 'R GOSH CV HAEMO DIASTOLE 1_mmHg')
print_allunique(flowsheet, 163)

#(164, 'R GOSH CV HAEMO DIASTOLE 2_mmHg')
print_allunique(flowsheet, 164)

#(192, 'R GOSH CV HAEMO EDP 1 NEW_mmHg')
print_allunique(flowsheet, 192)

#(193, 'R GOSH CV HAEMO EDP 1_mmHg')
print_allunique(flowsheet, 193)

#(255, 'R PED GIRLS DIASTOLIC BP PERCENTILE_%')
print_allunique(flowsheet, 255)

#(256, 'R PED GIRLS SYSTOLIC BP PERCENTILE_%')
print_allunique(flowsheet, 256)

#(361, 'R BP LOCATION')
print_allunique(flowsheet, 361)

#(385, 'R GOSH CV HAEMO MEAN 1_mmHg')
print_allunique(flowsheet, 385)

#(386, 'R GOSH CV HAEMO MEAN 2_mmHg')
print_allunique(flowsheet, 386)

#(411, 'R AN NIBP MEAN_mmHg')
print_allunique(flowsheet, 411)
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
print_allunique(flowsheet, 141)

#(142, 'R DEVICE CVP MEAN_mmHg')
print_allunique(flowsheet, 142)


#### Now do HR
"""
(271, 'R GOSH CAR HR')
(272, 'R GOSH IP NEONATAL PAIN HEART RATE')
(273, 'R GOSH IP HEART RATE ECG_beats per minute')
(274, 'R OPTIME HEART RATE STABLE')
(275, 'R OPTIME NEW HEART RATE STABLE')
(276, 'R GOSH PEWS HR SCORE')
(277, 'R HEART RATE SOURCE')
(302, 'R AN HEART RATE ECG_beats per minute')
(303, 'R AN SPO2 HR PULSE')
"""


#GCS/ AVPU

#CRT
"""
(98, 'R CAPILLARY REFILL: GENERAL')
(99, 'R CAPILLARY REFILL: ICU')
(100, 'R CAPILLARY REFILL: SECONDS')
(357, 'R PVS CAPILLARY REFILL LLE')
"""

#Sats

#Bodyweight

#Input/ output
#Will need to double check about duplicates here

#Other organ support


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