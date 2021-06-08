### PICU data wrangling
# Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
import os

#Import the files - start with the massive flowsheet one
flowsheet = pd.read_csv('Project/Project_data/files/caboodle_patient_selected_flowsheetrows_main_pivot.csv', sep= ',', parse_dates=['taken_datetime'])

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
    #Get the unique values flexibly
    if type(column) == int:
        unique_inputs = sheet.loc[:, sheet.columns[column]].unique()
        
        #Make sure original column is a string
        column = sheet.columns[column]

    elif type(column) == str:
        unique_inputs = sheet.loc[:, (column)].unique()
    else:
        raise ValueError('original_column should either be a numbered column or the correct name of a column')

    for i in enumerate(unique_inputs):
        yield print(i)

def print_allunique(sheet, column):
    """
    Simple function that prints proportion of NaNs, absolute number of Nans and unique values
    Could also add in functionality to say what proportion of each exist
    """

    #Get number of Nas
    isnan = sheet.iloc[:, (column)].isna()
    numNaNs = sheet.loc[isnan, :].shape[0]
    notNa = sheet.shape[0] - numNaNs
    propnotNa = notNa/sheet.shape[0]
    print('For {}'.format(sheet.columns[column]))
    print('{:2.2%} percent not Na'.format(propnotNa))
    print('{} of {} not Na'.format(notNa, sheet.shape[0]))

    #Print out the unique vals
    a = print_uniquevars(sheet, column)
    length = len(sheet.loc[:, sheet.columns[column]].unique())
    for i in range(length):
        next(a)






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
    
#flowsheet_pt_split = split_by_pt(flowsheet, 'project_id')




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


top16locs = pd.Series(sortedNaNs).isin(range(15))
top16 = sortedNaNs[top16locs] + 3




#Do some plotting
def plot_PICU_params(df, patient, labels = labels, title="", xlabel='Date', ylabel='Value', dpi=500):
    ''' 
    Function to do some plotting of timeseries data, plots multiple parameters for same patient
    Takes parameters df (pandas dataframe), x and y values, plots them
    You have to choose a label to print
    Uses the input from the index
    Note this function doesn't work atm - not sure why
    '''

    #Choose columns which have fewest NaNs
    pt_df = df[patient]
    colnames = pt_df.columns
    numNaNs = np.zeros([len(colnames) - 3])

    #Work  through columns and get numbers of Nans
    for i in range(3, len(colnames)):
        isnan = pt_df.iloc[:, (i)].isna()
        numNaNs[i - 3] = pt_df.loc[isnan, :].shape[0]

    #Sort the NaNs
    sortedNaNs = np.argsort(numNaNs)
    top16locs = pd.Series(sortedNaNs).isin(range(15))
    top16 = sortedNaNs[top16locs] + 3

    #Instantiate figure 
    fig, axs = plt.subplots(4, 4, figsize=(15,15), sharex=True)
    
    #ISSUE HERE HANDLING BLOOD PRESSURES - FIRST THING TO FIX!

    #Work through the different patients
    for i in range(4):
        for j in range(4):

            #Choose the correct column
            column = colnames[top16[4*i + j]]
            
            #First object in list for each pt is list of vars
            x = pt_df['taken_datetime']
            y = float(pt_df[column])
            axs[i, j].plot(x, y)
            axs[i, j].set_title(column)

    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='value')

    
    #Save it
    fig.savefig('Project/PICU_project/figs/PICU_plots' + str(patient) + '.png')


#Start by merging columns appropriately

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
    if type(original_column) == int:
        unique_inputs = sheet.loc[:, sheet.columns[original_column]].unique()
        
        #Make sure original column is a string
        original_column = sheet.columns[original_column]

    elif type(original_column) == str:
        unique_inputs = sheet.loc[:, (original_column)].unique()
    else:
        raise ValueError('original_column should either be a numbered column or the correct name of a column')

    #Allocate the new values
    if not inputIsoutput:
        i = 0
        
        #Work through the lists
        for key, value in input2output.items():
            new_column = new_cols[i]
            if not new_column in sheet.columns:

                #Should probably consider whether default should be zeros?
                sheet[new_column] = 0
            
            #Work  through the items in this list
            for j, k in enumerate(value):

                #Only take values that are in this column
                if isinstance(k, list):
                    for l, m in enumerate(k):
                        original_values = value[j]
                        sheet.loc[sheet[original_column].isin(original_values), (new_column)] = j + 1
                else:         
                    original_value = value[j]
                    sheet.loc[sheet[original_column] == original_value, (new_column)] = j + 1

            i += 1


#Start with ventilation and O2
print_all(flowsheet)

#'R GOSH ICU AIRWAY STATUS' 14
print_allunique(flowsheet, 14)
ventilated14 = [['Low Flow NC', 'Face Mask', 'SVIA;Low Flow NC', 'Face Mask;Low Flow NC', 'SVIA;Face Mask', 
                'Low Flow NC;SVIA'], 
                ['High Flow NC', 'CPAP', 'HFO', 'SVIA;CPAP', 'SVIA;High Flow NC', 'Face Mask;CPAP', 
                'HFO;NO', 'High Flow NC;SVIA', 'CPAP;SVIA'], 
                ['CMV', 'CMV;CPAP', 'SVIA;CMV', 'NO;CMV', 'CMV;NO']]
make_new_columns(14, flowsheet, ['Ventilation'], newcol1 = ventilated14)

#'R GOSH IP PANDA AIRWAY' 15
print_allunique(flowsheet, 15)
ventilated15 = [[], [], ['Intubation']]
make_new_columns(15, flowsheet, ['Ventilation'], newcol1 = ventilated15)

#'R GOSH NICU AIRWAY STATUS' 16
print_allunique(flowsheet, 16)
ventilated15 = [[], [], ['Intubation']]
make_new_columns(15, flowsheet, ['Ventilation'], newcol1 = ventilated15)

#'R GOSH RESUS ABCDE AIRWAY' 17
print_allunique(flowsheet, 15)
ventilated15 = [[], [], ['Intubation']]
make_new_columns(15, flowsheet, ['Ventilation'], newcol1 = ventilated15)

#'R ED AIRWAY (WDL)' 18
print_allunique(flowsheet, 15)
ventilated15 = [[], [], ['Intubation']]
make_new_columns(15, flowsheet, ['Ventilation'], newcol1 = ventilated15)

#'R AIRWAY LDA INSERTION ATTEMPTS' 19
print_allunique(flowsheet, 15)
ventilated15 = [[], [], ['Intubation']]
make_new_columns(15, flowsheet, ['Ventilation'], newcol1 = ventilated15)

#'R GOSH IP CATS AIRWAY STATUS' 20
print_allunique(flowsheet, 15)
ventilated15 = [[], [], ['Intubation']]
make_new_columns(15, flowsheet, ['Ventilation'], newcol1 = ventilated15)

#R VENT ALARM APNOEA SECONDS_Seconds 28

#R IP VENT APNOEA RATE 30

#R IP VENT APNOEA VOLUME_mL 31

#34, 'R GOSH IP ARE THEY INTUBATED AND VENTILATED?'

#127, 'R GOSH CPAP CMH2O_cmH2O'

#128, 'R GOSH IP FF CPAP/BIPAP MASK'

#129, 'R OPTIME RECOVERY CPAP/IPPV'

#136, 'R GOSH TRACHE CUFF'

#198, 'R GOSH ASTRAL EPAP_cm H2O') - Astral is NIV

#199, 'R GOSH EPAP'

#200, 'R GOSH NIPPY EPAP_cmH2O

#204, 'R GOSH ICU END TIDAL CO2_%

#205, 'R VENT ETCO2_kPa

#206, 'R IP VENT ALARM ETCO2/TCM MAX_mmHg

#207, 'R IP VENT ALARM ETCO2/TCM MIN_mmHg

#208, 'R GOSH ICU TRANSFER ETT

#209, 'R GOSH ICU TRANSFER ETT SECURE

#210, 'R ETT TO LIP_cm

#211, 'R ETT TO NOSE_cm'

#213, 'R EVENT SPO2'

#214, 'R AN AGENTS DESFLURANE EXPIRED_%'

#215, 'R AN AGENTS ISOFLURANE EXPIRED_%'

#216, 'R AN EXPIRED MINUTE VOLUME_L/min'

#217, 'R AN EXPIRED N2O_%'\\

#218, 'R AN AGENTS SEVOFLURANE EXPIRED_%'

#219, 'R GOSH EXP TV LITRES_L'

#220, 'R VENT EXP TIDAL VOLUME_mL'

#222, 'R GOSH ICU WITHDRAWAL EXTUBATION'

#234, 'R AN FIO2_%'

#235, 'R PERF FIO2_%'

#236, 'R GOSH ICU SANDWICH FIO2'

#237, 'R INSPIRED O2 FIO2 FOR NO_%'

#238, 'R GOSH IP FIO2 SETTING_%'

#239, 'R VENT INSP FLOWS_L/min'

#240, 'R GOSH VENT FLOW ML/KG_mL/Kg'

#241, 'R IP VENT FLOW OBS_L/min'

#242, 'R GOSH AIRVO FLOW_L/min' Highflow o2

#R GOSH VENT FABIAN O2 THERAPY FLOW_L/min HFO/ NIV

#(284, 'R GOSH HFO BASE FLOW READING')

#285, 'R GOSH HFO BASE FLOW SETTING')

#287, 'R GOSH HFO FREQUENCY READING_Hz')

"""(288, 'R GOSH IP HFV FREQUENCY SETTING_Hz')
(289, 'R GOSH IP HFV I:E RATIO SETTING')
(290, 'R GOSH IP HFV TIDAL VOLUME_mL')
(291, 'R GOSH HIGH FIO2_%') HFV is high frequency ventilation"""

"""R IP VENT HUMIDIFIER TEMP (SET)')
(306, 'R VENT I:E RATIO')
(307, 'R AN I:E RATIO')"""

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

def conv_BP(original_column, sheet, SysDia = 'Three'):
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
    if type(original_column) == int:
        unique_inputs = sheet.loc[:, sheet.columns[original_column]].unique()
        
        #Make sure original column is a string
        original_column = sheet.columns[original_column]

    elif type(original_column) == str:
        unique_inputs = sheet.loc[:, (original_column)].unique()
    else:
        raise ValueError('original_column should either be a numbered column or the correct name of a column')

    #If both need to split into sys and dia
    if SysDia == ('Three' | 'Both'):
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
        if SysDia == True:
            na_locs = (new_sheet.loc[:, (i)].isna() == False)
            new_values = new_sheet.loc[na_locs, (i)]
        else:
            na_locs = (new_sheet.isna() == False)
            new_values = new_sheet[na_locs]

        #Now assign
        sheet.loc[na_locs, (new_column)] = new_values

        i += 1


#Make a unified BP column
#4, 'R AN ARTERIAL BLOOD PRESSURE'
print_allunique(flowsheet, 4)
conv_BP(4, flowsheet, ['SysBP', 'DiaBP', 'MAP'])

#5, 'R AN MEAN ARTERIAL BLOOD PRESSURE_mmHg'
print_allunique(flowsheet, 5)
conv_BP(5, flowsheet, SysDia = False, ['MAP'])

#26, 'R AN CENTRAL AORTIC PRESSURE (CAP)'
print_allunique(flowsheet, 26)
conv_BP(26, flowsheet, 'SysBP', 'DiaBP', 'MAP')

#27, 'R AN CENTRAL AORTIC PRESSURE (CAP) MEAN_mmHg'
print_allunique(flowsheet, 27)
conv_BP(27, flowsheet, 'MAP', SysDia = False)

#(36, 'R ARTERIAL LINE WAVEFORM') - not used
print_allunique(flowsheet, 36)

#37, 'R GOSH IP MEAN ARTERIAL BLOOD PRESSURE_mmHg')
print_allunique(flowsheet, 37)
conv_BP(37, flowsheet, 'MAP', SysDia = False)

#39, 'R ARTERIAL LINE BLOOD PRESSURE')
print_allunique(flowsheet, 39)
conv_BP(39, flowsheet, 'SysBP', 'DiaBP', 'MAP')

#(40, 'R MAP A-LINE_mmHg')
print_allunique(flowsheet, 40)
conv_BP(4, flowsheet, 'MAP', SysDia = False)


"""
(39, 'R ARTERIAL LINE BLOOD PRESSURE')
(40, 'R MAP A-LINE_mmHg')
(68, 'R GOSH IP NEONATAL PAIN BLOOD PRESSURE')
(69, 'R GOSH PEWS BP SCORE')
(79, 'R PED BOYS DIASTOLIC BP PERCENTILE_%')
(80, 'R PED BOYS SYSTOLIC BP PERCENTILE_%')
(81, 'BLOOD PRESSURE')
(161, 'R PED DIASTOLIC BP PERCENTILE_%')
(162, 'R GOSH CV HAEMO DIASTOLE 1 NEW_mmHg')
(163, 'R GOSH CV HAEMO DIASTOLE 1_mmHg')
(164, 'R GOSH CV HAEMO DIASTOLE 2_mmHg')
(192, 'R GOSH CV HAEMO EDP 1 NEW_mmHg')
(193, 'R GOSH CV HAEMO EDP 1_mmHg')
(255, 'R PED GIRLS DIASTOLIC BP PERCENTILE_%')
(256, 'R PED GIRLS SYSTOLIC BP PERCENTILE_%')
(361, 'R BP LOCATION')
(385, 'R GOSH CV HAEMO MEAN 1_mmHg')
(386, 'R GOSH CV HAEMO MEAN 2_mmHg')
(411, 'R AN NIBP MEAN_mmHg')
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
(141, 'R AN CVP MEAN_mmHg')
(142, 'R DEVICE CVP MEAN_mmHg')
(636, 'R GOSH ICU TARGET CVP MAX_mmHg')
(637, 'R GOSH ICU TARGET CVP MIN_mmHg')
"""
print_allunique(flowsheet, 7)

vent14_unique = flowsheet.loc[:, flowsheet.columns[14]].unique()
ventilated = vent14_unique[[2, 12]]
flowsheet.loc[:, flowsheet.columns[14]]

ventilated = flowsheet.columns[(14, 15, 16, 17, 18, 19, 20, )]
MAP_BP = flowsheet.columns[(4, 5, )]



