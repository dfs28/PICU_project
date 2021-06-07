### PICU data wrangling
# Setup

import pandas as pd
import numpy as np


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
    a = print_uniquevars(sheet, column)
    length = len(sheet.loc[:, sheet.columns[column]].unique())
    for i in range(length):
        next(a)

#Function to split up by patient


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
                sheet[new_column] = 0
            
            #Work  through the items in this dict
            for j, k in enumerate(value):
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



vent14_unique = flowsheet.loc[:, flowsheet.columns[14]].unique()
ventilated = vent14_unique[[2, 12]]
flowsheet.loc[:, flowsheet.columns[14]]

ventilated = flowsheet.columns[(14, 15, 16, 17, 18, 19, 20, )]
MAP_BP = flowsheet.columns[(4, 5, )]



