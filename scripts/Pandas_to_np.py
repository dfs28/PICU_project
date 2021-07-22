#### Little script to convert big flowsheet file into np arrays for use
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import math
from datetime import datetime
from scipy import stats
import scipy
from progress.bar import Bar


#In reality import the big flowsheet to use thing
flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_zscores.csv', parse_dates = ['taken_datetime'])
print('Flowsheet loaded: ', datetime.now().strftime("%H:%M:%S"))

#Fix issue with ethnicity, sex and died
died = {'N': 0, 'Y': 1}
flowsheet['died'] = flowsheet['died'].replace(died)
sex = {'F': 0, 'M': 1, 'I': 2}
flowsheet['sex'] = flowsheet['sex'].replace(sex)
ethnicity = {j:i[0] for i, j in np.ndenumerate(flowsheet['ethnicity'].unique())}
flowsheet['ethnicity'] = flowsheet['ethnicity'].replace(ethnicity)
print('Correction to float applied: ', datetime.now().strftime("%H:%M:%S"))

#Make 3d training array
def make_3d_array(array, length, all_cols, point_cols, series_cols, percentile = True):
    """
    Function to make an pandas into a 3d np.array of slices and stack them \n
    Takes a dataframe, slices it by the length specified 
    Expects the time series to be longer than the number of vars
    Specify the length
    Percentile scales by rank - this should mean that fluctuations within the normal range are still visible
    """

    #Get the shape, work out what shape the new array should be
    all_used_cols = [i for i in array.columns if i in all_cols]
    print(all_used_cols)

    array_use = np.array(array.loc[:, all_used_cols])
    i_point_cols = [i for i, j in enumerate(all_used_cols) if j in point_cols]
    j_point_cols = [j for i, j in enumerate(all_used_cols) if j in point_cols]
    i_series_cols = [i for i, j in enumerate(all_used_cols) if j in series_cols]
    j_series_cols = [j for i, j in enumerate(all_used_cols) if j in series_cols]

    print('Converted to np array: ', datetime.now().strftime("%H:%M:%S"))

    #Monitor progress
    bar = Bar('Normalising data', max=array_use.shape[1])

    #Make this an np array with no nans
    for i in range(array_use.shape[1]):
        nas = np.isnan(array_use[:, i])
        if sum(nas == False) < 1:
            print(i)
            continue

        #Scale to percentile
        if percentile:
            array_use[nas == False, i] = stats.rankdata(array_use[nas == False, i])

        min = np.min(array_use[nas == False, i])
        max = np.max(array_use[nas == False, i])
        array_use[nas, i] = min

        #Now normalise 0-1
        array_use[:, i] -= min
        if not (max - min) == 0:
            array_use[:, i] /= (max - min)

        bar.next()
    bar.finish()


        
    shape = array_use.shape
    z_dim = math.floor(np.max(shape)/length)
    
    #Get the slices to use
    to_use = list()
    unique_patients = array['project_id'].unique()
    slicesPerPatient = np.zeros(len(unique_patients))

    bar = Bar('Getting useable slices', max=z_dim)
    for i in range(z_dim):
        start_position = i*length
        end_position = (i + 1)*length
        
        #Skip if more than one patient
        patients = array.loc[range(start_position, end_position), 'project_id'].unique()
        if patients.shape[0] == 1:
            to_use.append(i)
        
            #Record number of slices per patient (in order)
            patient_loc = unique_patients == patients[0]
            slicesPerPatient[patient_loc] += 1

        bar.next()
    bar.finish()

    x_dim = length
    y_dim = np.min(shape)
    array_3d = np.empty((len(to_use), len(i_series_cols), x_dim))
    array_2d = np.empty((len(to_use), len(i_point_cols)))
    array_characteristics = np.empty((len(to_use), len(i_series_cols)*2))
    splines = np.empty((len(to_use), 8*len(i_series_cols)))

    #Outcomes
    outcomes = np.zeros((len(to_use), 12))
    pt_slices = list()

    bar = Bar('Slicing and outcomes', max=len(to_use))
    for position, i in enumerate(to_use):
        start_position = i*length
        end_position = (i + 1)*length
        temp_array = array_use[start_position:end_position, i_series_cols]
        array_3d[position, :, :] = temp_array.transpose()
        array_2d[position, :] = array_use[end_position - 1, i_point_cols]


        ##Build the outcomes
        #Make sure you can see which patients these are coming from
        patient_id = np.where(array.loc[end_position, 'project_id'] == unique_patients)[0]
        project_id = array.loc[end_position, 'project_id']
        outcomes[position, 0] = patient_id[0]

        #Age of the patient
        outcomes[position, 1] = array.loc[end_position, 'Age_yrs']
        
        #Time to death as triplicate value - probably can't assume that death is within this admission?
        if array.loc[end_position, 'time_to_death'] <= 2/365:
            outcomes[position, 2] = 1
        elif array.loc[end_position, 'died'] == 'Y':
            outcomes[position, 3] = 1
        else: 
            outcomes[position, 4] = 1

        #Length of stay as triplicate
        end_of_section = array.loc[end_position, 'taken_datetime']
        all_dates = array.loc[array['project_id'] == project_id, 'taken_datetime']
        discharge_date = all_dates[all_dates.index[-1]]
        time_to_discharge = discharge_date - end_of_section

        #Correct for death prior to discharge

        #Now assign depending on how soon:
        if (time_to_discharge < np.timedelta64(2, 'D')) and (outcomes[position, 2] != 1):
            outcomes[position, 5] = 1
        elif time_to_discharge < np.timedelta64(7, 'D') and (array.loc[end_position, 'time_to_death'] > 7/365):
            outcomes[position, 6] = 1
        else:
            outcomes[position, 7] = 1

        #Now do something with PEWS as triplicate - get all PEWS after end of this slice
        pt_locs = np.where(array['project_id'] == project_id)

        #Get all slots from end of slice to end of patient
        next_slices = np.intersect1d(pt_locs[0], range(end_position, array.shape[0]))
        all_PEWS = array.loc[next_slices, 'PEWS']
        
        #Get maximum PEWS from current slice
        max_pews = np.max(array.loc[start_position:end_position, 'PEWS'])

        #Get all PEWS deterioration
        PEWSover8 = all_PEWS > (max_pews + 1)
        next_dates = all_dates[PEWSover8.index]
        PEWSdate = next_dates[PEWSover8]

        if len(PEWSdate) > 0 or array.loc[end_position, 'time_to_death'] <= 1/365:
            
            #As above with discharge date
            time_to_PEWS = PEWSdate[PEWSdate.index[0]] - end_of_section
            
            #Currently setting cutoffs to <6h, 6-24h, >24h
            if time_to_PEWS < np.timedelta64(6, 'h') or array.loc[end_position, 'time_to_death'] <= 0.25/365:
                outcomes[position, 8] = 1
            elif time_to_PEWS < np.timedelta64(1, 'D') or array.loc[end_position, 'time_to_death'] <= 1/365:
                outcomes[position, 9] = 1
            else:
                outcomes[position, 10] = 1
        else:
            outcomes[position, 10] = 1

        #Time to death as outcome (if doesn't die then 70yrs to death)
        if np.isnan(array.loc[end_position, 'time_to_death']):
            outcomes[position, 11] = 70
        else:
            outcomes[position, 11] = array.loc[end_position, 'time_to_death']

        ##Now get pointwise variables - could do median and mad depending on how scale to percentile looks?
        means = [np.mean(temp_array[:, i]) for i in range(np.shape(temp_array)[1])]
        st_devs = [np.std(temp_array[:, i]) for i in range(np.shape(temp_array)[1])]
        array_characteristics[position, :] = np.array(means + st_devs)
        
        ##Fit splines to approximate curve
        for j in range(temp_array.shape[1]):

            #Get polynomial values, smoothing seems to make this all the same length (8, will need to make this adaptable if change length)
            polynomials = scipy.interpolate.splrep(x = range(len(temp_array[:, j])), y = temp_array[:, j], s= 10000)[1]
            splines[position, range(j*8, (j+1)*8)] = polynomials[:8]

        bar.next()
        
    bar.finish()
    
    na_loc = np.isnan(array_3d)
    array_3d[na_loc] = 0
    na_loc2 = np.isnan(array_2d)
    array_2d[na_loc2] = 0
    na_loc_char = np.isnan(array_characteristics)
    array_characteristics[na_loc_char] = 0

    return array_3d, array_2d, array_characteristics, splines, outcomes, slicesPerPatient, all_used_cols, j_point_cols, j_series_cols

#Note that HCT. is HCT#
point_cols = ['ALT', 'Albumin', 'AlkPhos','AST', 'Aspartate', 'Amylase', 'APTT', 'Anion_gap', 'Base_excess', 'Basophils', 'Bicarb',
            'pH', 'Blood_culture', 'CRP', 'Ca2.', 'Cl', 'Eosinophils', 'FHHb', 'FMetHb', 'FO2Hb', 'Glucose', 'HCT.', 'HCT', 'INR',
            'Lactate', 'Lymphs', 'Mg', 'Monocytes', 'Neuts', 'P50', 'PaCO2', 'PcCO2', 'PmCO2', 'PaO2', 'PcO2', 'PmO2', 'PO2', 
            'PvCO2', 'PcO2.1', 'Phos', 'Plts', 'K.', 'PT', 'Retics', 'Na.', 'TT', 'Bili', 'WCC', 'Strong_ion_gap', 'Age_yrs', 'sex', 
            'ethnicity', 'Weight_z_scores']


series_cols =  ['Ventilation', 'HFO', 'IPAP', 'EPAP', 'Tracheostomy', 'ETCO2', 'FiO2', 'O2Flow.kg', 'Ventilation_L_min', 'Ventilation.ml.',
                'MeanAirwayPressure', 'Ventilation_missing', 'O2Flow.kg_missing', 'IPAP_missing', 'EPAP_missing', 'FiO2_missing', 'HFO_missing',
                'Tracheostomy_missing', 'Ventilation.ml._missing', 'MeanAirwayPressure_missing', 'ETCO2_missing', 'SBP_zscore', 'DBP_zscore', 
                'MAP_zscore', 'SysBP_missing', 'DiaBP_missing', 'MAP_missing', 'HR_zscore', 'HR_missing', 'Comfort.Alertness', 'Comfort.BP', 'Comfort.Calmness',
                'Comfort', 'Comfort.HR', 'Comfort.Resp', 'AVPU', 'GCS_V', 'GCS_E', 'GCS_M', 'GCS', 'GCS_missing', 'AVPU_missing', 'CRT', 'CRT_missing',
                'SpO2', 'SpO2_missing', 'interpolated_ht_m', 'ECMO', 'ECMO_missing', 'Inotropes_kg', 'Inotropes_missing', 'RR_zscore', 'RR_missing',
                'dialysis', 'dialysis_missing', 'Temp', 'Temp_missing', 'Urine_output_kg', 'Urine_output_missing_kg', 'PEWS']

other_cols = ['time_to_death', 'died']

all_cols = point_cols + series_cols + other_cols

#Now run it
array3d, array2d, array_characteristics, splines, outcomes, slicesPerPatient, all_used_cols, j_point_cols, j_series_cols = make_3d_array(flowsheet, 180, all_cols, point_cols, series_cols)
print('Slicing done: ', datetime.now().strftime("%H:%M:%S"))
np.savez('/store/DAMTP/dfs28/PICU_data/np_arrays.npz', d3 = array3d, d2 = array2d, chars = array_characteristics, splines = splines, 
         outcomes = outcomes, per_pt = slicesPerPatient, all_used = np.array(all_used_cols), point_cols = np.array(j_point_cols), series_cols = np.array(j_series_cols))
print('Saved: ', datetime.now().strftime("%H:%M:%S"))

#Plot PEWS to decide where to set the cutoff
#Need to also work out how well PEWS itself performs in predicting outcomes
plt.hist(flowsheet['PEWS'])
plt.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/PEWS_hist.png')
