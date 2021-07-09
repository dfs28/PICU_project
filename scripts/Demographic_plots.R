#### Some demographic plots and data exploration

#Setup 
library(ggplot2)
library(lubridate)

demographics <- read.csv('~/Documents/Masters/Course materials/Project/PICU project/caboodle_patient_demographics.csv')
admissions <- read.csv('~/Documents/Masters/Course materials/Project/PICU project/caboodle_patient_hospital_admissions.csv')
ward_stays <- read.csv('~/Documents/Masters/Course materials/Project/PICU project/caboodle_patient_ward_stays.csv')

#Combine patients?
encounter_locs = match(demographics$project_id, ward_stays$project_id)
demographics$encounter_key = ward_stays$encounter_key[encounter_locs]

#Remove duplicate rows and 

encounter_ids = match(demographics$project_id, ward_stays$project_id)

#Calculate birthdates
demographics$ages = NA
demographics$PICU_time = 0
demographics$CICU_time = 0
demographics$NICU_time = 0


for (i in demographics$project_id){
  #Work through all patients and pull their total age, ward locations
  
  rownum = demographics$project_id == i
  birthday = demographics$birth_date[rownum]
  admission_date = admissions$start_datetime[admissions$project_id == i]
  
  #Get age
  age = birthday %--% admission_date / years(1)
  demographics$ages[rownum] = age
  
  #Now get ward they spent most time in
  PICU_locs = (ward_stays$project_id == i) & (ward_stays$ward_code %in% c('PICU'))
  CICU_locs = (ward_stays$project_id == i) & (ward_stays$ward_code %in% c('CICU'))
  NICU_locs = (ward_stays$project_id == i) & (ward_stays$ward_code %in% c('NICU'))
  
  #Now put the times
  if (sum(PICU_locs) > 0){
    demographics$PICU_time[rownum] = sum(ward_stays$ward_stay_days[PICU_locs])}
  if (sum(NICU_locs) > 0){
    demographics$NICU_time[rownum] = sum(ward_stays$ward_stay_days[NICU_locs])}
  if (sum(CICU_locs) > 0){
    demographics$CICU_time[rownum] = sum(ward_stays$ward_stay_days[CICU_locs])}
  
}

#Think about proportion in PICU, NICU, total time in ITU, total time in each, gender, ethnicity


#Now do some relevant plotting
ggplot(demographics) + geom_histogram(aes(x = ages, fill = sex), position = 'dodge') + 
  xlab('Age') + theme_bw() + labs(title = 'Histogram of Ages')
