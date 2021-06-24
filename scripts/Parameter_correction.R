#### Adjust BP, other parameters using childsds package

#Setup
library(childsds)

#Read in the sheet
flowsheet = read.csv('/store/DAMTP/dfs28/PICU_data/flowsheet_output_touse.csv', header = TRUE, sep = ',')

#Will need to go through and only do corrections for that age range, then do the corretions for younger children

#Make corrections
flowsheet$Weight_z_scores =  sds(flowsheet$interpolated_weight_kg, 
            age = flowsheet$Age_yrs, 
            sex = flowsheet$sex,
            male = 'M', 
            female = 'F', 
            ref = who.ref, 
            item = 'weight', 
            type = 'SDS')


#### Build corrections using tables - do I need an over 15 case?
BP_df = data.frame(age = c(0.008, 1, 2, 5, 8, 12, 15), 
                   SBP_lower = c(67, 72, 86, 89, 97, 102, 110), 
                   SBP_higher = c(84, 104, 106, 112, 115, 120, 131), 
                   DBP_lower = c(35, 37, 42, 46, 57, 61, 64), 
                   DBP_higher = c(53, 56, 63, 72, 76, 80, 83), 
                   MAP_lower = c(45, 50, 50, 58, 66, 71, 73), 
                   MAP_higher = c(60, 62, 62, 69, 72, 79, 84))

normalise <- function(location, input, age, type, BP_df){
                #Function which finds normalised distance from median

                switch(type, 
                        'SBP' = {BP_df <- BP_df[, c('age', 'SBP_lower', 'SBP_higher')]}, 
                        'DBP' = {BP_df <- BP_df[, c('age', 'DBP_lower', 'DBP_higher')]}, 
                        'MAP' = {BP_df <- BP_df[, c('age', 'MAP_lower', 'MAP_higher')]})

                #Get age range
                age_range = which(BP_df$age > age[location])[1]
                if (is.na(age_range)) {age_range = dim(BP_df)[1]}

                #Get median and approx variance
                med = (BP_df[age_range, 2] + BP_df[age_range, 3])/2
                dev = med - BP_df[age_range, 2]

                #Get distance from med
                distance = input[location] - med
                return(distance/dev)

}

flowsheet$SBP_zscore = unlist(sapply(1:dim(flowsheet)[1], normalise, input = flowsheet$SysBP, age = flowsheet$'Age.yrs.', type = 'SBP', BP_df = BP_df))
flowsheet$DBP_zscore = unlist(sapply(1:dim(flowsheet)[1], normalise, input = flowsheet$DiaBP, age = flowsheet$'Age.yrs.', type = 'DBP', BP_df = BP_df))
flowsheet$MAP_zscore = unlist(sapply(1:dim(flowsheet)[1], normalise, input = as.numeric(flowsheet$MAP), age = flowsheet$'Age.yrs.', type = 'MAP', BP_df = BP_df))

#Need to read in the other things
HR_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/HR_meansd.csv', sep = ',', header = TRUE)
RR_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/RR_meansd.csv', sep = ',', header = TRUE)

#Now calculate z-scores
calc_zscore <- function(row, sheet, input_col, age_col, scortab) {
  #Function to calculate z-scores from table of mean and sd
  
  #Get age range
  age_range = which(scortab$age > sheet[row, age_col])[1]
  if (is.na(age_range)) {age_range = dim(scortab)[1]}
  
  #Get absolute distance from mean
  dev = sheet[row, input_col] - scortab$mean[age_range]
  
  #Get distance from med
  return(dev/scortab$sd[age_range])
  
}

calc_zscore(1, flowsheet, 'HR', 'Age_yrs', HR_meansd)
flowsheet$HR_zscore = sapply(1:dim(flowsheet)[1], calc_zscore, sheet = flowsheet, input_col = 'HR', age_col = 'Age_yrs', scortab = HR_meansd)
flowsheet$RR_zscore = sapply(1:dim(flowsheet)[1], calc_zscore, sheet = flowsheet, input_col = 'RR', age_col = 'Age_yrs', scortab = RR_meansd)

write.csv(flowsheet, '/store/DAMTP/dfs28/PICU_data/flowsheet_zscores.csv')