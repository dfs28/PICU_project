#### Adjust BP, other parameters using childsds package

#Setup
library(childsds)
library(rriskDistributions)

#Read in the sheet
flowsheet = read.csv('Documents/Masters/Course materials/Project/PICU project/flowsheet_output.csv')

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


### Do HR, RR correction
age = c(0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 15, 25)
RR_cutoffs <- data.frame(age, 
                         first = c(25, 24, 23, 22, 21, 19, 18, 17, 17, 16, 14, 12, 11), 
                         tenth = c(34, 33, 31, 30, 28, 25, 22, 21, 20, 18, 16, 15, 13), 
                         quarter = c(40, 38, 36, 35, 32, 29, 25, 23, 21, 20, 18, 16, 15),
                         median = c(43, 41, 39, 37, 35, 31, 28, 25, 23, 21, 19, 18, 16), 
                         threequarter = c(52, 49, 47, 45, 42, 36, 31, 27, 25, 23, 21, 19, 18), 
                         ninetieth = c(57, 55, 52, 50, 46, 40, 34, 29, 27, 24, 22, 21, 19), 
                         ninety9th = c(66, 64, 61, 58, 53, 46, 38, 33, 29, 27, 25, 23, 22))

HR_cutoffs <- data.frame(age,
                         first = c(107, 104, 98, 93, 88, 82, 76, 70, 65, 59, 52, 47, 43), 
                         tenth = c(123, 120, 114, 109, 103, 98, 92, 86, 81, 74, 67, 62, 58), 
                         quarter = c(133, 129, 123, 118, 112, 106, 100, 94, 89, 82, 75, 69, 65), 
                         median = c(143, 140, 134, 128, 123, 116, 110, 104, 98, 91, 84, 78, 73), 
                         threequarter = c(154, 150, 143, 137, 132, 126, 119, 113, 108, 101, 93, 87, 83), 
                         ninetieth = c(164, 159, 152, 145, 140, 135, 128, 123, 117, 111, 103, 96, 92), 
                         ninety9th = c( 181, 175, 168, 161, 156, 149, 142, 136, 131, 123, 115, 108, 104))


get_meansd <- function(q, p){
  #Function to return mean and sd
  a <- get.norm.par(p, q)
  c(a[1], a[2])
}

#Should probably export these curves and use them
RR_meansd <- t(apply(RR_cutoffs[, 2:dim(HR_cutoffs)[2]], 1, get_meansd, p = c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
RR_meansd <- data.frame(age, mean = RR_meansd[,1], sd = RR_meansd[,2])
HR_meansd <- t(apply(HR_cutoffs[, 2:dim(HR_cutoffs)[2]], 1, get_meansd, p = c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
HR_meansd <- data.frame(age, mean = HR_meansd[,1], sd = HR_meansd[,2])





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
