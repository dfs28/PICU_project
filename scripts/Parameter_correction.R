#### Adjust BP, other parameters using childsds package

#Setup
library(childsds)

#Read in the sheet
flowsheet = read.csv('~/Project/Project_data/files/flowsheet_output.csv')

#Will need to go through and only do corrections for that age range, then do the corretions for younger children

#Make corrections
weight =  sds(flowsheet$interpolated_weight_kg, 
            age = flowsheet$Age_yrs, 
            sex = flowsheet$sex,
            male = 'M', 
            female = 'F', 
            ref = kro.ref, 
            item = 'weight', 
            type = 'SDS')




#### Build corrections using tables
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

which(is.na(SBP_zscore) & !is.na(flowsheet$SysBP))[1]











SBP = sds(flowsheet$SysBP, 
        age = flowsheet$Age_yrs, 
        sex = flowsheet$sex,
        male = 'M', 
        female = 'F', 
        ref = bp_wuehl_age.ref, 
        item = 'SBP_24h', 
        type = 'SDS')

DiaBP = sds(flowsheet$SysBP, 
        age = flowsheet$Age_yrs, 
        sex = flowsheet$sex,
        male = 'M', 
        female = 'F', 
        ref = bp_wuehl_age.ref, 
        item = 'DBP_24h', 
        type = 'SDS')

#Calculate and correct MAP
MAP = abs(as.numeric(flowsheet$MAP))
na_locs = which(!is.na(flowsheet$SysBP) & !is.na(flowsheet$DiaBP))
MAP[na_locs] = as.numeric(flowsheet$DiaBP)[na_locs]*(2/3) + as.numeric(flowsheet$SysBP)[na_locs]/3

MAP_sds = rep(NA, length(MAP))


#Try this with try and keep making the gap smaller
for (i in seq(1, length(MAP), 10000)){
    MAP_sds[i:(i + 9999)] = try(sds(MAP[i:(i + 9999)], 
                            age = flowsheet$Age_yrs[i:(i + 9999)], 
                            sex = flowsheet$sex[i:(i + 9999)],
                            male = 'M', 
                            female = 'F', 
                            ref = bp_wuehl_age.ref, 
                            item = 'MAP_24h', 
                            type = 'SDS'))

        
        if (all(is.na(as.numeric(MAP_sds[i:(i + 9999)])))){
                for (j in seq(i, i + 9999, 100)){
                        MAP_sds[j:(j + 99)] =  try(sds(MAP[j:(j + 99)], 
                                                age = flowsheet$Age_yrs[j:(j + 99)], 
                                                sex = flowsheet$sex[j:(j + 99)],
                                                male = 'M', 
                                                female = 'F', 
                                                ref = bp_wuehl_age.ref, 
                                                item = 'MAP_24h', 
                                                type = 'SDS'))

                }

                if (all(is.na(as.numeric(MAP_sds[j:(j + 99)])))){
                        for (k in seq(j, j + 99)){
                        MAP_sds[k] =        try(sds(MAP[k], 
                                                age = flowsheet$Age_yrs[k], 
                                                sex = flowsheet$sex[k],
                                                male = 'M', 
                                                female = 'F', 
                                                ref = bp_wuehl_age.ref, 
                                                item = 'MAP_24h', 
                                                type = 'SDS'))

        
                        }
                }
        }
}
error_message <- which(!is.na(MAP_sds) & is.na(as.numeric(MAP_sds)))
sum(MAP_sds[error_message] == "Error in if (nu != 0) z <- (((q/mu)^nu - 1)/(nu * sigma)) else z <- log(q/mu)/sigma : \n  argument is of length zero\n")

#Going to have to find some normal values for different ages and then build a reference table as below for children under 5

pdf("~/Project/PICU_project/figs/SBP_hist_SDs.pdf")
par(mfrow = c(2, 1))
hist(SBP, main = 'Histogram of systolic blood pressure standard deviations', xlab = 'SD')
hist(DiaBP, main = 'Histogram of diastolic blood pressure standard deviations', xlab = 'SD')
dev.off()

anthro <- data.frame(age = c(11.61,12.49,9.5,10.42,8.42,10.75,9.57,10.48),
                    height = c(148.2,154.4,141.6,145.3,146,140.9,145.5,150),
                    sex = sample(c("male","female"), size = 8, replace = TRUE),
                    weight = c(69.5,72.65,47.3,51.6,45.6,48.9,53.5,58.5))
                    
anthro$height_sds <- sds(anthro$height,
                        age = anthro$age,
                        sex = anthro$sex, 
                        male = "male", 
                        female = "female",
                        ref = kro.ref,
                        item = "height",
                        type = "SDS")

anthro$bmi <- anthro$weight/(anthro$height**2) * 10000
anthro$bmi_perc <- sds(anthro$bmi,age = anthro$age,sex = anthro$sex, male = "male", female = "female",ref = kro.ref,item = "bmi",type = "perc")

data(who.ref)
x <- data.frame(height=c(50,100,60,54),
                            sex=c("m","f","f","m"),
                            age=c(0,2.9,0.6,0.2))
                            
sds(value = x$height, age = x$age, sex = x$sex, male = "m", female = "f", ref = who.ref, item = "height")

data(kiggs.ref)
print(kiggs.ref)
data(ukwho.ref)
print(ukwho.ref)
data(who.ref)
print(who.ref)