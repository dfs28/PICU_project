#### Adjust BP, other parameters using childsds package

#Setup
library(childsds)

#Read in the sheet
flowsheet = read.csv('~/Project/Project_data/files/flowsheet_output.csv')

#Make corrections
weight =  sds(flowsheet$interpolated_weight_kg, 
            age = flowsheet$Age_yrs, 
            sex = flowsheet$sex,
            male = 'M', 
            female = 'F', 
            ref = kro.ref, 
            item = 'weight', 
            type = 'SDS')

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
MAP = as.numeric(flowsheet$MAP)
na_locs = which(!is.na(flowsheet$SysBP) & !is.na(flowsheet$DiaBP))
MAP[na_locs] = as.numeric(flowsheet$DiaBP)[na_locs]*(2/3) + as.numeric(flowsheet$SysBP)[na_locs]/3
MAP_nas = which(!is.na(MAP))
MAP_sds = rep(NA, length = length(MAP))
MAP_sds[MAP_nas] = sds(MAP[MAP_nas], 
        age = flowsheet$Age_yrs[MAP_nas], 
        sex = flowsheet$sex[MAP_nas],
        male = 'M', 
        female = 'F', 
        ref = bp_wuehl_age.ref, 
        item = 'MAP_24h', 
        type = 'SDS')

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

data(who.ref)x <- data.frame(height=c(50,100,60,54),
                            sex=c("m","f","f","m"),
                            age=c(0,2.9,0.6,0.2))
                            
sds(value = x$height, age = x$age, sex = x$sex, male = "m", female = "f", ref = who.ref, item = "height")
