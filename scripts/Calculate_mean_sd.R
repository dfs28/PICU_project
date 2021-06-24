#Generate HR, RR z-scores
library(rriskDistributions)


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

write.csv(RR_meansd, '~/Documents/Masters/Course materials/Project/PICU_project/')
