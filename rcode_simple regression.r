
############ Introduction   ##########################
############ Read: Chapter 2 and 3

# Read data into R
# Summary statistics
# Displaying the data

##### Simple regression (a quick review)
  
  #Case study: Billion dollar Billy Beane
    #Model specification
    #LS estimates and properties
    #R-squared and RSE
    #Confidence intervals for coef.
    #Prediction intervals
    #Caution about reverse regression
    #Appendices

# Case study: (Sports/Baseball) Will a team perform better when they are paid more?
# Is Billy Beane worth 12 million dollars as predicted?
# Read an article: Billion dollar Billy Beane
# http://fivethirtyeight.com/features/billion-dollar-billy-beane/

# Data: MLPayData_Total.csv: consists of winning records and the payroll of all
# 30 ML teams from 1998 to 2014. There are 162 games in each season.
# payroll: total pay up to 2014 in billion dollars
# avgwin: average winning percentage for the span of 1998 to 2014
# p2014: total pay in 2014
# X2014: number of games won in 2014. 
# X2014.pct: percent winning in 2014. We only need one of the two from above
# Goal: 1) How does payroll relate to the performance (avgwin)?
#       2) Is Billy Beane worth 12 million dollars as argued in the article?
#       3) Given payroll= .84, on average what would be the mean winning percentage
#       and what do we expect the avgwin for such A team? 

############ Get familiar with R  ##############
################################################

### 	One of the most important aspects of using R is to know how to import data. 
### 	Most of the data is available in a table form as a csv file already. 
###		So the simplest way to do so is to use read.csv or read.table if the data is available in a table format.


### 	Here is one example that I try to read MLPayData_Total.csv into R. The data is stored in
###		/Users/lindazhao/Desktop/Dropbox/STAT471/Data 

###	 	What is the current working directory? R will find files or save files to the wk director
getwd() 

### Set a working directoray
dir="E:/Data Mining- STAT 571"  # my laptop
dir="/Users/lindazhao/Desktop/Dropbox/STAT471/Data"  # school

setwd(dir)   #same as setwd("/Users/lzhao/Desktop/Dropbox/Data") 
getwd()


###	 If the data is in the working dir then we can read it directly: 
# datapay = read.csv("MLPayData_Total.csv", header=T, as.is=T) 
datapay = read.csv("MLPayData_Total.csv", header=T, as.is=F) 
# numbers > numbers; characters > factor

### Most importantly we need to know how to use Help: 	

??read.csv      
help(read.csv)  # The argument needs to be a function
apropos("read")	#	List all the functions with "read" as part of the function. Very useful!
#google: r: how to import a data set?
args(read.csv) # list all the arguments
str(read.csv)


### Packages (skip this for the moment)  
install.packages("ggplot2")
install.packages("MASS")     # Many packages available. We only need a few.
library(MASS)  # load package MASS
help(package=MASS)  # get information about MASS
library(lattice) # xyplot(), bwplot()

installed.packages() # packages installed
pckgs.all <- available.packages("http://cran.r-project.org/bin/windows/contrib/3.2")
grep("knn", rownames(pckgs.all), val=T) # get all the package with knn as part of the name
grep("plot", rownames(pckgs.all), val=T )
### Packages 



### Before you do any analysis it is always wise to take a quick look at the data 
### and try to spot anything abnormal.

### Find the structure the data and have a quick summary
class(datapay)  
str(datapay) # make sure the variables are correctly defined.
summary(datapay) # a quick summary of each variable

### Get a table view
fix(datapay)     # need to close the window to move on
View(datapay)

###	Look at the first six rows or first few rows
head(datapay) # or tail(datapay)
head(datapay, 2) # first two rows

### Find the size of the data or the numbers or rows and columns
dim(datapay)

### Get the variable names
names(datapay) # It is a varaible by itself


### Work with a subset which includes relevant variables
datapay[1,1] # payroll for team one
datapay$payroll # call variable payroll
datapay[, 1] # first colunm 
datapay[, c(1:3)] # first three columns



### rename "Team.name.2014" to "team"
names(datapay)[3]="team"


### The data set was manually cleaned already
sum(is.na(datapay))  # check number of missing values

### data2 is a subset without X1998 to X2014 and with the rest of the col names sorted 
#that you may find it useful.
data1=datapay[, -(21:37)] # take X1998 to X2014 out
data2=data1[, sort(names(data1)[21-37])] # sort the col names
names(data2)
###  We don't need this yet.



#### We will concentrate on three variables
# payroll, avgwin and team 


# Descriptive statistics
mean(datapay$payroll)
sd(datapay$payroll)
quantile(datapay$payroll)


# Displaying variables

hist(datapay$payroll, breaks=5) 
hist(datapay$payroll, breaks=10, col="blue") # make larger number of classes to see the details
datapay$team[which.max(datapay$payroll)] # find the team name with the max payroll

boxplot(datapay$payroll)

### Explore the relationship between payroll (x) and avgwin (y)
###  Make a scatter plot. 
plot(datapay$payroll, datapay$avgwin, pch=16, cex=1.2,
     col="blue",
     xlab="Payroll", ylab="Win Percentage",
     main="MLB Teams's Overall Win Percentage vs. Payroll")

identify(datapay$payroll, datapay$avgwin, labels=datapay$team, pos=3) 
# label some points
text(datapay$payroll, datapay$avgwin, labels=datapay$team, cex= .7, pos=1) 
# label all points

### LS estimates: functin lm() will be used extensively   

myfit0 <- lm(avgwin~payroll, data=datapay)    # avgwin is y and payroll is x
names(myfit0) # it outputs many statistics
str(myfit0) # myfit0 is a list

summary(myfit0)   # it is another object that is often used
results=summary(myfit0)
names(results) 
str(results)
# Notice the output of myfit0 and that of summary(myfit0) are different

### Scatter plot with the LS line added
  par(mgp=c(1.8,.5,0), mar=c(3,3,2,1)) 
  plot(datapay$payroll, datapay$avgwin, pch=16, 
       xlab="Payroll", ylab="Win Percentage",
       main="MLB Teams's Overall Win Percentage vs. Payroll")
  abline(myfit0, col="red", lwd=4)         # many other ways. 
  abline(h=mean(datapay$avgwin), lwd=5, col="blue") # add a horizontal line, y=mean(y)
  identify(datapay$payroll, datapay$avgwin, labels=datapay$team, cex= .7, pos=1) 
  data.frame(datapay$team,datapay$payroll,datapay$avgwin, myfit0$fitted,
             myfit0$res)[15:25,] # show a few rows
### HERE is how the article concludes that
### Beane is worth as much as the GM in Red Sox.
### By looking at the above plot, Oakland A's win pct is
### more or less same as that of Red Sox, so based
### on the LS equation, the team should have paid 2 billion.
### Do you agree?????

### More about the LS method: RSE and Rsqured

myfit0 <- lm(avgwin~payroll, data=datapay)
RSS=sum((myfit0$res)^2) # residual sum of squares
RSS
RSE=sqrt(RSS/myfit0$df) # residual standard error
RSE
TSS=sum((datapay$avgwin-mean(datapay$avgwin))^2) # total sum of sqs
TSS
Rsquare=(TSS-RSS)/TSS    # Percentage reduction of the total errors
Rsquare

Rsquare=(cor(datapay$avgwin, myfit0$fitt))^2 # Square of the cor between response and fitted values
Rsquare

# We can also get 
RSE=summary(myfit0)$sigma
RSE
Rsquare=summary(myfit0)$r.squared
Rsquare


### Inference for the coefficients, beta_1 and beta_0
### Under the model assumptions:
### i) y_i ind, normally distributed
### ii) the mean of y given x is linear
### iii) the var of y does not depend on x
### THEN we have nice properties about the LS esimates b1, b0. 
######## t intervals and t tests for beta's
######## use RSE to estimate the true sigma.

myfit0 <- lm(avgwin~payroll, data=datapay)
summary(myfit0)  # Tests and CI for the coefficients
confint(myfit0)  # Pull out the CI for the coefficients


### Confidence and Prediction intervals


datapay = read.csv("MLPayData_Total.csv", header=T, as.is=T)
myfit0 <- lm(avgwin~payroll, data=datapay)
attach(datapay)   # Caution: 1) datapay won't be changed if we do some calculations
                  # 2) Can't tell the difference if two var's have the same names
                  # but from diff data.frame's
                  # 3) detach() as soon as we are done

### 95% CI for the mean of y given x=.841. I.e., find a CI for the mean avewin
#for teams like Oakland A's

new <- data.frame(payroll=c(.841))  #new <- data.frame(payroll=c(1.24))
CImean <- predict(myfit0, new, interval="confidence", se.fit=TRUE)  
CImean


### 95% CI for a future y given x=.841
new <- data.frame(payroll=c(.841))
CIpred <-predict(myfit0, new, interval="prediction", se.fit=TRUE)
CIpred    # a 95 prediciton interval varies from .474 to .531. for a team like Oakland A. But
          # its avewin is .5445. So it is somewhat unusual but not that unusual!

CIpred_99=predict(myfit0, new, interval="prediction", se.fit=TRUE, level=.99)
CIpred_99 # a 99% prediciton interval would contain .5445!


#################################
#################################
#################################



############### Please go through the following codes about reverse reg, LS might be sensitive to
############### some extreme observations

###  Reverse regression. Warning: one can not solve x using the above LS equation. 
### Also put two LS lines in the same graph

### Plot the first LS line
par(mgp=c(1.8,.5,0), mar=c(3,3,2,1)) 
plot(datapay$payroll, datapay$avgwin, pch=16, 
     xlab="Payroll", ylab="Win Percentage",
     main="MLB Teams's Overall Win Percentage vs. Payroll")
abline(lm(avgwin~payroll, data=datapay), col="red", lwd=4) 

### Do reverse regression and overlay the LS line
myfit1 <- lm(payroll~avgwin, data=datapay)
summary(myfit1)
beta0 <- myfit1$coefficients[1]
beta1 <- myfit1$coefficients[2]
avgwin2 <- (datapay$payroll-beta0)/beta1   
lines(datapay$payroll, avgwin2, col="green", lwd=4)

legend("bottomright", legend=c("y=winpercent", "y=payroll"),
       lty=c(1,1), lwd=c(2,2), col=c("red","green"))

text(datapay$payroll, datapay$avgwin, labels=datapay$team, cex= 0.7, pos=2) # label teams


###  We may want to get the LS equation w/o Oakland first. Scatter plot with both LS lines

subdata <- datapay[-20,]

myfit2 <- lm(avgwin~payroll, data=subdata)
summary(myfit2)
plot(subdata$payroll, subdata$avgwin, pch=16,
     xlab="Payroll", ylab="Win Percentage",
     main="The effect of Oakland")
lines(subdata$payroll, predict(myfit2), col="blue", lwd=3)
abline(myfit0, col="red", lwd=3)
legend("bottomright", legend=c("Reg. with Oakland", "Reg. w/o Oakland"),
       lty=c(1,1), lwd=c(2,2), col=c("red","blue"))


###  We may want to check the effect of Yankees  (2.70, .58)

subdata1 <- datapay[-19,]
myfit3 <- lm(avgwin~payroll, data=subdata1)
summary(myfit3)

plot(subdata$payroll, subdata$avgwin, pch=16,
     xlab="Payroll", ylab="Win Percentage",
     main="The effect of Yankees")
abline(myfit3, col="blue", lwd=3)	
abline(myfit0, col="red", lwd=3)
legend("bottomright", legend=c("Reg. All", "Reg. w/o Yankees"),
       lty=c(1,1), lwd=c(2,2), col=c("red","blue"))





####### End of ML pay data analyses ##############
##################################################

### Appendix 1
### Difference between z and t with df=n. The distribution of z is 
### similar to that t when df is large, say 30.

par(mfrow=c(2,1))
z=rnorm(1000)   
par(mgp=c(1.8,.5,0), mar=c(3,3,2,1)) 
hist(z, freq=FALSE, col="red", breaks=30, xlim=c(-5,5))
#qqnorm(z)    # check normality
#qqline(z)

df=30
t=rt(1000, df)   # see what a t variable looks like
hist(t, freq=FALSE, col="blue", breaks=50, xlim=c(-5,5),
     main=paste("Hist of t with df=",df))

### Appendix 2
#### Investigate R-Squared

### case I: a perfect model between X and Y but it is not liner
### R-squared=.837  here y=x^3 with no noise!  

dev.new()
par(mfrow=c(3, 1))

x=seq(0, 3, by=.05) # or x=seq(0, 3, length.out=61)
y=x^3 

myfit=lm(y~x)
myfit.out=summary(myfit)
rsquared=myfit.out$r.squared

par(mgp=c(1.8,.5,0), mar=c(3,3,2,1)) 
plot(x, y, pch=16, ylab="",
     xlab="No noise in the data",
     main=paste("R squared= ",round(rsquared,3),sep=""))
abline(myfit, col="red", lwd=4)


### case II: a perfect linear model between X and Y but with noise
###   here y= 2+3x + e, e is iid N(0, var=9). 
### run this repeatedly

#par(mfrow=c(3,1))
x=seq(0, 3, by=.02)
e=3* rnorm(length(x))   # Normal random errors with mean 0 and sigma=3
y= 2+3*x + 3* rnorm(length(x)) 

myfit=lm(y~x)
myfit.out=summary(myfit)
rsquared = round(myfit.out$r.squared,3)
hat_beta_0=round(myfit$coe[1], 2)
hat_beta_1=round(myfit$coe[2], 2)
par(mgp=c(1.8,.5,0), mar=c(3,3,2,1)) 
plot(x, y, pch=16, ylab="",
     xlab="True lindear model with errors", 
     main=paste("R squared= ",rsquared, 
                "LS est's=",hat_beta_0, "and", hat_beta_1))

abline(myfit, col="red", lwd=4)

### End of Case II     


### Case III: Same as that in Case II, but lower the sigma=1

#par(mfrow=c(3,1))
x=seq(0, 3, by=.02)
e=3* rnorm(length(x))   # Normal random errors with mean 0 and sigma=3
y= 2+3*x + 1* rnorm(length(x)) 

myfit=lm(y~x)
myfit.out=summary(myfit)
rsquared = round(myfit.out$r.squared,3)
b1=round(myfit.out$coe[2], 3)
par(mgp=c(1.8,.5,0), mar=c(3,3,2,1)) 
plot(x, y, pch=16, ylab="",
     xlab=paste("LS estimates, b1=", b1, ",  R squared= ", rsquared),
     main="The true model is y=2+3x+n(0,1)")
abline(myfit, col="red", lwd=4)


### Appendix 3: What do we expect to see even all the model assumptions are met?
  # a) Variability of the ls estimates b's
  # b) Variability of the R squares
  # c) Model diagnosis: through residuals 

### We demonstrate this through a simulation.
### Here is a case that all the linear model assumptions 
### are met. Once again everything can be checked by examining the residual plots

par(mfrow=c(1,3)) # make three plot windows: 1 row and three columns
### Set up the simulations
x=runif(100)      # generate 100 random numbers from [0, 1]
y=1+2*x+rnorm(100,0, 2) # generate response y's 
### The true line is y=1+2x, i.e., beta0=1, beta1=2, sigma=2

fit=lm(y~x)
fit.perfect=summary(lm(y~x))
rsquared=round(fit.perfect$r.squared,2)
hat_beta_0=round(fit.perfect$coefficients[1], 2)
hat_beta_1=round(fit.perfect$coefficients[2], 2)

plot(x, y, pch=16, 
     ylim=c(-8,8),
     xlab="a perfect linear model: true mean: y=1+2x in blue, LS in red",
     main=paste("R squared= ",rsquared, 
                ", LS estimates b1=",hat_beta_1, "and b0=", hat_beta_0)
    )
abline(fit, lwd=4, col="red")
lines(x, 1+2*x, lwd=4, col="blue")

# Residual plot
plot(fit$fitted, fit$residuals, pch=16,
     ylim=c(-8, 8),
     main="residual plot")
abline(h=0, lwd=4, col="red")

# Check normality
qqnorm(fit$residuals, ylim=c(-8, 8))
qqline(fit$residuals, lwd=4, col="blue")
par(mfrow=c(1,1))

par(mfrow=c(1,1))



