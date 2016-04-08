### Multiple Regression

# 1. Introduction to multiple regression
##    a. Quick review of simple regression
##    b. Interpretation of coefficients in multiple regression
##    c. Least squares estimation of coefficients
##    d. A few models
##    e. Confidence and Prediction intervals
##    f. Model diagnostics for multiple regression

# 2. More about multiple regression  
##    a. Adding categorical predictors to the multiple regression
##    b. Modeling non-linear relationships
##    c. Transformation of vars. to meet linear model assumptions 
##    d. Outliers, Leverage points

# 3. Introduction to Model selection (Move to a separate file)
##    a. Forward, backward, all subset selections
##    b. Cp, BIC and AIC rules
##    c. Post Model Inference (POSI)


# 4. Appendices
##    Appendix 1: Clean dataset Cars_04 > car_04_relular.csv
##    Appendix 2: Using formulae to obtain LS estimates
##    Appendix 3: Models with or without interactions
##    Appendix 4: Nonlinearity in predictors: transformations
##    Appendix 5: Model Diagnoses via a perfect linear model scenario
##    Appendix 6: Colinearity
##    Appendix 7: F distribution

# 5. library
library(car)   # Anova(): report F statistics
library(leaps) # regsubsets(): model selection

#######################################################
### Data: car_04_regular.csv                        ###
### Case Study: Fuel efficiency in automobils       ###
### New design: 
# Continent="Am"   
# Horsepower=225
# Weight=4
# Length=180
# Width=80
# Seating=5
# Cylinders=4
# Displacement=3.5
# Transmission="automatic"

### Question of the interests: Response is Y=MPG_City
###     1) Effects of each feature on Y
###     2) Estimate the mean MPG for all such cars
###     3) Predict Y for this particular car


### Explore the data 
# It is always crucial to look at the data first. For the purpose of 
# shortening the lecture time, I put this part in Appendix 1. 
# There the original data car_04_regular.csv is cleaned and a new 
# version called car_04_relular.csv is written out. 


### Read the data car_04_relular.csv              

### Set working directory
rm(list=ls()) # Remove all the existing variables
dir=c("E:/Data Mining- STAT 571")   # my laptop
setwd(dir)

### Read the data 

data1=read.csv("car_04_regular.csv", header=TRUE)
names(data1)
dim(data1)   # 226 cars and 13 variables
str(data1)

### Define a data frame for the new design so we can use it for predictions

newcar = data1[1, ]  # Create a new row with same structure as in data1
newcar[1]="NA" # Assign features for the new car
newcar[2]="Am"
newcar["MPG_City"]="NA"
newcar["MPG_Hwy"]="NA"
newcar[5:11]=c(225, 4, 180, 80, 5, 4, 3.5)
newcar$Make = "NA"
newcar[13]="automatic"
newcar


### model 1: Simple reg, MPG_City vs. Length 

# MPG_City = beta0 + beta1* Length + e

# Model assumptions

#     1) Linearity: given Length the mean of MPG_City is described as
#         beta0 + beta1* Length
#     2) Equal variance: the var's of MPG_City is same for any Length
#     3) Normality: MPG's are independent and normally distributed. 
#     Combining 1) - 3) it says: the errors are i.i.d N(0, sigma)

fit1=lm(MPG_City~ Length, data=data1)    # fit model 1

par(mgp=c(1.8,.5,0), mar=c(3,3,2,1))    # scatter plot with the LS line
plot(data1$MPG_City~ Length, data=data1, 
     pch=16, xlab="Length", ylab="MPG_City")
abline(fit1, lwd=4, col="red")

summary(fit1)
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
#   (Intercept) 45.31383    2.89753  15.639   <2e-16 ***
#   Length      -0.13983    0.01552  -9.011   <2e-16 ***

# Residual standard error: 3.175 on 224 degrees of freedom
# Multiple R-squared:  0.266,	Adjusted R-squared:  0.2628 
# F-statistic: 81.19 on 1 and 224 DF,  p-value: < 2.2e-16

# Note, coef for Length is estimated as -.13983
# We say on average MPG drops .13983 if a car is 1" longer.

### A quick model diagnostic check: residuals vs. fitted y
plot(fit1$fitted.values, fit1$residual, pch=16)   # Residual plot
abline(h=0, lwd=5, col="red")

# 1) The symmetry pattern w.r.t h=0 indicating the linearity
# assumption is ok.
# 2) A wider spread when fitted y (hat y) is in the middle seems to challenge
# the homoscedasticity.

qqnorm(fit1$residuals)
qqline(fit1$residuals)
# 3) A reasonably well fitted line indicates normality errors.

### We continue to produce confidence and prediction intervals:
# Mean intervals:

predict(fit1, newcar,  interval="confidence", se.fit=TRUE) # Mean intervals

#   fit      lwr     upr
# 1 20.1439 19.68629 20.6015 
# $se.fit
# [1] 0.2322149       # se for b0_b1*180
# $residual.scale
# [1] 3.17523         # approx.= se of the response or est. of sigma.

# The interpretation is for all the cars with Length=180, a 95% CI for the mean
# PMG is (19.68629 20.6015) 

# Interval to predict the MPG for this particular new car
predict(fit1, newcar,  interval="predict", se.fit=TRUE) # future prediction intervals

#   fit      lwr      upr
# 1 20.1439 13.87004 26.41775
# $se.fit
# [1] 0.2322149
# $residual.scale
# [1] 3.17523

# The interpretation is for this new desgin, a 95% CI for its PMG can be 
# (13.87004 26.41775). It is much wider than the mena CI: (19.68629 20.6015) 

##### End of model 1 ##########################

### Model 2: two features Length and Horsepower
# MPG_City = beta0 + beta1* Length + beta2* Housepower + e

fit2=lm(MPG_City~ Length+Horsepower, data=data1) 
# 1) The order for predictors plays no role.
# 2) The output of lm() is similar regardless how many predictors are used.
# 3) The model assumptions extended automatically here. 

summary(fit2)

# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   38.580721   2.219625  17.382  < 2e-16 ***
#   Length      -0.061651   0.012970  -4.754 3.58e-06 ***
#   Horsepower  -0.036942   0.002766 -13.355  < 2e-16 ***
#   
# Residual standard error: 2.372 on 224 degrees of freedom
# Multiple R-squared:  0.5908,	Adjusted R-squared:  0.5871 
# F-statistic: 161.7 on 2 and 224 DF,  p-value: < 2.2e-16

### NOTICE: Comparing fit2: Length+Horsepower to fit1: Length
# 1) The coef for Length changed to -0.061651 from -0.13983!!!!
# 2) R-squared is 0.5908 from 0.266 - a huge increase. (Never decreasing, why?)
# 3) RSE (Residual standard error) is now 2.372 decreased from 3.175

###################################################
# The effects of variables ( beta's) are defined within 
# the model. They all depend on what else are 
# included or accounted for!
###################################################

### The residual plots of model 2
par(mgp=c(1.8,.5,0), mar=c(3,3,2,1)) 
plot(fit2$fitted.values, fit2$residuals, pch=16,
     main="Residuals for model 2: MPG vs. Length+Horsepowwer")
abline(h=0, lwd=6, col="red")

# The pattern seems to be similar to that form model 1 except for
# the smaller mse.

# Use the default plots to put two residuals together
par(mfrow=c(2, 1)) 
plot(fit1, 1, pch=16)
plot(fit2, 1, pch=15)
par(mfrow=c(1, 1)) 

### In addition to the t tests for each coef we now have 
### F-tests: The null can be a set of parameters

# Ex1, H0: beta_HP=0 vs. H1: beta_HP != 0 
# Or   H0: lm(MPG~Length), H1: lm(MPG~Length+Horsepower)

# Ex2, H0: beta_Length=0
# Ex3, H0: beta_HP=beta_Length=0

# F_stat=((RSS(H0)-RSS(H1))/df1)/(RSS(H1)/df2)
# df1= # parameters in H1 - # parameters in H0
# df2= # of the obs's - # parameters in H1
# Use anova{base} or Anova{car} to get F_stat

# Ex1, H0: beta_HP=0
anova(fit1, fit2)

#     Model 1: MPG_City ~ Length
#     Model 2: MPG_City ~ Length + Horsepower

#   Res.Df    RSS       Df    Sum of Sq      F    Pr(>F)    
# 1    225    2263.0                                  
# 2   df2=224 1259.8  df1=1    1003.1      178.36  < 2.2e-16 ***

# F_stat=178.36
# p-vavle=pf(178.36, 1, 224, lower.tail=F)
# We reject the null (say at .0001 level) and conclude that Horsepower is
# useful after accounting for Length.
# End of Ex1. 

# We can also use Anova() from the package car to 
# get F test for each variable.

library(car) # Make the package available to the session.

Anova(fit2) 
# Response: MPG_City
#             Sum Sq  Df F value    Pr(>F)    
# Length      127.09   1  22.596 3.575e-06 ***
# Horsepower 1003.14   1 178.361 < 2.2e-16 ***
# Residuals  1259.82 224  

########## End of model 2 ###################



### Model 3:  with several continuous variables
fit3=lm(MPG_City~Horsepower+Length+Width+Seating+
          Cylinders+Displacement, data=data1)

summary(fit3) 
# Coefficients:
#                 Estimate Std. Error t value Pr(>|t|)    
# (Intercept)    45.076045   3.974102  11.342  < 2e-16 ***
#   Horsepower   -0.019881   0.004075  -4.878 2.05e-06 ***
#   Length        0.046711   0.017072   2.736  0.00672 ** 
#   Width        -0.337551   0.066222  -5.097 7.42e-07 ***
#   Seating      -0.248463   0.149239  -1.665  0.09736 .  
#   Cylinders    -0.319086   0.226297  -1.410  0.15994    
#   Displacement -0.896245   0.368305  -2.433  0.01575 *  

#   Residual standard error: 2.036 on 219 degrees of freedom
#   Multiple R-squared:  0.705,	Adjusted R-squared:  0.6969 
#   F-statistic: 87.22 on 6 and 219 DF,  p-value: < 2.2e-16


#   Q1:The coeff for Length is 0.046711. Is this estimate wrong?
#   Q2: a)  What does the F test do here?
#       b)  What does a t-test do?
#       c)  How to interpret the Rsquared and RSE?
#   Q3:Is Horsepower THE most important variable, Length the second, etc?
#   Q4:Can we conclude that none of the Seating or Cylinders is needed?

#   Answer to Q4: Not really!!!!!!
#         H0: beta_Width=beta_Cylinder=0
    fit3.0=lm(MPG_City~Horsepower+Length+Width+Displacement,
                     data=data1)

    anova(fit3.0, fit3) # At .1 level we reject the null hypothesis.

########## End of model 3 ###################


### Multiple regression with categorical variables
### Let's use Continent as one variable. It has three categories.
### Are Asian cars more efficient? How does region affect the MPG?

attach(data1)
levels(Continent)

### Model with a categorical variable. This is same as a "One Way ANOVA"

### Get the sample means and sample sd for each group
tapply(MPG_City, Continent, mean) # Oops, missing values
tapply(MPG_City, Continent, mean, na.rm=T)
tapply(MPG_City, Continent, sd, na.rm=T)
detach((data1))

### We use 2=(I-1) indicator predictors to analyse the effect of Continent.
summary(fit.continent)  	

# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)    
#   (Intercept)  18.7200     0.4160  44.998  < 2e-16 ***
#   ContinentAs   1.5249     0.5527   2.759  0.00628 ** 
#   ContinentE   -0.4607     0.6430  -0.717  0.47440    

# Residual standard error: 3.603 on 224 degrees of freedom
# Multiple R-squared:  0.05554,	Adjusted R-squared:  0.04711 
# F-statistic: 6.587 on 2 and 224 DF,  p-value: 0.001661

# 1) The intercept beta_0=mean(MPG|Am) is the mean for the base Am
# 2) beta_As = mean(MPG|As)-mean(MPG|Am): the increment between As and Am
# 3) beta_E = mean(MPG|E)-mean(MPG|Am): the increment between Europe and Am

# The F test here is to test there are no difference among the three regions
# We can also see the coding from here:

model.matrix(fit.continent) # Am is the base, each coeff captures the effect of the level vs. Am

### To estimate the mean difference between two regions, we may use package contrast()
library(contrast)
contrast(fit.continent, list(Continent='Am'), list(Continent='As'))
contrast(fit.continent, list(Continent='As'), list(Continent='E'))

# We can choose the base level as we want. 
### Set up European cars as the base category:
data1$Continent=factor(data1$Continent, levels=c("As", "Am", "E"))  
summary(lm(MPG_City~Continent, data1)) 
# Coefficients:
#           Estimate Std. Error t value Pr(>|t|)    
#   (Intercept)  20.2449     0.3639  55.627  < 2e-16 ***
#   ContinentAm  -1.5249     0.5527  -2.759  0.00628 ** 
#   ContinentE   -1.9856     0.6106  -3.252  0.00132 ** 

# Residual standard error: 3.603 on 224 degrees of freedom
# Multiple R-squared:  0.05554,	Adjusted R-squared:  0.04711 
# F-statistic: 6.587 on 2 and 224 DF,  p-value: 0.001661

# 1) The F test should be the same. 
# 2) The coef's are different but the mean est's are the same.

data1$Continent=factor(data1$Continent, levels=c("Am", "As", "E"))  
# set the levels back

### End of Multiple regression with categorical variables



### Model with both continuous or categorical variables

# We now try a model which includes all variables

names.exclude=names(data1) %in% c("Make.Model","MPG_Hwy","Make")
data2=data1[!names.exclude]  # Take a subset with all var's but "Make.Model","MPG_Hwy","Make"

fit.all=lm(MPG_City~., subset=data1[!names.exclude])
fit.all=lm(MPG_City~., data2)
summary(fit.all)

### The t-table is pretty messy: many variables are not significant 
# 1) The t test doesn't test the hypothesis that one variable is not needed
# 2) Need F tests, one for each variable.

library(car)
Anova(fit.all) #Anova{car}

# Response: MPG_City
# Sum Sq  Df F value    Pr(>F)    
#   Continent     10.08   2  1.7865 0.1700234    
#   Horsepower    33.15   1 11.7504 0.0007289 ***
#   Weight       232.44   1 82.3920 < 2.2e-16 ***
#   Length        14.14   1  5.0107 0.0262153 *  
#   Width          0.06   1  0.0210 0.8850498    
#   Seating       14.02   1  4.9702 0.0268215 *  
#   Cylinders      3.03   1  1.0749 0.3010105    
#   Displacement   0.48   1  0.1710 0.6796015    
#   Transmission   5.02   2  0.8906 0.4119297    
#   Residuals    606.54 215           

# 1) Continent is not needed after accounting for all other variables
# 2) Can we drop all the variables with p-value larger than .05???

#### Model diagnostic: check if three assumptions of 
# linear model are met.

plot(fit.all, 1) # residuals vs. fitted 
plot(fit.all, 2) # qqnormal plot of the residuals

predict(fit.all, newcar, interval = "confidence")
predict(fit.all, newcar, interval = "prediction", se.fit=TRUE)

### End of Model with both continuous or categorical variables

#### BIGGEST Question: WHICH MODEL TO USE?????












########## Appendices #####################################

### Appendix 1: Cleaning the original data Car_04.csv and 
### output a cleaner dataset: car_04_regular.csv

#dir=c("/Users/lindazhao/Desktop/Dropbox/STAT471/Data") # school
dir=c("/Users/lzhao/Dropbox/STAT471/Data")   # my laptop
setwd(dir)

data = read.csv("Cars_04.csv", header=T, as.is=F)
str(data)   # There are 242 cars in the data
data.comp=na.omit(data)  # It deletes any car with a missing value - not a good idea.
str(data.comp)  #182 cars with complete records for all the variables. 

### Explore the data
names(data)
head(data)
summary(data)

par(mgp=c(1.8,.5,0), mar=c(3,3,2,1)) 
hist(data$Horsepower, breaks=20, col="blue")  # notice that there are some super powerful cars

### Let's find out which are the super powerful cars
attach(data)  # Put the names to the path. 
Make.Model[data$Horsepower > 400] # Show models with Horsepower > 400
Make.Model[data$Horsepower > 390]
Make.Model[data$Horsepower <= 100]
detach(data)  # Names are no longer available. 

### We could find cars with horsepower > 400
data[data$Horsepower > 400, "Make.Model"] 

### Let's concentrate on  "regular" cars and exclude those super expensive ones 
datacar=data[(data$Horsepower <=390) & (data$Horsepower > 100), ]

### Take a subset with relevant variables
variables = c("Make.Model", "Continent", "MPG_City", "MPG_Hwy",
              "Horsepower","Weight","Length", "Width",
              "Seating","Cylinders","Displacement",
              "Make", "Transmission")

data1=datacar[, variables]  # subset
str(data1)
write.csv(data1, file="car_04_regular.csv", row.names = FALSE)
                            # Output the data and name it car_04_regular.csv


### Appendix 2: Using formulae to obtain LS estimates, covariance for model 3

### this session is to reproduce the LS estimates together with cov. matrix etc. 

design.x= model.matrix(fit3)  #fit3$model gives us Y and x1 x2...xp 
y=fit3$model[, 1]
design.x[,1]

beta=  (solve( t(design.x) %*% design.x)) %*% t(design.x) %*% y # reconstruct LS estimates
beta
rse=(summary(fit3))$sigma
#Cov matrix
sd.beta=sqrt(diag(cov.beta))    # check to see this agrees with the sd for each betahat
sd.beta

summary(fit3)$cov.unscaled   # inverse (X' X)
cov.beta=(rse^2) * (summary(fit3)$cov.unscaled)  # alternatively we can get the cov matrix this way


### Appendix 3: Models with or without interactions
# Let's take a look at the model which only includes HP and Continent.

## 1) Model without interaction
#   MPG=beta_0 + beta_As + beta_E + beta_1 HP + e
#   We assume that the effect of HP on MPG is the same regardless
#   which reagion the cars are produced.

fit.no.interation=lm(MPG_City~Horsepower+Continent, data1)
summary(fit.no.interation)

## 2) Model with interactions
#  MPG=beta_0 + beta_As + beta_E + 
#        beta_1_0 HP + beta_1_As HP + beta_1_E HP+ e
#  Here the effect of HP over PMG depends on Continent.
#  Similarly, beta_1_As is the increment of slope between Asian and American...

fit.with.interaction=lm(MPG_City~Horsepower*Continent, data1)
summary(fit.with.interaction)

## 3) To test there is no interaction is same as to test
#     H0: beta_1_As=beta_1_E=0

anova(fit.no.interation, fit.with.interaction)

# Model 1: MPG_City ~ Horsepower + Continent
# Model 2: MPG_City ~ Horsepower * Continent

#   Res.Df    RSS Df Sum of Sq      F   Pr(>F)   
# 1    223 1340.5                                
# 2    221 1262.2  2    78.247 6.8501 0.001299 **

# We do reject the null hypothesis with a p-value to be .0013, no terribaly small.
# That indicates some interaction effect of HP and Continent.

## 4) The following plots help us to understand the two model assumptions

## i) Model without interactions
attach(data1)
plot(Horsepower, MPG_City, pch=16, 
     main="Model w/o interactions",
     col=as.factor(Continent))
fit.no.interation=lm(MPG_City~Continent+Horsepower, data1)
coefs=(summary(fit.no.interation))$coefficients[, 1]
abline(a=coefs[1], b=coefs[4], col=1, lwd=3)
abline(a=coefs[1]+coefs[2], b=coefs[4], col=2, lwd=3)
abline(a=coefs[1]+coefs[3], b=coefs[4], col=3, lwd=3)
legend("topright", legend=c("Am", "As", "E"),
       lty=c(1,1), lwd=c(2,2), col=as.factor(sort(unique(Continent))))


## ii)  Model with interactions: the slopes for HP might be different for each region

plot(Horsepower, MPG_City, pch=16, col=as.factor(Continent),
     main="Model w interactions")
fit.with.interaction=lm(MPG_City~Continent*Horsepower, data1)
coefs=(summary(fit.with.interaction))$coefficients[, 1]
abline(a=coefs[1], b=coefs[4], col=1, lwd=3)
abline(a=coefs[1]+coefs[2], b=coefs[4]+coefs[5], col=2, lwd=3)
abline(a=coefs[1]+coefs[3], b=coefs[4]+coefs[6], col=3, lwd=3)
legend("topright", legend=c("Am", "As", "E"),
       lty=c(1,1), lwd=c(2,2), col=as.factor(sort(unique(Continent))))
detach(data1)

### End of Appendix: Model with or without interactions
#######################################################


### Appendix 4:  Nonlinearity in predictors
# x variable may relate to y through a function of x, say log(x) or x^2.
# We can handle this by transforming x. 
# First let's find out if there is some non-linear relationship there by looking at 
# pairwise scatter plots

pairs(data1[, -1]) # Horsepower may need a transformation
attach(data1)

plot(Horsepower, MPG_City, pch=16) # oops, a curve would have fitted the data better! 1/HP?
plot(1/Horsepower, MPG_City, pch=16)  #looks better!

fit.transform=lm(MPG_City~ I(1/Horsepower), data1)  # Use I()
plot(fit.transform, 1)

fit12=lm(MPG_City~Horsepower+I(Horsepower^2), data1) # fit a quadratic function
summary(fit12)
plot(fit12$fitted, fit12$residuals, pch=16)
abline(h=0, lwd=4, col="red")
plot(fit12)


##### Appendix 5: Model diagnoses

### What do we expect to see even all the model assumptions are met?
### Here is a case that all the linear model assumptions 
### are met. Once again everything can be checked by examining the residual plots

par(mfrow=c(1,3))

x=runif(100)
y=1+2*x+rnorm(100,0, 2)
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


# The following example tells us we can't look at y directly to check the
# normality assumption. Why not?
par(mfrow=c(4,1))
x=runif(1000)
y=1+20*x+rnorm(1000,0, 2)
plot(x, y, pch=16)
hist(y, breaks=40)
mean(y)
sd(y)
qqnorm(y, main="qqnorm for y")
qqline(y)

# We really need to exam residuals!
fit=lm(y~x)
qqnorm(fit$residuals, main="qqnorm for residuals")
qqline(fit$residuals)
par(mfrow=c(1,1))

#################



### Appendix 6: Colinearity
### When some x's are highly correlated we can't separate the effect. But
### it is still fine for the purpose of prediction.

### A simulation to illustrate some consequences of colinearity
###  Each p-value for x1 and x2 is large but the null of both
### coef's =0 are rejected....  Because x1 and x2 are highly correlated.

par(mfrow=c(2,1))
x1=runif(100)
x2=2*x1+rnorm(100, 0,.1)    # x1 and x2 are highly correlated
y=1+2*x1+rnorm(100,0, .7)   # model

newdata.cor=cbind(y, x1,x2) # to see the strong correlations..
pairs(newdata.cor, pch=16)


summary(lm(y~x1))  # Each var is a useful predictor
summary(lm(y~x2))  # Each var is a useful predictor
summary(lm(y~x1+x2)) # But putting both highly correlated var's together
# we can't separate the effect of each one, though the model is still useful!
cor(x1, x2)


##### Appendix 7: a glance at F distribution

### a quick look at an F distribution. One may change df1=4 and df2=221 to see
## the changes in the distribution.
hist(rf(10000, 4, 221), freq=FALSE, breaks=200)   # pretty skewed to the right

Fstat=(summary(fit3))$fstat    # The Fstat, df1, df2
Fstat
pvalue=1-pf(Fstat[1], 5, 219) 
pvalue #or
pf(Fstat[1], 5, 219, lower.tail=FALSE)

qf(.95, 5, 219) # As long as Fstat is larger than 2.26, we reject the null at .05 level.


