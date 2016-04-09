
######## Model selection ##########

# 1) Exploring the data
# 2) Transformation on y's or some x's
# 3) Model building: forward, backward or all subsets
# 4) Criterion of accuracy for the models: Cp, BIC, Rsquared, RSS
# 5) Findings/reports

rm(list=ls())
library(ISLR)   # load data from ISLR
library(leaps)  # regsubsets()
library(car)    # Anova()

# Let's try ISLR's baseball players' salaries data
# Goal: How do players' performance affect their salaries?
# Base Ball Data: 20 variables about players including performance in 1986 
# or before, salaries in 1987...


  library(ISLR)    # A data called Hitters is available here
  
  ### 1. Exploring the data first / transformation
  help(Hitters)         # Get some basic information about the data. Q: Should we
                        # look into average career statistics instead of total numbers?
  dim(Hitters)          # 322 players with 20 variables
  names(Hitters)        # Var's 1-7: infor in 86
  str(Hitters)          # Take a quick look at the data
  pairs(Hitters)        # Way too busy
  pairs(Hitters[, c(19,1:7)])     # So many with very low salaries
  cor(Hitters[, c(19,1:13, 17,18)]) # All pairwise cor's
  hist(Hitters$Salary, breaks=40)  # Missing values?
  
  sum(is.na(Hitters[-19])) # All the features are available
  sum(is.na(Hitters[19]))  # 59 out of 322 players with salary missing
  rownames(Hitters)[is.na(Hitters$Salary)] # players without the salary
  Hitters["-Billy Beane", ] # Billy Beane doesn't have a salary here
  
  # Ignore the players without a salary
  data.comp=na.omit(Hitters)   # Not a very good idea in general!!!!! 
  dim(data.comp)            # We are keeping 263 players
	cor(data.comp[, c(19,1:13, 16, 17,18)]) # Pairwise cor's among all quantitative var's
  t(cor(data.comp$Salary, data.comp[,c(1:13, 16,17,18) ])) # only get cor's of Salary vs. the rest var's
   
  pairs(data.comp[c("Salary","CHmRun", "Hits")], pch=16) # Zoom in some scatter plots
  data1=cbind(log(data.comp$Salary), data.comp)  
  names(data1)[1]="LogSalary" # Rename it
  # Make a log transformation to Salary and a new data frame. 
  # Note: we mainly care about percentage changes in salary. Difference in log salary is equivalent to
  # percentage changes in salaries. WHY??

  pairs(data1[c("LogSalary","CHmRun", "Hits")]) # We could make some transformations for features...
  pairs(data1)
  
  plot(lm(LogSalary~Hits, data=data1), 1)  # Some selected residual plots look much better in log salary
 
  plot(lm(Salary~Hits, data=data1), 1) # The residual plot without log transformation on Salary

  data2=data1[, -20]      # Taking Salary out of the data1 so that
  # we can use all the variables as predictors
  names(data2)
  
  # 3) Model building
  
  # Given a set of p predictors, there will be 2^p (In this case there are 2^19=524288)
  # many possible models.
  # If we use RSS as a criteria, then the best model would be using
  # all variables!!!! (In this case there are 2^19=524288)
  
  
  ### Let's now try to select a good model
  library(leaps)         # leaps gives us model selection functions
  help(regsubsets)      
  
  ##### All subset selection: for each model size, report the one with the smallest RSS 
  ##### Pro: identify the "best" model. Con: computation expensive.  In our example: 2^19=524288 models
  fit.exh=regsubsets(LogSalary~.,data2, nvmax=25, method="exhaustive")
            # nvmax=8 is the default. 
            # method=c("exhaustive", "backward", "forward", "seqrep")  exhaustive is the default setting
            # nbest: output candidate models whose RSS's are similar.
  summary(fit.exh) # List the model with the smallest RSS among each size of the model
  f.e=summary(fit.exh)
  names(f.e)

  f.e$which
  f.e$rsq
  f.e$rss
  f.e$bic
  
  par(mfrow=c(2,1))     # Compare different criterions: as expected rsq ^ when p is larger
  plot(f.e$rsq, xlab="Number of predictors", ylab="rsq", col="red", type="p", pch=16)
  plot(f.e$rss, xlab="Number of predictors", ylab="rss", col="blue", type="p", pch=16)
  par(mfrow=c(1,1)) 
  
  #rsq or rss will not be good criteria to use in order to find a model which
  #has the least average prediction squared error. For that purposes we have
  #Cp. AIC is approximately equivalent to Cp. (BIC is used for different criterion.)
  #Adjusted R-squred is not a good criterion. 
  
  ### Regardless which criterions to be used, given a fixed number of predictors, we will have the same set of 
  ### covariates which achieves the min value of RSS. 
  
  coef(fit.exh,6)
  #(Intercept)         AtBat          Hits         Walks         Years         CHits     DivisionW 
  #4.6553433960 -0.0021219641  0.0119367533  0.0074019533  0.0484718580  0.0003712366 -0.1679589552 
  
  coef(fit.exh,7)
  #(Intercept)         AtBat          Hits         Walks         Years         CRuns        CWalks       PutOuts 
  #4.5376278168 -0.0027440504  0.0122944610  0.0102448096  0.0625324536  0.0013517450 -0.0011200860  0.0003313385 
  
  coef(fit.exh,8)
  #(Intercept)         AtBat          Hits         Walks         Years         CRuns        CWalks     DivisionW 
  #4.6224724768 -0.0025307251  0.0115232541  0.0101101617  0.0633255046  0.0013560086 -0.0011474205 -0.1616445726 
  #      PutOuts 
  #      0.0003309151 
  
  
  ## To find a model with smallest Cp or BIC, we only need to compare the Cp or BIC values
  # for each of the candidate model having a fixed number of predictors.
  
  # Here are the plots of cp vs number of predictors. Similarly we have the plots of BIC v.s. 
  # number of the predictors
  
  par(mfrow=c(3,1))
  plot(f.e$cp, xlab="Number of predictors", 
       ylab="cp", col="red", type="p", pch=16)
  plot(f.e$bic, xlab="Number of predictors", 
       ylab="bic", col="blue", type="p", pch=16)
  plot(f.e$adjr2, xlab="Number of predictors", 
       ylab="adjr2", col="green", type="p", pch=16)
  par(mfrow=c(1,1))
  
  ## Notice that the final model can be different in terms of the number of
  # predictors depending on which criterion to use. 
  # BIC tends to give the model with least number of predictors.
  # In this case we may use five variable models.
  
  fit.exh=regsubsets(LogSalary~.,data2, nvmax=25, method="exhaustive")
  fit.exh.var=summary(fit.exh)$which 
  fit.exh.var[5,]  # This gives us the 5 varialbes selected
  
  # To pull out the final model, we could do
  fit.exh.5=lm(LogSalary~., data2[fit.exh.var[5,]])   # Check to see it takes care of categorical var's right.
  summary(fit.exh.5)  # Note: there is no guarantee that all the var's in the final model are significant.
  
  library(car)
  Anova(fit.exh.5) # Once again this gives us the test for each var at a time.
  
  # Note: there is no guarantee that all the var's in the final model will be significant.
  ####################################################################################
  
  
  # When p is too large or in the situation p is even larger than n,
  # it is impossible to search all subsets to find the least RSS model for each given number
  # of predictors. One possibility is through forward selection.
  
  
  ### Forward selection 
  fit.forward=regsubsets(LogSalary~., data2, nvmax=19, method="forward")
  fit.forward
  f.f=summary(fit.forward)
  f.f
  
  ## At any given number, the predictors selected may vary depending
  #on the selection method. 
  
  f.e  # output from the exhausted search: 
  # p=1, CRuns. p=2, Hits and CAtBat
  f.f  # forward selection.
  # p=1, CRuns. p=2, CRuns and Hits etc...
 
  # For any fixed number, the model selected from all subset selection 
  #will have larger rsq (or smaller rss) than that from forward selection. 
  plot(f.f$rsq, ylab="rsq", col="red", type="p", pch=16,
       xlab="Forward Selection")
  lines(f.e$rsq, ylab="rsq", col="blue", type="p", pch=16,
     xlab="All Subset Selection")
  par(mfrow=c(1,1))
  
  
  # If we decided to use a model with 6 predictors by forward selection here it is:
  coef(fit.forward, 6)
  # (Intercept)          Hits         Walks         Years         CRuns     DivisionW       PutOuts 
	# 4.4761860545  0.0056601793  0.0045587527  0.0595159333  0.0006006461 -0.1750653729  0.0003107519 
  summary(lm(LogSalary~Hits+Walks+Years+CRuns+Division+PutOuts, data2))
  
  # all the above variables have coef's sig. different from 0 at .05 level
  
  
  ### Backward selection, especially useful when p is large (still smaller than n)!!!
  
  fit.backward=regsubsets(LogSalary~., data2, nvmax=19, method="backward")
  f.b=summary(fit.backward)
  par(mfrow=c(3,1))
  plot(f.f$bic,  col="red", type="p", pch=16,
     xlab="Forward Selection")
  plot(f.b$bic,  col="red", type="p", pch=16,
     xlab="Backward Selection")
  plot(f.e$bic,  col="blue", type="p", pch=16,
     xlab="All Subset Selection")
  par(mfrow=c(1,1))


  coef(fit.backward, 6)
	# (Intercept)         AtBat          Hits         Walks         Years         CRuns       PutOuts 
	# 4.5285712503 -0.0025085020  0.0131783468  0.0061555796  0.0563731950  0.0006373218  0.0003257478 
  summary(lm(LogSalary~AtBat+Hits+Walks+Years+CRuns+PutOuts, data2))
  
	coef(fit.exh,6)
	#(Intercept)         AtBat          Hits         Walks         Years         CHits     DivisionW 
	#4.6553433960 -0.0021219641  0.0119367533  0.0074019533  0.0484718580  0.0003712366 -0.1679589552 
	summary(lm(LogSalary~AtBat+Hits+Walks+Years+CHits+Division, data2))
  
  #####  Final model
  ## Since the total number of parameters is not that big, we use all subset selection

  fit.exh=regsubsets(LogSalary~.,data2, nvmax=25, method="exhaustive")

  summary(fit.exh)
  f.e=summary(fit.exh)
  
  par(mfrow=c(2,1))     # Compare different criteria 
  plot(f.e$cp, xlab="Number of predictors", ylab="cp", col="red", type="p", pch=16)
  plot(f.e$bic, xlab="Number of predictors", ylab="bic", col="blue", type="p", pch=16) # penalizing the model size
  par(mfrow=c(1,1)) 

  fit.exh.var=f.e$which
  ### Use a five var's model as my final model
  
  
  fit.final=lm(LogSalary~Hits+Walks+Years+CHits+Division, data2)    # Division has two levels, so Division is same as DivisionW
  summary(fit.final)
  
  fit.final=lm(LogSalary~., data2[fit.exh.var[5,]])
  summary(fit.final)
  
  
  ### Model diagnostics: examine the residual plot for linearity and equal variances, normal plot of the residuals for normality.
  
  par(mfrow=c(2,1))
  plot(fit.final,1)     # Everything seems to be fine. I only use the first two plots.
  plot(fit.final,2)
  par(mfrow=c(1,1))
  
  ### Final report
  
  # 1) Collectively the five features:Hits, Walks, Years, CHits and Division do a good job to predict the
  # salaries in log scale.
  # 2) Interpret each LS estimates of the coefficients
  # 3) We may want to estimate the mean salary or individual salary. For example for one with the following
  # predictor values Hits=75, Walks=50, Years=4, CHits=1200 and Division=E

  #  a) We first make a dataframe: 
player=data2[1,] # get the right format and the varialbe names
player[1]=NA
player$Hits=75
player$Walks=50
player$Years=4
player$Chits=1200
player$Division=as.factor("E")   # To make sure it is a dataframe

  #  b) Get a 95% CI for the mean salaries for all such players
player.m=predict(fit.final, player, interval="confidence", se.fit=TRUE) 
player.m  # in log scale
exp(player.m$fit)  # 319.6469 (250.2053, 408.3612) as the mean salary after transforming back to the original scale

  #  c) Get a 95% CI of the salary for the player 
player.p=predict(fit.final, player, interval="prediction", se.fit=TRUE) 
player.p  # in log scale
exp(player.p$fit) #319.6469 (91.55162, 1116.028)

    
  
  ### Remarks
  
  # 1. If you want to only start with a collection of predictors, you may
  #use the usual formula: y~x1+x2 inside the regsubsets. For example if you 
  #only want to search a best model using AtBat,Hits, Runs and AtBat^2, 
  #you may do the following
  
  summary(regsubsets(LogSalary~AtBat+Hits+Runs+I(AtBat^2), data2))
  
  
  # 2. We could also restrict the maximum possible model size.
  
  summary(regsubsets(LogSalary~., nvmax=5, method="exhaus", data2)) #We restrict a model with no more than nvmax=5 variables.
  
  