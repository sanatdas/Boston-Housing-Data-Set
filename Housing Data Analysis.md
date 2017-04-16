House Price Prediction
=========================================



## Overview

This project aims are finding a statistical model to estimate the median value of a house in Boston area given property related paraments.To achieve the accuracy of prediction accuracy, the model performances are verified and compared. 

The data for this project is taken from UCI Machine Learning Repository.

The project covers the following sections:   

(1) Data Processing :  
This section covers the process for data downloading, exploring the data, correlation analysis to find out variable with strong correlation with median house price variable "Medv" and selecting the required attributes for processing.  
(2) Model Builing and Comparison:  
For this project, the following statistical methods are used for predicting the median price of the house:
  * Linear Gression
  * Lasso Regression
  * Random Forest
  The performance of the models are verifies using RMSE value
(3) Conclusion:  
This section includes the results of the analysis.
After completing the analysis, it was found that Randowm Forest gives a better result than linear regression and lasso regression.

## Data Processing

### Loading Required packages and Libraries


```r
install.packages("corrplot", repos = "http://cran.us.r-project.org")
install.packages("car", repos = "http://cran.us.r-project.org")  
install.packages("Metrics", repos = "http://cran.us.r-project.org")  
install.packages("randomForest", repos = "http://cran.us.r-project.org")  
install.packages("lars", repos = "http://cran.us.r-project.org")   
install.packages("ggplot2", repos = "http://cran.us.r-project.org") 
```


```r
library(corrplot)  
library(car)   
library(Metrics)  
library(randomForest)  
library(lars)   
library(ggplot2) 
```

### Loading Data

```r
urlfile <-'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'

if(!exists("boston.data")){
        boston.data = read.table(urlfile)
}
#### Rename the columns to meaningful names
names(boston.data) <- c("crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat","medv")
head(boston.data)
```

```
##      crim   zn indus chas   nox    rm  age    dis rad tax ptratio  black
## 1 0.02731  0.0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90
## 2 0.02729  0.0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83
## 3 0.03237  0.0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63
## 4 0.06905  0.0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90
## 5 0.02985  0.0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12
## 6 0.08829 12.5  7.87    0 0.524 6.012 66.6 5.5605   5 311    15.2 395.60
##   lstat medv
## 1  9.14 21.6
## 2  4.03 34.7
## 3  2.94 33.4
## 4  5.33 36.2
## 5  5.21 28.7
## 6 12.43 22.9
```

### Exploratory Data Analysis 

```r
dim(boston.data)
```

```
## [1] 505  14
```
The dataset has 506 observations and 14 column attributes


```r
str(boston.data)
```

```
## 'data.frame':	505 obs. of  14 variables:
##  $ crim   : num  0.0273 0.0273 0.0324 0.0691 0.0299 ...
##  $ zn     : num  0 0 0 0 0 12.5 12.5 12.5 12.5 12.5 ...
##  $ indus  : num  7.07 7.07 2.18 2.18 2.18 7.87 7.87 7.87 7.87 7.87 ...
##  $ chas   : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ nox    : num  0.469 0.469 0.458 0.458 0.458 0.524 0.524 0.524 0.524 0.524 ...
##  $ rm     : num  6.42 7.18 7 7.15 6.43 ...
##  $ age    : num  78.9 61.1 45.8 54.2 58.7 66.6 96.1 100 85.9 94.3 ...
##  $ dis    : num  4.97 4.97 6.06 6.06 6.06 ...
##  $ rad    : int  2 2 3 3 3 5 5 5 5 5 ...
##  $ tax    : num  242 242 222 222 222 311 311 311 311 311 ...
##  $ ptratio: num  17.8 17.8 18.7 18.7 18.7 15.2 15.2 15.2 15.2 15.2 ...
##  $ black  : num  397 393 395 397 394 ...
##  $ lstat  : num  9.14 4.03 2.94 5.33 5.21 ...
##  $ medv   : num  21.6 34.7 33.4 36.2 28.7 22.9 27.1 16.5 18.9 15 ...
```

#### Checking for null values

```r
mean(is.na(boston.data))
```

```
## [1] 0
```

#### Corrrelation analysis to check how the variables are related to each other


```r
correlations<- cor(boston.data[,-1],use="everything")
corrplot(correlations, method="number", type="lower",  sig.level = 0.01, insig = "blank")
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png)

The response variable being variable "medv", we can infer from the above that variables rm, ptratio, and lstat and most relevant in estimating the value of "medv", which represents the median value of the house price.

#### Plotting the correlation between variables

```r
pairs(~medv + rm +ptratio+lstat,data=boston.data,main="Scatterplot Matrix")
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png)

#### Spliting Data into train dataset and test dataset (70% Training data; 30% Test data)

```r
subset <- sample(nrow(boston.data), nrow(boston.data) * 0.7)
boston.train = boston.data[subset, ]
boston.test = boston.data[-subset, ]
```

## Model Building and Comparision

### Model1 : Linear Regression
The linear regression model is built, the multicollinearity is checked and relevant variable if any is removed, the 
#building first linear regression model with train data with all the variables

```r
LR_model1 <- lm(medv ~ . , data=boston.train) 
summary(LR_model1) 
```

```
## 
## Call:
## lm(formula = medv ~ ., data = boston.train)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -14.2595  -2.7722  -0.5507   1.7032  27.9756 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  33.912638   6.170072   5.496 7.64e-08 ***
## crim         -0.106049   0.037557  -2.824 0.005028 ** 
## zn            0.046093   0.016660   2.767 0.005974 ** 
## indus        -0.036531   0.072892  -0.501 0.616583    
## chas          1.417883   0.993014   1.428 0.154253    
## nox         -15.554955   4.632989  -3.357 0.000876 ***
## rm            4.113419   0.513018   8.018 1.76e-14 ***
## age          -0.013705   0.015839  -0.865 0.387508    
## dis          -1.558159   0.243781  -6.392 5.43e-10 ***
## rad           0.278864   0.082747   3.370 0.000838 ***
## tax          -0.011780   0.004570  -2.578 0.010367 *  
## ptratio      -0.923609   0.161344  -5.724 2.29e-08 ***
## black         0.008991   0.003317   2.711 0.007050 ** 
## lstat        -0.446704   0.062663  -7.129 6.15e-12 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 4.794 on 339 degrees of freedom
## Multiple R-squared:  0.7319,	Adjusted R-squared:  0.7217 
## F-statistic:  71.2 on 13 and 339 DF,  p-value: < 2.2e-16
```

#### Checking collinearity of model1

```r
vif(LR_model1)
```

```
##     crim       zn    indus     chas      nox       rm      age      dis 
## 1.696971 2.222184 3.795760 1.069990 4.405560 2.057437 3.008109 3.901707 
##      rad      tax  ptratio    black    lstat 
## 8.085046 9.340837 1.773260 1.347312 3.135972
```

Looking at VIF(variance inflation factor) values from the model, the variable "tax"" having highest VIF value at 9.365086, it is removed.

#### Removing the variable "tax" with highest collinearity from the model

```r
LR_model2 <- lm(medv ~ . -tax, data=boston.train)
summary(LR_model2)
```

```
## 
## Call:
## lm(formula = medv ~ . - tax, data = boston.train)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -14.7965  -2.8833  -0.4757   1.8103  28.2214 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  32.232069   6.186253   5.210 3.28e-07 ***
## crim         -0.105908   0.037867  -2.797  0.00545 ** 
## zn            0.037937   0.016492   2.300  0.02204 *  
## indus        -0.107864   0.067993  -1.586  0.11358    
## chas          1.747532   0.992885   1.760  0.07930 .  
## nox         -17.268119   4.622977  -3.735  0.00022 ***
## rm            4.266837   0.513767   8.305 2.40e-15 ***
## age          -0.015301   0.015958  -0.959  0.33832    
## dis          -1.607508   0.245037  -6.560 2.00e-10 ***
## rad           0.108073   0.049980   2.162  0.03129 *  
## ptratio      -0.953407   0.162260  -5.876 1.00e-08 ***
## black         0.009222   0.003343   2.759  0.00611 ** 
## lstat        -0.435070   0.063017  -6.904 2.49e-11 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 4.833 on 340 degrees of freedom
## Multiple R-squared:  0.7267,	Adjusted R-squared:  0.717 
## F-statistic: 75.33 on 12 and 340 DF,  p-value: < 2.2e-16
```

#### Removing variables that are not statistically significant:
Based on high p-value, the variables age, indus are also removed from the  model

```r
LR_model <-  lm(formula=medv ~ . -tax-age-indus,data=boston.train)
```

#### Diagnosis of residuals

```r
layout(matrix(c(1,2,3,4), 2, 2, byrow = TRUE))
plot(LR_model)
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png)

```r
par(mfrow=c(1,1))
```

#### Checking the performance of linear regression model with RMSE value.

```r
Prediction_1<- predict(LR_model, newdata= boston.test)
lr_rmse <- rmse(log(boston.test$medv),log(Prediction_1)) 
```

* RMSE value for Linear Regression : 0.226143286242468

### Model2 : Lasso Regression

```r
Ind_Var<- as.matrix(boston.train[,1:13])
Dep_Var<- as.matrix(boston.train[,14])
Las_model<- lars(Ind_Var,Dep_Var,type = 'lasso')
plot(Las_model)
```

![plot of chunk unnamed-chunk-17](figure/unnamed-chunk-17-1.png)

#### Addressing multicollinearity of the model

```r
best_step<- Las_model$df[which.min(Las_model$Cp)]
Prediction_2<- predict.lars(Las_model,newx =as.matrix(boston.test[,1:13]), s=best_step, type= "fit")
```

#### Checking the performance of the model with RMSE value

```r
las_rmse <- rmse(log(boston.test$medv),log(Prediction_2$fit))
```

* RMSE value for Lasso Regression : 0.229642343958865

### Model3 : Random Forest

```r
RF_model <- randomForest(medv~.,data= boston.train)
Prediction_3 <- predict(RF_model, newdata=boston.test)
```

#### Check the performance of Random Forest model with RMSE value

```r
rf_rmse<-rmse(log(boston.test$medv),log(Prediction_3)) 
```

* RMSE value for Lasso Regression : 0.174635205074729


## Conclusion
Considering the above machine learning models, the random forest regression model is better for estimating the median value of the housing price.  

