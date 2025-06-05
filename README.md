## Introduction

Below is the proposed solution to the challenge. The solution consists of two key components: 

* The first component deals with the QAQC of the dataset. Specifically, there are two different types of missing data.
    + Below detection limit values for Au (<0.005); and below detection limit values for Mo (-999) with unknown value for its detection limit
    + A number of missing data are marked as NA. I am going to assume that these elements were not analysed.
        
        
* The second component of the challenge is building a classifier model  to predict the class A or B for the samples marked a *?*.


Prior to that, I will use common geological knowledge to assess how minerals that host these elements are zoned with respect to each other. Based on the given assays, I expect the elements to be hosted by pyrite (FeS2), arsenopyrite (FeAsS), pyrrhotite (FeS), sphalerite ([ZnFeS]), chalcopyrite (CuFeS2), and galena (PbS). Such mineralogy will explain high values of Fe (wt% level). I will also expect Pb (galena) to be elevated proximally to the ore zone and Zn (sphalerite) to be distal. I will also examine the geochemical differences between A and B Class.


## Step 1 Read-in the Data

```{r }
#read in data
data <- read.csv("data_for_distribution.csv",header=T,sep=',')

#run a summary
summary(data)
```

From the summary of the data, I notice a few things. Firstly, *Au* column is read-in as character due to the presence of '<' sign. Secondly *holeid* and *Class* columns are being read-in is character data type and they need to be converted to factor data type.

## Step 2: Clean-up of the Data
```{r }
# replacing Au <0.005 with -0.005 and changing the data type to factor
x = data['Au']
data['Au'] <- replace(x, x=='<0.005', '-0.005')
data['Au'] <- as.numeric(data[,'Au'])
remove(x)

#changing holeid and Class data type to factor
data[,('holeid')] <- as.factor(data[,c('holeid')])
data[,('Class')] <- as.factor(data[,c('Class')])

#creating summary of the data to check the data types
#summary(data)
```

## Step 3 Imputation
Initially, I will be creating a missing map to visually check all the NA values in the data, this map ignores Au values which are marked as -0.005. I will be missmap() function from Amelia package
```{r }
#loading required libraries
library(Amelia)
library(zCompositions)

missmap(data,col = c("blue", "white"))
```

Arsenic has the highest number of *NA* values, ~30% which means that element needs to be used with caution when making predictive models. 

For the imputation of the data, I chose to use lrEMplus algorithm. It is a log-ratio expectation maximization algorithm (equivalent of the multiple linear regression). Please refer to page 14 <https://cran.r-project.org/web/packages/zCompositions/zCompositions.pdf>. This algorithm deals with left-censored (below detection limit values) and missing data simultaneously. Within the input dataset, the values below detection limit need to be marked as 0 and missing data needs to be marked as NA.

For the algorithm to start, I need at least one of the columns to not to have any missing data. Zinc has 9 *NA* values and I am going to remove those from the dataset in order to be able to run the imputation algorithm. This is not the ideal solution.

```{r}
#dropping rows which have NA zinc values
data2 <- subset(data, !is.na(data$Zn))

#creating a vector that stores names of the elements
x <- c('Au','As','Pb','Fe','Mo','Cu','S','Zn')

#creating a subset of the data that contains elements
dataToImp <- data2[,x]

#substituting 0.005 with 0 in the data subset
dataToImp[,'Au'] <- ifelse(dataToImp[,'Au'] == -0.005, 0, dataToImp[,'Au'])

#replacing -999 with 0 because it is a minimum value present for Mo, assuming it is a detection limit.
dataToImp[,'Mo'] <- ifelse(dataToImp[,'Mo'] == -999, 0, dataToImp[,'Mo'])
remove(x)

#creating a vector storing the detection limit values, in this case it is 0.005 for Au and 1 for Mo
dl <- c(0.005,0,0,0,1,0,0,0)

#running the algorithm
set.seed(1234)
dataImputed <- lrEMplus(dataToImp, dl = dl)

#merging imputed data back with hole and sample id columns
x <- c('Unique_ID','holeid','from','to','Class')
dataImputed <- cbind(dataImputed, data2[,x])
remove(x,dl,dataToImp)
```

## Exploratory Data Analysis
Generating median values per Class to see which elements show the most differences in A and B. Median values is used as opposed to mean because the distribution of minor and trace element data has lognormal and hence median is more informative than mean.
```{r }
library(dplyr)

dataImputed %>% 
  group_by(Class) %>% 
  summarize(medianAu = median(Au),
            medianAs = median(As),
            medianPb = median(Pb),
            medianFe = median(Fe),
            medianMo = median(Mo),
            medianCu = median(Cu),
            medianS = median(S),
            medianZn = median(Zn))
```
Class A is proximal to ore body and is characterized by relatively higher Pb (galena), Fe (pyrrhotite and other Fe-bearing sulphides) and Mo (molybdenite?). Distal Class B is characterized by relatively higher As (30% of data was imputed), Zn and S (sphalerite) median values.
```{r }
#importing libraries
library(ggplot2)
library(cowplot)

#generating plots
p1 <- ggplot(dataImputed, aes(x =log10(Pb), y = log10(Fe),
                   color = factor(Class))) + geom_point(shape=1,size=2)

p2 <- ggplot(dataImputed, aes(x =log10(Au), y = log10(As),
                   color = factor(Class))) + geom_point(shape=1,size=2)

p3 <- ggplot(dataImputed, aes(x =log10(S), y = log10(Zn),
                   color = factor(Class))) + geom_point(shape=1,size=2)

p4 <- ggplot(dataImputed, aes(x =log10(Mo), y = log10(Cu),
                   color = factor(Class))) + geom_point(shape=1,size=2)
#combining plots
plot_grid(p1, p2, p3, p4, labels = "AUTO")

```

Scatter plots of log-transformed data above as part of exploratory data analysis. It is visible that data is discretized. It is evident that class A (proximal to ore body) has higher Pb and Mo.

## Step 4 Subsetting the Data
I am splitting data into training and prediction sets. The training set is all of the data that has Class A and B, whereas prediciton set is all of the data that has unknown Class.
```{r}
#subset the data
dataImputedA <- subset(dataImputed, Class=="A") 
dataImputedB <- subset(dataImputed, Class=="B")
TrainSet <- rbind(dataImputedA,dataImputedB)
PredictSet <- subset(dataImputed, Class=="?")

#create a vector x that stores names of the columns that are necessary
x <- c('Au','As','Pb','Fe','Mo','Cu','S','Zn','Class')

#subset the data and drop unnecessary columns
TrainSet <- droplevels(TrainSet[,x])
PredictSet <- droplevels(PredictSet[,x])
```

## Step 5 Logistic Regression Model
Logistic regression was used as part of the exploratory data analysis. Since the target variable has two classes—A and B—a binary logistic regression model is appropriate to identify which elements are good predictors of class membership. Because logistic regression is a parametric method that assumes a normal distribution of the input variables, I applied a logarithmic transformation to the elemental data to better meet this assumption.

```{r}

model_glm = glm(Class ~ log10(Au)+log10(As)+log10(Pb)+log10(Fe)+log10(Mo)+log10(Cu)+log10(S)+log10(Zn), family="binomial", data = TrainSet)
summary(model_glm)

```

The summary of the logistic regression model shows that Fe, Zn, and possibly Cu are not significant predictors for distinguishing between Class A and B. This is consistent with the small differences in the median values of these elements across the two classes. If I were to build a classifier based on the assumption of normally distributed data, I would likely exclude Fe and Zn from the model.

## Step 6 Random Forest model
In order to predict Class A and B in the PredictSet, I am building a Random Forest model
```{r}
library(randomForest)
#Create a Random Forest model with default parameters
set.seed(1234)
model1 <- randomForest(Class ~ ., data = TrainSet, importance = TRUE)
model1

importance(model1)
```
From model importance it appears that Pb, As and Mo are the most important variables for model's ability to predict the class
```{r}
#Predicting the class
predictions <- predict(model1, PredictSet, type = 'response')
```
The Out-of-Bag (OOB) estimate is ~14% suggesting that the model might missclassify approximately 14% of the instances when making predictions on unseen data.

## Step 7 Conclusions
The QAQC process for the dataset proved to be quite challenging. To run the imputation algorithm—which requires at least one complete column—I had to drop nine rows of Zn data. The algorithm I used is capable of handling both left-censored and missing data simultaneously. However, an alternative approach could have involved using two separate algorithms or applying a regression model to predict Au values below 0.005, followed by imputing the remaining missing data.

Although it is possible to build a predictive model using the geochemical data and the A/B labels, the model currently achieves only about 14% accuracy, which is far from ideal. Some elements—particularly Pb, As, and Mo—show a greater influence on the model. This suggests that accuracy might be improved by removing less important elements such as Zn and Fe.
