Practial Machine Learning Course Project
================
Tongtian Fan
24 October, 2020

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it.

This project aims to use the data collected in the data and build a
predictive model to predict the `classe`. The analysis contains data
pre-processing, exploratory data analysis, model selections, and final
model predcitions.

## Data Pre-processing

In this section, my approaches are:

  - Load data and packages

<!-- end list -->

``` r
library('readr')
library('caret')
library('dplyr')
library('GGally')
library('Metrics')
library('ggplot2')
data <- read_csv('pml-training.csv')
```

  - Check num of columns and rows

<!-- end list -->

``` r
print(paste('This dataset has: ', dim(data)[2], 'columns and',
           dim(data)[1], 'rows.' ))
```

    ## [1] "This dataset has:  160 columns and 19622 rows."

  - Check num of predicted classes

<!-- end list -->

``` r
distinct(data, classe)
```

    ## # A tibble: 5 x 1
    ##   classe
    ##   <chr> 
    ## 1 A     
    ## 2 B     
    ## 3 C     
    ## 4 D     
    ## 5 E

  - This rest processes only applies the training data. The train and
    validation spilt is 70/30.

<!-- end list -->

``` r
set.seed(666)
trainIndex <- createDataPartition(y = data$classe, 
                                  p = 0.7, 
                                  list = F,
                                  times = 1)
# split
Train <- data[trainIndex,]
```

    ## Warning: The `i` argument of ``[`()` can't be a matrix as of tibble 3.0.0.
    ## Convert to a vector.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_warnings()` to see where this warning was generated.

``` r
Val  <- data[-trainIndex,]
```

  - Remove columns whose missing perecentages are over 50%

<!-- end list -->

``` r
# check missing pct of each variable
missing_cols = c()
for (i in c(1:dim(Train)[2])){
  missing_cols <- bind_rows(
    missing_cols,
    data.frame(
      'variable' = names(Train)[i],
      'missing_pct' = mean(is.na(Train%>%select(i)))
    )
  )
}

#remove columns with over 50% missing value 
missing_cols_rm <- missing_cols %>% 
  filter(missing_pct > 0.5) %>%
  select(variable)

for (i in missing_cols_rm){
  Train <- Train %>% 
    select(-i)
}
```

  - Remove near Zero Variables

<!-- end list -->

``` r
# remove near0 value
near0 <- nearZeroVar(Train, names = T)

Train <- Train %>% 
  select(-near0)
```

  - Remove high pairwise correlated variase by cutoff of 0.75

<!-- end list -->

``` r
# remove cor varibale
corvar <- findCorrelation(x = Train %>%
                  select_if(is.numeric) %>% 
                  cor(), 
                cutoff = 0.75,
                names = T)

Train <- Train %>% 
  select(-corvar)
```

  - Remove first 6 columns as there are more like ID\# type of columns

<!-- end list -->

``` r
#remove first 6 columns

Train <- Train %>%
  select(- c(1:6))
```

## Exploratory Data Analysis

This section aims to find the association bewteen features and labels in
order to select potential useful predictors. As there are way many
variables, I plot them in three rounds.

### First 10 variables + label

``` r
ggpairs(Train,
        columns = c(1:10,31),
        ggplot2::aes(colour=classe),
        lower = list(
          continuous = wrap("smooth", 
                            alpha = 0.3,
                            size=0.1)))
```

![](Course-Project_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

### Second 10 variables + label

``` r
ggpairs(Train,
       columns = c(11:20,31),
       ggplot2::aes(colour=classe),
       lower = list(
         continuous = wrap("smooth", 
                           alpha = 0.3,
                           size=0.1)))
```

![](Course-Project_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

### Last 10 variables + label

``` r
ggpairs(Train,
       columns = c(21:31),
       ggplot2::aes(colour=classe),
       lower = list(
         continuous = wrap("smooth", 
                           alpha = 0.3,
                           size=0.1)))
```

![](Course-Project_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

From the three plots, it is hardly to find one or two variables that can
effective differentiate the predicted classes. In this case, I plan to
try tree models as these models are robust to learn large numbers of
features. In specific, I will try the basic decision tree, a bagging
tree (random forest), and a boosting tree (XG Boost).

## Model Selection

This section aims pick the best models among decision tree, random
forest, and xg boost.

In each modeling, a K-fold cross validation will be applied and the K is
set to be 5. After this, the most balanced models of these three
algorthim will be trained. In the end, I will use validation set to test
the prediction accuracy and pick the best model.

### Decision Tree

``` r
set.seed(666)
control <- trainControl(method='cv',
                        number=5, 
                        savePredictions = T)

DT <- train(as.factor(classe)~ ., 
            data=Train, 
            method='rpart', 
            metric= 'Accuracy',
            trControl = control)

Val <- Val %>%
  select(names(Train))

DT_pred <- predict.train(DT, 
                   Val[, DT$coefnames],
                   type = 'raw' )

accuracy(DT_pred, Val$classe)
```

    ## [1] 0.5148683

### Random Forest

``` r
set.seed(666)
control <- trainControl(method='cv',
                        number=5, 
                        savePredictions = T)

RF <- train(as.factor(classe)~ ., 
            data=Train, 
            method='rf', 
            metric= 'Accuracy',
            trControl = control)


RF_pred <- predict(RF, 
                   Val[, RF$coefnames],
                   type = 'raw' )

accuracy(RF_pred, Val$classe)
```

    ## [1] 0.9930331

### XG Boost

``` r
set.seed(666)
control <- trainControl(method='cv',
                        number=5, 
                        savePredictions = T)

XGB <- train(as.factor(classe)~ ., 
            data=Train, 
            method='xgbTree', 
            metric= 'Accuracy',
            trControl = control)

XGB_pred <- predict(XGB, 
                    Val[, RF$coefnames],
                   type = 'raw' )

accuracy(XGB_pred, Val$classe)
```

    ## [1] 0.9916737

### Validation Accuracy Comparsions

From the model comparsions, both xg boost and random forest has decent
performance. I will pick random forest as:

  - Random Forest has the best validation accuracy
  - Bagging models is less likely to get overfitting than boosting
    models

<!-- end list -->

``` r
dt <- data.frame(
  'Model' = c('Decison Tree', 'Random Forest', 'XG Boost'),
  'Accuracy' = round(c(accuracy(DT_pred, Val$classe),
                 accuracy(RF_pred, Val$classe),
                 accuracy(XGB_pred, Val$classe)
                 ),3)
) 

ggplot(dt, aes(Model, Accuracy, fill = Model))+
  geom_bar(stat = 'identity', position = 'dodge') +
  geom_label(aes(label = Accuracy), 
            vjust=3, color="#CACFD2",fontface = "bold") +
  ggtitle('Model Performance Comparsions') +
  theme(plot.title = element_text(hjust = 0.5))+
  xlab('') + ylab('') +
  ylim(c(0,1)) +
  scale_fill_manual(values=c("#78281F","#512E5F", "#154360", "#0B5345" ))
```

![](Course-Project_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

### Out-of-sample Errors Estimation

The out-of-sampe estimate ranges is projected to be:

``` r
range(RF$finalModel$err.rate)
```

    ## [1] 0.001280082 0.129554656

My point estimate is :

``` r
1- accuracy(RF_pred, Val$classe)
```

    ## [1] 0.006966865

## Final Model Prediction

Here is the final model predictions for the testing datasets.

``` r
Test <- read_csv('pml-testing.csv')

predict(RF, 
        Test[, RF$coefnames],
        type = 'raw' )
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
