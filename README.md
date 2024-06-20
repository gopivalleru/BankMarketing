
<!-- TOC -->

- [The Client Will Subscribe to a Term Deposit or Not?](#the-client-will-subscribe-to-a-term-deposit-or-not)
  - [Source](#source)
  - [Prerequisites](#prerequisites)
    - [Tools and Environment](#tools-and-environment)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Understanding the Business Problem](#understanding-the-business-problem)
  - [Data Overview](#data-overview)
  - [Numerical Features Description](#numerical-features-description)
  - [Categorical Features Description](#categorical-features-description)
  - [Correlation Matrix](#correlation-matrix)
  - [Count of Term Deposits Subscriptions](#count-of-term-deposits-subscriptions)
  - [Boxplot of Numerical Features vs Term Deposit Subscription](#boxplot-of-numerical-features-vs-term-deposit-subscription)
  - [Count Plot of Categorical Features vs Term Deposit Subscription](#count-plot-of-categorical-features-vs-term-deposit-subscription)
- [Data Pre-Processing](#data-pre-processing)
- [Train and Test Split](#train-and-test-split)
- [Model Building](#model-building)
  - [Model Performance](#model-performance)
- [Conclusion](#conclusion)
  - [Feature Importance](#feature-importance)
- [Recommendations](#recommendations)
- [Next Steps](#next-steps)

<!-- /TOC -->
## The Client Will Subscribe to a Term Deposit or Not?

This project investigates a dataset related to direct marketing campaigns of a Portuguese banking institution. Often, multiple contacts with the same client were required to assess if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The classification goal is to predict if the client will subscribe (yes/no) to a term deposit (variable y).

### Source

[Moro et al., 2014] S. Moro, P. Cortez, and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.

### Prerequisites

#### Tools and Environment

**Jupyter Notebook:** Preferably via Anaconda-Navigator or any IDE supporting Jupyter Notebooks.

**Python Version:** 3.11.5

Essential Libraries

```text
matplotlib 3.7.2
seaborn 0.12.2
pandas 2.0.3
numpy 1.21.5
sklearn 1.0.2
```

## Exploratory Data Analysis

### Understanding the Business Problem

The primary objective is to predict whether a client will subscribe to a term deposit. This involves analyzing the effectiveness of marketing campaigns and identifying key factors influencing the subscription decision.

### Data Overview

The dataset is available in two versions:

1. **bank-additional-full.csv**: Contains all examples (41,188 rows) and 20 inputs, ordered by date (from May 2008 to November 2010).
2. **bank-additional.csv**: A smaller dataset (4,119 rows) used for testing more computationally demanding machine learning algorithms (e.g., SVM).

For this project, the smaller dataset was used to reduce computational complexity and time.

### Numerical Features Description

Descriptive statistics for numerical features:

| Statistic         | Age         | Duration     | Campaign     | Pdays       | Previous    | Emp. Var. Rate | Cons. Price Idx | Cons. Conf. Idx | Euribor3m   | Nr. Employed |
|-------------------|-------------|--------------|--------------|-------------|-------------|----------------|-----------------|-----------------|-------------|--------------|
| Count             | 4119        | 4119         | 4119         | 4119        | 4119        | 4119           | 4119            | 4119            | 4119        | 4119         |
| Mean              | 40.11       | 256.79       | 2.54         | 960.42      | 0.19        | 0.08           | 93.58           | -40.50          | 3.62        | 5166.48      |
| Std               | 10.31       | 254.70       | 2.57         | 191.92      | 0.54        | 1.56           | 0.58            | 4.59            | 1.73        | 73.67        |
| Min               | 18          | 0            | 1            | 0           | 0           | -3.40          | 92.20           | -50.80          | 0.64        | 4963.60      |
| 25%               | 32          | 103          | 1            | 999         | 0           | -1.80          | 93.08           | -42.70          | 1.33        | 5099.10      |
| 50%               | 38          | 181          | 2            | 999         | 0           | 1.10           | 93.75           | -41.80          | 4.86        | 5191.00      |
| 75%               | 47          | 317          | 3            | 999         | 0           | 1.40           | 93.99           | -36.40          | 4.96        | 5228.10      |
| Max               | 88          | 3643         | 35           | 999         | 6           | 1.40           | 94.77           | -26.90          | 5.05        | 5228.10      |

### Categorical Features Description

Descriptive statistics for categorical features:

| Statistic | Job      | Marital  | Education          | Default | Housing | Loan   | Contact  | Month | Day of Week | Poutcome    | Y   |
|-----------|----------|----------|--------------------|---------|---------|--------|----------|-------|-------------|-------------|-----|
| Count     | 4119     | 4119     | 4119               | 4119    | 4119    | 4119   | 4119     | 4119  | 4119        | 4119        | 4119|
| Unique    | 12       | 4        | 8                  | 3       | 3       | 3      | 2        | 10    | 5           | 3           | 2   |
| Top       | admin.   | married  | university.degree  | no      | yes     | no     | cellular | may   | thu         | nonexistent | no  |
| Freq      | 1012     | 2509     | 1264               | 3315    | 2175    | 3349   | 2652     | 1378  | 860         | 3523        | 3668|

### Correlation Matrix

![Correlation Matrix](/images/correlation_matrix.png)

### Count of Term Deposits Subscriptions

![Count of term deposits subscriptions](/images/term_deposit_subscribe.png)

### Boxplot of Numerical Features vs Term Deposit Subscription

* Duration vs Term Deposit Subscription: Clients who subscribed to term deposits had a higher median duration compared to those who did not.
* Consumer Price Index vs Term Deposit Subscription: Subscriptions to term deposits looks good when consumer price index is low.
* Euro Interbank Offered Rate 3 Months vs Term Deposit Subscription: Subscriptions to term deposits are higher when the Euribor 3-month rate is low.

![Boxplot of Numerical Features vs Term Deposit Subscription](/images/boxplot_numerical_columns_vs_term_deposit.png)

### Count Plot of Categorical Features vs Term Deposit Subscription

![Count plot of Categorical Features vs Term Deposit Subscription](/images/countplt_1_colums_vs_term_deposits.png)
![Count plot of Categorical Features vs Term Deposit Subscription](/images/countplt_2_colums_vs_term_deposits.png)

## Data Pre-Processing

1. Imputed missing values for columns 'job', 'marital', 'education', 'default', 'housing', 'loan' using `SimpleImputer` with the most frequent strategy.
2. Excluded 'duration' for predictive modeling as it is only known after the call is performed.
3. Converted `y` to integer type (0 for 'no' and 1 for 'yes').
4. Applied one-hot encoding for categorical columns and standard scaling for numerical columns as part of the pipeline to avoid data leakage.

## Train and Test Split

* Split the data into training and testing sets with a test size of 30%.
* Used `random_state` set to 25 for reproducibility
* Employed `stratify=y` to maintain class distribution in training and testing sets.

## Model Building

Given the imbalance in the dataset (11% 'yes' class), just accuracy is not a reliable metric. **Even dummy model is giving 89.03% of accuracy.**
The focus is on recall and precision, aiming for a higher F1 score to balance false positives and correctly identified subscriptions.

Initial models with default hyperparameters and tuned models using GridSearchCV with 5-fold cross-validation and F1 score as the metric were tested.

### Model Performance

| Model           | Train Time | Train Accuracy | Test Accuracy | Train Recall | Test Recall | Train Precision | Test Precision | Train F1 | Test F1 |
|-----------------|------------|----------------|---------------|--------------|-------------|-----------------|----------------|----------|---------|
| Logistic        | 0.519693   | 0.902879       | 0.908576      | 0.240506     | 0.251852    | 0.655172        | 0.739130       | 0.351852 | 0.375691|
| KNN             | 0.020142   | 0.912591       | 0.889968      | 0.332278     | 0.207407    | 0.719178        | 0.491228       | 0.454545 | 0.291667|
| DecisionTree    | 0.040505   | 0.999306       | 0.828479      | 0.993671     | 0.303704    | 1.000000        | 0.257862       | 0.996825 | 0.278912|
| SVC             | 1.560230   | 0.910163       | 0.902913      | 0.218354     | 0.177778    | 0.851852        | 0.727273       | 0.347607 | 0.285714|
| Logistic F1     | 0.522171   | 0.901838       | 0.904531      | 0.237342     | 0.244444    | 0.641026        | 0.673469       | 0.346420 | 0.358696|
| KNN F1          | 0.065956   | 0.999306       | 0.901294      | 0.993671     | 0.222222    | 1.000000        | 0.638298       | 0.996825 | 0.329670|
| DecisionTree F1 | 0.029450   | 0.940340       | 0.889159      | 0.481013     | 0.214815    | 0.950000        | 0.483333       | 0.638655 | 0.297436|
| SVC F1          | 6.816899   | 0.951093       | 0.893204      | 0.556962     | 0.251852    | 0.994350        | 0.523077       | 0.713996 | 0.340000|

Out of all the models evaluated, Logistic Regression with default hyperparameters proved to be the best, achieving the highest F1 score of 0.375691.

Below are the performance metrics presented in bar plots for all the models:

**Training Time**

* SVC has the longest training time at 1.560230 seconds, making it the least efficient in terms of training speed.
* Logistic Regression follows with a training time of around 0.52 seconds, which is relatively moderate.
* KNN and Decision Tree models are much faster, with training times of 0.020142 and 0.040505 seconds, respectively, indicating their efficiency in training.

![Training time](/images/models_comp_by_train_time.png)

**Accuracy**

* Decision Tree shows the highest training accuracy at 0.999306 but significantly drops in test accuracy to 0.828479, indicating overfitting.
* Logistic Regression and SVC have similar training and test accuracies (Logistic: 0.902879 train, 0.908576 test; SVC: 0.910163 train, 0.902913 test), suggesting good generalization.
* KNN has a slightly lower test accuracy (0.889968) compared to its training accuracy (0.912591), indicating a moderate generalization capability.

![Accuracy](/images/models_comp_by_test_accuracy.png)

**F1 Score**

* Decision Tree has the highest training F1 score (0.996825) but a lower test F1 score (0.278912), indicating overfitting.
* Logistic Regression maintains a reasonable balance with training (0.351852) and test (0.375691) F1 scores, suggesting good generalization.
* KNN and SVC have relatively lower test F1 scores (KNN: 0.291667, SVC: 0.285714), with KNN showing a significant drop from its training F1 score (0.454545).


![F1 Score](/images/models_comp_by_test_f1.png)

**ROC Curve**

The ROC curves visually indicate that Logistic Regression, Logistic Regression with F1, and KNN with F1 have high AUC values.

![ROC Curve](/images/roc_curve.png)

However, when calculating the ROC AUC scores, both Logistic Regression with default hyperparameters and Logistic Grid achieved the highest scores of 0.757628 and 0.758126, respectively. Although these two models are very close in performance, Logistic Regression with default hyperparameters has a higher F1 score.

| Model                  | ROC AUC Score |
|------------------------|---------------|
| Logistic Simple        | 0.757628      |
| KNN Simple             | 0.697322      |
| Decision Tree Simple   | 0.597948      |
| SVM Simple             | 0.701187      |
| Logistic Grid          | 0.758126      |
| KNN Grid               | 0.752306      |
| Decision Tree Grid     | 0.616450      |
| SVC Grid               | 0.638504      |

**ROC AUC Scores**
![ROC AUC](/images/ROC_AUC.png)

## Conclusion

The best performing model is **Logistic Regression with default hyperparameters**, achieving an F1 score of 0.38 and an accuracy of 90.87%. This model can effectively predict whether a client will subscribe to a term deposit.

### Feature Importance

| Feature                   | Coefficient | Absolute Coefficient |
|---------------------------|-------------|----------------------|
|  month_mar              | 1.184       | 1.184                |
|  contact_telephone      | -0.937      | 0.937                |
|  emp.var.rate           | -0.807      | 0.807                |
|  month_dec              | 0.725       | 0.725                |
|  poutcome_nonexistent   | 0.637       | 0.637                |
|  poutcome_success       | 0.585       | 0.585                |
|  month_oct              | -0.579      | 0.579                |
|  month_nov              | -0.521      | 0.521                |
|  cons.price.idx         | 0.519       | 0.519                |
|  job_entrepreneur        | -0.496      | 0.496                |
|  job_retired             | -0.450      | 0.450                |
|  job_unemployed          | 0.428       | 0.428                |
|  marital_single         | 0.402       | 0.402                |
|  month_may              | -0.401      | 0.401                |
|  job_self-employed       | -0.351      | 0.351                |
|  month_sep              | -0.343      | 0.343                |
|  job_housemaid           | -0.336      | 0.336                |
|  job_blue-collar         | -0.320      | 0.320                |
|  job_services            | -0.314      | 0.314                |
|  age                    | 0.259       | 0.259                |
|  job_management          | -0.245      | 0.245                |
|  education_basic.6y     | 0.232       | 0.232                |
|  pdays                  | -0.214      | 0.214                |
|  month_jun              | 0.213       | 0.213                |

![Feature Importance](/images/feature_importances.png)

## Recommendations

1. **Focus on Key Months**: Marketing campaigns should be intensified in March, December, October, November, and September as these months show a higher likelihood of term deposits.
2. **Leverage Previous Campaign Outcomes**: Clients with 'success' or 'nonexistent' outcomes in previous campaigns are more likely to subscribe.
3. **Optimize Contact Type**: Prioritize cellular contacts, which have a higher success rate.
4. **Consider Economic Indicators**: Monitor the consumer price index and employment variation rate as they are significant predictors.

## Next Steps

1. Collect more data to enhance model performance.
2. Incorporate additional features to capture more predictive power.
3. Experiment with advanced algorithms and ensemble methods like Random Forest, Gradient Boosting, etc.
4. Implement techniques to handle class imbalance more effectively.
5. Assess the model's performance on the full dataset.
6. Automate the data pipeline and model training process for real-time predictions.
7. Implement the model in a real-world scenario and monitor its performance.
8. Keep the model updated with new data and retrain it periodically.
9. Conduct a cost-benefit analysis to determine the optimal threshold for the model.
