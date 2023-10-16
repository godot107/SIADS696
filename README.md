# SIADS 696 Milestone II Project Report

# Title of Project

Willie Man (manwill@umich.edu), James Conner (jamcon@umich.edu), Ruwan Jayasumana (ruwanjay@umich.edu)

# Introduction

In the US, cars are the most utilized modes of transportation for every day activity and life. Compared to European countries like Amsterdam that uses bicycles, Paris, France on subways, and Rome, Italy on high speed rails, US citizens for the most-part rely solely on cars. This raises concern on the scalability of urban planning and affordable housing to various areas in the US.

Using government curated and publically accessible datasets, our goal is to Analyze the characteristics of highly accessible population centers in the United State with the hope to identify the characteristics that can improve the daily lives of their citizens by increasing the accessibility of amenities such as jobs, entertainment, and shopping. It will also be useful for home owners attempting to identify regions of a city which feature a category of mobility that suits their particular need.

The outcome of this report is to identify the characteristics which makes a city more accessible than others, in order to create a model that predicts the accessibility of a population center with those characteristics.

# Related Work

1. [The Influence of Land Use on Travel Behavior](https://www.sciencedirect.com/science/article/abs/pii/S0965856400000197) (Boarnet and Crane 2001)
  1. The Influence of Land Use on Travel Behavior (Boarnet and Crane 2001) seeks to identify key factors determining the likelihood of consumers using mass transit over single family automobiles. The study regresses trip behavior (number of trips by walking, mass transit and automobile) of Southern California residents with three classes of independent variables: 1) Price of travel and the income level of the individual household, 2) Socio-demographic "taste variables such as gender, education levels, age and number of people per household and 3) Land use and urban design characteristics (as regressed in previous similar studies)
  2. Our project will incorporate social-demographic variables such as urban or not, access to food, and urban design such as road network density. However, instead of using income level of individual households, we will use housing prices by FIPS code.
2. [The economic value of neighborhoods: Predicting real estate prices from the urban environment](https://www.mdpi.com/2071-1050/12/2/593)
  1. Use home listings, oepnstreetmap, and census data on population, buildings, and industries to predict home value.
  2. In terms of differences, the research scope are individual property and neighborhood. Our research is scoped to census tracts.
3. [Neighborhood Walkability and Housing Prices: A Correlation Study](https://www.mdpi.com/2071-1050/12/2/593)
4. [Walking the Walk: How Walkability Raises Home Values in US Cities](https://nacto.org/docs/usdg/walking_the_walk_cortright.pdf)

# Data Sources

There were two major data sources for this project and two additional sources used to supplement each. The first major source is one [EPA Smart Location Database](https://edg.epa.gov/EPADataCommons/public/OA/SLD/SmartLocationDatabaseV3.zip). The second major source was [Zillow Home Prices](https://www.zillow.com/research/data/). The additional sources are [USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data/) and [FRED Monthly Supply of New Houses in the US](https://fred.stlouisfed.org/series/MSACSR).

Each of the data assets contained at a minimum the State and County level FIPS codes. These codes were used as the join keys between the various data assets, with State and County being the coarsest grain, and State/County/Census Block being the lowest grain. These codes were also used as the groupby key for much of the ML and analytics work

## EPA Smart Location Database

Originally in GDB format. Using geopandas, the dataset is formatted to CSV and processed into parquet format for faster reading. The EPA SLD contains a variety of features, including one we were using as a dependent variable, the National Walkability Index. Other features included indexes composed of granular demographic and transit features. EPA SLD contains data from a variety of sources and times, including the Census LEHD 2017, the Census 2019 Tiger, Census ACS 5yr from 2018, and GFTS 2020.

## Zillow Home Prices

Can be exported as CSV. The Zillow Home Value Index (ZHVI) contained times series data at the State/County FIPS grain regarding the "middle tier" (ie 35th to 65th percentile) typical home value on a per month basis. Zillow ranges from Jan-01-2012 to July-01-2023.

## USDA Food Atlas

The USDA Food Atlas is a comprehensive data asset in and of itself, but we only used it for a few of its features, namely the binary variables that defined a FIPS as Urban, Low Access to Food, Low Income, and Vehicle Accessibility. USDA Food Atlas was downloaded as a multi-sheet Excel file. USDA Food Atlas was updated in 4/27/2021.

## Pre-Processing

- Removed records that are not states (Territories & Holdings)
- SLD
  - Dropped records where features is out-of-range from the study. E.G., All CBGs with population-weighted centroids that were further than three-quarter miles from a transit stop were assigned a value of "-99999.
- Zillow
  - combine identifiers StateCodeFIPS and MunicipalCodeFIPS to create 'FIPS'. This allows merging with SLD and others.
  - Flatten cols using pd.melt by Year
  - Extract year and month from date
  - Some Zillow records were missing, especially for the older data and less popular FIPS locations. This decision was made to use 10 years of data for training (from 2012 to 2022), which largely removed the majority of nan values. Since we were using Prophet for time series analysis, there was some tolerance for missing data, so the few remaining nan values were subsequently dropped for that analysis.

# Feature Engineering

Food Desert

Urban Percentages

Home Values Differences

# Supervised Learning

## Objective

As part of the goal of understanding the characteristics of highly accessible population centers in the United States, we looked at the National Walkability Index (NWI) metric that was created by the US Environmental Protection Agency in their Smart Location Database (SLD). The NWI is a complex index that is a composite of four other metrics that are related to Employment (D2A\_Ranked, D2B\_Ranked), Transit Accessibility (D3B\_Ranked), and Street Density (D4A\_Ranked). It requires a great deal of data from a variety of sources that can be laborious to retrieve and recompose, if a researcher wanted to rebuild the Index themselves.

Our goal became to see if it was possible to approximate the NWI without using the same data resources that the SLD was built from (Census, Labor statistics, Transit statistics and Mapping data). The NWI therefore became our dependent variable that we were trying to predict.

## Data Sources

Refer to the **Data Sources** , **Preprocessing** and **Feature Engineering** sections above to see details on the Zillow, Smart Location Database, and the Food Atlas data source.

## Methods and Evaluation

We use several different regression algorithms to create models that attempt to approximate the National Walkability Index.

1. Ridge Regression - This is the baseline regression algorithm that was used to validate the other algorithms are performing adequately. The Scikit-Learn version of the algorithm was used in this analysis.
2. Random Forest Regression - A tree ensemble algorithm, this modeling method leverages the concept of "bagging", where multiple subsets of the data source are sampled with replacement to train individual trees. The individual trees are how regularization is introduced to the model, as the various trees are then aggregated to produce a final output, which reduces overall variance. It is capable of handling missing values without requiring imputation, and can also capture non-linear relationships between the dependent and independent variables.
3. XGBoost Regression - XGBoost is also a tree based algorithm, but unlike RandomForest, it builds each tree sequentially with the focus of correcting the errors of the previous tree. This method leverages the concept of using the gradient of the loss function to direct the construction of the next tree. It uses L1 and L2 regularization to prevent overfitting, and like RandomForest, it is insensitive to missing data.
4. PyTorch Neural Net Regression - The PyTorch Neural Network that was used in this report was built from scratch, using only one hidden layer and ReLU Linear activation functions. An additional Dropout layer was included in each model architecture, along with an Early Stopping function to add a method of regularization and to prevent overfitting. The element of using Dropout layers makes the model inherently non-deterministic, due to the randomness introduced into the training process, despite setting a seed variable. The PyTorch NN was the only model that was trained on a GPU.

Each of the methods listed above utilized a hyperparameter tuning method, and leveraged 5 fold cross validation, using either Hyperopt or Scikit-Learn's (SKLearn) HalvingGridSearchCV or ParamGrid. The choices for the hypertuning library were quite deliberate.

We chose the HalvingGridSearchCV method for RandomForest models because of the independent nature of the tree ensembles. With 32 processors available, each processor could work on an independent tree with sample data, and each iteration would leverage larger samples of the data on the most successful of the models. The HalvingGridSearchCV hyperparameter tuning method has the advantage of evaluating a large number of model candidates using a small amount of sample data, then iteratively selects the best candidates for training on larger samples over multiple rounds.

The Hyperopt library with Tree-based Parzen Estimators (TPE) was selected for XGBoost as a result of XGBoost being not quite as parallelizable as RandomForest, due to its boosting process being sequential (though its node splitting process is indeed parallelized). The Hyperopt TPE method implements an iterative Bayesian optimization process that was able to search through a grid space effectively and determine the most effective model with substantially fewer iterations than HalvingGridSearchCV. This optimization process can be seen in Figure # below, where promising hyperparameters are discovered early in the process, and the variation of MAE scores decreases towards the end of the run parameters are refined. ![](RackMultipart20231016-1-mhfzue_html_5623da23428abdfe.png)

The standard ParameterGrid method from Scikit-Learn was used for hypertuning the PyTorch model. This particular model required a large amount of fine tuning and tweaking of the parameter grid space for the number of hidden nodes, learning rate and dropout percentages, which meant it was more efficient to iterate over every permutation in the space to discover the optimal parameters, which is what the standard ParameterGrid method allowed. It did have an advantage over the others, in that it could use GPU resources, which drastically reduced the overall training time required. Figure # below shows the results of the 5-Fold cross validation used to generate the best Neural Net model, with the various spikes in each fold indicating the impact of dropout, and the differing number of epochs per fold illustrating the early stopping method. The maximum number of epochs per fold was set to 1000, so the early stopping algorithm was utilized to stop overfitting in each fold. ![](RackMultipart20231016-1-mhfzue_html_fe122640d77e7a1.png)

### Loss Function & Evaluation Metric

The Mean Absolute Error (MAE) method was used as the objective function and as the evaluation metric in our analysis across all models. This selection was driven by the fact that the values returned by MAE are in the same unit of measure as the dependent variable, and were consequently easily interpreted against the National Walkability Index.

### Model Determination

Despite the variety of algorithms and hyperparameter training, the non-baseline models resulted in fairly similar scores, but the XGBoost model had the lowest MAE and standard deviation scores, coming in at 0.7570 ±0.026 MAE. For this reason, the XGBoost was used for the rest of the analysis process.

| **Model** | **MAE of CV 5 Mean** | **Std Dev** | **Training (seconds)** | **Hyper Param Models** |
| --- | --- | --- | --- | --- |
| Ridge (Baseline) | 1.1 | ±0.25 | 1 | 24 |
| XGBoost | 0.757 | ±0.026 | 1052 | 1000 |
| Random Forest | 0.808 | ±0.060 | 2632 | 15246 |
| PyTorch NN | 0.798 | ±0.050 | 1022 | 706 |

### Feature Analysis

Shapely Additive Explanations (SHAP) was used to calculate the overall importance of features and provide insights into some of the highest ranked.

The SHAP "Beeswarm" plot in Figure # indicates that the _Urban\_Pct_feature is by far the most important, followed by _LATracts\_half\_Pct_ and _Home Value_.

Taking a closer look at the dependency plot for the _Urban\_Pct_ feature (colored by _Home Value_ in Figure #), we can see that there's a non-linear impact to the SHAP values. For observations with small values, the SHAP impact is nearly -1, which slowly increases to 0 when the values of the feature reach roughly 0.33. At this point, there is a plateau from ~0.33 to ~0.75, after which the impact to the SHAP value increases dramatically from 0 to 4+. ![](RackMultipart20231016-1-mhfzue_html_798e21508f1e1384.png)

Per SHAP's interaction values, the feature that has the most interactive effects with _Urban\_Pct_ is _LATracts\_half\_Pct_. Looking at the interactive effects dependency plot for those two features (Figure #) indicates a negligible impact to the SHAP value from the combination until the values of both features reach roughly 0.75, which then results in a mostly positive, but slight (maximum 0.4+) increase to the SHAP value. ![](RackMultipart20231016-1-mhfzue_html_7c1e48bd2e97ef84.png)

![](RackMultipart20231016-1-mhfzue_html_1b16e3f42763f3c6.png)

### Ablative Analysis of Top 5 Features

An ablative analysis was performed, where the Top 5 features, as reported by XGBoost in Figure #, were removed from the training and test datasets in order to measure the impact against the XGBoost model. The scores here were surprising, indicating a resiliency to the model that was unexpected. ![](RackMultipart20231016-1-mhfzue_html_2809adb3eaaf809f.png)

| **Ablated Features** | **MAE (5-Fold CV)** | **Std Dev** | **Test Data MAE** | **Test Accuracy Loss vs NWI Range** |
| --- | --- | --- | --- | --- |
| None | 0.7570 | ±0.0263 | 0.7739 | Baseline |
| Urban\_Pct | 0.8132 | ±0.0267 | 0.8132 | -0.35% |
| Home Value | 0.7818 | ±0.0278 | 0.8023 | -0.16% |
| Urban\_Pct, Home Value | 0.8727 | ±0.0290 | 0.8794 | -0.73% |
| Urban\_Pct, Home Value, LATracts\_half\_Pct | 0.8898 | ±0.0313 | 0.8860 | -0.83% |
| Urban\_Pct, Home Value, LATracts\_half\_Pct, LATracts10\_Pct | 0.9005 | ±0.0302 | 0.9031 | -0.9% |
| Urban\_Pct, Home Value, LATracts\_half\_Pct, LATracts10\_Pct, LAhalfand10\_Pct | 1.1205 | ±0.0215 | 1.1035 | -2.28% |

Key Findings:

1. The Urban Percentage has significantly more impact on the MAE score than Home Value. This was surprising, given the initial assumptions that Home Values would tend to be more expensive in rural areas, causing Urban Percentage to be of secondary importance.
2. Despite the loss of both Urban Percentage and Home Value, the model is still fairly resilient
3. It is not until the top 5 features (Urban\_Pct, Home Value, LATracts\_half\_Pct, LATracts10\_Pct, LAhalfand10\_Pct), as reported by the XGBoost Feature Importance variable, until there is significant decay in the MAE score. It is at this point that the XGBoost model performance is worse than the Ridge Regression Baseline (1.1 MAE ± 0.25).

### Hyperparameter Sensitivity

As Hyperopt was used for the training of the XGBoost model, it was important to look at some of the interactions between parameters that occurred to

![](RackMultipart20231016-1-mhfzue_html_cd017d82f2cd168f.png) ![](RackMultipart20231016-1-mhfzue_html_6f6581d73acc0688.png)

##


Adding the Food Atlas data, especially the Urban\_Pct feature

## Failure Analysis

1. By itself, the Home Value data was not enough to produce viable predictions. A large amount of time was spent building Time Series models for each FIPS code (3000+) using Facebook's Prophet library and training on 10 years of Zillow data to produce features for Home Value trends and predictions with confidence intervals. This input was used as features to build the regression models, but MAE accuracy was poor due to the volatility of the housing market in the past several years. Removing the features produced by Prophet and adding the Food Atlas features into the model improved performance substantially

| **Model** | **Home Value Alone
 MAE CV 5 Mean & StdDev **|** Home Value + Food Atlas ****MAE CV 5 Mean & StdDev** | **Improvement in**  **MAE CV Mean** |
| --- | --- | --- | --- |
| Ridge (Baseline) | 1.874 ± 0.764 | 1.1 ± 0.25 | 0.774 |
| XGBoost | 1.254 ± 0.0415 | 0.757 ± 0.026 | 0.497 |
| Random Forest | 1.389 ± 0.093 | 0.808 ± 0.060 | 0.581 |
| PyTorch NN | 1.244 ±0.011 | 0.798 ± 0.050 | 0.446 |

1. Data leakage occurred during initial model development due to the time series nature of the Zillow data In short, observations for single FIPS codes were appearing in both Train and Test datasets, and the FIPS code categorical variable was included in the dataset as well. This caused exceptionally low MAE scores that were suspicious. Selecting a single month of observations (Jan 2023) and removing FIPS codes remediated this issue.
2. Dealing with a specific failure with the final predictions, the worst performing prediction, both in terms of magnitude and percentage, was for FIPS code "31043", which corresponds to the Sioux City metropolitan region. Looking at the SHAP force plot gives us an idea of the code and the reasons for the prediction of 6.18 That is a 5.34 point miss from the actual score of 11.51. The reasoning behind this discrepancy comes down to the problems with bringing all of the data up to the grain of State/County for the FIPS codes, to match against the Zillow Home Value parameters. The Smart Location Database (SLD) has very granular data, with their FIPS codes going down to the Census Block Group level, while the Food Atlas is slightly less granular at the Census Tract level. In the case of FIPS 31043, it contains a large number of Census Block Groups in the SLD that have a very high National Walkability Index (8 out of 13 have an NWI \> 15), but there are only four Census Tracts in the Food Atlas, which is where the Urban\_Pct flag is derived from. There is an inequality in the grain of the data which causes the prediction to suffer in this case. ![](RackMultipart20231016-1-mhfzue_html_cef40c36435169d6.png)

In order to remediate this issue for future analysis, we would remove the Home Value feature from Zillow, which would raise the 5-fold CV mean MAE from 0.750 ±0.0263 to 0.7817 ±0.0278. This would allow the grain of the data to be one level deeper down to the

#


# Unsupervised Learning

## Motivation

As part of the unsupervised learning section of this project, we have decided to create several levels of clusters to understand how transit and food infrastructure intersect with the health of a housing market in a particular county.In order to do this, three k-means clustering models were created:

1. Clustering around transit infratructure, walkability, and urban density
2. Clustering based around the availability of food in a particular county
3. Clustering based around the sensitivity of a county's housing market to macro-economic shifts.

After this, we examined how each group of clusters overlapped with one another to gain an understanding of the relationship between each domain.

## Methods description

**Hyperparameter Tuning**

Initially, we set a specific number of clusters to create for each model (n = 5), but quickly discovered that those clusters were not well differentiated from one another. In fact, there was such an overlap in clusters that it was difficult to describe the differences between different centroids. This led us to conduct hyperparameter tuning with the objective of maximizing the silhouette score.

The Silhouette score is defined as: (b-a) / max(a,b) where:

a = the average eucledian distance between each point within a cluster

b = the average distance between clusters

Scores closer to one indicate a well differentiated model with clear definitions for clusters. Scores closer to -1 indicate overlapping and undifferentiated clusters. Silhouette scores were calculated for k-means models containing the following combinations of hyperparameters:

1. Algorithms (Lloyd and Elkan)- The Lloyd algorithm is the most common K-means optimization method and involves an iterative process of recalculating centroids as the mean over their closest data-points until the model reaches convergence. This has risk of getting stuck on a local optima (depending on initialization). Elkan uses a triangle inequality to speed up the performance of training the k-means algorithm. Because our feature space and observations for clustering is relatively low, all models were successfully trained with the Lloyd algorithm.
2. N-Clusters (3 to 10)- Because these clusters inform our exploratory data analysis, we did not want to create too many of them. It would be difficult to comprehend what each cluster means if we separated them into groups of more than ten. In addition, having too few (2 clusters) may create groups too large to perform any meaningful analysis. In fact, when we tried using 2 clusters, it merely split county into rural and urban counties (without differentiating between access to transit, walkability, etc).
3. Tolerance (1e-6, 1e-4, 1e-3, 1e-2)- represents the tolerance for when Scikit-learn declares convergence between two iterations. A smaller tolerance means that the same centroids would need to be closer to converge and stop iteration. A larger tolerance indicates the iterations of centroids could be further apart.

## Unsupervised evaluation

## Failure Analysis

# Discussion

## Supervised ![](RackMultipart20231016-1-mhfzue_html_e0e20a41ba671a28.png)

An early hypothesis was developed around the relationship between home prices and the National Walkability Index. The initial assumption was that homes that were in a FIPS code region with a higher NWI would see a substantial increase in home value during the past few years.

At this point, we decided to perform a correlation analysis to determine if there was any significant relationship between Zillow's Home Prices, the NWI as well as the 4 metrics that are used to build the NWI.

The data sources that are used to compose the NWI data points for each FIPS code not only come from a variety of sources, as previously mentioned, but also come from a variety of times. The data sources used to compose the Smart Location Database range from 2017 to 2022.

_First Models - Prophet for independent variables_

_Second model(s) - Regressions using Home Value_

_Third models - Regressions using Home Value and Food Atlas Urban\_Pct_

_Fourth models - Time Series on Zillow ZHVI as inputs for the Regressions (this included trend and predictions with upper and lower boundaries) … again using Jan 2023 as the comparison._

Since the data that we had from Zillow was in a time series format, we decided to use the Facebook Prophet library to build time series models on the data. We used 10 years of data, from January 2012, to December 2022 as the training data, and then used January 2023 to July 2023 as the test. Upon investigating the dataset, we discovered that there were 35,256 observations missing Home Values in the training data (out of 427,842 total observations), and only 106 observations missing from the test data.

Since Prophet is insensitive to missing data, we opted to drop incomplete records from the Zillow dataset for the purpose of building the Prophet models.

Given the lack of consistent dates used to construct the SLD dataset, we decided there was no point in attempting to align the date used from the Zillow dataset for our analysis. The Zillow data is organized in monthly periodicity time series observations, so we arbitrarily selected January 2023 observations for the correlation analysis with the SLD, and to build initial regression models.

One complication to our process was that the granularity of data between the Zillow ZHVI and SLD data sources were not aligned. The SLD dataset goes down to the FIPS Code for State/County/Census Tract/Census Block, whereas the Zillow dataset stops at State/County. Thus our analysis required us to roll up the SLD data to the State/County level, which meant we were introducing a degree of inaccuracy into our correlations and predictions from the start. The method selected for the rollup was to simply perform a mean of the features, grouped by the State/County FIPS codes. This had a side effect of reducing the range of values for the National Walkability Index from 0 - 20 to 2.72 - 15.96.

The correlation matrix between the aggregated SLD features and the Zillow Home Values can be seen in Fig. 1. Home Value has a 0.47 correlation to the National Walkability Index, and has a 0.27 correlation to the 8 Tier Employment Entropy feature (D2B\_Ranked), a 0.32 correlation to the Street Intersection Density feature (D3B\_Ranked) and a 0.44 correlation with the Distance to nearest Transit feature (D4A\_Ranked). This gave us a reasonable amount of confidence that we could predict the National Walkability Index from the Zillow Home Value.

## ![](RackMultipart20231016-1-mhfzue_html_2ef532514e96720.png)

## Unsupervised

# Ethical considerations

# Statement of Work

# References

# Appendix

11
