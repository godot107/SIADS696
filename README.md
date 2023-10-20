# SIADS 696 Milestone II Project Report

# Analyze the characteristics of highly accessible population centers in the United States

Willie Man (manwill@umich.edu), James Conner (jamcon@umich.edu), Ruwan Jayasumana (ruwanjay@umich.edu)

# Introduction

In the US, cars are the most utilized modes of transportation for every day activity and life. Compared to European countries like Amsterdam that use bicycles, Paris, France on subways, and Rome, Italy on high speed rails, US citizens for the most-part rely solely on cars. This raises concern on the scalability of urban planning and affordable housing to various areas in the US. Using government curated and publically accessible datasets, our hope to improve the daily lives of their citizens by increasing the accessibility of amenities such as jobs, entertainment, and shopping.

The outcome of this report is to identify the characteristics which makes a city more accessible than others to create models that predict the accessibility of a population center with those characteristics. Realizing that the scope is beyond the allotted time, we decided to focus on _walkability_ as our metric for accessibility.

# Related Work

1. [The Influence of Land Use on Travel Behavior](https://www.sciencedirect.com/science/article/abs/pii/S0965856400000197) (Boarnet and Crane 2001)
  1. The Influence of Land Use on Travel Behavior (Boarnet and Crane 2001) seeks to identify key factors determining the likelihood of consumers using mass transit over single family automobiles. The study regresses trip behavior (number of trips by walking, mass transit and automobile) of Southern California residents with three classes of independent variables: 1) Price of travel and the income level of the individual household, 2) Socio-demographic "taste variables such as gender, education levels, age and number of people per household and 3) Land use and urban design characteristics (as regressed in previous similar studies)
  2. Our project will incorporate social-demographic variables such as urban or not, access to food, and urban design such as road network density. However, instead of using income level of individual households, we will use housing prices by FIPS code.
2. [The economic value of neighborhoods: Predicting real estate prices from the urban environment](https://arxiv.org/abs/1808.02547)
  1. Use home listings, openstreetmap, and census data on population, buildings, and industries to predict home value. In terms of differences, the research scope are individual property and neighborhood. Our research is scoped to census tracts.
3. [Neighborhood Walkability and Housing Prices: A Correlation Study](https://www.mdpi.com/2071-1050/12/2/593)
  1. The study regresses housing prices using walkability score using OLS regression. Different categories were used to derive walkability. For example, the study uses access to restaurants, shopping, schools, and recreation. The study invokes a penalty based on pedestrian friendliness using intersection density and average block length.

# Data Sources

## Environmental Protection Agency's Smart Location Database (SLD)

The [EPA's Smart Location Database](https://edg.epa.gov/EPADataCommons/public/OA/SLD/SmartLocationDatabaseV3.zip) is an aggregation dataset, containing data from a variety of sources and timeframes, including [Longitudinal Employer Household Dynamics](https://lehd.ces.census.gov/) from 2017, the [Census 2019 Tiger/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2019.html#list-tab-790442341), [Census ACS 5 year](https://www.census.gov/programs-surveys/acs/technical-documentation/table-and-geography-changes/2018/5-year.html) from 2018, and [GTFS](https://gtfs.org/) data from providers such as "[TransitFeeds](https://transitfeeds.com/)" and "[TransitLand](https://www.transit.land/)" in 2020. The SLD was created to help facilitate efforts such as travel demand and transportation planning studies. It is composed of more than 90 different variables associated with employment, transportation, land use, and demographics. This dataset contained our dependent variable, the National Walkability Index (NWI).

## Zillow Home Value Index (ZHVI)

The [Zillow Home Value Index (ZHVI)](https://www.zillow.com/research/data/), which is made available as a csv file, contains time series data at the State/County FIPS grain for the "middle tier" (i.e. 35th to 65th percentile) typical home value with a monthly periodicity. The dataset ranges from Jan-01-2000 to Jul-01-2023, but due to missing values for small regions, especially in early years, we have decided to only use data from January 2012 or later. This dataset provided the "Home Value" feature for our analysis.

## U.S. Department of Agriculture's Food Access Research Atlas (Food Atlas)

The [USDA Food Access Research Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data/) (hereafter referred to as "Food Atlas") is a data asset focused on identifying Census Tracts of low income and low accessibility to food resources, originally created to measure the impact of food deserts. It contains numerous features, but we concentrated on binary variables that define a Census Tract as Urban, Low Access to Food, Low Income, or Low Vehicle Accessibility. The Food Atlas was downloaded as a multi-sheet Excel file from their website, which was updated on 4/27/2021.

## Pre-Processing Steps for Data Sources

EPA SLD

- Converted the dataset from [ESRI GDB](https://desktop.arcgis.com/en/arcmap/latest/manage-data/administer-file-gdbs/file-geodatabases.htm) format using geopandas.
- Dropped shapefile columns that make the dataset excessively large.
- Removed records that are not states (US territories & holdings).
- Created Census Tract and county level FIPS codes to join against Zillow and Food Atlas.
- Dropped records where features are out-of-range from the study, for example all CBGs with population-weighted centroids that were further than three-quarter miles from a transit stop were assigned a value of "-99999.

Zillow ZHVI

- Combine StateCodeFIPS and MunicipalCodeFIPS to create a county level FIPS column.
- Melt the data by Year in order to transform the dataset from a wide structure to a long structure.
- Extract the years and months as independent variables from the date column for easy filtering.

USDA Food Atlas

- Created the county level FIPS from the Census Tract identifier to join against Zillow and EPA SLD.

# Feature Engineering

Feature engineering tasks were straightforward with these particular datasets, given the overall good hygiene of the data in general. Due to the size of the data, and the fact we were taking relatively few columns from large datasets, the [Apache Parquet](https://parquet.apache.org/) data format was utilized as part of the workflow, which is columnar optimized, features built in compression algorithms, and is well supported by the Pandas library.

For the Food Atlas dataset, we focused on the binary features for census tracts representing Urban, low access to food by distance, low income, or low vehicle accessibility to food. For these features, we grouped the tracts together by their county level FIPS identities, and then took the percentage representations for each of the binary features. For example, if a county level FIPS contained 4 census tracts with 3 tracts having the "Urban" variable set to 1, the resulting output would be 0.75 (3 Urban tracts / 4 total tracts).

There were two additional features derived from the Food Atlas, one of which was taking the mean of the PovertyRate feature, grouped by the county FIPS. The other additional feature was a binary flag created to identify potential food desert locations, applied at the tract level and then grouped to a percentage like the other binary features.

From Zillow, there were four features that were created. The first feature was a simple filter of the data to Jan-2023 observations for the supervised analysis. The other three features percent change indicators that were built by using annual percentage growth from [FRED Median Sales Price of Houses Sold for the United States](https://fred.stlouisfed.org/series/MSPUS) to identify years of high/medium/low macro home value trends and apply that to the Zillow dataset.

Please see [Appendix B - Full Feature List](#qwbpbmtpzp7t) for a listing of all features.

# Supervised Learning

## Objective

As part of the goal of understanding the characteristics of highly accessible population centers in the United States, we looked at the National Walkability Index (NWI) metric that was created by the US Environmental Protection Agency in their Smart Location Database (SLD). The NWI is a complex index that is a composite of four other metrics that are related to Employment (_D2A\_Ranked_, _D2B\_Ranked_), Transit Accessibility (_D3B\_Ranked_), and Street Density (_D4A\_Ranked_). It requires a great deal of data from a variety of sources that can be laborious to retrieve and recompose, if a researcher wanted to rebuild the Index themselves.

Our goal became to see if it was possible to approximate the NWI without directly using the same data resources that the SLD was built from (Census Demographics, Labor statistics, and Transit statistics). The NWI therefore became our dependent variable that we were trying to predict.

The workflow consisted of building a development environment for creating notebooks on a local server with 32 CPU Cores and an NVidia 3090 GPU. Using Anaconda, a virtual environment with Python 3.10 was created for this project, with Jupyter Lab and all additional libraries installed via pip. After notebooks were developed on the local server, the notebooks, data, hyperparameter objects, models, and visualizations were uploaded to a project space on Deepnote for review by the team. The data manipulation portions of the notebooks were not able to be directly executed on the Deepnote environment, due to memory limitations of the service when loading the original large datasets, but result sets were usable in the shared environment.

## Feature Representations

The Features used for this analysis are the ones in the " **Feature Engineering**" section of this report. With the exception of the Zillow _Home Value_ feature, all features were composed from the USDA Food Atlas dataset, and were aggregated from the Census Tract level to the county level. The aggregation method used was to take percentages of the binary features in all cases except for the _PovertyRate_, which took the mean. Final representations are continuous floats, ranging from 0 to 1, and 0 to100 for _PovertyRate_.

The Zillow _Home Value_ feature was built by filtering the Zillow dataset so that it was limited to January 2023 observations, and the 16 (out of 3078) observations missing the _Home Value_ data point were dropped from the analysis. This value was represented as a continuous variable with no limits.

## Method

Because the _NatWalkInd_ dependent variable is continuous within the range of 1 to 20, multiple regression modeling methods were selected for evaluation in approximating the National Walkability Index.

1. Ridge Regression - This is the baseline regression algorithm that was used to validate the other algorithms were performing adequately. Its primary advantages are that it is simple to use, is quick to execute, and that it should handle multicollinearity from the Food Atlas data fairly well.
2. Random Forest Regression - A tree ensemble algorithm, this modeling method leverages the concept of "bagging", where multiple subsets of the data source are sampled with replacement to train individual trees. The individual trees are how regularization is introduced to the model, as the various trees are then aggregated to produce a final output, which reduces overall variance. It is capable of handling missing values without requiring imputation, can also capture non-linear relationships, and is highly parallelizable.
3. XGBoost Regression - XGBoost is also a tree based algorithm, but unlike RandomForest, it builds each tree sequentially with the focus of correcting the errors of the previous tree. This method leverages the concept of using the gradient of the loss function to direct the construction of the next tree. It uses L1 and L2 regularization to prevent overfitting, and like RandomForest, it is insensitive to missing data and has some parallelism capabilities.
4. PyTorch Neural Net Regression - The PyTorch Neural Network (NN) that was used in this report was built from scratch, using only one hidden layer and ReLU Linear activation functions. An additional Dropout layer was included in each model architecture, along with an Early Stopping function to add a method of regularization and to prevent overfitting. The element of using Dropout layers makes the model inherently non-deterministic, due to the randomness introduced into the training process, despite setting a seed variable. The PyTorch NN was the only model that was trained on a GPU.

Each of the algorithms listed above utilized a hyperparameter tuning method, and leveraged 5 fold cross validation with 70/30 train/test split of data, using either Hyperopt, Scikit-Learn's HalvingGridSearchCV or ParamGrid. The selections for the hypertuning methods were quite deliberate, as explained below.

HalvingGridSearchCV was used for RandomForest models because of the independent nature of the tree ensembles. With 32 processors available, each processor could work on an independent tree with sample data, and each iteration would leverage larger samples of the data on the most successful of the models. The HalvingGridSearchCV hyperparameter tuning method evaluates a large number of model candidates using a small amount of sample data, then iteratively selecting the best candidates for training on larger samples over multiple rounds. This combination created a very efficient parameter search process.

The Hyperopt library with Tree-based Parzen Estimators (TPE) was selected for XGBoost as a result of XGBoost being not quite as parallelizable as RandomForest, due to its boosting process being sequential (though its node splitting process is parallelized). The Hyperopt TPE method implements an iterative Bayesian optimization process that was able to search through a grid space efficiently and determine the most effective model with substantially fewer iterations than HalvingGridSearchCV. This optimization process can be seen in Fig. 1 below, where promising hyperparameters are discovered early in the process, and the variability of MAE scores decreases as parameters are refined.

![](RackMultipart20231020-1-cxpey8_html_5623da23428abdfe.png)

![Shape 2](RackMultipart20231020-1-cxpey8_html_7ca98a122de74044.gif)

Fig. 1

The standard ParameterGrid method from Scikit-Learn was used for hypertuning the PyTorch model. This particular model required a large amount of fine tuning and tweaking of the parameter grid space for the number of hidden nodes, learning rate and dropout percentages, which meant it was more efficient to iterate over every permutation in the space to discover the optimal parameters, which is what the standard ParameterGrid method allowed. This model did have an advantage in that it could use GPU resources, which reduced the overall training time required. A [visualization](#_es9wik286it) is available in [Appendix E](#kipwtc1u6agx), showing the results of the 5-Fold cross validation used to generate the best Neural Net model, with the various spikes in each fold indicating the impact of dropout, and the differing number of epochs per fold illustrating the early stopping method preventing the model from hitting the maximum 1000 epoch limit, preventing overfitting.

## Evaluation

### Loss Function & Evaluation Metric

The Mean Absolute Error (MAE) method was used as the objective function and the evaluation metric in our analysis across all models. This selection was driven by the fact that the values returned by MAE are in the same unit of measure as the dependent variable, and were consequently easily interpreted against the NWI.

![Shape 2](RackMultipart20231020-1-cxpey8_html_407f4fc3ab78677b.gif)

Table 1

| **Model** | **MAE of CV 5 Mean** | **Std Dev** | **Training Time (seconds)** | **Hyperparam Models** |
| --- | --- | --- | --- | --- |
| Ridge (Baseline) | 1.1 | ±0.25 | 1 | 24 |
| XGBoost | 0.757 | ±0.026 | 1052 | 1000 |
| Random Forest | 0.808 | ±0.060 | 2632 | 15246 |
| PyTorch NN | 0.798 | ±0.050 | 240 | 706 |

### Model Evaluation and Determination

![Shape 2](RackMultipart20231020-1-cxpey8_html_1a69d0d5d4b03a0a.gif)

Table 1

Despite the variety of algorithms and hyperparameter training, the non-baseline models resulted in fairly similar scores (Table 1), with XGBoost having the slightly lower MAE and standard deviation overall, coming in at 0.757 ± 0.026 MAE. There were multiple important tradeoffs to consider for this model selection process, when comparing the top two performing models, XGBoost and PyTorch NN.

1. The training time for PyTorch models is significantly lower than the time required for XGBoost, but the accuracy of the XGBoost models is higher.
2. The PyTorch model utilized a modern GPU and there is additional substantial expense associated with this hardware. If standard CPUs are used instead, the amount of time to train complex models with more epochs would increase dramatically. The latest version of XGBoost (currently 2.0.0) can also leverage GPU hardware for training, but using it actually increased training times.
3. The explainability of XGBoost is substantially superior to that of PyTorch's NN through its "feature\_importance\_" function, but NN can often more accurately model non-linear relationships.
4. The PyTorch model's architecture was extremely simple, and a more complex architecture could certainly surpass the existing XGBoost model. The tradeoff here is the amount of development time it takes to fine tune a Neural Network model to achieve those gains, versus quickly delivering a model that is sufficient for the purpose.
5. The XGBoost model is deterministic, whereas the PyTorch model is non-deterministic due to its use of dropout and early stopping. This makes XGBoost extremely repeatable, with only the parameters required to rebuild the model. For the PyTorch NN model, to recreate the results you would need to load the model's state dictionary, or a copy of the entire model itself. The model cannot be re-created by simply reusing the same parameters and architecture. The tradeoff here is therefore the storage space required to persist or recreate the model. For XGBoost, it would be measured in bytes, regardless of complexity, and for the NNs, it is measured in kilobytes for this simple model. Larger NN models would require substantially more storage, mega or gigabytes.

Considering the tradeoffs listed above, the XGBoost model was selected due to its better MAE score, its explainability, its deterministic and repeatable nature, and that it did not require additional time to refine.

### Feature Analysis ![](RackMultipart20231020-1-cxpey8_html_798e21508f1e1384.png)

Shapely Additive Explanations (SHAP) was used to calculate the overall importance of features and provide insights into some of the highest ranked.

The SHAP "Beeswarm" plot in Fig. 2 indicates that the _Urban\_Pct_feature is by far the most important, followed by _LATracts\_half\_Pct_ and _Home Value_.

![Shape 2](RackMultipart20231020-1-cxpey8_html_7ca98a122de74044.gif)

Fig. 2

Taking a closer look at the dependency plot for the _Urban\_Pct_ feature (colored by _Home Value_ in Fig. 3), we can see that it has a non-linear impact on the SHAP values. For observations with small _Urban\_Pct_ values, the impact to SHAP is nearly -1, which slowly increases to 0 when the values of the feature reach roughly 0.33. At this point, there is a plateau from ~0.33 to ~0.75, after which the impact to the SHAP value increases dramatically from 0 to 4+.

Per SHAP's interaction values, the feature that has the most interactive effects with _Urban\_Pct_ is _LATracts\_half\_Pct_. Looking at the interactive effects dependency plot for those two features (Fig.4) indicates a negligible impact to the SHAP value from the combination until the values of both features reach roughly 0.7, which then results in a mostly positive, but slight (maximum 0.4+) increase to the SHAP value. This is the strongest interactive effect, for the most impactful feature, and it only accounts for a slight increase to the overall score, indicating that most of the

![](RackMultipart20231020-1-cxpey8_html_1b16e3f42763f3c6.png) ![](RackMultipart20231020-1-cxpey8_html_7c1e48bd2e97ef84.png)

###


### ![Shape 2](RackMultipart20231020-1-cxpey8_html_7ca98a122de74044.gif) ![Shape 2](RackMultipart20231020-1-cxpey8_html_7ca98a122de74044.gif)

Fig. 4

Fig. 3

###


### Model Accuracy

![Shape 2](RackMultipart20231020-1-cxpey8_html_7ca98a122de74044.gif)

Fig. 5

The MAE results of 0.757 ± 0.026 indicated that the model should be capable of providing reasonable results at predicting the aggregated National Walkability Index (NWI), which has a possible range of 1 to 15.96 points. The scatterplot results in Fig. 5 has the dependent variable (_NatWalkInd_) on the X axis, the feature with the highest SHAP value (_Urban\_Pct_) on the Y axis, and the plot is colored by the difference between the actual NWI values and our predicted values. The shape of the data is similar to a sigmoidal curve, and our model has clearly captured this non-linearity, as evidenced by observing the narrow band of green in the diagram, which represents a near zero difference in actuals versus predictions. ![](RackMultipart20231020-1-cxpey8_html_aa735820d6138971.png)

### Ablative Analysis of Top 5 Features

![Shape 2](RackMultipart20231020-1-cxpey8_html_12444bd0979d1a00.gif)

# Table 2

An ablative analysis was performed on the top 5 features as reported by SHAP in Figure 2. These features were removed from the training and test datasets in order to measure their MAE impact, using the XGBoost model. The MAE resulting from the ablation tests were compared against the baseline MAE, and the differences between the two are displayed in Table 3.

| **Feature** | **Urban\_Pct** | **LATracts\_half\_Pct** | **Home Value** | **LATracts10\_Pct** | **LAhalfand10\_Pct** |
| --- | --- | --- | --- | --- | --- |
| **Importance per XGBoost** | 32% | 21% | 10% | 10% | 6% |

While holding the XGBoost model parameters static, each feature(s) in the table below were dropped from the training and testing datasets, and another 5-Fold cross validation mean MAE was generated.

| **Ablated Tests** | **MAE (5-Fold CV)** | **Std Dev** | **Difference from Baseline MAE** | **Percent Difference from Baseline MAE** |
| --- | --- | --- | --- | --- |
| None (XGBoost Baseline) | 0.7570 | ±0.0263 | 0 | 0 |
| Urban\_Pct | 0.8132 | ±0.0267 | -0.0563 | -7.43% |
| Home Value | 0.7818 | ±0.0278 | -0.0248 | -3.29% |
| Urban\_Pct, Home Value | 0.8727 | ±0.0290 | -0.1157 | -15.29% |
| Urban\_Pct, Home Value, LATracts\_half\_Pct | 0.8898 | ±0.0313 | -0.1328 | -17.55% |
| Urban\_Pct, Home Value, LATracts\_half\_Pct, LATracts10\_Pct | 0.9005 | ±0.0302 | -0.1435 | -18.96% |
| Urban\_Pct, Home Value, LATracts\_half\_Pct, LATracts10\_Pct, LAhalfand10\_Pct | 1.1205 | ±0.0215 | -0.3635 | -48.03% |

![Shape 2](RackMultipart20231020-1-cxpey8_html_12444bd0979d1a00.gif)

# Table 3

Key Findings:

1. _Urban\_Pct_ has significantly more impact on the MAE score than _Home Value_. This was not entirely unexpected given the SHAP analysis, but the magnitude of the difference was surprising, given the initial assumptions that _Home Value_ would tend to be more expensive in rural areas, causing Urban Percentage to be of secondary importance.
2. Despite the loss of both _Urban\_Pct_ and _Home Value_, the model is still fairly resilient, with an accuracy drop of only -15.29%. The resulting mean MAE of 0.8727 is still better than the Ridge regression model's baseline score of 1.1 MAE with all of the features. This is an indication that there is likely a good amount of multicollinearity involved in this data.
3. It is not until the top 5 features (_Urban\_Pct_, _Home Value_, _LATracts\_half\_Pct_, _LATracts10\_Pct_, _LAhalfand10\_Pct_), as reported by the XGBoost Feature Importance variable, until there is significant decay in the MAE score. It is at this point that the XGBoost model performance is worse than the Ridge Regression Baseline (MAE 1.1 ± 0.25).

### Hyperparameter Sensitivity ![](RackMultipart20231020-1-cxpey8_html_e26a937ea5e097e1.png)

![Shape 2](RackMultipart20231020-1-cxpey8_html_7ca98a122de74044.gif)

Fig. 6

With Hyperopt selecting the optimal parameters used for the training of the XGBoost model, it was important to look at how sensitive some parameters were to being changed. A model that is exceptionally sensitive to slight parameter changes indicates it might be over-fitted and would perform poorly when making predictions on unseen data. All other parameters were held constant when performing these tests.

![](RackMultipart20231020-1-cxpey8_html_b1a89e00aa49d271.png)

![Shape 2](RackMultipart20231020-1-cxpey8_html_7ca98a122de74044.gif)

Fig. 7

 With regards to the L1 regularization parameter (reg\_alpha), Hyperopt selected the value of "1" as being the most optimal value, but this indicates there is little Lasso regression occurring in our model. According to the tests (Fig. #), our model is moderately insensitive to changing the reg\_alpha parameter from 1 to 100, resulting in a change from MAE 0.757 ± 0.03 to an MAE of 0.83 ± 0.03. Of course, since regularization by definition is making the model more generalizable, this result is not unexpected, and scaling from 1 to 100 is a fairly large change.

Performing a similar analysis on the max\_depth hyperparameter (Fig. #) shows negligible changes from a depth of 2, which is the value selected for the best model, versus 12. The model\_depth parameter is stable and is not significantly sensitive.

## Failures Analysis

1. Data leakage occurred during initial regression model development due to the time series nature of the Zillow data Observations for single FIPS codes were appearing in both Train and Test datasets, and the FIPS code categorical variable was included in the dataset as well. This caused exceptionally good MAE scores that were suspicious. Selecting a single month of observations (Jan 2023) and removing FIPS codes from training datasets remediated this issue.
2. The worst performing under prediction was for FIPS code 31043, which corresponds to the Sioux City metropolitan region in South Dakota. Looking at the SHAP force plot (Figure #) gives us an idea of the reasons for the prediction of 6.18, which is a 5.34 point miss from the actual NWI score of 11.51. The reasoning for this discrepancy comes down to issues with aggregating data up to the grain of State/County for the FIPS codes, so the Zillow _Home Value_ feature can be added to the analysis. The Smart Location Database (SLD) has very granular data, with their FIPS codes going down to the Census Block Group level, while the Food Atlas is slightly less granular at the Census Tract level. Both have to be aggregated upwards to accommodate the Zillow data.

![](RackMultipart20231020-1-cxpey8_html_cef40c36435169d6.png)

 ![Shape 2](RackMultipart20231020-1-cxpey8_html_7ca98a122de74044.gif)

Fig.8

In the case of FIPS 31043, it contains a large number of Census Block Groups in the SLD that have a very high National Walkability Index (8 out of 13 CBGs have an NWI \> 12), pushing the actual NWI mean number quite high. Looking at the features data, only two of the four Food Atlas Census Tracts for this FIPS are considered "Urban", so the critical _Urban\_Pct_ feature produces a fairly low score of 0.5. This is further exacerbated by the fairly low _Home Value_ for a metropolitan area, as seen in the force plot, and a high score (0.75) for the Low Accessibility to grocery stores at 1 mile for Urban or 10 miles for Rural feature (_LA1and10\_Pct_). These factors combined mean the prediction fails quite badly in this scenario.

In order to remediate this issue for future analysis, we would remove the _Home Value_ feature from Zillow, which would allow us to use the Census Tract grain of data. The feature ablation tests indicated that the _Home Value_ feature provides only nominal value, and the increased degree of accuracy gained by using a lower grain of data would most likely offset that loss.

1. The worst over prediction, in terms of percentage, was for FIPS code 51181. This particular region was quite rural (Surry County, VA), with 7 Census Block Groups in the SLD dataset, 2 of which had NWI ratings slightly above 5 points, and 3 groups had rankings in the 1 point range. This gave the county an aggregated NWI of 3.07, but the prediction was 5.42, an over prediction of 2.35 points. Reviewing the [force plot](#_i5lmqfpw6x1b) revealed that while the prediction score was pushed down from the base value of 6.5 by a lack of positive scores in the _Urban\_Pct_ and _LATracts\_half\_Pct_, it wasn't very substantial, and no other features have enough power to push the prediction further down.

Unlike the situation for FIPS 31043, while using a lower grain of data should help in this situation, it is doubtful that this particular prediction can be remediated without the use of additional features. The most likely candidate features would be acreage of the tract and population & households. This would help identify physically large tracts with small populations that are truly rural, as well as differentiate physically large tracts with large populations that are suburbs.

1. The FIPS code with the highest over prediction by magnitude is 22087, which is the area east of New Orleans. The NWI aggregated mean value was 7.08, and our model's prediction was 10.77. Reviewing the [force plot](#_sdyxxsmqzx18) reveals the value is significantly impacted by _Urban\_Pct_ and related features. Investigating this issue further, there are 17 Census Tracts within the county reported in the Food Atlas data source, with 16 listed as Urban. There are also 50 Census Block Groups in the SLD, with NWI values ranging from 1 to 16. While part of this problem comes down to the issue of aggregating the NWI and features, as previously mentioned, there also seems to be a data classification problem after reviewing maps of the region. As an example, tract 22087.0302.04 is predominantly swampland, but is classified as "Urban" with a single state road passing through it with a handful of auto salvage yards, scrap metal yards, and collision repair businesses. Additionally, the majority of the other tracts appear to be suburban or residential in nature, which would typically have poor NWI scores.

Additional features that would help with this problem would include the population and acreage, which were mentioned as part of the FIPS 51181 failure analysis, but also a count of Census Block Groups within a tract, whether or not the tract is within the boundaries of a Core Based Statistical Area (CBSA), and if they are within a Combined Statistical Area (CSA).

# Unsupervised Learning

## Motivation

As part of the unsupervised learning section of this project, we have decided to create several levels of clusters to understand how transit and food infrastructure intersect with the health of a housing market in a particular county.In order to do this, three k-means clustering models were created:

1. Clustering around transit infrastructure, walkability, and urban density
2. Clustering based around the availability of food in a particular county
3. Clustering based around the sensitivity of a county's housing market to macro-economic shifts.

After this, we examined how each group of clusters overlapped with one another to gain an understanding of the relationship between each domain.

## Methods description

### Hyperparameter Tuning

Initially, we set a specific number of clusters to create for each model (n = 5), but quickly discovered that those clusters were not well differentiated from one another. In fact, there was such an overlap in clusters that it was difficult to describe the differences between different centroids. This led us to conduct hyperparameter tuning with the objective of maximizing the silhouette score.

The Silhouette score is defined as: (b-a) / max(a,b) where:

a = the average euclidean distance between each point within a cluster

b = the average distance between clusters

Scores closer to one indicate a well differentiated model with clear definitions for clusters. Scores closer to -1 indicate overlapping and undifferentiated clusters. Silhouette scores were calculated for k-means models containing the following combinations of hyperparameters:

1. Algorithms (Lloyd and Elkan)- The Lloyd algorithm is the most common K-means optimization method and involves an iterative process of recalculating centroids as the mean over their closest data-points until the model reaches convergence. This has risk of getting stuck on a local optima (depending on initialization). Elkan uses a triangle inequality to speed up the performance of training the k-means algorithm. Because our feature space and observations for clustering is relatively low, all models were successfully trained with the Lloyd algorithm.
2. N-Clusters (3 to 10)- Because these clusters inform our exploratory data analysis, we did not want to create too many of them. It would be difficult to comprehend what each cluster means if we separated them into groups of more than ten. In addition, having too few (2 clusters) may create groups too large to perform any meaningful analysis. In fact, when we tried using 2 clusters, it merely split county into rural and urban counties (without differentiating between access to transit, walkability, etc).
3. Tolerance (1e-6, 1e-4, 1e-3, 1e-2)- represents the tolerance for when Scikit-learn declares convergence between two iterations. A smaller tolerance means that the same centroids would need to be closer to converge and stop iteration. A larger tolerance indicates the iterations of centroids could be further apart.

## Unsupervised evaluation

We evaluated each clustering model based off their individual silhouette scores. The transit and infrastructure model, housing market model, and food availability models scored .43, .24, and .3 respectively. Additionally, based on the visualizations described in the discussion section, we can easily see how defined each cluster is. For example, the transit and infrastructure model created well defined, easily decipherable clusters. This is reflected in the relatively high silhouette score. However, the housing market health model is less defined, both in the visualizations as well as the silhouette score. The scatter plots- across each pair of features contains many overlapping clusters in this case.

## Failure Analysis

The failure of the housing market model can be attributed to the reasons:

1. Feature engineering and selection- low growth, medium growth, and high growth periods were defined using single years provided in the Zillow dataset. For example, the low growth period was the year with the _lowest_ growth rate in the national housing market between 2010 and present (in this case, the year 2018. Medium and high growth periods were defined as the years 2013 and 2011, respectively. As such, 7 years of the Zillow dataset were excluded from the clustering model. Alternatively, we could have created thresholds defining low, medium, and high growth periods and categorized all available years within those buckets. This would have allowed us to utilize the entire time series in our model, and potentially given us a better understanding of regional housing market resilience compared to the national average.
2. In the above point, growth was defined as the average housing cost at the beginning of the year compared to the average housing cost at the end of the year. This ignored intra-year fluctuations, and potentially missed dramatic housing events. Calculating monthly growth periods would have allowed us to catch short lived housing spikes and depressions.
3. We primarily optimized this model using a discrete K-means algorithm. Potentially, a more robust clustering model could have been achieved by exploring alternative clustering approaches, such as fuzzy k-means, hierarchical clustering, or gaussian models

# Discussion

## Supervised

An early hypothesis was developed around using _Home Value_ to predict the National Walkability Index (NWI), with an initial assumption that homes in FIPS code regions with a higher NWI would see a substantial increase in home value during the past few years. Using Facebook's Prophet library, we created time series models for all 3000+ FIPS codes to generate trends and predictions, along with their respective confidence intervals to use as features in regression models to predict the NWI index for a FIPS State/County location.

This process produced mediocre results (MAE 1.253 ± 0.043), so we pivoted from leveraging time series to a classic regression approach, still using _Home Value_, but augmented with features built from the Food Atlas. The decision to pivot from the Prophet time series approach was ultimately correct, but that process consumed a considerable amount of time, manpower, and compute cycles; we should have decided to change tack earlier in the project.

A major complication to our analysis was the lack of alignment in the granularity of data between the Zillow ZHVI, SLD and Food Atlas data sources. Rolling up our dependent variable (NWI) by two levels inherently introduced a degree of inaccuracy into the predictions from the start. Despite this issue, the model was ![](RackMultipart20231020-1-cxpey8_html_e989d45976da4065.png)surprisingly successful in predicting the NWI, with a clear normalized curve in the predictions, and an overall mean of differences in NWI actuals versus predictions of 0.034 ± 0.967.

![Shape 2](RackMultipart20231020-1-cxpey8_html_7ca98a122de74044.gif)

Fig. 9

Another surprising aspect of this analysis was the exceptionally strong correlation (0.86) between the _Urban\_Pct_ feature and the _D3B\_Ranked_ variable from the SLD. _DB3\_Ranked_ is a street intersection density ranking, and it is one of the 4 indexes that is used to build the NWI. These two data points come from completely different resources, where _D3B\_Ranked_ is built based on mapping data from [HERE](https://www.here.com/)'s NAVSTREETS product, the Urban flag in the Food Atlas is determined by a simple calculation of whether or not the geographic centroid of the census tract contains more or less than 2500 people. The strength of this correlation that was the reason that _Urban\_Pct_ was the most important feature, according SHAP.

After the feature ablation tests were performed, it was clear that the Zillow _Home Value_ feature was not providing enough value (3.29%) to justify the complexities caused by the aggregation, and that a finer grain of data could potentially provide better predictions. The NWI would still need to be rolled up, but only one step, from Census Block Group to Tract level, in order to be joined with the features from the Food Atlas This would only reduce the NWI range by a minimal amount, 1 to 19.83 (instead of the 1 to 15.96 caused by the two steps roll up), which is a more accurate reflection of the actual 1 to 20 range. That being said, model times would also increase as a result of the increased number of records available for training (roughly a 2,400% increase), so there is a potential tradeoff between time to train and accuracy to be considered.

Another factor to be considered is that resiliency of the MAE scores with respect to the Feature Ablation process indicates the features likely have high multicollinearity. Utilizing Principal Component Analysis (PCA) to perform dimensionality reduction would create orthogonal components that explain the maximum variance while simplifying the feature space.

And finally, building a slightly more complex PyTorch Neural Network would almost certainly achieve better MAE results, especially with the finer grain of data mentioned above, along with the additional features mentioned in the **Failures Analysis** section for Supervised Learning. A simple test on the existing data with two additional hidden layers using ReLU activators, and changing the dimension inputs to those layers through simple multiplication, produced a 5-fold CV mean MAE of 0.7153 ± 0.0251, which is a 6% improvement over the best XGBoost model. With more time, it would be interesting to more fully investigate the possibilities available with Neural Networks with more features and observations.

## Unsupervised

### Walkability and Transit Infrastructure

The first model's goal was to understand the different levels of transit infrastructure (such as availability to public transportation) and urban density amongst different U.S. counties. In order to do this, several factors were selected from the smart location database. Since many of the factors are highly correlated with the national walkability index (see figure below), we decided to keep the list of features low.

![Shape 2](RackMultipart20231020-1-cxpey8_html_c179342a5891d654.gif)

Fig. 10

 ![](RackMultipart20231020-1-cxpey8_html_ac6d65828063ed62.png)

![](RackMultipart20231020-1-cxpey8_html_760c89e01f26a6d5.png)

![Shape 2](RackMultipart20231020-1-cxpey8_html_6039852edb3bb6af.gif)

Fig. 11

This also helped us have a conceptual understanding of the unique characteristics of each cluster. The features on the model are: **Total county land acreage** , **D3B** (Street intersection density), the number of street intersections per square mile, which is a good proxy for understanding the urban density of a county, **D4BO50** , Proportion of CBG employment within a square mile of fixed-guideway transit stop, which provides a measure of the public transit availability in commercial centers, and **National Walkability Index.**

![](RackMultipart20231020-1-cxpey8_html_b64fb971e3aa933d.png) ![Shape 2](RackMultipart20231020-1-cxpey8_html_1a69d0d5d4b03a0a.gif)

Fig. 12

Based on the silhouette optimized model, the model parameters and centroids are defined below. Low acreage counties generally correspond to high transit availability and walkability (cluster 0). Clusters 1 and 2 differentiate high acreage counties between their unique transit characteristics. In this case, cluster 1 represents rural counties with limited walkability and transit availability where cluster 2 represents suburban counties that are connected to urban centers by transit infrastructure.

### Food Availability

The second model's goal is to understand how individuals within a specific county have access to food- either through vehicle travel or walking. This will be used to understand the impact that food availability has on housing prices and market stability in a regional context. The features we selected are less auto-correlated than the smart location dataset and do not necessarily require principal component analysis.

![Shape 2](RackMultipart20231020-1-cxpey8_html_1a69d0d5d4b03a0a.gif)

Fig. 13

 ![](RackMultipart20231020-1-cxpey8_html_3d90cbce416c4cdd.png)

![](RackMultipart20231020-1-cxpey8_html_88cb3e9464383ca5.png)

The features for clustering by food availability are as follows:

1. The aggregate percent of low income tracts within a county
2. The aggregate percent of census tracts that are categorized as a "food desert" by the FDA
3. The aggregate percent of urban tracts that are within a county
4. The aggregate percent of tracts that have low access to food and have low vehicle ownership. This is defined as tracts where \>= 100 of households do not have a vehicle, and beyond 1/2 mile from supermarket; or \>= 500 individuals are beyond 20 miles from supermarket ; or \>= 33% of individuals are beyond 20 miles from supermarkets
 ![Shape 2](RackMultipart20231020-1-cxpey8_html_1a69d0d5d4b03a0a.gif)

Fig. 14

Based on the silhouette optimized model, the model parameters and centroids are defined below. In this case, there are 8 clusters within the optimal model. There are several unique clusters of note. First, cluster 0 represents a high income, urban county with high accessibility to food. We may expect these areas to have more robust housing markets. On the contrary, cluster 1 consists of counties with a poorer population that have significantly limited accessibility to food. One interesting feature on the below clusters is that food availability related to vehicle ownership does not seem to be concentrated in poorer areas. In fact, certain wealthier counties seem to have low food accessibility and vehicle ownership.

### Housing Market Health

The goal of the 3rd model is to understand the health of housing markets within a respective county. Specifically, how regional housing markets react to price swings within the broader national housing environment. The Zillow home price dataset was used to create this set of clusters. The features for this model are listed below:

1. Home value- the average normalized home value within a specific county (most recent FYE)
2. pct \_change\_diff\_low- the average percent change in home value compared to the U.S. percent change in home value during periods of negative growth in the U.S. housing market (between 2013 and 2022)
3. pct \_change\_diff\_med- the average percent change in home value compared to the U.S. percent change in home value during periods of low to no growth in the U.S. housing market (between 2013 and 2022)
4. pct \_change\_diff\_high- the average percent change in home value compared to the U.S. percent change in home value during periods of high growth in the U.S. housing market (between 2013 and 2022) ![](RackMultipart20231020-1-cxpey8_html_2048babfee7e82e1.png)

We generally define healthy regional housing markets as those with resilient home values during periods of economic decline and perform on par with national trends during high and no growth periods. Additionally, we have reduced all the features above, using Principal Component Analysis to further understand where each cluster is separated visually. Based on the silhouette optimized model, the model parameters and centroids are defined below. Unlike the transit model, this model contains (n=7) clusters.

![Shape 2](RackMultipart20231020-1-cxpey8_html_1a69d0d5d4b03a0a.gif)

Fig. 15

**Cluster relatedness and takeaways**

After running the above models, we looked at the overlap between the housing stability, food availability, and transit cluster sets. Specifically, we looked to see if there are unique characteristics associated with stable housing markets that are present in other datasets. We will look at 3 housing stability clusters.

**Housing cluster 0** , a middle class price-stable dataset, has high overlap with transit clusters 1 (very rural, high acreage, low density, low transit availability) and 2 (Moderate acreage, high walkability, high accessibility). Additionally, this group overlaps heavily with food availability clusters 1 (moderately urban, low vehicle accessibility to food, many food deserts) This fits in line with middle-to-high class suburban and semi-rural communities that rely heavily on cars.

**Housing cluster 2** (very low income, underperforms the national housing market), however, tells a different story. In this case, the transit infrastructure is approximately the same as with transit cluster 0. However, there is much higher overlap with urban and suburban clusters clusters that have low food availability (in this case, food availability clusters 6 and 4). Not surprisingly, this indicates that the accessibility to food corresponds to higher instability in a regional housing market.

The **healthiest housing cluster, 1** (upper class, resistant to macroeconomic swings) has a very high (40%)overlap with food availability cluster 7, which is a group representing high income, low food desert, and high food availability. This is in line with our understanding that wealthier areas generally have high quality supermarkets and many local businesses that could support the population. It also has a very high overlap with transit cluster 3, representing high density, walkable, transit accessible regions. In general, this indicates that wealthy, stable housing markets are mostly in areas that have food and have a high degree of urban mobility.

We were surprised to see that in terms of transit infrastructure, middle and low class counties were not as differentiated as we hoped. In fact, housing cluster 0 (middle income) and housing cluster 2 (low income) both were represented by a mix of high walkability and low walkability clusters. This could potentially mean that transit does not do a good enough job of differentiating between class. This could be because larger cities contain a diverse mix of individuals and classes. Because the baseline of our analysis considered characteristics at the _county_ level, we may have missed certain nuanced class considerations between specific neighborhoods. In this case, if we had more time, we may have been able to find data sources at a census tract, or even census block level. This would allow us to differentiate between very different neighborhoods within the same county.

One challenge we faced was the fact that we needed to compare time series data, specifically, the Zillow home price dataset, with point-in-time data. This prevented us from merely joining the primary datasets together. We solved this by making an assumption that point-in-time regional housing market health can be defined in terms of its past performance. For example, the three features that were created (performance during low growth, medium growth, and high growth periods) were no longer time series data points, but rather, became point in time features that we could compare to our other datasets.

# Ethical considerations

For both supervised and unsupervised applications, we examined areas within:

1. Privacy: All data sources are publically accessible and aggregated by FIPS county codes and census tracts. Therefore, no individual names or addresses are expected to be identified based on home values or infrastructure characteristics.
2. Protected Classes such as race, religion, sex, and age were not used as part of analysis or model building. Only macro-economic features were used such as Urban Percentage, Food Access, and Home Values were used in model building and analysis
3. Transparency and reproducibility: source code and notebooks have been provided, along with feature importance via SHAP analysis. Aside from the random-forest regression and pytorch neural network, all other models are deterministic.
4. The Government of Canada Algorithm Impact Assessment has been performed and can be found in Appendix.

Impact consideration and unintended consequences

We acknowledge that our findings could have policy making implications for city planners and zoning that can affect individuals within the FIPS county. For those policy-making decisions, we recommend performing a disparate impact by merging census data that contains protected classes and evaluate disproportional impacts. Unfortunately, due to time and scope, we are unable to provide the analysis.

# Statement of Work

| Willie Man | James Conner | Ruwan Jayasumana |
| --- | --- | --- |
| Administrative and team member support
 Documentation, Code Review, Prophet Time Series, Report Authoring | Data Pre-processing and core data cleansing, Feature Engineering, Supervised Learning (Ridge, PyTorch, XGBoost, Random Forest, Prophet Time Series) Visualizations, Report Authoring | Unsupervised Learning using k-nn, PCA, feature selection, and model evaluation, Report Authoring |

#


# References

#


# Appendix

# Appendix A - Notebook Catalog

A read only copy of notebooks can be accessed at Deepnote with this link and [github repo](https://github.com/godot107/SIADS696).

Below table list the notebook names and their functionalities

| Notebook Category | Notebook Description | Notebook Name |
| --- | --- | --- |
| Data Preprocessing |
 |
 |
|
 |
 |
 |
| Unsupervised Learning |
 |
 |
|
 |
 |
 |
| Supervised Learning |
 |
 |
|
 |
 |
 |

#


# Appendix B - Full Feature List

| **Feature Name** | **Description** | **Data Type** | **Data Source** |
| --- | --- | --- | --- |
| Home Value | The [mid-tier ZHVI](https://www.zillow.com/research/zhvi-methodology-2019-deep-26226/) estimate from Zillow for Jan. 2023. | Numeric | [Zillow ZHVI](https://www.zillow.com/research/data/) |
| Urban\_Pct | Percent of Census Tracts flagged Urban within the County level FIPS, as defined by the Bureau of the Census's '_urbanized area definitions_'. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| Food\_Desert | Percent of Census Tracts in the County FIPS code where the poverty rate \>= 20%, and there was low accessibility to a grocery store ( no access within 1 mile Urban, or 10 miles Rural). | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| LowIncTracts\_Pct | Percentage of Census Tracts in the County FIPS code where the poverty rate \>= 20%, or median family income \< 80% of surrounding metro median family income. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| LA1and10\_Pct | Percentage of Census Tracts in the County FIPS code with at least 500 people or 33% of the population, live more than 1 mile (urban) or 10 miles (rural) from a grocery store. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| LAhalfand10\_Pct | Percentage of Census Tracts in the County FIPS code with at least 500 people or 33% of the population, live more than 0.5 miles (urban) or 10 miles (rural) from a grocery store. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| LA1and20\_Pct | Percentage of Census Tracts in the County FIPS code with at least 500 people or 33% of the population, live more than 1 mile (urban) or 20 miles (rural) from a grocery store. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| LATracts\_half\_Pct | Percentage of Census Tracts in the County FIPS code with at least 500 people or 33% of the population, live more than 0.5 miles (urban) from a grocery store. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| LATracts1\_Pct | Percentage of Census Tracts in the County FIPS code with at least 500 people or 33% of the population, live more than 1 mile (urban) from a grocery store. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| LATracts10\_Pct | Percentage of Census Tracts in the County FIPS code with at least 500 people or 33% of the population, live more than 10 miles (rural) from a grocery store. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| LATracts20\_Pct | Percentage of Census Tracts in the County FIPS code with at least 500 people or 33% of the population, live more than 20 miles (rural) from a grocery store. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| LATractsVehicle\_20\_Pct | Percentage of Census Tracts in the County FIPS code with at least 100 households more than 0.5 miles from the nearest supermarket with no vehicle access; or at least 500 people or 33% of the population, live more than 20 miles (rural) from a grocery store, regardless of vehicle accessibility. | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| PovertyRate\_FIPS | The mean PovertyRate for all Census Tracts in the County FIPS code. Any missing values were filled in with "0.00". | Numeric | Derived from
[USDA Food Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/documentation/) |
| Ac\_Land | Total land area (acres) | Numeric | Derived from [Smart Location Database](https://www.epa.gov/smartgrowth/smart-location-mapping#SLD) |
| D3b | Street intersection density (weighted, auto-orientedintersections eliminated) | Numeric | Derived from [Smart Location Database](https://www.epa.gov/smartgrowth/smart-location-mapping#SLD) |
| D4b050 | Proportion of CBG employment within ½ mile of fixed guideway transit stop | Numeric | Derived from [Smart Location Database](https://www.epa.gov/smartgrowth/smart-location-mapping#SLD) |
| NatWalkInd | Walkability index comprised of weighted sum of the ranked values of employment and household entropy, static eight-tier employment entropy, street intersection density, and distance to nearest transit stop | Numeric | [Smart Location Database](https://www.epa.gov/smartgrowth/smart-location-mapping#SLD) |
| pct \_change\_diff\_low | The average percent change in home value compared to the U.S. percent change in home value during periods of negative growth in the U.S. housing market (between 2013 and 2022) | Numeric | Derived from [Zillow ZHVI](https://www.zillow.com/research/data/) |
| pct \_change\_diff\_med | The average percent change in home value compared to the U.S. percent change in home value during periods of low to no growth in the U.S. housing market (between 2013 and 2022) | Numeric | Derived from [Zillow ZHVI](https://www.zillow.com/research/data/) |
| pct \_change\_diff\_high | The average percent change in home value compared to the U.S. percent change in home value during periods of high growth in the U.S. housing market (between 2013 and 2022) | Numeric | Derived from [Zillow ZHVI](https://www.zillow.com/research/data/) |

#


# Appendix C - Supervised Feature Ablation Table

| **Removed\_Feature** | **Mean\_MAE** | **Mean\_MAE\_Change** | **Std\_MAE** | **Test\_MAE** | **MAE\_Pct\_Change** |
| --- | --- | --- | --- | --- | --- |
| Home Value | 0.7818 | -0.0249 | 0.0278 | 0.8023 | -3.2861 |
| LowIncTracts\_Pct | 0.7584 | -0.0014 | 0.0248 | 0.7698 | -0.1896 |
| Urban\_Pct | 0.8132 | -0.0563 | 0.0267 | 0.8179 | -7.4332 |
| Desert\_Pct | 0.7557 | 0.0013 | 0.0283 | 0.7703 | 0.1677 |
| LA1and10\_Pct | 0.7558 | 0.0012 | 0.0247 | 0.7744 | 0.1573 |
| LAhalfand10\_Pct | 0.754 | 0.003 | 0.0265 | 0.7689 | 0.3908 |
| LA1and20\_Pct | 0.7538 | 0.0031 | 0.0297 | 0.7671 | 0.4154 |
| LATracts\_half\_Pct | 0.7548 | 0.0022 | 0.0308 | 0.7697 | 0.2895 |
| LATracts1\_Pct | 0.754 | 0.003 | 0.0292 | 0.7703 | 0.3926 |
| LATracts10\_Pct | 0.75 | 0.0069 | 0.0297 | 0.7654 | 0.9146 |
| LATracts20\_Pct | 0.7531 | 0.0038 | 0.0304 | 0.7644 | 0.504 |
| LATractsVehicle\_20\_Pct | 0.7536 | 0.0034 | 0.032 | 0.768 | 0.4468 |
| PovertyRate\_FIPS | 0.7644 | -0.0074 | 0.036 | 0.7756 | -0.9788 |
| Urban\_Pct, Home Value | 0.8727 | -0.1157 | 0.029 | 0.8794 | -15.2881 |
| Urban\_Pct, Home Value, LATracts\_half\_Pct | 0.8898 | -0.1328 | 0.0313 | 0.886 | -17.5502 |
| Urban\_Pct, Home Value, LATracts\_half\_Pct, LATracts10\_Pct | 0.9005 | -0.1435 | 0.0302 | 0.9031 | -18.9566 |
| Urban\_Pct, Home Value, LATracts\_half\_Pct, LATracts10\_Pct, LAhalfand10\_Pct | 1.1205 | -0.3636 | 0.0215 | 1.1035 | -48.0299 |

#


# Appendix D - Supervised Hyperparameter Sensitivity Table

| **Modified Parameter** | **Modified Value** | **Original Value** | **Mean MAE** | **Mean MAE Change** | **Std MAE** | **MAE Pct\_Change** |
| --- | --- | --- | --- | --- | --- | --- |
| max\_depth | 2 | 2 | 0.757 | 0 | 0.0263 | 0 |
| max\_depth | 3 | 2 | 0.7554 | 0.0016 | 0.0259 | 0.2114 |
| max\_depth | 4 | 2 | 0.7537 | 0.0032 | 0.0327 | 0.4259 |
| max\_depth | 5 | 2 | 0.7528 | 0.0041 | 0.0295 | 0.5444 |
| max\_depth | 6 | 2 | 0.76 | -0.0031 | 0.0281 | -0.407 |
| max\_depth | 7 | 2 | 0.761 | -0.004 | 0.0282 | -0.5329 |
| max\_depth | 8 | 2 | 0.7701 | -0.0132 | 0.0283 | -1.7392 |
| max\_depth | 9 | 2 | 0.7677 | -0.0107 | 0.028 | -1.4198 |
| max\_depth | 10 | 2 | 0.7712 | -0.0143 | 0.0283 | -1.886 |
| max\_depth | 11 | 2 | 0.7645 | -0.0076 | 0.0296 | -0.9998 |
| max\_depth | 12 | 2 | 0.7677 | -0.0108 | 0.0271 | -1.4207 |
| colsample\_bytree | 0.01 | 0.5541 | 0.795 | -0.0381 | 0.0115 | -5.0311 |
| colsample\_bytree | 0.1 | 0.5541 | 0.795 | -0.0381 | 0.0115 | -5.0311 |
| colsample\_bytree | 0.2 | 0.5541 | 0.7702 | -0.0132 | 0.0221 | -1.7496 |
| colsample\_bytree | 0.3 | 0.5541 | 0.7656 | -0.0086 | 0.0308 | -1.1356 |
| colsample\_bytree | 0.4 | 0.5541 | 0.7604 | -0.0035 | 0.0269 | -0.4589 |
| colsample\_bytree | 0.5 | 0.5541 | 0.7571 | -0.0001 | 0.025 | -0.0193 |
| colsample\_bytree | 0.6 | 0.5541 | 0.757 | 0 | 0.0263 | 0 |
| colsample\_bytree | 0.7 | 0.5541 | 0.7548 | 0.0022 | 0.0311 | 0.2858 |
| colsample\_bytree | 0.8 | 0.5541 | 0.7518 | 0.0052 | 0.0313 | 0.682 |
| colsample\_bytree | 0.9 | 0.5541 | 0.7509 | 0.0061 | 0.0304 | 0.7993 |
| colsample\_bytree | 1 | 0.5541 | 0.7491 | 0.0078 | 0.0292 | 1.032 |
| reg\_lambda | 0.01 | 0.5629 | 0.756 | 0.001 | 0.0273 | 0.1261 |
| reg\_lambda | 0.1 | 0.5629 | 0.7568 | 0.0002 | 0.0277 | 0.0252 |
| reg\_lambda | 0.2 | 0.5629 | 0.755 | 0.0019 | 0.0276 | 0.2575 |
| reg\_lambda | 0.3 | 0.5629 | 0.7544 | 0.0026 | 0.028 | 0.341 |
| reg\_lambda | 0.4 | 0.5629 | 0.7573 | -0.0003 | 0.0252 | -0.039 |
| reg\_lambda | 0.5 | 0.5629 | 0.756 | 0.001 | 0.0255 | 0.1322 |
| reg\_lambda | 0.6 | 0.5629 | 0.7571 | -0.0001 | 0.0264 | -0.0143 |
| reg\_lambda | 0.7 | 0.5629 | 0.7581 | -0.0012 | 0.0263 | -0.1565 |
| reg\_lambda | 0.8 | 0.5629 | 0.7585 | -0.0016 | 0.024 | -0.2059 |
| reg\_lambda | 0.9 | 0.5629 | 0.7567 | 0.0003 | 0.0288 | 0.0343 |
| reg\_lambda | 1 | 0.5629 | 0.7567 | 0.0003 | 0.0278 | 0.034 |
| reg\_alpha | 1 | 1 | 0.757 | 0 | 0.0263 | 0 |
| reg\_alpha | 10 | 1 | 0.7661 | -0.0091 | 0.0284 | -1.2011 |
| reg\_alpha | 20 | 1 | 0.7776 | -0.0206 | 0.0326 | -2.7233 |
| reg\_alpha | 30 | 1 | 0.7856 | -0.0287 | 0.0357 | -3.785 |
| reg\_alpha | 40 | 1 | 0.7919 | -0.0349 | 0.0364 | -4.615 |
| reg\_alpha | 50 | 1 | 0.7974 | -0.0404 | 0.0361 | -5.3361 |
| reg\_alpha | 60 | 1 | 0.8017 | -0.0448 | 0.0367 | -5.913 |
| reg\_alpha | 70 | 1 | 0.8066 | -0.0496 | 0.0359 | -6.5561 |
| reg\_alpha | 80 | 1 | 0.8132 | -0.0563 | 0.0349 | -7.4319 |
| reg\_alpha | 90 | 1 | 0.8226 | -0.0657 | 0.0347 | -8.6756 |
| reg\_alpha | 100 | 1 | 0.83 | -0.073 | 0.0341 | -9.6495 |

#


# Appendix E - Supporting Visualizations

## Supervised Hyperparameter Training Visualizations:

### PyTorch Hyperparemter Training using KFold, Early Stopping and Dropout

![](RackMultipart20231020-1-cxpey8_html_fe122640d77e7a1.png)

## Supervised Failure Analysis Visualizations:

### SHAP Force Plot for FIPS 51181

![](RackMultipart20231020-1-cxpey8_html_c56c4d5cc03b3636.png)

### SHAP Force Plot for FIPS 22087

![](RackMultipart20231020-1-cxpey8_html_f34179b6d23d0977.png)

![](RackMultipart20231020-1-cxpey8_html_b00693e3b142f21f.png)

#


#


### SHAP Waterfall Plot for FIPS 22087

![](RackMultipart20231020-1-cxpey8_html_3940b4ea0868ac66.png)

## Supervised Model Accuracy

### Choropleth of Walk Difference Scores by Points

![](RackMultipart20231020-1-cxpey8_html_ecd26d6d9e6fbd81.png)

### Distribution of Difference in Actual/Predicted Walkability Index by Points

![](RackMultipart20231020-1-cxpey8_html_96329079c39e08b5.png)

### Distribution of Difference in Actual/Predicted Walkability Index by Percentage

![](RackMultipart20231020-1-cxpey8_html_198842eb3f71c2f8.png)

### Distribution of errors within Urban\_Pct 5% bins

![](RackMultipart20231020-1-cxpey8_html_33a81d759c81ac53.png)

### Average prediction/actuals difference errors by State

![](RackMultipart20231020-1-cxpey8_html_bbc2c974bf8782c7.png)

# Appendix F - Algorithmic Impact Assessment Results

Version: 0.10.0

Project Details

1. Name of Respondent

Willie Man

2. Job Title

Student

3. Project Title

Analyze the characteristics of highly accessible population centers in the United States

4. Project Phase

Implementation

[Points: 0]

5. Please provide a project description:

In the US, cars are the most utilized modes of transportation for every day activity and life. Compared to European countries like Amsterdam that use bicycles, Paris, France on subways, and Rome, Italy on high speed rails, US citizens for the most-part rely solely on cars. This raises concern on the scalability of urban planning and affordable housing to various areas in the US.

Using government curated and publically accessible datasets, we hope to improve the daily lives of their citizens by increasing the accessibility of amenities such as jobs, entertainment, and shopping.

The outcome of this report is to identify the characteristics which makes a city more accessible than others to create models that predicts the accessibility of a population center with those characteristics. Realizing that the scope is beyond the allotted time, we decided to focus on walkability as our metric for accessibility.

About The System

6. Please check which of the following capabilities apply to your system.

Risk assessment: Analyzing very large data sets to identify patterns and recommend courses of action and in some cases trigger specific actions

**Section 1: Impact Level : 2**

Current Score: 49

Raw Impact Score: 49

Mitigation Score: 11

Section 2: Requirements Specific to Impact Level 2

Peer review

Consult at least one of the following experts and publish the complete review or a plain language summary of the findings on a Government of Canada website:

qualified expert from a federal, provincial, territorial or municipal government institution

qualified members of faculty of a post-secondary institution

qualified researchers from a relevant non-governmental organization

contracted third-party vendor with a relevant specialization

a data and automation advisory board specified by Treasury Board of Canada Secretariat.

OR

Publish specifications of the automated decision system in a peer-reviewed journal. Where access to the published review is restricted, ensure that a plain language summary of the findings is openly available.

Gender-based Analysis Plus

Ensure that the Gender-based Analysis Plus addresses the following issues:

impacts of the automation project (including the system, data and decision) on gender and/or other identity factors;

planned or existing measures to address risks identified through the Gender-based Analysis Plus.

Notice

Plain language notice posted through all service delivery channels in use (Internet, in person, mail or telephone).

Human-in-the-loop for decisions

Decisions may be rendered without direct human involvement.

Explanation

In addition to any applicable legal requirement, ensure that a meaningful explanation is provided to the client with any decision that results in the denial of a benefit or service, or involves a regulatory action. The explanation must inform the client in plain language of:

the role of the system in the decision-making process;

the training and client data, their source, and method of collection, as applicable;

the criteria used to evaluate client data and the operations applied to process it;

the output produced by the system and any relevant information needed to interpret it in the context of the administrative decision; and

a justification of the administrative decision, including the principal factors that led to it.

Explanations must also inform clients of relevant recourse options, where appropriate.

A general description of these elements must also be made available through the Algorithmic Impact Assessment and discoverable via a departmental website.

Training

Documentation on the design and functionality of the system.

IT and business continuity management

None

Approval for the system to operate

None

Other requirements

The Directive on Automated Decision-Making also includes other requirements that must be met for all impact levels.

Link to the Directive on Automated Decision-Making

Contact your institution's ATIP office to discuss the requirement for a Privacy Impact Assessment as per the Directive on Privacy Impact Assessment.

**Section 3: Questions and Answers**

**Section 3.1: Impact Questions and Answers**

Reasons for Automation

1. What is motivating your team to introduce automation into this decision-making process? (Check all that apply)

Improve overall quality of decisions

2. What client needs will the system address and how will this system meet them? If possible, describe how client needs have been identified.

average home prices by county, access to food, city characteristics such as road network density and access to transit.

3. Please describe any public benefits the system is expected to have.

Promote city planning and equitable outcomes by identifying characteristics of a FIPS county that is considered accessible via walkability over a different FIPS county that is considered low-walkable.

4. How effective will the system likely be in meeting client needs?

Slightly effective

[Points: +2]

5. Please describe any improvements, benefits, or advantages you expect from using an automated system. This could include relevant program indicators and performance targets.

Reduce overhead of gathering data, report findings, and reduce time to consumers such as the public, city planners, and government officials.

6. Please describe how you will ensure that the system is confined to addressing the client needs identified above.

NA

7. Please describe any trade-offs between client interests and program objectives that you have considered during the design of the project.

Disparate analysis is considered for protected classes such as race, age, gender, and religion. Recommend using FairLearn (an open-source application) and Google Cloud What-If Tool.

8. Have alternative non-automated processes been considered?

No

[Points: +1]

9. What would be the consequence of not deploying the system?

Service costs are too high

[Points: 0]

Service quality is not as high

[Points: 0]

Service cannot be delivered in a timely or efficient manner

[Points: +2]

Risk Profile

10. Is the project within an area of intense public scrutiny (e.g. because of privacy concerns) and/or frequent litigation?

No

[Points: +0]

11. Are clients in this line of business particularly vulnerable?

No

[Points: +0]

12. Are stakes of the decisions very high?

Yes

[Points: +4]

13. Will this project have major impacts on staff, either in terms of their numbers or their roles?

No

[Points: +0]

14. Will the use of the system create or exacerbate barriers for persons with disabilities?

No

[Points: +0]

Project Authority

15. Will you require new policy authority for this project?

Yes

[Points: +2]

About the Algorithm

16. The algorithm used will be a (trade) secret

No

[Points: +0]

17. The algorithmic process will be difficult to interpret or to explain

No

[Points: +0]

About the Decision

18. Please describe the decision(s) that will be automated.

Deterministic algorithms are employed such as ridge regression, auto-regressive time series forecasting, and k-nearest neighbors.

19. Does the decision pertain to any of the categories below (check all that apply):

Access and mobility (security clearances, border crossings)

[Points: +1]

Employment (recruitment, hiring, promotion, performance evaluation, monitoring, security clearance)

[Points: +1]

Impact Assessment

20. Which of the following best describes the type of automation you are planning?

Full automation (the system will make an administrative decision)

[Points: +4]

21. Please describe the role of the system in the decision-making process.

The role of the system is to assess a FIPS county's walkability metric and clustered with related counties of similar walkability

22. Will the system be making decisions or assessments that require judgement or discretion?

Yes

[Points: +4]

23. Please describe the criteria used to evaluate client data and the operations applied to process it.

Clients will be given an opportunity to evaluate outcomes.

24. Please describe the output produced by the system and any relevant information needed to interpret it in the context of the administrative decision.

The client can compare walkability metrics against the existing walkability evaluation metrics defined by the Smart Location Database. [https://www.epa.gov/smartgrowth/smart-location-mapping#SLD](https://www.epa.gov/smartgrowth/smart-location-mapping#SLD)

25. Will the system perform an assessment or other operation that would not otherwise be completed by a human?

Yes

[Points: +2]

26. If yes, please describe the relevant function(s) of the system.

Walkability Index has been defined by EPA using employment, transit access, and network density.

27. Is the system used by a different part of the organization than the ones who developed it?

Yes

[Points: +4]

28. Are the impacts resulting from the decision reversible?

Reversible

[Points: +1]

29. How long will impacts from the decision last?

Some impacts may last a matter of months, but some lingering impacts may last longer

[Points: +2]

30. Please describe why the impacts resulting from the decision are as per selected option above.

The application is intended to have humans-in-the-loop to evaluate its performance against existing walkability methodology.

31. The impacts that the decision will have on the rights or freedoms of individuals will likely be:

Moderate impact

[Points: +2]

32. Please describe why the impacts resulting from the decision are as per selected option above.

The impact is dependent on policy-makers, city planners, and housing and urban development. The goal of this application is to provide tools for decision-making on walkability.

33. The impacts that the decision will have on the equality, dignity, privacy, and autonomy of individuals will likely be:

Moderate impact

[Points: +2]

34. Please describe why the impacts resulting from the decision are as per selected option above.

More research is required for protected classes. However, privacy is protected, since the scope of the application is based on FIPS and Census-Tract Codes.

35. The impacts that the decision will have on the health and well-being of individuals will likely be:

Moderate impact

[Points: +2]

36. Please describe why the impacts resulting from the decision are as per selected option above.

To be assessed.

37. The impacts that the decision will have on the economic interests of individuals will likely be:

High impact

[Points: +3]

38. Please describe why the impacts resulting from the decision are as per selected option above.

Housing and real estate are impacted, since home value prices were used in the application.

39. The impacts that the decision will have on the ongoing sustainability of an environmental ecosystem, will likely be:

High impact

[Points: +3]

40. Please describe why the impacts resulting from the decision are as per selected option above.

By concentrating on walkability, one hope of improvement is to reduce the reliance of small vehicles and use public transit or walk to areas of interest such as economic centers, entertainment, and food.

About the Data - A. Data Source

41. Will the Automated Decision System use personal information as input data?

No

[Points: +0]

42. What is the highest security classification of the input data used by the system? (Select one)

None

[Points: +0]

43. Who controls the data?

Federal government

[Points: +1]

44. Will the system use data from multiple different sources?

Yes

[Points: +4]

45. Will the system require input data from an Internet- or telephony-connected device? (e.g. Internet of Things, sensor)

No

[Points: +0]

46. Will the system interface with other IT systems?

No

[Points: +0]

47. Who collected the data used for training the system?

Your institution

[Points: +1]

48. Who collected the input data used by the system?

Your institution

[Points: +1]

49. Please describe the input data collected and used by the system, its source, and method of collection.

Input data was collected by using publically available data. They were downloaded in their native format and processed using local compute.

About the Data - B. Type of Data

50. Will the system require the analysis of unstructured data to render a recommendation or a decision?

No

[Points: 0]

**Section 3.2: Mitigation Questions and Answers**

Consultations

1. Internal Stakeholders (federal institutions, including the federal public service)

No

[Points: +0]

2. External Stakeholders (groups in other sectors or jurisdictions)

No

[Points: +0]

De-Risking and Mitigation Measures - Data Quality

3. Do you have documented processes in place to test datasets against biases and other unexpected outcomes? This could include experience in applying frameworks, methods, guidelines or other assessment tools.

Yes

[Points: +2]

4. Is this information publicly available?

Yes

[Points: +1]

5. Have you developed a process to document how data quality issues were resolved during the design process?

Yes

[Points: +1]

6. Is this information publicly available?

Yes

[Points: +1]

7. Have you undertaken a Gender Based Analysis Plus of the data?

No

[Points: +0]

8. Is this information publicly available?

Yes

[Points: +1]

9. Have you assigned accountability in your institution for the design, development, maintenance, and improvement of the system?

No

[Points: +0]

10. Do you have a documented process to manage the risk that outdated or unreliable data is used to make an automated decision?

No

[Points: +0]

11. Is this information publicly available?

No

[Points: +0]

12. Is the data used for this system posted on the Open Government Portal?

Yes

[Points: +2]

De-Risking and Mitigation Measures - Procedural Fairness

13. Does the audit trail identify the authority or delegated authority identified in legislation?

No

[Points: +0]

14. Does the system provide an audit trail that records all the recommendations or decisions made by the system?

No

[Points: +0]

15. Are all key decision points identifiable in the audit trail?

No

[Points: +0]

16. Are all key decision points within the automated system's logic linked to the relevant legislation, policy or procedures?

No

[Points: +0]

17. Do you maintain a current and up to date log detailing all of the changes made to the model and the system?

Yes

[Points: +2]

18. Does the system's audit trail indicate all of the decision points made by the system?

No

[Points: +0]

19. Can the audit trail generated by the system be used to help generate a notification of the decision (including a statement of reasons or other notifications) where required?

No

[Points: +0]

20. Does the audit trail identify precisely which version of the system was used for each decision it supports?

No

[Points: +0]

21. Does the audit trail show who an authorized decision-maker is?

No

[Points: +0]

22. Is the system able to produce reasons for its decisions or recommendations when required?

No

[Points: +0]

23. Is there a process in place to grant, monitor, and revoke access permission to the system?

No

[Points: +0]

24. Is there a mechanism to capture feedback by users of the system?

No

[Points: +0]

25. Is there a recourse process established for clients that wish to challenge the decision?

No

[Points: +0]

26. Does the system enable human override of system decisions?

No

[Points: +0]

27. Is there a process in place to log the instances when overrides were performed?

No

[Points: +0]

28. Does the system's audit trail include change control processes to record modifications to the system's operation or performance?

No

[Points: +0]

29. Have you prepared a concept case to the Government of Canada Enterprise Architecture Review Board?

No

[Points: +0]

De-Risking and Mitigation Measures - Privacy

30. If your system uses or creates personal information, have you undertaken a Privacy Impact Assessment, or updated an existing one?

No

[Points: +0]

31. Have you undertaken other types of privacy assessments for your automation project? Please describe any relevant efforts.

The system does not use PII.

32. Have you designed and built security and privacy into your systems from the concept stage of the project?

Yes

[Points: +1]

33. Is the information used within a closed system (i.e. no connections to the Internet, Intranet or any other system)?

No

[Points: +0]

34. If the sharing of personal information is involved, has an agreement or arrangement with appropriate safeguards been established?

No

[Points: +0]

35. Will you de-identify any personal information used or created by the system at any point in the lifecycle?

No

47