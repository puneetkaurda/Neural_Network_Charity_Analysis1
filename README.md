
# Neural Network Charity Analysis
## Overview
The purpose of this project was to create a binary classifier, using a neural network, to attempt to predict whether applicant to a charity organization, Alphabet Soup, will be successful with their funding. The starting dataset consisted of a CSV of over 34,000 organizations that received funding from Alphabet Soup. The columns in the dataset included:

EIN and NAME — Identification columns
APPLICATION_TYPE — Alphabet Soup application type
AFFILIATION — Affiliated sector of industry
CLASSIFICATION — Government organization classification
USE_CASE — Use case for funding
ORGANIZATION — Organization type
STATUS — Active status
INCOME_AMT — Income classification
SPECIAL_CONSIDERATIONS — Special consideration for application
ASK_AMT — Funding amount requested
IS_SUCCESSFUL — Was the money used effectively

The libraries used were:

Pandas
Scikit-learn
Tensorflow
The goal was to attempt to develop a model with over 75% accuracy in predicting success from the given data.

# Result
Preprocessing
The first step was to examine and preprocess the provided dataset.

The target variable was the IS_SUCCESSFUL column.
The unnecessary data for the purposes of this model were the EIN and NAME columns.
All other columns were considered potential features for the model. The next steps were to bin, encode, and scale the data.
Any APPLICATION_TYPE with less than 1000 entries were binned into OTHER.
Any CLASSIFICATION with less than 1000 entries were binned into OTHER.
All "object" type columns were encoded using OneHotEncoder.
The SPECIAL_CONSIDERATIONS_N column was dropped, as it was redundant to the SPECIAL_CONSIDERATIONS_Y column.
All columns were then scaled using StandardScaler.
Compiling, Training, and Evaluating
In the initial model I used:

two layers -- one with 80 neurons, the second with 45 neurons -- providing me with 6,891 total and trainable parameters;
both layers used 'relu' activation functions;
the output layer used 'sigmoid' activation function. Unfortunately I was only able to achieve 72.5% accuracy with this model.
I tried three more models in an attempt to reach 75% accuracy. In my subsequent attempts I attempted:

binning INCOME_AMT values greater than $5 million into a '5M+' bin;
adding a third hidden layer;
increasing the total number of trainable parameters to as high as 9,411;
increasing training epochs from 100 to 150, then as high as 300;
trying out both the 'adamax' and 'nadam' optimizers when compiling the model;
using 'tanh' activation functions on the hidden layers;
un-binning certain values by lowering the threshold from 1000 values to 700 values on both APPLICATION_TYPE and CLASSIFICATION.
Across all four of my attempts I never managed to raise my models' accuracy above 72.8%.

Summary

