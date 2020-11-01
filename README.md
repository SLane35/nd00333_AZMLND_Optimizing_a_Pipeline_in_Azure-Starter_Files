# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about bank customers and seeks to predict if a customer with subscribe to a fixed term deposit. 

The best-performing model used the VotingEnsemble algorithm and achieved an accuracy of 91.77%. The AutoML run outperformed the Azure ML pipeline using Hyperdrive.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
First, the data was cleaned. This included dropping certain columns and creating dictionaries for other columns as well as other functions. Then I set a parameter sampler, early-termination policy and estimator. I set the parameter sampler to RandomParameterSampling since it is quicker than Grid or Bayesian sampling and achieves excellent results. For the early-termination policy, I used BanditPolicy since it will terminate the run once it reaches the desired accuracy. I set the estimator to the train.py file, and this is what trains the model.

The algorithm used is Logistic Regression with Accuracy as the primary metric, and the goal is to get the maximum accuracy possible. I set the max total runs to 5 since I thought that would be enough to achieve a good accuracy, and the max concurrent runs to 3 so that the model will train faster.


## AutoML
The VotingEnsemble achieved the highest accuracy with 91.77%. I used a cross-validation value of 5, the task was set to classification with a primary metric of Accuracy. It used k-fold cross validation. 

## Pipeline comparison
The accuracies of the models were very similar - the AutoML model achieved an accuracy of 91.77% and the Hyperdrive model achieved an accuracy of 91.34%. However, the AutoML run was much easier to set up since all I had to do was set the training data, the type of task, the primary metric, and the number of cross validations. For the Hyperdrive run, I needed to decide which parameters to use, which options there should be for each parameter, and how each parameter should be sampled. I also needed to set the estimator to train the model. 

On the other hand, the Hyperdrive run lasted for about 9 minutes, and the AutoML run lasted a little more than a half hour. In this case it didn't make a big difference but it's possible that on a larger scale the difference would be more significant.

The AutoML model used a VotingEnsemble algorithm, and the Hyperdrive model used a LogisticRegression algorithm and I set the C and max_iter parameters.

Although the models achieved similar results, they went about getting the results in very different ways.

## Future work
If I had the opportunity to experiment more with this, I would try using different parameters for the Logistic Regression in the Hyperdrive run and also perhaps use the Bayesian sampler. I would also like to experiemnt with different primary metrics. 
