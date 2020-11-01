# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about bank customers and seeks to predict if a customer with subscribe to a fixed term deposit. The data is an Azure sample dataset and can be downloaded from https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv.

The best-performing model used the VotingEnsemble algorithm and achieved an accuracy of 91.67%. The AutoML run outperformed the Azure ML pipeline using Hyperdrive.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
First, the data was cleaned. This included dropping certain columns and creating dictionaries for other columns as well as other functions. Then I set a parameter sampler, early-termination policy and estimator. I set the parameter sampler to RandomParameterSampling since it is quicker than Grid or Bayesian sampling (and also cheaper, since it doesn't consume as many resources or run for as long) and achieves excellent results. For the early-termination policy, I used BanditPolicy since it will terminate the run based on a slack factor. This means that it terminates any run that doesn't reach the slack factor of the evaluation metric with respect to the best performing run. I set the estimator to the train.py file, and this is what trains the model using a Logistic Regression algorithm. It uses Hyperdrive to automatically tune the hyperparameters to achieve the highest value in the set primary metric (in our case, accuracy). I set a Hyperdrive configuration to include as parameters the parameter sampler, early-termination policy and the estimator. I set the primary metric to accuracy with the goal to get the maximum accuracy possible. I set the max total runs to 5 since I thought that would be enough to achieve a good accuracy, and the max concurrent runs to 3 so that the model will train faster.


## AutoML
The VotingEnsemble achieved the highest accuracy with 91.77%. The AutoML parameters that I set were task, primary_metric, experiment_timeout_minutes, training_data, label_column_name, and n_cross_validations. Task can be classification, regression or forecasting. In our case, we are trying to predict whether a customer will subscribe to a fixed term deposit, which is a classification problem (yes or no). I set the primary metric to accuracy to match the primary metric used in the Hyperdrive run, and the experiment timeout was set to 30 minutes. I used 5 for the cross validations since that is a standard number to use.

## Pipeline comparison
The accuracies of the models were very similar - the AutoML model achieved an accuracy of 91.67% and the Hyperdrive model achieved an accuracy of 91.24%. However, the AutoML run was much easier to set up since all I had to do was set the training data, the type of task, the primary metric, and the number of cross validations. For the Hyperdrive run, I needed to decide which parameters to use, which options there should be for each parameter, and how each parameter should be sampled. I also needed to set the estimator to train the model. 

On the other hand, the Hyperdrive run lasted for about 9 minutes, and the AutoML run lasted a little more than a half hour. In this case it didn't make a big difference but it's possible that on a larger scale the difference would be more significant. This also had to do with that the AutoML model uses more complex algorithms than the Logistic Regression algorithm that was used with the Hyperdrive run.

The AutoML model used a VotingEnsemble algorithm, and the Hyperdrive model used a LogisticRegression algorithm and I set the C and max_iter parameters.

Although the models achieved similar results, they went about getting the results in very different ways.

## Future work
If I had the opportunity to experiment more with this, I would try using the Bayesian sampler instead of Random since it is more thorough and could achieve better results. I would also like to experiment with different primary metrics, like AUC. Since the data in this dataset is imbalanced, the accuracy does not really give a good measure of performance.
