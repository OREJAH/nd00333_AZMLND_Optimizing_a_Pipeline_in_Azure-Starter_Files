# Optimizing an ML Pipeline in Azure

Table of contents
## Project overview

## Summary
     > Problem description
     > Solution
    
## Scikit-learn pipeline
     > Pipeline architecture
     > Data
     > Hyperparameter tuning
     > Classification algorithm
     > Benefits of random sampling
     > Benefits of bandit policy
     
## AutoML
    
## Pipelines comparison
     > Comparison (accuracy)
     > Difference in architecture
     > Reasons for differences
    
## Future work

## Project overview

This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

![project overview](https://github.com/OREJAH/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/project%20overview.PNG)


## Summary
 The bank marketing dataset contains 32951 instances of client data about bank marketing campaigns that are based on phone calls. We seek to predict if the client will subscribe to a bank term deposit.
In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."

## Scikit-learn Pipeline

### Pipeline architecture.
    
 This involves setting up the python training script through the following processes below:

•	Check for packages needed to be passed on to the script.

from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

•	Create a dictionary for cleaning the dataset.  

def clean_data(data):

         #Dict for cleaning data

   Months = {“jan”:1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

•	Apply OneHotEncoder preprocessing technique to the dataframe.

  Clean and one hot encode data
    
    x_df = data.to_pandas_dataframe().dropna()
    
    jobs = pd.get_dummies(x_df.job, prefix="job")
    
    x_df.drop("job", inplace=True, axis=1)
    
    x_df = x_df.join(jobs)
    
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    
    x_df.drop("contact", inplace=True, axis=1)
    
    x_df = x_df.join(contact)
    
    education = pd.get_dummies(x_df.education, prefix="education")
    
    x_df.drop("education", inplace=True, axis=1)
    
    x_df = x_df.join(education)
    
    x_df["month"] = x_df.month.map(months)
    
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    
    return x_df,y_df
    

•	Import the bank marketing dataset csv file using the TabularDatasetFactory

  Data is located at: "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

src= "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

ds = TabularDatasetFactory.from_delimited_files(path=src)


•	Use the clean data function to clean the data

x, y = clean_data(ds)


•	Split the dataset into train sets of 80% and test sets of 20%.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)


•	Create a run object in the script to train the hyperdrive model.

run = Run.get_context()


•	Parse arguments (“--C:” Regularization strength and “--max_iter:” Maximum number of iterations taken for the solvers to converge.)

def main():
    # Add arguments to script
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    
    run.log("Max iterations:", np.int(args.max_iter))
    

•	Create a scikit learn logistic regression model that tests for accuracy.

model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    accuracy = model.score(x_test, y_test)
    
    run.log("Accuracy", np.float(accuracy))
    

•	Create a source directory to save the generated hyperdrive model.

os.makedirs('outputs', exist_ok=True)

    joblib.dump(model,"outputs/hyperdrive_model.joblib")
    
if __name__ == '__main__':

    main()
         
 ### Data. 

The bank marketing dataset includes bank records with 21 observations per record. Each record includes 20 explanatory observations about the client contacted and 1 response (y) observation of whether the client subscribed to a bank term deposit with boolean values(yes or no).

### Hyperparameter tuning.
 
 This was done by building a hyperdrive service using jupyter notebook through the following processes:
 
Initializing the azure machine learning workspace, creating a compute cluster to run the experiments on and check for existing cluster, since an existing cluster was found, it was used for the experiments instead of creating a new cluster.

Specified parameter sampler as RandomParameterSampling and also a policy for early stopping (early_termination_policy). Created a scikit-learn estimator for the training script.
Created a HyperDriveConfig with the following parameters:

(
hyperparameter_sampling=parameter_sampling,
primary_metric_name=’Accuracy’,
primary_metric_goal=PrimaryMetricGoal_MAXIMIZE,
max_total_runs=4,
max_concurrent_runs=4
)

Next, submitted the HyperDriveConfig and used RunDetails to monitor run progress. Selected the best hyperparameters for the hyperdrive model and saved the best model which had an accuracy of 0.9072837632776934.
 
### Classification algorithm.

  For the hyperdrive pipeline, logistic regression was the algorithm used. This is a machine learning binary classification algorithm used to predict the probability of a categorical dependent variable.In this case, we'll be getting outputs in boolean form of either yes or no.
  
### Benefits of parameter sampler.
 In this hyperdrive pipeline, we made use of the random parameter sampling method which helped to avoid bias, choose the best hyperparameters and also optimize for speed versus accuracy.
 
### Benefits of early stopping policy.
  Lastly, the bandit policy was applied to the hyperdrive pipeline and two parameters were passed,
  
 i:  evaluation_interval: This is the frequency for applying the bandit policy, in this case, an integer value of 2 was passed to the parameter.
 
 ii: slack_factor: This is the ratio used to calculate the allowed distance from the best performing experiment run, a decimal value of 0.2 was passed to the parameter.
  
The bandit policy helped to avoid burning up a lot of resources while trying to find an optimal parameter, it terminates any run that does not fall within the slack factor's range.

The hyperdrive workflow diagram is seen below:

![hyperdrive](https://github.com/OREJAH/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Hyperdrive%20workflow.png)

 ## AutoML 

The automl pipeline produced its best performing model known as Voting Ensemble at the 40th iteration of the experiment with an accuracy of 0.9180576631259484, it is a useful technique which comes especially handy when a single model shows some kind of bias. The Voting Ensemble estimates multiple base models and uses voting to combine the individual predictions to arrive at the final ones.  

## Pipeline comparison

### Comparison (accuracy)

The hyperdrive service automatically adjusted the hyperparameters, found the best parameters and used them in training the scikit learn logistic regression model with an accuracy of 0.9072837632776934 while the automl model automatically took the data, went through in selecting the best categorical data, including the label and gave a predicting model with an accuracy of 0.9180576631259484. The AutoML has a better model accuracy.

### Difference in architecture:

The hyperdrive made use of a python training script to optimize the parameters for the logistic regression algorithm and provide its accuracy while the AutoML used several learning algorithms to obtain its accuracy.


### Reasons for difference.

AutoML's ability to automatically and efficiently identify the algorithms or models that work best for the dataset through an iterative process of several algorithms can be a reason for the difference in accuracy. It is not restricted to just one maching learning algorithm as in the case of hyperparameter tuning.

## Future work

Some areas of improvement for future experiments are the use of the grid sampling method and the bayesian sampling method.

These improvements might help the model in the following ways:

Grid sampling: It helps in selecting a criteria and choose the best result for the model.

Bayesian sampling: It makes it possible to use a different kind of statistical technique to improve the hyperparameter.
