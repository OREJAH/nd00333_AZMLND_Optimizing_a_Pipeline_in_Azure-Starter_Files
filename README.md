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
         > Benefits of parameter sampler
         > Benefits of early stopping policy
         
    ## AutoML
        
    ## Pipelines comparison
         > Comparison
         > Differences(accuracy)
         > Differences(architecture)
         > Reasons for differences
        
    ## Future work
    
        
## Project overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary 
     > The bank marketing dataset contains 32950 instances of clients data about bank marketing campaigns that are based on phone calls. We seek to predict if the client will subscribe to a bank term deposit.
**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

## Scikit-learn Pipeline

     > Pipeline architecture.
         - Setting up the training script.
             . Checked for parameters needed to be passed on to the script.
             . Imported the bank marketing dataset csv file using TabularDatasetFactory.
             . Dictionary for cleaning data.
             . OneHotEncoder processing technique applied to the dataframe.
             . Split the data into train sets(80%) and test sets(20%).
             . Argument parser construction to parse the arguments (--C: Regularization strength, --max_iter: Maximum number of iterations taken for the solvers to converge.)
             . Create a logistic regression model that tests for accuracy.
             . Source directory to save the generated custom-coded model
             
     > Data.
         -

     > Hyperparameter tuning.
         - Build hyperdrive service using jupyter notebook
             . Initialize workspace
             . Create a compute cluster to run the experiments on and check for existing cluster.
             . Specify parameter sampler
             . Specify a policy for early stopping.
             . Create an estimator for the training script.
             . Create a HyperDriveConfig with the following parameters 
             . Submit the hyperdriveconfig to run.
             . Use RunDetails to monitor run prrogress.
             . Select the best hyperparameters for the custom-coded model.
             . Save the best model.
     
      > Classification algorithm.
          - Logistic regression.
               . This is a machine learning binary classification algorithm that is used to predict the probability of a categorical dependent variable.  
           
      > Benefits of parameter sampler.
          - RandomParameterSampling.
              . It hepls to avoid bias.
              . It chooses the best hyperparameters.
              . It optimizes for speed versus accuracy.
               
      > Benefits of early stopping policy.
          - BanditPolicy.
              . It helps to avoid burning up a lot of resources while trying to find an optimal parameter
              

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
