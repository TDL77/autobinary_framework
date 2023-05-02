# Autobinary Framework version 1.0.11

The Autobinary library is a set of tools that allow you to automate the process of building a model to solve certain business problems.

## Autobinary allows you:
-----------------

  1. To conduct a primary exploratory analysis and process factors;
  2. To conduct a primary selection of factors from all available;
  3. To conduct a primary training according to the required cross-validation scheme;
  4. To search for the optimal set of hyperpatmers;
  5. To conduct a deep selection of factors to finalize the model;
  6. To calibrate the final model if necessary;
  7. To visualize optimization and business metrics;
  8. To conduct an interpretative analysis of the factors.

## How to use:
-----------------

#### Installation script:
  1. Move the installation file autobinary-1.0.10.tar.gz to the required folder
  2. Install with: !pip install autobinary-1.0.10.tar.gz
  3. Import the library with: import autobinary


#### Manual adjustment:
  1. Move "autobinary" folder to local space;
  2. Set the path to the "autobinary" folder;
  3. Import the necessary tools from the autobinary library.

## Requirements:
-----------------

  * pandas >= 1.3.1
  * numpy >= 1.21.5 
  * catboost >= 0.25.1
  * matplotlib >= 3.1.0
  * sklearn >= 0.24.2 and <1.2.0
  * pdpbox == 0.2.0

## The repository folders provide detailed examples of using the library:
-----------------

  1. 01_Feature_vs_Target:

    * Examples of analysis of the target variable with respect to the factor for classification problems;

    * Examples of analysis of the target variable relative to the factor for regression problems.

  2. 02_CV_importances_for_trees:

    * Examples of training various algorithms for solving classification problems according to a cross-validation scheme;
    
    * Examples of training various algorithms for solving regression problems using a cross-validation scheme;
    
    * Examples of training various algorithms for solving multiclassification problems using a cross-validation scheme;

    * Calculation of the importance of factors after learning the algorithm;

  3. 03_Tuning_parameters_Optuna:

    * Examples of finding the optimal set of hyperparameters using the Optuna library.
    
  4. 04_Explaining_output:
  
    * Examples of interpretation of the influence of factors on the target variable using the Shap library;
    
    * Examples of interpretation of the influence of factors on the target variable using the PDPbox library.
    
  5. 05_Uplift_models:
  
    * Examples of Solo model for solving uplift problems with the necessary cross-validation scheme;
    
    * Examples of Two models (Vanilla) for solving uplift problems with the necessary cross-validation scheme;
    
    * Examples of Two models (DDR control) for solving uplift problems with the necessary cross-validation scheme;
    
    * Examples of Two models (Treatment control) for solving uplift problems with the necessary cross-validation scheme.
    
  6. 06_Base_uplift_calibration:
  
    * Calibration examples for response tasks;
    
    * Calibration examples for uplift tasks;
    
    * Calibration examples for other types of tasks;

  7. 07_Feature_selection:

    * Examples of primary selection of factors from all available using gap analysis, correlation analysis, tree depth analysis, as well as the Permutation Importance method (for binary classification, regression and multiclass classification);
    
    * Examples of deep selection of factors using the Forward and Backward selection methods;

    * Examples of factor selection using the Target Permutation method.

  8. 08_Custom_metrics:

    * Examples of visualization of known and custom metrics for a detailed understanding of the quality of the algorithm in binary classification and uplift tasks;

    * Examples of visualization of known and custom metrics for a detailed understanding of the quality of the algorithm in regression problems.

  9. 09_Finalization_calibration:

    * An example of the finalization and calibration of the model with the existing model and training with the given parameters for the binary classification problem;
    
    * An example of the finalization and calibration of the model with the existing model and training with the given parameters for the regression problem;

  10. 10_Full_Fitting_model:

    * An example of the entire process of building and finalizing a model in a laptop for a binary classification problem (probability of surviving a Titanic crash).


#### Authors:
* Vasily Sizov - https://github.com/Vasily-Sizov
* Dmitry Timokhin - https://github.com/dmitrytimokhin
* Pavel Zelenskiy - https://github.com/vselenskiy777
* Ruslan Popov - https://github.com/RuslanPopov98
