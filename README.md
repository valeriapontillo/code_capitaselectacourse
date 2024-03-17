# code_capitaselectacourse
This repository contains an example of a machine learning pipeline for the prediction of the presence or absence of a test refactoring in a test class. 
The pipeline can be run by means of the class "ml_run_configurations.py". The various classifiers and balancing techniques are reported in the "configuration.txt" file. I reported the combinations for just one project (emissary), but in the folder dataset there are other three different.

The ml_main.py class contains the input for the other part of the pipeline. If you want to change the binary variable to predict, you must modify part in which the variable "isRefactored" is reported. For the textual columns that cannot be scale or for columns that are not used as independent variable, you can report the information in the lines (always in ml_main.py) in which you see this: 

['App','Repository','SHA','Tag','TestFilePath','isRefactored','group']