# Linear-Regression
California House Price Prediction
This project explores the California Housing dataset to build a predictive model that estimates median house prices based on various demographic and geographic features. The notebook follows a standard machine learning workflow, including data exploration, visualization, model training, and evaluation.

Table of Contents
Project Objective

Dataset Overview

Technologies Used

Project Workflow

Model Evaluation

Usage

Project Objective
The primary goal of this project is to use Linear Regression to predict the median house value for California districts based on the 1990 U.S. Census data.

Dataset Overview
The project utilizes the California Housing dataset fetched from sklearn.datasets.

Instances: 20,640

Features: 8 numeric predictive attributes:

MedInc: Median income in block group

HouseAge: Median house age in block group

AveRooms: Average number of rooms per household

AveBedrms: Average number of bedrooms per household

Population: Block group population

AveOccup: Average number of household members

Latitude: Block group latitude

Longitude: Block group longitude

Target Variable: SalePrice (Median house value in hundreds of thousands of dollars).

Technologies Used
The following Python libraries are used for data manipulation, visualization, and modeling:

Data Handling: pandas, numpy

Visualization: matplotlib, seaborn, plotly

Machine Learning: scikit-learn (LinearRegression, train_test_split, mean_squared_error)

Project Workflow
Data Loading: Importing the dataset and converting it into a structured pandas DataFrame.

Exploratory Data Analysis (EDA):

Checking data dimensions and data types.

Statistical summaries using describe().

Identifying missing values (the dataset contains no null values).

Data Visualization: Using seaborn and plotly to understand feature distributions and relationships.

Model Training:

Splitting the data into training and testing sets.

Training a Linear Regression model on the features to predict SalePrice.

Evaluation: Testing the model's accuracy on unseen data.

Model Evaluation
The model's performance is measured using standard regression metrics:

Mean Squared Error (MSE): ~0.0503.

Root Mean Squared Error (RMSE): Calculated as the square root of MSE.

Sample output comparison:

Actual Value: 0.477

Model Predicted Value: 0.627

Usage
To run this project:

Ensure you have Python installed.

Install dependencies:

Bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn
Open Untitled-1.ipynb in a Jupyter environment (VS Code, JupyterLab, etc.) and run all cells.
