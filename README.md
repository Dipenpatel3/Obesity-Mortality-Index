# Obesity Mortality Index
This project predicts obesity-related mortality rates based on other diseases using a linear regression model. The application is built with Streamlit for the frontend interface.

## Features
1. Visualize the total number of death rates by various diseases.
2. Display correlation matrix heatmap for the selected country.
3. Train a linear regression model to predict obesity mortality rates based on selected diseases and country.
4. Predict future obesity mortality rates by adjusting the parameters.

## How to Use the Application
1. Load the dataset: The application will automatically load the dataset number-of-deaths-by-risk-factor.csv(You just need to download the dataset and load in repo).

2. Select Year: Use the slider to select the start year.

3. Visualize Data:
-Total Number of Death Rate by Disease: Bar graph visualization of death rates.
-Correlation Matrix Heatmap: Select a country to see the correlation matrix of different diseases.

4. Train the Model:
-Select a disease for training the model.
-The application will display the R2 score and RMSE of the trained model along with a scatter plot of actual vs predicted values.

5. Predict Obesity Mortality:
-Select a country and year.
-Adjust the values to predict future obesity mortality rates.
-The application will display the predicted obesity death rate and percentage change.

## Output Screenshot:
1. Actual vs Predicted
<img width="804" alt="Screenshot 2024-07-09 at 4 50 12 PM" src="https://github.com/Dipenpatel3/Obesity-Mortality-Index/assets/60914088/87c21c5f-5e05-49f6-b540-b6dfa58ba3b9">

2. For a Country (Obesity vs Blood Pressure)
<img width="734" alt="Screenshot 2024-07-09 at 4 50 45 PM" src="https://github.com/Dipenpatel3/Obesity-Mortality-Index/assets/60914088/162bff52-12ff-4e78-9178-e807ab9f9093">

## Clone the repository:
git clone https://github.com/your-username/obesity-mortality-index.git
cd obesity-mortality-index

## Run the streamlit app:
streamlit run Obesity_Mortality_Index.py

## Installation
To run this project, you need to have Python installed along with the following packages:

1. streamlit
2. pandas
3. scikit-learn
4. matplotlib
5. seaborn
## You can install the necessary packages using:
```sh
pip install streamlit
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn


