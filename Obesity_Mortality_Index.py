import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('number-of-deaths-by-risk-factor.csv')
centered_header = """
<div style="text-align: center;">
    <h2>Obesity Mortality Index</h2>
</div>
"""

st.markdown(centered_header, unsafe_allow_html=True)

#Disease Columns
disease_columns = ['Blood_Pressure',
    'Diet_High_Sodium', 'Diet_Low_Whole_grains', 'Alcohol',
    'diet_low_fruits', 'unsafe_water_source', 'secondhand_smoke',
    'low_birth_weight', 'child_wasting', 'unsafe_sex',
    'diet_low_nuts_seeds', 'air_pollution_solid_fuels',
    'diet_low_vegetables', 'smoking', 'high_fasting_plasma_glucose',
    'air_pollution', 'Obesity', 'unsafe_sanitatin', 'drug_use',
    'low_bone_mineral_density', 'vitamin_deficiency', 'child_stunting',
    'non-exclusive_breastfeeding', 'iron_deficiency',
    'ambient_particulate_Matter_pollution', 'low_physical_activity',
    'handwashing_facility', 'high_idl_cholesterol']

st.markdown('### Total Number of Death Rate by Disease')
# Function to preprocess data
def preprocess_data(data):
    # Renaming columns
    new_columns_name={'Entity':'Country_Name',
                 'Code':'Country_Code',
                 'Year':'Year',
                 'Deaths that are from all causes attributed to high systolic blood pressure, in both sexes aged all ages':'Blood_Pressure',
                 'Deaths that are from all causes attributed to diet high in sodium, in both sexes aged all ages':'Diet_High_Sodium',
                 'Deaths that are from all causes attributed to diet low in whole grains, in both sexes aged all ages':'Diet_Low_Whole_grains',
                 'Deaths that are from all causes attributed to alcohol use, in both sexes aged all ages':'Alcohol',
                 'Deaths that are from all causes attributed to diet low in fruits, in both sexes aged all ages':'diet_low_fruits',
                 'Deaths that are from all causes attributed to unsafe water source, in both sexes aged all ages':'unsafe_water_source',
                 'Deaths that are from all causes attributed to secondhand smoke, in both sexes aged all ages':'secondhand_smoke',
                 'Deaths that are from all causes attributed to low birth weight, in both sexes aged all ages':'low_birth_weight',
                 'Deaths that are from all causes attributed to child wasting, in both sexes aged all ages':'child_wasting',
                 'Deaths that are from all causes attributed to unsafe sex, in both sexes aged all ages':'unsafe_sex',
                 'Deaths that are from all causes attributed to diet low in nuts and seeds, in both sexes aged all ages':'diet_low_nuts_seeds',
                 'Deaths that are from all causes attributed to household air pollution from solid fuels, in both sexes aged all ages':'air_pollution_solid_fuels',
                 'Deaths that are from all causes attributed to diet low in vegetables, in both sexes aged all ages':'diet_low_vegetables',
                 'Deaths that are from all causes attributed to smoking, in both sexes aged all ages':'smoking',
                 'Deaths that are from all causes attributed to high fasting plasma glucose, in both sexes aged all ages':'high_fasting_plasma_glucose',
                 'Deaths that are from all causes attributed to air pollution, in both sexes aged all ages':'air_pollution',
                 'Deaths that are from all causes attributed to high body-mass index, in both sexes aged all ages':'Obesity',
                 'Deaths that are from all causes attributed to unsafe sanitation, in both sexes aged all ages':'unsafe_sanitatin',
                 'Deaths that are from all causes attributed to drug use, in both sexes aged all ages':'drug_use',
                 'Deaths that are from all causes attributed to low bone mineral density, in both sexes aged all ages':'low_bone_mineral_density',
                 'Deaths that are from all causes attributed to vitamin a deficiency, in both sexes aged all ages':'vitamin_deficiency',
                 'Deaths that are from all causes attributed to child stunting, in both sexes aged all ages':'child_stunting',
                 'Deaths that are from all causes attributed to non-exclusive breastfeeding, in both sexes aged all ages':'non-exclusive_breastfeeding',
                 'Deaths that are from all causes attributed to iron deficiency, in both sexes aged all ages':'iron_deficiency',
                 'Deaths that are from all causes attributed to ambient particulate matter pollution, in both sexes aged all ages':'ambient_particulate_Matter_pollution',
                 'Deaths that are from all causes attributed to low physical activity, in both sexes aged all ages':'low_physical_activity',
                 'Deaths that are from all causes attributed to no access to handwashing facility, in both sexes aged all ages':'handwashing_facility',
                 'Deaths that are from all causes attributed to high ldl cholesterol, in both sexes aged all ages':'high_idl_cholesterol'}
    data.rename(columns=new_columns_name, inplace=True)
    
    # # Drop rows with missing values
    # data.dropna(inplace=True)
    
    # Label encoding for 'Country_Name'
    le_country = LabelEncoder()
    data['Country_Code'] = le_country.fit_transform(data['Country_Name'])
    
    return data
preprocess_data(data)
# Slider for selecting the year range
start_year = st.slider('Select start year:', 1990, 2019, 1990)

# Function to filter data by year range and plot bar graph
def plot_cases_by_disease(data, disease_columns, start_year):
    data = data[(data['Year'] ==start_year)]

    # Calculate total cases for each disease
    total_cases = data[disease_columns].sum(axis=0)
    total_cases_sorted = total_cases.sort_values(ascending=True) / 1000000  # Divide by 1,000,000 to convert to million

    # Plotting the bar graph using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    total_cases_sorted.plot(kind='barh', ax=ax)

    # Add labels to the bars
    for i, v in enumerate(total_cases_sorted):
        ax.text(v, i, f'{v:.2f} million', va='center', ha='left', color='black')

    ax.set_xlabel('Number of Cases (Millions)')
    ax.set_ylabel('Disease or Health Risk Factor')
    ax.set_title('Total Number of Cases by Disease or Health Risk Factor (Sorted)')

    # Display the plot using Streamlit
    st.pyplot(fig)


# Plot the bar graph
plot_cases_by_disease(data, disease_columns, start_year)

# Streamlit app
st.markdown('### Correlation of Country with the deaths')

# Dropdown for selecting country
selected_country = st.selectbox('Select a country', data['Country_Name'].unique())

def cor(data,selected_country):
    numeric_cols = [
        'Alcohol', 'Blood_Pressure', 'Diet_High_Sodium', 'Diet_Low_Whole_grains',
        'diet_low_fruits', 'diet_low_nuts_seeds', 'diet_low_vegetables',
        'drug_use', 'smoking', 'vitamin_deficiency', 'Obesity'
    ]
    # Filter rows where 'Country_Name' is 'Afghanistan' and calculate correlation matrix
    data_Country = data[data['Country_Name'] == selected_country]
    correlation_matrix = data_Country[numeric_cols].corr()

    # Plot the heatmap using Streamlit
    st.write("## Correlation Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

cor(data,selected_country)

# Function to train linear regression model
## Start of Model 

st.markdown('### Select the Disease for Training the Model')
selected_disease_model=st.selectbox('Select Disease',disease_columns)

X=data[['Year','Country_Code',selected_disease_model]]
y=data['Obesity']

def train_model(X, y,selected_disease_model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared = False)
    r2 = r2_score(y_test, y_pred)
    st.write('The r2 square value for model is:',r2)
    #st.write('RMSE value of the model is:',rmse)
    # Plotting actual vs predicted values
    fig,ax=plt.subplots(figsize=(10, 6))

    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line representing perfect prediction
    ax.set_title(f'Actual vs Predicted Obesity given {selected_disease_model}')
    ax.set_xlabel('Actual Obesity')
    ax.set_ylabel('Predicted Obesity')
    plt.grid(True)
    st.pyplot(fig)
    return model
x1=train_model(X,y,selected_disease_model)

st.markdown('### Select the parameter to predict the death of the obesity with the selected Disease')
selected_country_model=st.selectbox('Select a country for predicting', data['Country_Name'].unique())
selected_year_values=st.slider('Select a year for the values',1990, 2019, 1990)

data_all=data[data['Country_Name']==selected_country_model]['Country_Code'].values
country_code=data_all[0]

# country_code=country_code_1[0]
selected_year_values=int(selected_year_values)

data_disease_country=data[data['Year']==selected_year_values]

data_disease_Value=data_disease_country[data_disease_country['Country_Name']==selected_country_model][selected_disease_model].values
st.write(f'The given value for {selected_disease_model} Disease for the year {selected_year_values} is',data_disease_Value[0])

selected_value_predict=st.slider('Select a value to insert',int(data_disease_Value)-10000,int(data_disease_Value[0])+100000)
difference_value=selected_value_predict-int(data_disease_Value[0])

st.write(f'Difference in the value from {selected_year_values} year is:{difference_value}')

selected_predict_year=st.slider('Select a year for prediction',int(selected_year_values),2030)

selected_value_predict=int(selected_value_predict)
percentage_value=difference_value/int(data_disease_Value[0])*100
percentage_value=round(percentage_value,2)

# Determine the color based on whether the value is positive or negative
if percentage_value > 0:
    color = 'green'
else:
    color = 'red'

# Create the text with the color and two-digit precision
styled_text = f"""
    Percentage change from the year {selected_year_values} to the predicted year {selected_predict_year} is: 
    <span style="color: {color};">{percentage_value}%</span>
"""

# Display the styled text in Streamlit
st.markdown(styled_text, unsafe_allow_html=True)

def predict_obesity_count(x1, selected_predict_year, selected_value_predict, country_code, selected_disease_model):
    new_data = pd.DataFrame({
        'Year': [selected_predict_year],
        'Country_Code': [country_code],
        selected_disease_model: [selected_value_predict]
    })
    obesity_pred = x1.predict(new_data)
    return obesity_pred

final_value=predict_obesity_count(x1,selected_predict_year,selected_value_predict,country_code,selected_disease_model)[0]
final_value=int(final_value)
# # Display the predicted obesity count
if final_value>0:
    st.markdown(f'Death rate entered for {selected_disease_model}: {selected_value_predict}')
    st.markdown(f'Predicted Obesity death rate : {final_value}')
    
numeric_cols = [
        'Country_Name','Country_Code','Year',
        'Alcohol', 'Blood_Pressure', 'Diet_High_Sodium', 'Diet_Low_Whole_grains',
        'diet_low_fruits', 'diet_low_nuts_seeds', 'diet_low_vegetables',
        'drug_use', 'smoking', 'vitamin_deficiency', 'Obesity'
    ]
filter_data_country=data[data['Country_Name']==selected_country_model]
filter_data_country=filter_data_country[numeric_cols]

def predict_visulization(selected_disease_model,selected_country_model,x1,filter_data_country):
    # Creating a scatter plot with the selected disease model
    # st.markdown('### Actual vs Predict Plot of Obesity given ' + selected_disease_model+' for Country '+selected_country_model)
    for col in numeric_cols:
        if col==selected_disease_model:
            X_diseases=filter_data_country[col].values.reshape(-1,1)
            y_diseases=filter_data_country['Obesity'].values.reshape(-1,1)    
            
            model=x1.fit(X_diseases,y_diseases)
            Y_Pred=model.predict(X_diseases)
            
            #plotting linear regression model
            fig,ax=plt.subplots(figsize=(10, 6))
            
            sns.scatterplot(x=filter_data_country[col], y=filter_data_country['Obesity'], color='blue', label='Actual data')
            plt.plot(X_diseases, Y_Pred, color='red', linewidth=2, label='Linear regression')
            # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line representing perfect prediction
            ax.set_title(f'Obesity given {selected_disease_model} for country {selected_country_model}')
            ax.set_xlabel(f'{selected_disease_model}')
            ax.set_ylabel('Obesity')
            ax.grid(True)
            st.pyplot(fig)
    
predict_visulization(selected_disease_model,selected_country_model,x1,filter_data_country)