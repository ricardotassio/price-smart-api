import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb

# Function to load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
model_path = 'C:/Users/japje/Downloads/model.pkl'
model = load_model(model_path)

# Define the feature names as in the trained model
FEATURES = [
    'Feedback Score', 'Category_Encoded', 'Feedback_Quality', 'Positive Feedback %',
    'Price_in_USD', 'Price_Condition', 'Is_Trusted_Seller', 'Shipping Cost', 
    'Cost_to_Price_Ratio', 'Condition_Encoded', 'Shipping Type_Encoded', 
    'Store URL_flag', 'Listing Type_Encoded'
]

# Streamlit interface
st.title("Item Price Prediction")
st.write("Enter the details of the item to get the predicted output.")

# Collect user inputs dynamically using a dictionary to match the feature names
user_inputs = {}
for feature in FEATURES:
    user_inputs[feature] = st.number_input(f"Enter {feature}", format="%.2f", min_value=0.0)

# Convert user inputs into a pandas DataFrame
input_data = pd.DataFrame([user_inputs])

# Display the collected input data
st.write("You entered the following data:")
st.write(input_data)

# Convert the DataFrame to DMatrix for prediction
dinput = xgb.DMatrix(input_data)

# Prediction logic
if st.button("Predict"):
    prediction = model.predict(dinput)
    st.write(f"Prediction Result: {prediction[0]:.2f}")

# Styling (optional)
st.markdown("""
    <style>
        .css-ffhzg2 {text-align: center;}
    </style>
""", unsafe_allow_html=True)
