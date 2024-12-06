import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb

# Load the trained model using pickle
with open('C:/Users/raman/Desktop/Price_Smart/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Feature names used for training the model
feature_names = [
    'Feedback Score', 'Category_Encoded', 'Feedback_Quality',
    'Positive Feedback %', 'Price_in_USD', 'Price_Condition',
    'Is_Trusted_Seller', 'Shipping Cost', 'Cost_to_Price_Ratio',
    'Condition_Encoded', 'Shipping Type_Encoded', 'Store URL_flag',
    'Listing Type_Encoded'
]

# Streamlit UI for input
st.title("Prediction App")
st.write("Enter the input features for the model:")

# Collect user input using Streamlit widgets
feedback_score = st.number_input("Feedback Score", min_value=0.0, format="%.2f")
category_encoded = st.number_input("Category Encoded", min_value=0.0, format="%.2f")
feedback_quality = st.number_input("Feedback Quality", min_value=0.0, format="%.2f")
positive_feedback_percent = st.number_input("Positive Feedback %", min_value=0.0, format="%.2f")
price_in_usd = st.number_input("Price in USD", min_value=0.0, format="%.2f")
price_condition = st.number_input("Price Condition", min_value=0.0, format="%.2f")
is_trusted_seller = st.number_input("Is Trusted Seller (0 or 1)", min_value=0, max_value=1, format="%.0f")
shipping_cost = st.number_input("Shipping Cost", min_value=0.0, format="%.2f")
cost_to_price_ratio = st.number_input("Cost to Price Ratio", min_value=0.0, format="%.2f")
condition_encoded = st.number_input("Condition Encoded", min_value=0.0, format="%.2f")
shipping_type_encoded = st.number_input("Shipping Type Encoded", min_value=0.0, format="%.2f")
store_url_flag = st.number_input("Store URL Flag (0 or 1)", min_value=0, max_value=1, format="%.0f")
listing_type_encoded = st.number_input("Listing Type Encoded", min_value=0.0, format="%.2f")

# Create a DataFrame with the input data in the correct feature order
input_data = pd.DataFrame([{
    'Feedback Score': feedback_score,
    'Category_Encoded': category_encoded,
    'Feedback_Quality': feedback_quality,
    'Positive Feedback %': positive_feedback_percent,
    'Price_in_USD': price_in_usd,
    'Price_Condition': price_condition,
    'Is_Trusted_Seller': is_trusted_seller,
    'Shipping Cost': shipping_cost,
    'Cost_to_Price_Ratio': cost_to_price_ratio,
    'Condition_Encoded': condition_encoded,
    'Shipping Type_Encoded': shipping_type_encoded,
    'Store URL_flag': store_url_flag,
    'Listing Type_Encoded': listing_type_encoded
}], columns=feature_names)

# Convert the DataFrame to a DMatrix for prediction
dinput = xgb.DMatrix(input_data)

# Predict and display the result
if st.button("Predict"):
    prediction = model.predict(dinput)
    st.write("Predicted Output:", prediction)








