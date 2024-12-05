import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib

# Set page config
st.set_page_config(page_title="Product Comparison", layout="wide")

# Load model
def load_model():
    try:
        model = joblib.load("./model.pkl")  # Ensure the path is correct
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define the mapping dictionary for predictions
prediction_mapping = {0: '0-30 days', 1: '30-90 days', 2: '90-360 days', 3: '>360 days'}

# Define category mapping (update with actual category names)
category_mapping = {
    1000: 'Category A',
    1500: 'Category B',
    2000: 'Category C',
    2500: 'Category D'
}

# Define input fields with corrected feature names
fields = {
    "Product Name": "Product 1",
    "Feedback Score": 52000,
    "Category_Encoded": [1000, 1500, 2000, 2500],
    "Feedback_Quality": 1000,
    "Positive Feedback %": 98.5,
    "Price_in_USD": 25.00,
    "Price_Condition": 20.00,
    "Is_Trusted_Seller": ["Yes", "No"],
    "Shipping Cost": 5.00,
    "Cost_to_Price_Ratio": 0.20,
    "Condition_Encoded": 3,
    "Shipping Type_Encoded": 1,
    "Store URL_flag": ["Yes", "No"],
    "Listing Type_Encoded": 1,
}

# Main content area

# Add centered space with the PriceSmart platform name (and logo if available)
st.markdown("""
    <div style='text-align: center; margin-bottom: 40px;'>
        <!-- Replace 'logo_url' with the actual URL of the PriceSmart logo -->
        <!-- <img src='logo_url' alt='PriceSmart Logo' style='width:100px;'> -->
        <h1 style='font-size: 2.5em; color: #1e90ff;'>PriceSmart</h1>
    </div>
    """, unsafe_allow_html=True)

# Title for the app
st.markdown("<h2 style='text-align: center;'>Product Comparison</h2>", unsafe_allow_html=True)

# Create two columns for input forms
col1, col2 = st.columns(2)

input_data1 = {}
input_data2 = {}

# Process input fields for Product 1
with col1:
    st.subheader("Product 1")
    input_data1["Product Name"] = st.text_input("Product Name", value="Product 1")
    for field, default in fields.items():
        if field == "Product Name":
            continue  # Already handled
        if isinstance(default, list):  # Handle list fields
            if field == "Category_Encoded":
                user_input = st.selectbox(f"{field}", options=default)
                input_data1[field] = user_input
            else:
                user_input = st.radio(field, options=default)
                input_data1[field] = 1 if user_input == "Yes" else 0
        else:  # Handle numeric fields
            input_data1[field] = st.number_input(f"{field}", value=default, key=f"{field}_1")

# Process input fields for Product 2
with col2:
    st.subheader("Product 2")
    input_data2["Product Name"] = st.text_input("Product Name", value="Product 2", key="Product_Name_2")
    for field, default in fields.items():
        if field == "Product Name":
            continue  # Already handled
        if isinstance(default, list):  # Handle list fields
            if field == "Category_Encoded":
                user_input = st.selectbox(f"{field}", options=default, key=f"{field}_2")
                input_data2[field] = user_input
            else:
                user_input = st.radio(field, options=default, key=f"{field}_2")
                input_data2[field] = 1 if user_input == "Yes" else 0
        else:  # Handle numeric fields
            input_data2[field] = st.number_input(f"{field}", value=default, key=f"{field}_2")

# Submit button and prediction
if st.button("Compare Products"):
    if model:
        try:
            # Prepare data for Product 1
            df_input1 = pd.DataFrame([input_data1])
            df_input1_model = df_input1[[
                'Feedback Score',
                'Category_Encoded',
                'Feedback_Quality',
                'Positive Feedback %',
                'Price_in_USD',
                'Price_Condition',
                'Is_Trusted_Seller',
                'Shipping Cost',
                'Cost_to_Price_Ratio',
                'Condition_Encoded',
                'Shipping Type_Encoded',
                'Store URL_flag',
                'Listing Type_Encoded'
            ]]
            dmatrix_input1 = xgb.DMatrix(df_input1_model)
            prediction1 = model.predict(dmatrix_input1)
            label1 = prediction_mapping[int(prediction1[0])]
            category_name1 = category_mapping.get(int(df_input1['Category_Encoded'][0]), "Unknown Category")

            # Prepare data for Product 2
            df_input2 = pd.DataFrame([input_data2])
            df_input2_model = df_input2[[
                'Feedback Score',
                'Category_Encoded',
                'Feedback_Quality',
                'Positive Feedback %',
                'Price_in_USD',
                'Price_Condition',
                'Is_Trusted_Seller',
                'Shipping Cost',
                'Cost_to_Price_Ratio',
                'Condition_Encoded',
                'Shipping Type_Encoded',
                'Store URL_flag',
                'Listing Type_Encoded'
            ]]
            dmatrix_input2 = xgb.DMatrix(df_input2_model)
            prediction2 = model.predict(dmatrix_input2)
            label2 = prediction_mapping[int(prediction2[0])]
            category_name2 = category_mapping.get(int(df_input2['Category_Encoded'][0]), "Unknown Category")

            # Determine the winning product (product with lower score)
            if int(prediction1[0]) < int(prediction2[0]):
                winner = df_input1['Product Name'][0]
                trophy1 = "🏆"
                trophy2 = ""
            elif int(prediction2[0]) < int(prediction1[0]):
                winner = df_input2['Product Name'][0]
                trophy1 = ""
                trophy2 = "🏆"
            else:
                winner = "Tie"
                trophy1 = trophy2 = ""

            # Add centered title to the result
            st.markdown("<h2 style='text-align: center; color: #1e90ff; margin-top: 50px;'>Best Product with Shortest Time in Stock</h2>", unsafe_allow_html=True)

            # Display results using custom HTML and CSS
            st.markdown("""
                <style>
                .card {
                    background-color: #ffffff;
                    border: 1px solid #e0e0e0;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    margin-bottom: 20px;
                }
                .card h2 {
                    margin-top: 0;
                    color: #1e90ff;
                }
                .card h3 {
                    margin: 0;
                    color: #555555;
                    font-weight: normal;
                    margin-bottom: 10px;
                }
                .card p {
                    color: #333333;
                }
                .trophy {
                    font-size: 1.5em;
                }
                .vs {
                    font-size: 2em;
                    text-align: center;
                    margin-top: 100px;
                    color: #777777;
                }
                </style>
                """, unsafe_allow_html=True)

            result_col1, vs_col, result_col2 = st.columns([1, 0.1, 1])

            with result_col1:
                st.markdown(f"""
                    <div class="card">
                        <h2>{df_input1['Product Name'][0]} <span class="trophy">{trophy1}</span></h2>
                        <h3>{category_name1}</h3>
                        <p><strong>Prediction:</strong> {label1}</p>
                    </div>
                    """, unsafe_allow_html=True)

            with vs_col:
                st.markdown(f"<div class='vs'>vs</div>", unsafe_allow_html=True)

            with result_col2:
                st.markdown(f"""
                    <div class="card">
                        <h2>{df_input2['Product Name'][0]} <span class="trophy">{trophy2}</span></h2>
                        <h3>{category_name2}</h3>
                        <p><strong>Prediction:</strong> {label2}</p>
                    </div>
                    """, unsafe_allow_html=True)

            if winner != "Tie":
                st.success(f"**{winner}** is the better choice!")
            else:
                st.info("It's a tie between the two products.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model is not loaded. Please verify the model file.")
