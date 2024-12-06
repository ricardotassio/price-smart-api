import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
from PIL import Image
import base64

# Set page config
st.set_page_config(page_title="Product Comparison", layout="wide")

# Load model
def load_model():
    try:
        model = joblib.load("./model.pkl")  # Ensure the path to your model is correct
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Prediction mapping
prediction_mapping = {
    0: '0-30 days',
    1: '30-90 days',
    2: '90-360 days',
    3: '>360 days'
}

# Category mapping
category_mapping = {
    '12-Volt Portable Appliances': 0, '2D Nail Art Design': 9, '3D TV Glasses & Accessories': 10, 
    '40K Miniatures': 11, '40K Rulebooks & Publications': 12, '40K Spare Bits & Pieces': 13, 
    '40K Starter Sets': 14, '40K Terrain & Scenery': 15, '50p': 16, '8-Track Players': 17, 
    'A/V Cleaning Cloths & Brushes': 18, 'A/V Cleaning Kits': 19, 'AC/DC': 20, 'Accessories': 21, 
    'Accessories & Dice': 22, 'Accordions': 23, 'Acoustic Electric Guitars': 24, 'Acoustic Guitars': 25, 
    'Action Figures': 26, 'Action/ Adventure': 27, 'Activity Centers': 28, 'Adapters': 29, 
    'Adapters & Converters': 30, 'Address & Shipping Labels': 31, 'Adult & Drinking Games': 32, 
    'Advent Calendars': 33, 'Adventure Modules': 34, 'Advertisements': 35, 'Africa': 36, 
    'Agility Training': 37, 'Air Force': 38, 'Air Rifles': 39, 'American': 40, 
    'Ammunition Belts & Bandoliers': 41, 'Amplifier Kits': 42, 'Amplifiers & Preamps': 43, 
    'Animation Art': 44, 'Antennas': 45, 'Anti-Aging Products': 46, 'Antiquarian & Collectible': 47, 
    'Apparel': 48, 'Aromatherapy': 49, 'Art & Prints': 50, 'Art Drawings': 51, 'Art Photographs': 52, 
    'Art Posters': 53, 'Art Prints': 54, 'Art Sculptures': 55, 'Ashtrays': 56, 
    'Atmospheric Effects Fluids': 57, 'Atmospheric Effects Machines': 58, 'Audio Cables & Interconnects': 59, 
    'Audio Docks & Mini Speakers': 60, 'Audio Tapes': 61, 'Audio/Video Transmitters': 62, 
    'Autoharps & Zithers': 63, 'Axis & Allies': 64, 'BMW': 65, 'Baby Bibs & Burp Cloths': 66, 
    'Baby Books & Albums': 67, 'Baby Bottles': 68, 'Baby Boxes': 69, 'Baby Food': 70, 
    'Baby Gyms, Play Mats & Jigsaw Mats': 71, 'Baby Jumping Exercisers': 72, 'Baby Locks & Latches': 73, 
    'Baby Monitors': 74, 'Baby Picture Frames': 75, 'Baby Scales': 76, 'Baby Swings': 77, 
    'Baby Wipe Warmers': 78, 'Baby Wipes': 79, 'Backyard Poultry Supplies': 80, 'Badges': 81, 
    'Badges/ Patches/ Stickers': 82, 'Bag Tags': 83, 'Bagpipes': 84, 'Bags & Cases': 85, 
    'Baits & Lures': 86, 'Ball Markers': 87, 'Balls': 88, 'Banjos': 89, 'Banners & Flags': 90, 
    'Barbells & Attachments': 91, 'Base Layers & Thermals': 92, 'Baseball Shirts & Jerseys': 93, 
    'Baseball-MLB': 94, 'Baseballs': 95, 'Basketball Hoops': 96, 'Basketball-NBA': 97, 
    'Bass Guitars': 98, 'Bassinets': 99
}

category_list = list(category_mapping.keys())

# Condition mapping
Condition = {'Almost new': 0, 'Defective': 1, 'New': 2, 'Unknown': 3, 'Used': 4}
condition_list = list(Condition.keys())

# Shipping Type mapping
Shipping_Type = {
    'Calculated': 0,
    'CalculatedDomesticFlatInternational': 1,
    'Flat': 2,
    'FlatDomesticCalculatedInternational': 3,
    'Free': 4,
    'FreePickup': 5,
    'Freight': 6,
    'NotSpecified': 7
}
shipping_type_list = list(Shipping_Type.keys())

# Listing Type mapping
Listing_Type = {'Auction': 0, 'AuctionWithBIN': 1, 'FixedPrice': 2, 'StoreInventory': 3}
listing_type_list = list(Listing_Type.keys())

# We will not use Shipping_Listing_Type_Encoded here since the model was not trained with it

# Define input fields
fields = {
    "Product Name": "Product 1",
    "Feedback Score (num)": 52000,
    "Category": category_list,
    "Feedback Quality (num)": 1000,
    "Positive Feedback (%)": 98.5,
    "Price (USD)": 25.00,
    "Price Condition (USD)": 20.00,
    "Trusted Seller?": ["Yes", "No"],
    "Shipping Cost (USD)": 5.00,
    "Cost/Price Ratio": 0.20,
    "Condition": condition_list,
    "Shipping Type": shipping_type_list,
    "Has Store URL?": ["Yes", "No"],
    "Listing Type": listing_type_list
}
fields2 = {
    "Product Name": "Product 2",
    "Feedback Score (num)": 1200,
    "Category": category_list,
    "Feedback Quality (num)": 100,
    "Positive Feedback (%)": 42.5,
    "Price (USD)": 250.00,
    "Price Condition (USD)": 200.00,
    "Trusted Seller?": ["Yes", "No"],
    "Shipping Cost (USD)": 50.00,
    "Cost/Price Ratio": 30.20,
    "Condition": condition_list,
    "Shipping Type": shipping_type_list,
    "Has Store URL?": ["Yes", "No"],
    "Listing Type": listing_type_list
}

# Load and encode the image (adjust the path and filename as needed)
image_path = "./price_smart.jpg"  # Ensure the image file is present
with open(image_path, "rb") as img_file:
    image_data = base64.b64encode(img_file.read()).decode()

# Add the logo and title at the top (centered)
st.markdown(f"""
    <div style='text-align: center; margin-top: 20px; margin-bottom: 20px;'>
        <img src='data:image/png;base64,{image_data}' alt='PriceSmart Logo' style='border-radius:50%; width:150px; height:150px; margin-bottom:20px;'>
        <h1 style='font-size: 2.5em; color: #1e90ff;'>PriceSmart</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Product Comparison</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

input_data1 = {}
input_data2 = {}

# A helper mapping from display labels to original feature names
feature_name_map = {
    "Feedback Score (num)": "Feedback Score",
    "Category": "Category_Encoded",
    "Feedback Quality (num)": "Feedback_Quality",
    "Positive Feedback (%)": "Positive Feedback %",
    "Price (USD)": "Price_in_USD",
    "Price Condition (USD)": "Price_Condition",
    "Shipping Cost (USD)": "Shipping Cost",
    "Cost/Price Ratio": "Cost_to_Price_Ratio",
    "Condition": "Condition_Encoded",
    "Shipping Type": "Shipping Type_Encoded",
    "Listing Type": "Listing Type_Encoded"
}

def process_input(fields, col, product_prefix="", keys_suffix=""):
    input_data = {}
    with col:
        st.subheader(product_prefix)
        input_data["Product Name"] = st.text_input("Product Name", value=product_prefix, key=f"Product_Name{keys_suffix}")
        for field, default in fields.items():
            if field == "Product Name":
                continue
            if isinstance(default, list):
                user_input = st.selectbox(field, options=default, key=f"{field}{keys_suffix}")
                # Map back to encoded values
                if field == "Category":
                    input_data["Category_Encoded"] = category_mapping[user_input]
                elif field == "Condition":
                    input_data["Condition_Encoded"] = Condition[user_input]
                elif field == "Shipping Type":
                    input_data["Shipping Type_Encoded"] = Shipping_Type[user_input]
                elif field == "Listing Type":
                    input_data["Listing Type_Encoded"] = Listing_Type[user_input]
                elif field == "Trusted Seller?":
                    input_data["Is_Trusted_Seller"] = 1 if user_input == "Yes" else 0
                elif field == "Has Store URL?":
                    input_data["Store URL_flag"] = 1 if user_input == "Yes" else 0
            else:
                # Numeric fields
                fname = feature_name_map.get(field, field)
                if field not in ["Trusted Seller?", "Has Store URL?", "Category", "Condition", "Shipping Type", "Listing Type"]:
                    input_data[fname] = st.number_input(field, value=default, key=f"{field}{keys_suffix}")
    return input_data

input_data1 = process_input(fields, col1, product_prefix="Product 1", keys_suffix="_1")
input_data2 = process_input(fields2, col2, product_prefix="Product 2", keys_suffix="_2")
col_left, col_center, col_right = st.columns([1,1,1])

with col_center:
    st.markdown("""
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
            background-color: #1e90ff;
            color: white;
                border-radius: 20px;
                padding: 10px 20px;
                border-color: transparent;
                font-size: 16px;
          
        }
        div.stButton > button:hover {
            background-color: #0077b6;
            color: white;
            border-color: transparent;
        }
        div.stButton > button:a:active {
            background-color: #0077b6;
            color: white;
            border-color: transparent;
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button("Compare Products"):
        if model:
            try:
                required_fields = [
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
                ]

                # Fill missing keys with 0 if not provided
                for rf in required_fields:
                    if rf not in input_data1:
                        input_data1[rf] = 0
                    if rf not in input_data2:
                        input_data2[rf] = 0

                # Print inputs before prediction
                # st.write("**Input data for Product 1 (raw):**", input_data1)
                df_input1 = pd.DataFrame([input_data1])
                df_input1_model = df_input1[required_fields]
                # st.write("**Model input for Product 1:**")
                # st.write(df_input1_model)

                dmatrix_input1 = xgb.DMatrix(df_input1_model)
                prediction1 = model.predict(dmatrix_input1)
                label1 = prediction_mapping[int(prediction1[0])]
                cat1_name = list(category_mapping.keys())[list(category_mapping.values()).index(df_input1['Category_Encoded'][0])]

                # st.write("**Input data for Product 2 (raw):**", input_data2)
                df_input2 = pd.DataFrame([input_data2])
                df_input2_model = df_input2[required_fields]
                # st.write("**Model input for Product 2:**")
                # st.write(df_input2_model)

                dmatrix_input2 = xgb.DMatrix(df_input2_model)
                prediction2 = model.predict(dmatrix_input2)
                label2 = prediction_mapping[int(prediction2[0])]
                cat2_name = list(category_mapping.keys())[list(category_mapping.values()).index(df_input2['Category_Encoded'][0])]

                # Determine winner
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

                # Add centered title for results
                st.markdown("<h2 style='text-align: center; color: #1e90ff; margin-top: 50px;'>Best Product with Shortest Time in Stock</h2>", unsafe_allow_html=True)

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
                        height: 400px;
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
                            <h3>{cat1_name}</h3>
                            <p><strong>Prediction:</strong> {label1}</p>
                        </div>
                        """, unsafe_allow_html=True)

                with vs_col:
                    st.markdown("<div class='vs'>vs</div>", unsafe_allow_html=True)

                with result_col2:
                    st.markdown(f"""
                        <div class="card">
                            <h2>{df_input2['Product Name'][0]} <span class="trophy">{trophy2}</span></h2>
                            <h3>{cat2_name}</h3>
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
    