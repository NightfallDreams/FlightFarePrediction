# -*- coding: utf-8 -*-
"""

@author: NightfallDreams
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load your trained model (replace with your model's actual path)
model = joblib.load('FlightPricePrediction/Model/flight_rf.pkl.gz')

# Title of the app
st.title("Flight Price Prediction App")
st.title("Enter the following details:")

# Create layout with two columns
col1, col2 = st.columns(2)

# Row 1: Total Stops
with col1:
    total_stops = st.slider('Total Stops', 0, 4, 0, 1, format="Total Stops: %d")

# Row 2: Journey Date and Month
with col1:
    journey_date = st.date_input('Journey Date', datetime.today())
    journey_day = journey_date.day
    journey_month = journey_date.month

# Row 3: Departure Time and Arrival Time
with col1:
    dep_time = st.time_input("Departure Time", datetime(2024, 12, 17, 12, 30))
    dep_hour = dep_time.hour
    dep_min = dep_time.minute

with col2:
    arrival_time = st.time_input("Arrival Time", datetime(2024, 12, 17, 14, 30))
    arrival_hour = arrival_time.hour
    arrival_min = arrival_time.minute

# Calculate Duration (based on dep_time and arrival_time)
dep_time = datetime.strptime(f"{dep_hour}:{dep_min}", "%H:%M")
arrival_time = datetime.strptime(f"{arrival_hour}:{arrival_min}", "%H:%M")

# If arrival time is before departure time, it means the flight passed midnight
if arrival_time < dep_time:
    arrival_time += pd.Timedelta(days=1)  # Add one day to arrival time to handle overnight flights

# Calculate the time difference (duration)
duration = arrival_time - dep_time

# Extract the duration in hours and minutes
duration_hours = duration.seconds // 3600
duration_minutes = (duration.seconds // 60) % 60

# Row 4: Number of Passengers and Seat Type
with col1:
    num_passengers = st.number_input('Number of Passengers', min_value=1, max_value=10, value=1, step=1)

with col2:
    seat_type = st.selectbox('Seat Type', ['Economy', 'Premium Economy', 'Business', 'First Class'])

# Initialize the one-hot encoded columns for airlines
airline_features = {
    'Airline_Air India': 0,
    'Airline_GoAir': 0,
    'Airline_IndiGo': 0,
    'Airline_Jet Airways': 0,
    'Airline_Multiple carriers': 0,
    'Airline_SpiceJet': 0,
    'Airline_Vistara': 0,
    'Airline_Trujet': 0,
}

# Set the selected airline's one-hot encoded feature to 1
airline = st.selectbox('Airline', ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Multiple carriers', 'SpiceJet', 'Vistara', 'Trujet'])

if airline == 'Air India':
    airline_features['Airline_Air India'] = 1
elif airline == 'GoAir':
    airline_features['Airline_GoAir'] = 1
elif airline == 'IndiGo':
    airline_features['Airline_IndiGo'] = 1
elif airline == 'Jet Airways':
    airline_features['Airline_Jet Airways'] = 1
elif airline == 'Multiple carriers':
    airline_features['Airline_Multiple carriers'] = 1
elif airline == 'SpiceJet':
    airline_features['Airline_SpiceJet'] = 1
elif airline == 'Vistara':
    airline_features['Airline_Vistara'] = 1
elif airline == 'Trujet':
    airline_features['Airline_Trujet'] = 1

# Initialize the one-hot encoded columns for source and destination
source = st.selectbox('Source City', ['Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
destination = st.selectbox('Destination City', ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi'])

source_features = {
    'Source_Chennai': 0,
    'Source_Delhi': 0,
    'Source_Kolkata': 0,
    'Source_Mumbai': 0,
}

destination_features = {
    'Destination_Cochin': 0,
    'Destination_Delhi': 0,
    'Destination_Hyderabad': 0,
    'Destination_Kolkata': 0,
    'Destination_New Delhi': 0,
}

# Set the selected source and destination's one-hot encoded features to 1
source_features[f'Source_{source}'] = 1
destination_features[f'Destination_{destination}'] = 1

# Button to trigger the prediction
if st.button('Predict Flight Price'):
    # Get feature names from the model
    feature_names = model.feature_names_in_

    # Prepare the input data as a DataFrame (include all features)
    input_data = pd.DataFrame({
        'Total_Stops': [total_stops],
        'Journey_day': [journey_day],
        'Journey_month': [journey_month],
        'Dep_hour': [dep_hour],
        'Dep_min': [dep_min],
        'Duration_hours': [duration_hours],
        'Duration_mins': [duration_minutes],
        'Arrival_hour': [arrival_hour],
        'Arrival_min': [arrival_min],
        **airline_features,
        **source_features,
        **destination_features,
    })

    # Ensure that the input data has the same columns as the model expects
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Make the prediction (ensure that the model is pre-trained to handle this input format)
    prediction = model.predict(input_data)

    # Display the output summary
    st.write(f"### Ticket Search Summary")
    st.write(f"*Airline*: {airline}")
    st.write(f"*Seat Type*: {seat_type}")
    st.write(f"*Journey Date*: {journey_date.strftime('%B %d, %Y')}")
    st.write(f"*Departure Time*: {dep_hour}:{dep_min:02d}")
    st.write(f"*Arrival Time*: {arrival_hour}:{arrival_min:02d}")
    st.write(f"*Duration*: {duration_hours} hours {duration_minutes} minutes")
    st.write(f"*Total Stops*: {total_stops}")
    st.write(f"*Source*: {source}")
    st.write(f"*Destination*: {destination}")
    
    # Highlight the predicted price
    st.markdown(f"<h2 style='color: #FF5733; text-align: center;'>${prediction[0]:,.2f}</h2>", unsafe_allow_html=True)

    # Trigger the balloon animation
    st.balloons()
