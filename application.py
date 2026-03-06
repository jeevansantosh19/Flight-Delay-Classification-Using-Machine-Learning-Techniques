import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Flight Delay Prediction")

# LOAD MODELS (FAST - CACHED)
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/label_encoder.pkl")
    lr = joblib.load("models/lr.pkl")
    knn = joblib.load("models/knn.pkl")
    dtc = joblib.load("models/dtc.pkl")
    rfc = joblib.load("models/rfc.pkl")
    accuracies = joblib.load("models/model_accuracy.pkl")
    return scaler, encoders, lr, knn, dtc, rfc, accuracies

scaler, encoders, lr, knn, dtc, rfc, accuracies = load_models()

# PAGE STATE
if "page" not in st.session_state:
    st.session_state.page = "input"

# INPUT PAGE
if st.session_state.page == "input":

    st.title("✈ Flight Delay Prediction")

    Year = st.selectbox(
        "Year",
        encoders["Year"].classes_,
        index=None,
        placeholder="Select Year"
    )

    Quarter = st.selectbox(
        "Quarter",
        encoders["Quarter"].classes_,
        index=None,
        placeholder="Select Quarter"
    )

    Month = st.selectbox(
        "Month",
        encoders["Month"].classes_,
        index=None,
        placeholder="Select Month"
    )

    DayofMonth = st.selectbox(
        "Day of Month",
        encoders["DayofMonth"].classes_,
        index=None,
        placeholder="Select Day"
    )

    DayOfWeek = st.selectbox(
        "Day of Week",
        encoders["DayOfWeek"].classes_,
        index=None,
        placeholder="Select Day of Week"
    )

    Reporting_Airline = st.selectbox(
        "Reporting Airline",
        encoders["Reporting_Airline"].classes_,
        index=None,
        placeholder="Select Airline"
    )

    Origin = st.selectbox(
        "Origin Airport",
        encoders["Origin"].classes_,
        index=None,
        placeholder="Select Origin Airport"
    )

    Dest = st.selectbox(
        "Destination Airport",
        encoders["Dest"].classes_,
        index=None,
        placeholder="Select Destination Airport"
    )

    DepPeriod = st.selectbox(
        "Departure Period",
        encoders["DepPeriod"].classes_,
        index=None,
        placeholder="Select Departure Period"
    )

    Distance = st.number_input("Distance", min_value=0.0)
    DepHour = st.number_input("Departure Hour", min_value=0, max_value=23)

    if st.button("Predict"):

        # VALIDATION
        if None in [Year, Quarter, Month, DayofMonth, DayOfWeek,
                    Reporting_Airline, Origin, Dest, DepPeriod]:
            st.warning("⚠ Please select all fields before prediction.")
            st.stop()

        input_data = np.array([[ 
            encoders["Year"].transform([Year])[0],
            encoders["Quarter"].transform([Quarter])[0],
            encoders["Month"].transform([Month])[0],
            encoders["DayofMonth"].transform([DayofMonth])[0],
            encoders["DayOfWeek"].transform([DayOfWeek])[0],
            encoders["Reporting_Airline"].transform([Reporting_Airline])[0],
            encoders["Origin"].transform([Origin])[0],
            encoders["Dest"].transform([Dest])[0],
            Distance,
            encoders["DepPeriod"].transform([DepPeriod])[0],
            DepHour
        ]])

        # Scale numeric columns
        input_data[:, [8, 10]] = scaler.transform(
            input_data[:, [8, 10]]
        )

        # Predictions
        st.session_state.pred_lr = lr.predict(input_data)[0]
        st.session_state.pred_knn = knn.predict(input_data)[0]
        st.session_state.pred_dtc = dtc.predict(input_data)[0]
        st.session_state.pred_rfc = rfc.predict(input_data)[0]

        st.session_state.page = "result"
        st.rerun()

# RESULT PAGE
elif st.session_state.page == "result":

    st.title("📊 Prediction Results")

    result_df = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "Prediction": [
            "Delayed" if st.session_state.pred_lr == 1 else "Not Delayed",
            "Delayed" if st.session_state.pred_knn == 1 else "Not Delayed",
            "Delayed" if st.session_state.pred_dtc == 1 else "Not Delayed",
            "Delayed" if st.session_state.pred_rfc == 1 else "Not Delayed"
        ]
    })

    st.subheader("Model Predictions")
    st.table(result_df)

    accuracy_df = pd.DataFrame(
        list(accuracies.items()),
        columns=["Model", "Accuracy"]
    )

    st.subheader("Model Accuracies")
    st.table(accuracy_df)

    if st.button("Go Back"):
        st.session_state.page = "input"
        st.rerun()