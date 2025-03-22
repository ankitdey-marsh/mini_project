import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(page_title="Urban Heat Island Prediction & Mitigation", page_icon="🌆", layout="wide")
save_dir = r"./plots"
st.title("🌆 Urban Heat Island (UHI) Prediction & Mitigation App")

# Sidebar option
option = st.sidebar.radio("Choose an option:", ["Upload CSV for Predictions", "Fetch Previous Predictions"])

# Display dataset name if available
dataset_name_path = os.path.join(save_dir, "dataset_name.txt")
if os.path.exists(dataset_name_path):
    with open(dataset_name_path, "r") as f:
        dataset_name = f.read().strip()
    st.sidebar.markdown(f"**📂 Last Trained Dataset:** `{dataset_name}`")

if option == "Fetch Previous Predictions":
    st.write("### 🔍 Displaying Previously Saved Predictions")
    forecast_plot_path = os.path.join(save_dir, "forecast_plot.png")
    components_plot_path = os.path.join(save_dir, "prophet_components.png")
    feature_plot_path = os.path.join(save_dir, "feature_components.png")
    
    if os.path.exists(forecast_plot_path):
        st.markdown("<h5>📅 Future UHI Forecasting (Prophet)</h5>", unsafe_allow_html=True)
        st.image(forecast_plot_path, caption="Future UHI Forecast", use_container_width=True)
    else:
        st.warning("⚠️ Forecast plot not found!")
    
    if os.path.exists(components_plot_path):
        st.markdown("<h5>📅 Seasonal UHI Forecasting (Prophet)</h5>", unsafe_allow_html=True)
        st.image(components_plot_path, caption="Seasonal UHI Components", use_container_width=True)
    else:
        st.warning("⚠️ Prophet components plot not found!")
    
    if os.path.exists(feature_plot_path):
        st.markdown("<h5>🔥 Feature Importance (Random Forest)</h5>", unsafe_allow_html=True)
        st.image(feature_plot_path, caption="Feature Importance (RF Model)", use_container_width=True)
    else:
        st.warning("⚠️ Feature importance plot not found!")
    
    st.stop()

# File Upload Section
st.sidebar.header("Upload CSV for Predictions")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
save_dir = r"./plots"
os.makedirs(save_dir, exist_ok=True)  # ✅ Ensure the directory exists before using it
if uploaded_file is not None:
    dataset_name = uploaded_file.name  # Get the dataset filename
    with open(dataset_name_path, "w") as f:
        f.write(dataset_name)  # Save dataset name

    try:
        df = pd.read_csv(uploaded_file)
        
        if df.empty:
            st.error("⚠️ The uploaded CSV file is empty. Please upload a valid file.")
            st.stop()
    
    except pd.errors.EmptyDataError:
        st.error("⚠️ No data found in the uploaded file. Please upload a valid CSV.")
        st.stop()
    
    except pd.errors.ParserError:
        st.error("⚠️ Error parsing the CSV file. Check the format and delimiters.")
        st.stop()
    
    df.rename(columns={"start_date": "Date"}, inplace=True)
    original_df = df.copy()
    df.drop(columns=["region", "end_date", "population_mean", "population_std"], errors='ignore', inplace=True)
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # Load trained models and scalers
    with open("models_and_scalers.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    rf_model = model_data["random_forest"]
    xgb_model = model_data["xgboost"]
    prophet_model = model_data["prophet"]
    X_scaler = model_data["X_scaler"]
    y_scaler = model_data["y_scaler"]

    model_features = list(X_scaler.feature_names_in_)
    available_features = [col for col in model_features if col in df.columns]

    if not available_features:
        st.error("❌ Error: No matching features found in uploaded CSV!")
        st.stop()

    X = df[available_features]
    X_scaled = X_scaler.transform(X)
    y_true = df["uhi"] if "uhi" in df.columns else None

    # Predictions
    rf_pred = rf_model.predict(X_scaled)
    xgb_pred = xgb_model.predict(X_scaled)
    rf_pred = y_scaler.inverse_transform(rf_pred.reshape(-1, 1)).flatten()
    xgb_pred = y_scaler.inverse_transform(xgb_pred.reshape(-1, 1)).flatten()

    df["RF_Prediction"] = rf_pred
    df["XGB_Prediction"] = xgb_pred

    if y_true is not None:
        y_true_scaled = y_scaler.transform(y_true.values.reshape(-1, 1)).flatten()
        rf_rmse = np.sqrt(mean_squared_error(y_true_scaled, rf_pred))
        xgb_rmse = np.sqrt(mean_squared_error(y_true_scaled, xgb_pred))
        
        st.write("### 📊 Model Performance (RMSE)")
        st.write(f"✔️ **Random Forest RMSE:** {rf_rmse:.2f}")
        st.write(f"✔️ **XGBoost RMSE:** {xgb_rmse:.2f}")

    df = original_df

    # Prophet Forecasting
    if "Date" in df.columns and "uhi" in df.columns:
        st.write("### 📅 Future UHI Forecasting (Prophet)")
        prophet_data = df[["Date", "uhi"]].rename(columns={"Date": "ds", "uhi": "y"})
        prophet_data["ds"] = pd.to_datetime(prophet_data["ds"])

        prophet_model = Prophet()
        prophet_model.add_seasonality(name="weekly", period=7, fourier_order=3)
        prophet_model.fit(prophet_data)

        future_dates = prophet_model.make_future_dataframe(periods=365 * (2030 - 2024), freq='D')
        forecast = prophet_model.predict(future_dates)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prophet_data["ds"], prophet_data["y"], 'k.', label="Historical UHI", color='green')
        ax.plot(forecast["ds"], forecast["yhat"], label="Predicted UHI", color='red')
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color='gray', alpha=0.3)
        ax.set_title(f"Predicted Future UHI up to 2030\nDataset: {dataset_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Urban Heat Island (UHI)")
        ax.legend()
        st.pyplot(fig)
        fig.savefig(os.path.join(save_dir, "forecast_plot.png"), format="png", dpi=300)

        st.write("### 📅 Seasonal UHI Forecasting (Prophet)")
        fig = prophet_model.plot_components(forecast)
        axes = fig.get_axes()
        for ax in axes:
            ax.set_ylabel("UHI")
        for ax in axes:
            ax.set_ylabel("NO2_Total")
            ax.set_xlabel("Years")
            break
        fig.suptitle(f"Seasonal UHI Components\nDataset: {dataset_name}", fontsize=12)
        st.pyplot(fig)
        fig.savefig(os.path.join(save_dir, "prophet_components.png"), format="png", dpi=300)

        st.write("### 🔥 Feature Importance (Random Forest)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=rf_model.feature_importances_, y=available_features, ax=ax, color='skyblue')
        ax.set_title(f"Feature Importance (RF Model)\nDataset: {dataset_name}")
        st.pyplot(fig)
        fig.savefig(os.path.join(save_dir, "feature_components.png"), format="png", dpi=300)
        st.write("### 🌱 UHI Mitigation Recommendations")

    # Extract unique city names for autocomplete
        city_list = df["region"].unique().tolist() if "region" in df.columns else []

        # Autocomplete-like search for city selection
        selected_city = st.sidebar.selectbox("Select a city to view recommendations:", options=[""] + city_list)

        if "region" in original_df.columns:
            # Encode categorical 'region' feature
            label_encoder = LabelEncoder()
            df["region_encoded"] = label_encoder.fit_transform(original_df["region"])
            region_mapping = dict(zip(df["region_encoded"], original_df["region"]))

            # Handle missing values
            df.fillna(df.mean(numeric_only=True), inplace=True)

            # Scale numerical features
            numerical_cols = df.select_dtypes(include=["float64"]).columns
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

            # Create new feature interactions
            df["ndvi_evi_ratio"] = df["ndvi_mean"] / (df["evi_mean"] + 1e-6)
            df["population_density"] = df["population_mean"] / (df["region_encoded"] + 1e-6)

            import random

            def suggest_cooling_actions(data, region_mapping):
                actions = {}

                for index, row in data.iterrows():
                    region_name = region_mapping[int(row['region_encoded'])]

                    if region_name not in actions:
                        actions[region_name] = set()  # Use set to prevent duplicates

                    # Define possible actions
                    import random

                    green_space_suggestions = [
                        "🌳 Expand Urban Green Spaces: Increase trees in parks, streets, and residential areas. \n",
                        "🏡 Implement Green Roofs: Encourage rooftop gardens to reduce surface temperatures. \n",
                        "🌿 Promote Vertical Greenery: Install green walls on buildings for cooling and air purification.\n",
                        "🌾 Reintroduce Native Vegetation: Use drought-resistant native plants for sustainable shading.\n",
                        "🍀 Establish Community Gardens: Encourage local participation in greening efforts.\n"
                    ]

                    reflective_surface_suggestions = [
                        "🏗 Encourage Cool Roofs: Use reflective roofing materials to minimize heat absorption.\n",
                        "🚶 Use Reflective Pavements: Replace dark asphalt with light-colored or permeable materials.\n",
                        "🏢 Apply Solar-Reflective Coatings: Paint building exteriors with heat-reflective coatings.\n",
                        "🏠 Incorporate Shading Elements: Use awnings, pergolas, and tree cover for natural cooling.\n",
                        "🌞 Utilize Light-Colored Materials: Light-colored walls and pavements absorb less heat.\n"
                    ]

                    high_density_suggestions = [
                        "💧 Develop Water Features: Introduce lakes, fountains, or artificial ponds for natural cooling.\n",
                        "🚇 Enhance Public Transport: Improve transit options to reduce vehicle emissions and heat.\n",
                        "🏙 Improve Urban Wind Flow: Plan open spaces and corridors to allow natural air circulation.\n",
                        "🔄 Adopt Heat-Resistant Urban Planning: Use heat-resistant materials and layouts to reduce UHI effects.\n",
                        "🏢 Increase Shaded Public Spaces: Construct shaded areas in parks, bus stops, and open spaces.\n"
                    ]

                    # Select one random suggestion from each category
                    random_suggestions = {
                        "Green Space": random.choice(green_space_suggestions),
                        "Reflective Surfaces": random.choice(reflective_surface_suggestions),
                        "High Density Areas": random.choice(high_density_suggestions)
                    }

                    # Print the randomized suggestions
                    for category, suggestion in random_suggestions.items():
                        print(f"{category}: {suggestion}")


                    # Apply conditions and add **randomized** suggestions
                    if row['ndvi_mean'] < 0.2:
                        actions[region_name].update(random.sample(green_space_suggestions, min(2, len(green_space_suggestions))))

                    if row['albedo_mean'] > 0.5:
                        actions[region_name].update(random.sample(reflective_surface_suggestions, min(2, len(reflective_surface_suggestions))))

                    if row['population_density'] > 1:
                        actions[region_name].update(random.sample(high_density_suggestions, min(2, len(high_density_suggestions))))

                return {region: list(suggestions) for region, suggestions in actions.items()}



            # Generate recommendations
            cooling_actions = suggest_cooling_actions(df, region_mapping)

            if selected_city and selected_city in cooling_actions:
                st.write(f"### Recommendations for {selected_city}")
                st.write("\n".join(cooling_actions[selected_city]))
