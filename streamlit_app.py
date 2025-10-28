import streamlit as st
import pandas as pd
import requests

st.title("Breast Cancer Prediction using ML Models")

with st.expander("📘 Introduction", expanded=True):
    st.markdown("""
This app demonstrates how classical machine-learning models can assist with **breast tumor classification** (**benign** vs **malignant**) using a small set of numeric features derived from digitized cell images.

### How to use
1) **Enter data**  
   • Type values manually **or** upload a **CSV** and review/edit the populated fields.  
2) **Choose a model** (Random Forest / KNN / SVC)  
3) **Run Prediction**  
   • The app sends the 8 numeric inputs and the chosen model to a secure API and returns the predicted class.  

### Interpreting results
- **Prediction**: `Benign` or `Malignant`  
- **Always** interpret results with domain expertise; this tool is for **learning and demonstration**, not diagnosis.
    """)
    st.info("This app is for educational purposes only and must **not** be used for medical decision-making.")

with st.expander("ℹ️ About the App & Methodology"):
    st.markdown("""
### Data & Features
We use features derived from the **Breast Cancer Wisconsin (Diagnostic) dataset (WDBC)**.  
From the full set, this demo focuses on **8 informative means**:

- `radius_mean` – average distance from centroid to perimeter
                
- `compactness_mean` – (perimeter² / area)  
- `concavity_mean` – severity of concave portions of the contour  
- `concave_points_mean` – number of concave portions 
- `texture_mean` – standard deviation of gray-scale values  
- `smoothness_mean` – local variation in radius lengths  

- `symmetry_mean` – symmetry of the mass  
- `fractal_dimension_mean` – “coastline approximation” of the contour


### Models available
- **Random Forest (RF)** — ensemble of decision trees; robust on tabular data and handles non-linear interactions well.  
- **Decision Trees** — Using branches and nodes, simulates a tree to make predictions.  
- **Support Vector Classifier (SVC)** — finds a maximum-margin boundary; strong in high-dimensional spaces.



### Privacy & data flow
Only the **8 numeric inputs** and the **selected model key** are sent to the Azure endpoint to compute a prediction.  
No files are stored by the app; uploaded CSVs are used **in-session** to populate fields.

""")
    

# Config
EXPECTED_FIELDS = [
    "radius_mean", "texture_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"
]

FIELD_LIMITS = {
    "radius_mean": (0.0, 120.0, 0.001),
    "texture_mean": (0.0, 120.0, 0.001),
    "smoothness_mean": (0.0, 1.0, 0.0001),
    "compactness_mean": (0.0, 1.0, 0.0001),
    "concavity_mean": (0.0, 1.0, 0.0001),
    "concave_points_mean": (0.0, 1.0, 0.0001),
    "symmetry_mean": (0.0, 1.0, 0.0001),
    "fractal_dimension_mean": (0.0, 1.0, 0.0001),
}


# Section 1: Data Entry
st.header("Step 1: Enter Patient Data")

tab1, tab2 = st.tabs(["📋 Manual Entry", "📂 CSV Upload"])

manual_features = None
csv_features = None

with tab1:
    manual_values = {}
    cols = st.columns(2)
    for i, field in enumerate(EXPECTED_FIELDS):
        mn, mx, step = FIELD_LIMITS[field]
        with cols[i % 2]:
            manual_values[field] = st.number_input(
                field.replace("_", " ").title(),
                min_value=float(mn),
                max_value=float(mx),
                step=float(step),
                value=0.0,
                key=f"manual_{field}"
            )
    manual_features = [manual_values[f] for f in EXPECTED_FIELDS]

with tab2:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        row_index = st.number_input("Row index", 0, len(df)-1, 0)
        selected_row = df.iloc[row_index]

        # Pre-fill values from CSV row
        defaults = [float(selected_row.get(f, 0.0)) for f in EXPECTED_FIELDS]

        st.markdown("**Populate fields (editable):**")
        cols = st.columns(2)
        csv_values = {}
        for i, field in enumerate(EXPECTED_FIELDS):
            mn, mx, step = FIELD_LIMITS[field]
            with cols[i % 2]:
                csv_values[field] = st.number_input(
                    field.replace("_", " ").title(),
                    min_value=float(mn),
                    max_value=float(mx),
                    step=float(step),
                    value=defaults[i],
                    key=f"csv_{field}"
                )
        csv_features = [csv_values[f] for f in EXPECTED_FIELDS]

# Decide which input to use (CSV takes precedence if provided)
input_features = csv_features if csv_features is not None else manual_features

# Section 2: Model Selection
st.header("Step 2: Choose Model")
model_choice = st.radio("Model:", ["Random Forest", "Decision Tree", "SVC"], horizontal=True)

# Display Model Info (your existing function)
from model_info import main
main(model_choice)

# Map UI label -> backend key used by Azure Function
MODEL_KEY = {
    "Random Forest": "rf",
    "Decision Tree":           "dt",
    "SVC":           "svc",
}[model_choice]

# -----------------------------
# Section 3: Predict
# -----------------------------
st.header("Step 3: Predict")
FUNC_URL = "https://ml-models-bc.azurewebsites.net/api/predict"  # single endpoint

if st.button("Run Prediction"):
    if input_features is None:
        st.error("Please enter data manually or upload a CSV first.")
    else:
        payload = {"model": MODEL_KEY, "features": input_features}
    

        try:
            r = requests.post(FUNC_URL, json=payload, timeout=20)
            if r.status_code == 200:
                result = r.json()
                st.success(f"Prediction: {result.get('prediction')}")
                # show probs if your API returns them
            else:
                # show server error body
                try:
                    st.error(r.json())
                except Exception:
                    st.error(r.text)
        except Exception as e:
            st.error(f"Request failed: {e}")
