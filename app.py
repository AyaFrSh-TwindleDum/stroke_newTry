import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# הגדרות עמוד ראשוניות (חייב להיות הדבר הראשון)
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="wide"
)

# הוספת CSS מותאם אישית לעיצוב הכללי
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_index=True)

# ======================
# Dataset (נשאר אותו דבר לוגית)
# ======================
@st.cache_data # מומלץ להוסיף כדי שהאתר לא יטען מחדש בכל לחיצה
def load_stroke_data():
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
    files = os.listdir(path)
    csv_path = os.path.join(path, files[0])
    df = pd.read_csv(csv_path)
    df = df[["stroke", "bmi", "avg_glucose_level", "age"]]
    df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
    df["avg_glucose_level"] = df["avg_glucose_level"].fillna(df["avg_glucose_level"].mean())
    df["age"] = df["age"].fillna(df["age"].mean())
    return df

df = load_stroke_data()

# ======================
# Model Training
# ======================
X = df[["bmi", "avg_glucose_level", "age"]]
y = df["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ======================
# Sidebar
# ======================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864344.png", width=100)
    st.title("Control Panel")
    page = st.radio(
        "Select Page:",
        ["📊 Dataset Overview", "📈 Model Performance", "🔮 Stroke Prediction"]
    )
    st.divider()
    st.info("This app uses Machine Learning to predict stroke probability based on health metrics.")

# ======================
# Page 1: Dataset
# ======================
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Analysis")
    st.write("Exploration of the stroke prediction dataset metrics.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Raw Data Summary")
        st.dataframe(df.head(100), use_container_width=True)
    
    with col2:
        st.subheader("Key Statistics")
        st.write(df.describe())

    st.divider()
    st.subheader("Distribution Visualization")
    st.scatter_chart(df, x="bmi", y="avg_glucose_level", color="stroke", use_container_width=True)

# ======================
# Page 2: Performance
# ======================
elif page == "📈 Model Performance":
    st.title("📈 Model Evaluation")
    
    # שימוש ב-Metric ו-Cards
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Model Type", value="KNN")
    col2.metric(label="Accuracy Score", value=f"{accuracy:.2%}")
    col3.metric(label="N-Neighbors", value="5")
    
    st.divider()
    st.subheader("Confusion Matrix Context")
    st.info("The model is trained to distinguish between 'Healthy' and 'At Risk' patients based on BMI, Glucose, and Age.")

# ======================
# Page 3: Prediction
# ======================
elif page == "🔮 Stroke Prediction":
    st.title("🔮 Predictive Diagnosis")
    st.write("Adjust the sliders below to check the prediction.")

    # חלוקה לעמודות עבור הסליידרים
    with st.container():
        c1, c2 = st.columns([1, 1])
        with c1:
            age = st.slider("Age", 0, 100, 45)
            bmi = st.slider("BMI Index", 10.0, 50.0, 24.0)
        with c2:
            avg_glucose_level = st.slider("Average Glucose Level", 50.0, 250.0, 100.0)
            st.write("---") # רווח קטן
            predict_btn = st.button("Analyze Results")

    if predict_btn:
        input_data = np.array([[bmi, avg_glucose_level, age]])
        prediction = knn.predict(input_data)[0]

        st.divider()
        
        if prediction == 0:
            st.success("### Results: Patient is likely Healthy")
            st.balloons()
        else:
            st.error("### Results: High Risk of Stroke Detected")
            st.warning("Please consult with a medical professional immediately.")

        # החזרת הגרף שביקשת (היה בהערה)
        st.subheader("Patient Position Relative to Data")
        new_point = pd.DataFrame({"bmi": [bmi], "avg_glucose_level": [avg_glucose_level], "stroke": [2]})
        plot_df = pd.concat([df, new_point], ignore_index=True)
        st.scatter_chart(plot_df, x="bmi", y="avg_glucose_level", color="stroke")
