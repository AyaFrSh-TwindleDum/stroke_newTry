import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# הגדרות עמוד (חייב להיות בתחילת הקוד)
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="wide"
)

# תיקון ה-CSS עם הפרמטר הנכון: unsafe_allow_html=True
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
    </style>
    """, unsafe_allow_html=True)

# ======================
# Dataset
# ======================
@st.cache_data
def load_stroke_data():
    # הורדה מ-Kaggle
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
    files = os.listdir(path)
    csv_path = os.path.join(path, files[0])
    
    df = pd.read_csv(csv_path)
    
    # בחירת עמודות רלוונטיות
    df = df[["stroke", "bmi", "avg_glucose_level", "age"]]
    
    # מילוי ערכים חסרים
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ======================
# Sidebar Navigation
# ======================
with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Choose a section",
        ["📊 Dataset", "📈 Model Performance", "🔮 Make Prediction"]
    )
    st.divider()
    st.info("AI-powered system for early stroke detection.")

# ======================
# Page 1: Dataset
# ======================
if page == "📊 Dataset":
    st.title("📊 Dataset Overview")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Data Table")
        st.dataframe(df, use_container_width=True)
    with col2:
        st.subheader("Statistics")
        st.write(df.describe())

    st.divider()
    st.subheader("Visualizing Correlation")
    st.scatter_chart(
        df,
        x="bmi",
        y="avg_glucose_level",
        color="stroke",
        use_container_width=True
    )

# ======================
# Page 2: Performance
# ======================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy Score", f"{accuracy:.2%}")
    col2.metric("Algorithm", "KNN")
    
    st.success(f"The model is performing with {accuracy:.2f} accuracy based on the test set.")

# ======================
# Page 3: Prediction
# ======================
elif page == "🔮 Make Prediction":
    st.title("🔮 Predict Stroke Risk")
    
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            bmi = st.slider("BMI", 10, 50, 25)
            age = st.slider("Age", 0, 110, 30)
        with c2:
            avg_glucose_level = st.slider("Average Glucose Level", 60, 220, 100)
            st.write("---")
            predict_btn = st.button("Run Prediction")

    if predict_btn:
        input_data = np.array([[bmi, avg_glucose_level, age]])
        prediction = knn.predict(input_data)[0]

        st.divider()
        if prediction == 0:
            st.balloons()
            st.success("### Prediction: Healthy (Low Risk)")
        else:
            st.error("### Prediction: High Risk of Stroke")
            st.warning("Please consult a doctor for a professional medical evaluation.")

        # ויזואליזציה של הנקודה החדשה
        st.subheader("Visualization")
        new_point = pd.DataFrame({
            "bmi": [bmi],
            "avg_glucose_level": [avg_glucose_level],
            "stroke": [2]  # צבע שונה לחיזוי החדש
        })
        plot_df = pd.concat([df, new_point], ignore_index=True)
        st.scatter_chart(plot_df, x="bmi", y="avg_glucose_level", color="stroke")
