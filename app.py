
# Streamlit Code:
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ======================
# Dataset
# ======================
import kagglehub
import pandas as pd
import os


def load_stroke_data():
    # הורדה
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")

    print("Path to dataset files:", path)

    # קבצים בתיקייה
    files = os.listdir(path)
    print("Files:", files)

    # נתיב לקובץ CSV
    csv_path = os.path.join(path, files[0])

    # קריאה לדאטה
    df = pd.read_csv(csv_path)

    # בחירת עמודות
    df = df[["stroke", "bmi", "avg_glucose_level", "age"]]

    # מילוי ערכים חסרים עם ממוצע
    df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
    df["avg_glucose_level"] = df["avg_glucose_level"].fillna(df["avg_glucose_level"].mean())
    df["age"] = df["age"].fillna(df["age"].mean())

    return df


# זימון הפונקציה
df = load_stroke_data()

# ==============================================================================================


# ======================
# Model
# ======================
from sklearn.model_selection import train_test_split

X = df[["bmi", "avg_glucose_level","age"]]
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# ==============================================================================================

# ======================
# Sidebar
# ======================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    ["Dataset", "Model Performance", "Make Prediction"]
)

# ======================
# Page 1: Dataset
# ======================
# if page == "Dataset":
#     st.title("Dataset")

#     st.subheader("Data")
#     st.dataframe(df)

#     st.subheader("Scatter Plot")

#     st.scatter_chart(
#         df,
#         x="blood_sugar",
#         y="blood_pressure",
#         color="diabetes"
#     )

# # ======================
# # Page 2: Performance
# # ======================
# elif page == "Model Performance":
#     st.title("Model Performance")

#     st.metric("Accuracy", f"{accuracy:.2f}")

# # ======================
# # Page 3: Prediction
# # ======================
# elif page == "Make Prediction":
#     st.title("Predict Diabetes")

#     blood_sugar = st.slider("Blood Sugar", 70, 220, 120)
#     blood_pressure = st.slider("Blood Pressure", 60, 130, 80)

#     if st.button("Predict"):
#         input_data = np.array([[blood_sugar, blood_pressure]])
#         prediction = knn.predict(input_data)[0]

#         if prediction == 0:
#             label = "Healthy"
#         else:
#             label = "Diabetes"

#         st.subheader(f"Prediction: {label}")

#         # ======================
#         # Add point to plot
#         # ======================
#         new_point = pd.DataFrame({
#             "blood_sugar": [blood_sugar],
#             "blood_pressure": [blood_pressure],
#             "diabetes": [2]  # new category
#         })

#         plot_df = pd.concat([df, new_point], ignore_index=True)

#         st.subheader("Visualization")

#         st.scatter_chart(
#             plot_df,
#             x="blood_sugar",
#             y="blood_pressure",
#             color="diabetes"
#         )
