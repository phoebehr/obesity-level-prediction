import math
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

sns.set_theme(style="whitegrid")
plt.rc("font", size=10)


st.set_page_config(
    page_title="Obesity Prediction System", page_icon="🏥", layout="wide"
)


@st.cache_data
def load_data():
    for filepath in ["ObesityDataSet_raw_and_data_sinthetic.csv", "obesity.csv"]:
        try:
            df = pd.read_csv(filepath)
            if "BMI" not in df.columns:
                df["BMI"] = df["Weight"] / (df["Height"] ** 2)
            return df
        except Exception:
            pass
    return None


df_raw = load_data()

try:
    dt_model = joblib.load("decision_tree.pkl")
    knn_model = joblib.load("knn.pkl")
    svm_model = joblib.load("svm.pkl")
    pre_scaler = joblib.load("pre_scaler.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception:
    dt_model = knn_model = svm_model = pre_scaler = scaler = None

try:
    label_encoder = joblib.load("label_encoder.pkl")
except Exception:
    label_encoder = None


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page", ["Home", "EDA", "Prediction", "Compare Models", "Extra"]
)


if page == "Home":
    st.title("🏥 Obesity Prediction System")
    st.markdown("---")

    col1, col2 = st.columns([1, 3]) 
    with col1: 
        st.image(r"kpj.jpeg")
    with col2: 
        st.header("The Background")
        st.write(
        """
KPJ Healthcare Berhad is one of Malaysia’s leading private healthcare providers. As a prominent organisation in the healthcare sector, KPJ is interested in leveraging machine learning to enhance healthcare services, support early disease detection, and improve preventive care. 

"""
    )

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("The Problem")
        st.write("""Obesity is an increasingly prevalent health concern and is associated with a range of serious health complications. Early identification of obesity risk can support timely intervention and preventive care. However, identifying individuals at risk can be challenging due to the range of demographic, dietary, and lifestyle factors that contribute to obesity.""")
    with col2: 
        st.header("The Solution")
        st.write("""We present the Obesity Prediction App, a machine learning–based tool designed to assist healthcare professionals with the preliminary assessment of an individual’s obesity level.The application analyses demographic characteristics, eating habits, and lifestyle behaviours to predict an individual's obesity category. By providing an initial risk assessment, the system aims to support healthcare professionals in identifying individuals who may benefit from further evaluation and early preventive intervention.""")

    st.warning("This application is a student-developed prototype and is not intended to provide medical diagnoses or replace professional medical advice. Rather, the project demonstrates how machine learning can be applied to support healthcare organisations, such as KPJ Healthcare Berhad, in the early identification and management of obesity risk.")
    

    st.markdown("---")
    st.header("📊 Dataset at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", "2,111")
    with col2:
        st.metric("Features", "17")
    with col3:
        st.metric("Obesity Classes", "7")
    with col4:
        st.metric("Best Model Accuracy", "96.88%")
    st.caption("Source: UCI Machine Learning Repository — Estimation of Obesity Levels Based on Eating Habits and Physical Condition")

    st.markdown("---")
    st.header("Machine Learning Models")
    st.caption("👇 Click on any model below to expand and view details:")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander("🌳 **Decision Tree**", expanded=False):
            st.markdown(
                """
            **CV Accuracy:** 0.977798 (97.78%)  
            **Test Accuracy:** 0.968825 (96.88%)  
            
            **Overview:**  
            A non-parametric supervised learning algorithm that splits data based on feature decision rules.
            
            **Key Features:**
            - **No Scaling Needed:** Operates independently of feature scale.
            - **High Interpretability:** Tree rules are easy to visualize and explain.
            - **Top CV Performer:** Achieved the highest cross-validation score.
            """
            )

    with col2:
        with st.expander("🧠 **Support Vector Machine (SVM)**", expanded=False):
            st.markdown(
                """
            **CV Accuracy:** 0.964009 (96.40%)  
            **Test Accuracy:** 0.968825 (96.88%)  
            
            **Overview:**  
            Finds the optimal decision boundary (hyperplane) that maximizes the margin between different classes.
            
            **Key Features:**
            - **High Dimensionality:** Effective in high-dimensional feature spaces.
            - **Generalization:** Strong robustness against overfitting.
            - **Feature Scaling:** Requires standardized input data.
            """
            )

    with col3:
        with st.expander("📍 **K-Nearest Neighbors (KNN)**", expanded=False):
            st.markdown(
                """
            **CV Accuracy:** 0.866812 (86.68%)  
            **Test Accuracy:** 0.896882 (89.69%)  
            
            **Overview:**  
            An instance-based algorithm that assigns a label based on the majority class of its nearest neighbors.
            
            **Key Features:**
            - **Instance-Based:** Simple lazy-learning architecture.
            - **Scale-Sensitive:** Requires feature standardization (StandardScaler).
            - **Distance Metrics:** Sensitive to noise and density variations.
            """
            )

    st.markdown("---")
    st.success("Select **Prediction** or **EDA** from the sidebar to begin.")

# ======================================
# PREDICTION PAGE
# ======================================
elif page == "Prediction":
    st.title("🤖 Obesity Level Prediction")
    st.write(
        "Enter your details below and choose a trained machine learning "
        "model to predict the obesity classification."
    )
    st.markdown("---")

    st.subheader("👤 Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=10, max_value=100, value=25)

    with col2:
        height = st.number_input(
            "Height (metres)",
            min_value=1.00,
            max_value=2.50,
            value=1.70,
            step=0.01,
        )
        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=70.0,
            step=0.5,
        )

    with col3:
        family_history = st.selectbox(
            "Family History of Overweight", ["no", "yes"]
        )

    bmi = weight / (height**2)
    st.info(f"📊 Calculated BMI: **{bmi:.2f}**")
    st.markdown("---")

    st.subheader("🥗 Eating Habits")
    col1, col2, col3 = st.columns(3)

    with col1:
        favc = st.selectbox(
            "Frequently Eat High-Calorie Food (FAVC)", ["no", "yes"]
        )
        fcvc = st.select_slider(
            "Vegetable Consumption (FCVC)",
            options=[1.0, 2.0, 3.0],
            value=2.0,
        )

    with col2:
        ncp = st.select_slider(
            "Number of Main Meals (NCP)",
            options=[1.0, 2.0, 3.0, 4.0],
            value=3.0,
        )
        caec = st.selectbox(
            "Food Between Meals (CAEC)",
            ["no", "Sometimes", "Frequently", "Always"],
        )

    with col3:
        ch2o = st.select_slider(
            "Daily Water Intake (CH2O)", options=[1.0, 2.0, 3.0], value=2.0
        )
        calc = st.selectbox(
            "Alcohol Consumption (CALC)",
            ["no", "Sometimes", "Frequently", "Always"],
        )
        smoke = st.selectbox("Do You Smoke?", ["no", "yes"])

    st.markdown("---")
    st.subheader("🏃 Lifestyle Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        scc = st.selectbox("Monitor Calorie Consumption (SCC)", ["no", "yes"])
        faf = st.select_slider(
            "Physical Activity Frequency (FAF)",
            options=[0.0, 1.0, 2.0, 3.0],
            value=1.0,
        )

    with col2:
        tue = st.select_slider(
            "Technology Usage Time (TUE)", options=[0.0, 1.0, 2.0], value=1.0
        )

    with col3:
        mtrans = st.selectbox(
            "Main Transportation",
            [
                "Automobile",
                "Bike",
                "Motorbike",
                "Public_Transportation",
                "Walking",
            ],
        )

    st.markdown("---")
    st.subheader("🧠 Select Machine Learning Model")

    selected_model = st.selectbox(
        "Choose Model",
        [
            "Decision Tree",
            "Support Vector Machine (SVM)",
            "K-Nearest Neighbors (KNN)",
        ],
    )

    gender_map = {"Female": 0, "Male": 1}
    yes_no_map = {"no": 0, "yes": 1}
    # The notebook uses LabelEncoder, whose codes are alphabetical.
    caec_map = {"Always": 0, "Frequently": 1, "Sometimes": 2, "no": 3}
    calc_map = {"Always": 0, "Frequently": 1, "Sometimes": 2, "no": 3}
    mtrans_map = {
        "Automobile": 0,
        "Bike": 1,
        "Motorbike": 2,
        "Public_Transportation": 3,
        "Walking": 4,
    }

    input_data = pd.DataFrame(
        [
            [
                gender_map[gender],
                age,
                height,
                weight,
                yes_no_map[family_history],
                yes_no_map[favc],
                fcvc,
                ncp,
                caec_map[caec],
                yes_no_map[smoke],
                ch2o,
                yes_no_map[scc],
                faf,
                tue,
                calc_map[calc],
                mtrans_map[mtrans],
                bmi,
            ]
        ],
        columns=[
            "Gender",
            "Age",
            "Height",
            "Weight",
            "family_history_with_overweight",
            "FAVC",
            "FCVC",
            "NCP",
            "CAEC",
            "SMOKE",
            "CH2O",
            "SCC",
            "FAF",
            "TUE",
            "CALC",
            "MTRANS",
            "BMI",
        ],
    )

    if st.button("🔍 Predict Obesity Level", use_container_width=True):
        active_model = None
        if selected_model == "Decision Tree":
            active_model = dt_model
        elif selected_model == "K-Nearest Neighbors (KNN)":
            active_model = knn_model
        else:
            active_model = svm_model

        if active_model is None:
            st.error(
                f"⚠️ {selected_model} model file is missing or failed to load. Check your .pkl files."
            )
        else:
            if pre_scaler is None:
                st.error(
                    "⚠️ pre_scaler.pkl is missing. This artifact is required "
                    "to reproduce the notebook's first preprocessing stage."
                )
                st.stop()

            prepared_input = input_data.copy().astype(float)

            prepared_input.loc[:, pre_scaler.feature_names_in_] = pre_scaler.transform(
                prepared_input[pre_scaler.feature_names_in_]
            )

            if hasattr(active_model, "feature_names_in_"):
                aligned_input = prepared_input[active_model.feature_names_in_]
            elif scaler is not None and hasattr(scaler, "feature_names_in_"):
                aligned_input = prepared_input[scaler.feature_names_in_]
            else:
                aligned_input = prepared_input

            if selected_model == "Decision Tree":
                model_input = aligned_input
            else:
                if scaler is not None:
                    model_input = scaler.transform(aligned_input)
                else:
                    st.warning(
                        "⚠️ Scaler is missing. Prediction quality may be affected."
                    )
                    model_input = aligned_input

            prediction_raw = active_model.predict(model_input)[0]

            if isinstance(prediction_raw, (str, np.str_)):
                prediction_label = str(prediction_raw)
            elif label_encoder is not None:
                prediction_label = label_encoder.inverse_transform(
                    [prediction_raw]
                )[0]
            else:
                alphabetical_map = {
                    0: "Insufficient_Weight",
                    1: "Normal_Weight",
                    2: "Obesity_Type_I",
                    3: "Obesity_Type_II",
                    4: "Obesity_Type_III",
                    5: "Overweight_Level_I",
                    6: "Overweight_Level_II",
                }
                prediction_label = alphabetical_map.get(
                    int(prediction_raw), f"Unknown ({prediction_raw})"
                )

            formatted_label = str(prediction_label).replace("_", " ")

            st.markdown("---")
            st.subheader("📋 Prediction Result")
            st.success(f"Predicted Obesity Level: **{formatted_label}**")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Model", selected_model)
            with col2:
                st.metric("BMI", f"{bmi:.2f}")
            with col3:
                st.metric("Prediction", formatted_label)

# ======================================
# COMPARE MODELS PAGE
# ======================================
elif page == "Compare Models":
    st.title("📊 Comprehensive Machine Learning Model Evaluation")
    st.markdown("---")

    metrics_data = {
        "Model": ["Decision Tree", "SVM", "KNN"],
        "CV Accuracy": [0.977798, 0.964009, 0.866812],
        "Test Accuracy": [0.968825, 0.968825, 0.896882],
        "CV Accuracy (%)": [97.78, 96.40, 86.68],
        "Test Accuracy (%)": [96.88, 96.88, 89.69],
        "RMSE": [0.177, 0.177, 0.539],
        "Requires Scaling": ["No", "Yes", "Yes"],
        "Interpretability": ["High", "Low (Black Box)", "Medium"],
    }

    df_metrics = pd.DataFrame(metrics_data)

    st.subheader("🏆 Model Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Top CV Accuracy",
            "0.977798",
            "Decision Tree (97.78%)",
        )
    with col2:
        st.metric(
            "Top Test Accuracy",
            "0.968825",
            "Decision Tree & SVM (96.88%)",
        )
    with col3:
        st.metric(
            "Fastest Training",
            "KNN",
            "Lazy Learner",
        )
    with col4:
        st.metric(
            "Best Feature Scaling Sensitivity",
            "Decision Tree",
            "No Scaling Needed",
        )

    st.markdown("---")
    st.subheader("🏆 9.1 Selection of Best Model")
    st.success(
        """
**Decision Tree** is selected as the best-performing model for deployment, based on
its cross-validation accuracy — the most reliable indicator of how well a model
generalises to unseen data.

- **Highest CV accuracy** (97.78%) among the three models, versus SVM (96.40%) and KNN (86.68%).
- **Tied with SVM** on test accuracy (96.88%) and RMSE (0.177), but with the added
  advantage of full interpretability through feature importance.
- **No feature scaling required**, simplifying the deployment pipeline.

SVM remains a strong secondary candidate. KNN is not recommended for deployment,
given its consistently lower accuracy, higher RMSE, and its specific weakness in
identifying Normal_Weight cases (recall = 0.54).
        """
    )

    st.markdown("---")
    st.subheader("📋 Model Accuracy Table (Exact Match)")

    display_df = df_metrics[
        [
            "Model",
            "CV Accuracy",
            "Test Accuracy",
            "CV Accuracy (%)",
            "Test Accuracy (%)",
            "RMSE",
            "Requires Scaling",
            "Interpretability",
        ]
    ]
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Dynamic Metric Comparison Chart")

    chart_metric = st.selectbox(
        "Select Metric to Visualize:",
        ["CV Accuracy", "Test Accuracy", "CV Accuracy (%)", "Test Accuracy (%)", "RMSE"],
        index=0,
    )

    if chart_metric == "RMSE":
        st.caption("ℹ️ Lower RMSE indicates better performance (unlike the accuracy metrics above).")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df_metrics, x="Model", y=chart_metric, palette="viridis", ax=ax
    )
    ax.set_title(
        f"Model Comparison: {chart_metric}", fontsize=12, fontweight="bold"
    )

    for p in ax.patches:
        val = p.get_height()
        if chart_metric == "RMSE":
            format_str = f"{val:.3f}"
        else:
            format_str = f"{val:.6f}" if val <= 1.0 else f"{val:.2f}%"
        ax.annotate(
            format_str,
            (p.get_x() + p.get_width() / 2.0, val),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    y_max = df_metrics[chart_metric].max()
    ax.set_ylim(0, y_max * 1.15)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("🔍 Qualitative Algorithm Analysis")

    tab1, tab2, tab3 = st.tabs(
        [
            "🌳 Decision Tree",
            "🧠 Support Vector Machine (SVM)",
            "📍 K-Nearest Neighbors (KNN)",
        ]
    )

    with tab1:
        st.markdown(
            """
        **Decision Tree Profile:**
        - **CV Accuracy:** `0.977798` (97.78%) — *Highest cross-validation accuracy*
        - **Test Accuracy:** `0.968825` (96.88%)
        - **RMSE:** `0.177` — *Tied for lowest (best) RMSE*
        - **Preprocessing Requirements:** Direct raw inputs. No feature scaling necessary.
        - **Pros:** Highly interpretable decision boundaries, zero scaling requirement, fast prediction speed.
        """
        )

    with tab2:
        st.markdown(
            """
        **Support Vector Machine (SVM) Profile:**
        - **CV Accuracy:** `0.964009` (96.40%)
        - **Test Accuracy:** `0.968825` (96.88%) — *Tied for highest test accuracy*
        - **RMSE:** `0.177` — *Tied for lowest (best) RMSE*
        - **Preprocessing Requirements:** Strictly requires standard scaling (`StandardScaler`).
        - **Pros:** Strong robustness against overfitting, highly reliable test accuracy.
        """
        )

    with tab3:
        st.markdown(
            """
        **K-Nearest Neighbors (KNN) Profile:**
        - **CV Accuracy:** `0.866812` (86.68%)
        - **Test Accuracy:** `0.896882` (89.69%)
        - **RMSE:** `0.539` — *Highest (worst) RMSE, roughly 3x the other two models*
        - **Preprocessing Requirements:** Strictly depends on `StandardScaler` transformations.
        - **Pros:** Simple lazy learner algorithm with instant training phase.
        """
        )

    # -----------------------------
    # FULL CLASSIFICATION REPORT
    # -----------------------------
    st.markdown("---")
    st.subheader("📄 Full Classification Report")
    st.caption(
        "Precision, Recall, F1-score, and Support for each obesity class, "
        "across all three models."
    )

    class_names = [
        "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
        "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II",
        "Obesity_Type_III",
    ]

    report_data = {
        "KNN": {
            "precision": [0.85, 0.88, 0.91, 0.92, 1.00, 0.86, 0.84],
            "recall":    [0.96, 0.54, 0.96, 1.00, 1.00, 0.89, 0.90],
            "f1-score":  [0.90, 0.67, 0.93, 0.96, 1.00, 0.88, 0.87],
            "support":   [53, 56, 70, 60, 65, 55, 58],
            "accuracy": 0.90, "macro avg": [0.89, 0.89, 0.89], "weighted avg": [0.90, 0.90, 0.89],
        },
        "Decision Tree": {
            "precision": [1.00, 0.95, 1.00, 0.94, 1.00, 0.93, 0.97],
            "recall":    [0.96, 0.93, 0.94, 1.00, 1.00, 0.95, 1.00],
            "f1-score":  [0.98, 0.94, 0.97, 0.97, 1.00, 0.94, 0.98],
            "support":   [53, 56, 70, 60, 65, 55, 58],
            "accuracy": 0.97, "macro avg": [0.97, 0.97, 0.97], "weighted avg": [0.97, 0.97, 0.97],
        },
        "SVM": {
            "precision": [0.98, 0.95, 0.99, 0.98, 1.00, 0.91, 0.96],
            "recall":    [0.96, 0.95, 1.00, 0.98, 0.98, 0.95, 0.95],
            "f1-score":  [0.97, 0.95, 0.99, 0.98, 0.99, 0.93, 0.96],
            "support":   [53, 56, 70, 60, 65, 55, 58],
            "accuracy": 0.97, "macro avg": [0.97, 0.97, 0.97], "weighted avg": [0.97, 0.97, 0.97],
        },
    }

    report_tabs = st.tabs(["📍 KNN", "🌳 Decision Tree", "🧠 SVM"])

    for tab, model_name in zip(report_tabs, report_data.keys()):
        with tab:
            r = report_data[model_name]

            df_report = pd.DataFrame({
                "Class": class_names,
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1-Score": r["f1-score"],
                "Support": r["support"],
            })

            # Append accuracy / macro avg / weighted avg rows
            summary_rows = pd.DataFrame({
                "Class": ["accuracy", "macro avg", "weighted avg"],
                "Precision": ["", r["macro avg"][0], r["weighted avg"][0]],
                "Recall": ["", r["macro avg"][1], r["weighted avg"][1]],
                "F1-Score": [r["accuracy"], r["macro avg"][2], r["weighted avg"][2]],
                "Support": [417, 417, 417],
            })

            df_report_full = pd.concat([df_report, summary_rows], ignore_index=True)
            st.dataframe(df_report_full, use_container_width=True, hide_index=True)

            if model_name == "KNN":
                st.warning(
                    "⚠️ Normal_Weight recall is only 0.54 — nearly half of actual "
                    "Normal_Weight cases were misclassified into a neighbouring class."
                )

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.markdown("---")
    st.subheader("🌟 Feature Importance (Decision Tree)")
    st.caption(
        "Since Decision Tree was selected as the best model, its feature "
        "importance is shown to explain what drives its predictions. KNN "
        "(distance-based) and SVM (margin-based) do not produce a directly "
        "comparable native importance score."
    )

    importance_data = {
        "Feature": [
            "BMI", "Gender", "Weight", "MTRANS", "Age", "NCP", "SCC",
            "FAF", "Height", "FAVC", "family_history_with_overweight",
        ],
        "Importance": [
            0.793066, 0.168085, 0.016843, 0.006295, 0.004331, 0.003721,
            0.002658, 0.001640, 0.001418, 0.001098, 0.000845,
        ],
    }
    df_importance = pd.DataFrame(importance_data).sort_values(
        "Importance", ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df_importance, y="Feature", x="Importance", palette="Blues_r", ax=ax)
    ax.set_title("Decision Tree Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "BMI (79.3%) and Gender (16.8%) together account for over 96% of the "
        "model's predictive power. This matches the EDA finding (Chart 3) that "
        "Obesity_Type_I is 99.7% female and Overweight_Level_II is 99.3% male, "
        "which the model likely uses as a near-perfect split for those two classes."
    )

# ======================================
# EDA PAGE (20 MATCHING JUPYTER CHARTS)
# ======================================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write(
        "Explore the 20 EDA visualisations exactly as they appear in "
        "Assignment Data Science.ipynb."
    )
    st.markdown("---")

    chart_titles = [
        "Height vs Weight by Obesity Category",
        "Age Distribution across Obesity Levels",
        "Gender Proportion within Each Obesity Level",
        "Impact of Family History on BMI Density Distribution",
        "Vegetable Consumption Frequency across Obesity Levels",
        "Main Meals & High-Calorie Food on BMI",
        "Snack Consumption vs Obesity Levels",
        "Daily Water Consumption across Key Categories",
        "Alcohol Consumption Frequency vs Mean BMI",
        "Physical Activity and Tech Use by Gender",
        "Physical Activity and Family History",
        "Physical Activity by Transportation Method",
        "Calorie Monitoring & Smoking on BMI",
        "Transportation Disparities between Extreme Weight Groups",
        "Gender & Alcohol Consumption on BMI",
        "Obesity Tier Composition by Smoking Status",
        "Family History & High-Caloric Food on BMI",
        "Cumulative Risk Factor Score vs Average BMI",
        "BMI Distribution across Transportation Modes",
        "Lifestyle Profile: Normal Weight vs Obesity Type III",
    ]
    chart_groups = [
        ("📍 Group 1: Body Characteristics (1–4)", 0, 4),
        ("📍 Group 2: Dietary Habits (5–9)", 4, 9),
        ("📍 Group 3: Lifestyle & Physical Activity (10–14)", 9, 14),
        ("📍 Section 4: Advanced Risk Profiling (15–20)", 14, 20),
    ]
    tabs = st.tabs([group[0] for group in chart_groups])
    graph_directory = Path(__file__).with_name("eda_graphs")

    for tab, (_, start, end) in zip(tabs, chart_groups):
        with tab:
            for chart_index in range(start, end):
                st.subheader(f"{chart_index + 1}. {chart_titles[chart_index]}")
                graph_path = graph_directory / f"chart_{chart_index + 1:02d}.png"
                if graph_path.exists():
                    st.image(str(graph_path))
                else:
                    st.error(f"⚠️ Missing notebook chart: {graph_path.name}")
                if chart_index + 1 < end:
                    st.markdown("---")

# Previous dynamic notebook renderer retained as a non-navigation reference.
elif page == "__notebook_runtime_EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write(
        "These are the same 20 EDA visualisations, in the same order and with "
        "the same plotting code, as the Jupyter notebook."
    )
    st.markdown("---")

    if df_raw is None:
        st.error("⚠️ Dataset file not found.")
    else:
        notebook_path = Path(__file__).with_name("Assignment Data Science.ipynb")

        try:
            with notebook_path.open(encoding="utf-8") as notebook_file:
                notebook = json.load(notebook_file)
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"⚠️ Unable to load the EDA notebook: {error}")
        else:
            # These are the exact 20 executed EDA chart cells in the notebook.
            chart_cells = [
                (94, "Height vs Weight by Obesity Category"),
                (96, "Age Distribution across Obesity Levels"),
                (98, "Gender Proportion within Each Obesity Level"),
                (100, "Impact of Family History on BMI Density Distribution"),
                (103, "Vegetable Consumption Frequency across Obesity Levels"),
                (105, "Main Meals & High-Calorie Food on BMI"),
                (107, "Snack Consumption vs Obesity Levels"),
                (109, "Daily Water Consumption across Key Categories"),
                (111, "Alcohol Consumption Frequency vs Mean BMI"),
                (114, "Physical Activity and Tech Use by Gender"),
                (116, "Physical Activity and Family History"),
                (118, "Physical Activity by Transportation Method"),
                (120, "Calorie Monitoring & Smoking on BMI"),
                (122, "Transportation Disparities between Extreme Weight Groups"),
                (125, "Gender & Alcohol Consumption on BMI"),
                (127, "Obesity Tier Composition by Smoking Status"),
                (129, "Family History & High-Caloric Food on BMI"),
                (131, "Cumulative Risk Factor Score vs Average BMI"),
                (133, "BMI Distribution across Transportation Modes"),
                (135, "Lifestyle Profile: Normal Weight vs Obesity Type III"),
            ]

            cells = notebook.get("cells", [])
            if len(cells) <= chart_cells[-1][0]:
                st.error("⚠️ The notebook does not contain the expected 20 EDA cells.")
            else:
                obesity_plot = df_raw.copy()
                obesity_order = [
                    "Insufficient_Weight",
                    "Normal_Weight",
                    "Obesity_Type_I",
                    "Obesity_Type_II",
                    "Obesity_Type_III",
                    "Overweight_Level_I",
                    "Overweight_Level_II",
                ]

                # Match the plotting setup used immediately before the 20 charts.
                sns.set_theme(style="whitegrid")
                plt.rcParams["figure.dpi"] = 120
                plt.rcParams["font.sans-serif"] = "DejaVu Sans"

                groups = [
                    ("📍 Group 1: Body Characteristics (1–4)", 0, 4),
                    ("📍 Group 2: Dietary Habits (5–9)", 4, 9),
                    ("📍 Group 3: Lifestyle & Physical Activity (10–14)", 9, 14),
                    ("📍 Section 4: Advanced Risk Profiling (15–20)", 14, 20),
                ]
                tabs = st.tabs([group[0] for group in groups])

                # Give the notebook cells the same variables/imports they had when
                # originally executed. Only the 20 reviewed chart cells are run.
                chart_namespace = {
                    "np": np,
                    "pd": pd,
                    "plt": plt,
                    "sns": sns,
                    "MinMaxScaler": MinMaxScaler,
                    "obesity_plot": obesity_plot,
                    "obesity_order": obesity_order,
                }

                original_show = plt.show

                def show_in_streamlit(*_args, **_kwargs):
                    figure = plt.gcf()
                    st.pyplot(figure)
                    plt.close(figure)

                try:
                    plt.show = show_in_streamlit
                    for tab, (_, start, end) in zip(tabs, groups):
                        with tab:
                            for chart_number in range(start, end):
                                cell_index, chart_title = chart_cells[chart_number]
                                st.subheader(f"{chart_number + 1}. {chart_title}")
                                source = "".join(cells[cell_index].get("source", []))
                                exec(
                                    compile(source, f"{notebook_path.name}:cell-{cell_index}", "exec"),
                                    chart_namespace,
                                )
                                if chart_number + 1 < end:
                                    st.markdown("---")
                except Exception as error:
                    st.error(
                        f"⚠️ Unable to render chart {chart_number + 1} "
                        f"from the notebook: {error}"
                    )
                finally:
                    plt.show = original_show

# Retained temporarily as a reference for the former hand-recreated charts.
elif page == "__legacy_EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write(
        "Explore visualisations from our Exploratory Data Analysis."
    )
    st.markdown("---")

    if df_raw is None:
        st.error("⚠️ Dataset file not found.")
    else:
        df = df_raw.copy()

        t1, t2, t3, t4 = st.tabs(
            [
                "📍 Group 1: Body Characteristics (1–4)",
                "📍 Group 2: Dietary Habits (5–9)",
                "📍 Group 3: Lifestyle & Physical Activity (10–14)",
                "📍 Section 4: Advanced Risk Profiling (15–20)",
            ]
        )

        # -----------------------------
        # GROUP 1: CHARTS 1–4
        # -----------------------------
        with t1:
            st.subheader("1. Height vs Weight by Obesity Category (Scatter Plot)")
            fig, ax = plt.subplots(figsize=(9, 5))

            sns.scatterplot(
                data=df,
                x="Height",
                y="Weight",
                hue="NObeyesdad",
                palette="Spectral",
                alpha=0.8,
                ax=ax
            )

            ax.set_title("Height vs Weight by Obesity Category", fontweight="bold", fontsize=12)
            ax.set_xlabel("Height")
            ax.set_ylabel("Weight")

            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0., title=None)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 2
            st.subheader("2. Age Distribution across Obesity Levels (Boxplot + Strip Plot)")
            fig, ax = plt.subplots(figsize=(9.5, 5))

            obesity_order = [
                'Insufficient_Weight',
                'Normal_Weight',
                'Overweight_Level_I',
                'Overweight_Level_II',
                'Obesity_Type_I',
                'Obesity_Type_II',
                'Obesity_Type_III'
            ]

            sns.boxplot(
                data=df,
                x="NObeyesdad",
                y="Age",
                order=obesity_order,
                palette="crest",
                showfliers=False,
                ax=ax
            )

            sns.stripplot(
                data=df,
                x="NObeyesdad",
                y="Age",
                order=obesity_order,
                color="black",
                alpha=0.25,
                jitter=0.2,
                size=2.5,
                ax=ax
            )

            ax.set_title("Age Distribution across Obesity Levels", fontweight="bold", fontsize=12)
            ax.set_xlabel("Obesity Category")
            ax.set_ylabel("Age (Years)")

            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 3
            st.subheader("3. Gender Proportion within Each Obesity Level (100% Stacked Bar)")
            fig, ax = plt.subplots(figsize=(9.5, 5))

            gender_ct = pd.crosstab(df['NObeyesdad'], df['Gender'], normalize='index') * 100
            gender_ct = gender_ct.reindex(obesity_order)

            bars = gender_ct.plot(
                kind='bar',
                stacked=True,
                color=['#FF6B8B', '#4A90E2'],
                ax=ax,
                width=0.6,
                edgecolor='white'
            )

            for i, category in enumerate(obesity_order):
                female_val = gender_ct.loc[category, 'Female']
                male_val = gender_ct.loc[category, 'Male']

                if female_val > 5:
                    ax.text(i, female_val / 2, f"{female_val:.1f}%",
                            ha='center', va='center', color='white', fontweight='bold', fontsize=9)

                if male_val > 5:
                    ax.text(i, female_val + (male_val / 2), f"{male_val:.1f}%",
                            ha='center', va='center', color='white', fontweight='bold', fontsize=9)

            ax.set_title("Gender Proportion within Each Obesity Level (100% Stacked Bar)", fontweight="bold", fontsize=12)
            ax.set_xlabel("Obesity Category")
            ax.set_ylabel("Percentage (%)")
            ax.set_ylim(0, 100)

            plt.xticks(rotation=30, ha='right')
            ax.legend(title="Gender", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 4
            st.subheader("4. Impact of Family History of Obesity on BMI (Violin Plot)")
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.violinplot(
                data=df,
                x="family_history_with_overweight",
                y="BMI",
                hue="family_history_with_overweight",
                palette="Set2",
                legend=False,
                inner="quartile",
                ax=ax,
            )
            ax.set_title("Impact of Family History of Obesity on BMI")
            st.pyplot(fig)
            plt.close(fig)

        # -----------------------------
        # GROUP 2: CHARTS 5–9
        # -----------------------------
        with t2:
            # Chart 5
            st.subheader("5. Vegetable Consumption Frequency (FCVC) across Obesity Levels")
            fig, ax = plt.subplots(figsize=(9.5, 5))

            df_chart5 = df.copy()
            df_chart5['FCVC_round'] = df_chart5['FCVC'].round()

            sns.boxplot(
                data=df_chart5,
                x="NObeyesdad",
                y="FCVC_round",
                order=obesity_order,
                hue="NObeyesdad",
                palette="YlGn",
                legend=False,
                ax=ax
            )

            ax.set_title("5. Vegetable Consumption Frequency (FCVC) across Obesity Levels", fontweight="bold", fontsize=12)
            ax.set_xlabel("Obesity Category")
            ax.set_ylabel("Frequency of Vegetable Consumption (FCVC)")

            plt.xticks(rotation=25, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 6
            st.subheader("6. Interaction Between High-Calorie Food Consumption (FAVC) and Number of Main Meals (NCP) on BMI")
            df_chart6 = df.copy()
            df_chart6["NCP_group"] = df_chart6["NCP"].round()

            fig, ax = plt.subplots(figsize=(9, 5))
            sns.pointplot(
                data=df_chart6,
                x="NCP_group",
                y="BMI",
                hue="FAVC",
                palette={"no": "#2ecc71", "yes": "#e74c3c"},
                markers=["o", "s"],
                linestyles=["-", "--"],
                capsize=0.15,
                err_kws={"linewidth": 1.5},
                ax=ax,
            )

            ax.set_title("Interaction Effect: Main Meals (NCP) & High-Calorie Food (FAVC) on BMI", fontsize=11, fontweight="bold")
            ax.set_xlabel("Number of Main Meals Daily (NCP)")
            ax.set_ylabel("Mean BMI ($kg/m^2$)")
            ax.legend(title="Frequent High-Calorie Food (FAVC)", bbox_to_anchor=(1.02, 1), loc="upper left")

            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 7
            st.subheader("7. Crosstab Heatmap: Snack Consumption (CAEC) vs Obesity Levels")
            fig, ax = plt.subplots(figsize=(9.5, 5.5))

            caec_order = ['no', 'Sometimes', 'Frequently', 'Always']
            ct = pd.crosstab(df['NObeyesdad'], df['CAEC'], normalize='index') * 100
            ct = ct.reindex(index=obesity_order, columns=caec_order).fillna(0)

            sns.heatmap(
                ct,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': 'Proportion within Obesity Category (%)'},
                ax=ax
            )

            ax.set_title("Crosstab Heatmap: Snack Consumption (CAEC) vs Obesity Levels", fontweight="bold", fontsize=12)
            ax.set_xlabel("Consumption of Food Between Meals (CAEC)")
            ax.set_ylabel("Obesity Category")

            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 8
            st.subheader("8. Daily Water Consumption (CH2O) Density Distribution Across Key Categories")
            fig, ax = plt.subplots(figsize=(9.5, 5))

            target_categories = ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_II', 'Obesity_Type_III']
            df_filtered = df[df['NObeyesdad'].isin(target_categories)].copy()

            sns.kdeplot(
                data=df_filtered,
                x="CH2O",
                hue="NObeyesdad",
                hue_order=target_categories,
                palette="Blues_r",
                fill=True,
                common_norm=False,
                alpha=0.4,
                linewidth=1.5,
                ax=ax
            )

            ax.set_title("Daily Water Consumption (CH2O) Density Distribution Across Key Categories", fontweight="bold", fontsize=12)
            ax.set_xlabel("Daily Water Intake (CH2O: 1 = Low, 2 = Moderate, 3 = High)")
            ax.set_ylabel("Density")

            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title="Obesity Category")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 9
            st.subheader("9. Alcohol Consumption Frequency (CALC) vs Mean BMI (with 95% CI)")
            fig, ax = plt.subplots(figsize=(9, 5.5))

            calc_order = ['no', 'Sometimes', 'Frequently', 'Always']

            sns.barplot(
                data=df,
                x="CALC",
                y="BMI",
                order=calc_order,
                palette="YlOrBr",
                capsize=0.1,
                err_kws={'linewidth': 1.8, 'color': '#333333'},
                ax=ax
            )

            means = df.groupby('CALC')['BMI'].mean().reindex(calc_order)
            for i, mean_val in enumerate(means):
                if not np.isnan(mean_val):
                    ax.text(i, mean_val / 2, f"{mean_val:.1f}", ha='center', va='center', color='black', fontweight='bold', fontsize=10)

            ax.set_title("9. Alcohol Consumption Frequency (CALC) vs Mean BMI (with 95% CI)", fontweight="bold", fontsize=12)
            ax.set_xlabel("Alcohol Consumption Frequency (CALC)")
            ax.set_ylabel(r"Mean BMI ($kg/m^2$)")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # -----------------------------
        # GROUP 3: CHARTS 10–14
        # -----------------------------
        with t3:
            # Chart 10
            st.subheader("10. Physical Activity Frequency (FAF) vs. Time Spent Using Technology Devices (TUE)")
            df_chart10 = df.copy()
            df_chart10["TUE_group"] = df_chart10["TUE"].round()
            df_chart10["FAF_group"] = df_chart10["FAF"].round()

            g = sns.catplot(
                data=df_chart10[df_chart10["TUE_group"].isin([0.0, 1.0, 2.0])],
                x="FAF_group",
                y="BMI",
                hue="Gender",
                col="TUE_group",
                kind="box",
                palette={"Female": "#ea99c6", "Male": "#71b1ea"},
                height=4.5,
                aspect=0.85,
                showfliers=True,
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
            )

            g.fig.subplots_adjust(top=0.82)
            g.fig.suptitle("BMI Distribution across Physical Activity (FAF) and Tech Use (TUE) by Gender", fontsize=12, fontweight="bold")
            g.set_axis_labels("Physical Activity (FAF: Days/Week)", "BMI ($kg/m^2$)")
            g.set_titles("Tech Use (TUE) = {col_name}")

            st.pyplot(g.fig)
            plt.close(g.fig)
            st.markdown("---")

            # Chart 11
            st.subheader("11. Effect of Physical Activity Frequency (FAF) on BMI by Family History of Obesity")
            df_chart11 = df.copy()
            df_chart11["FAF_group"] = df_chart11["FAF"].round()

            g = sns.catplot(
                data=df_chart11,
                x="FAF_group",
                y="BMI",
                col="family_history_with_overweight",
                hue="FAF_group",
                kind="box",
                palette="YlOrBr",
                legend=False,
                height=4.2,
                aspect=1.1,
                showfliers=True,
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.5}
            )
            g.fig.subplots_adjust(top=0.82)
            g.fig.suptitle("Effect of Physical Activity Frequency (FAF) on BMI by Family History", fontsize=12, fontweight="bold")
            g.set_axis_labels("Physical Activity (FAF)", "BMI")

            st.pyplot(g.fig)
            plt.close(g.fig)

            # Chart 12: Horizontal Barplot for MTRANS vs FAF 
            st.subheader("12. Average Physical Activity Frequency (FAF) by Transportation Method (MTRANS)")
            fig, ax = plt.subplots(figsize=(9.5, 5))

            df_mtrans = (
                df.groupby('MTRANS')['FAF']
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            custom_colors = ['#A3C863', '#EB8A66', '#92A1C3', '#6CB49C', '#D891C0']

            sns.barplot(
                data=df_mtrans,
                x='FAF',
                y='MTRANS',
                palette=custom_colors,
                ax=ax
            )

            
            for i, row in df_mtrans.iterrows():
                val = row['FAF']
                ax.text(
                    val / 2, 
                    i, 
                    f"{val:.2f} days", 
                    ha='center', 
                    va='center', 
                    color='black', 
                    fontweight='bold', 
                    fontsize=10
                )

            ax.set_title("Average Physical Activity Frequency (FAF) by Transportation Method (MTRANS)", fontweight="bold", fontsize=11)
            ax.set_xlabel("Mean Physical Activity Frequency (FAF: Days/Week)")
            ax.set_ylabel("Transportation Method (MTRANS)")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 13: Grouped Barplot for SCC & SMOKE on BMI 
            st.subheader("13. Combined Impact of Calorie Monitoring (SCC) & Smoking (SMOKE) on BMI")
            fig, ax = plt.subplots(figsize=(9, 5.5))

            df_chart13 = df.copy()
            if 'BMI' not in df_chart13.columns:
                df_chart13['BMI'] = df_chart13['Weight'] / (df_chart13['Height'] ** 2)

            scc_map = {'no': 'No SCC', 'yes': 'Monitors Calories'}
            smoke_map = {'no': 'Non-Smoker', 'yes': 'Smoker'}

            df_chart13['SCC_label'] = df_chart13['SCC'].map(scc_map)
            df_chart13['SMOKE_label'] = df_chart13['SMOKE'].map(smoke_map)

            scc_order = ['No SCC', 'Monitors Calories']
            smoke_order = ['Non-Smoker', 'Smoker']

            custom_colors = ['#169688', '#B55D61']

            sns.barplot(
                data=df_chart13,
                x='SCC_label',
                y='BMI',
                hue='SMOKE_label',
                order=scc_order,
                hue_order=smoke_order,
                palette=custom_colors,
                errorbar=None,
                edgecolor='white',
                ax=ax
            )

            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height) and height > 0:
                    ax.annotate(
                        f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 4),
                        textcoords='offset points',
                        fontweight='bold',
                        fontsize=10
                    )

            ax.set_title("Combined Impact of Calorie Monitoring (SCC) & Smoking (SMOKE) on BMI", fontweight="bold", fontsize=11)
            ax.set_xlabel("Calorie Consumption Monitoring (SCC)")
            ax.set_ylabel("Mean BMI (kg/m²)")

            ax.set_ylim(0, 58)

            ax.legend(title="Smoking Status", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 14: Transportation Method Disparities between Extreme Weight Groups
            st.subheader("14. Transportation Method Disparities between Extreme Weight Groups")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            df_chart14 = df.copy()

            mtrans_order = ['Motorbike', 'Walking', 'Public_Transportation', 'Automobile', 'Bike']
            legend_labels = ['Motorbike', 'Walking', 'Public Transportation', 'Automobile', 'Bike']
            custom_colors = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854']

            df_normal = df_chart14[df_chart14['NObeyesdad'] == 'Normal_Weight']['MTRANS'].value_counts().reindex(mtrans_order).fillna(0)
            df_obesity3 = df_chart14[df_chart14['NObeyesdad'] == 'Obesity_Type_III']['MTRANS'].value_counts().reindex(mtrans_order).fillna(0)

            wedges1, texts1, autotexts1 = ax1.pie(
                df_normal,
                colors=custom_colors,
                autopct=lambda p: f"{p:.1f}%" if p > 0.5 else '',
                startangle=90,
                counterclock=False,
                pctdistance=0.75,
                wedgeprops=dict(width=0.38, edgecolor='white', linewidth=1.5)
            )
            ax1.set_title("Normal Weight", fontweight="bold", fontsize=11)

            wedges2, texts2, autotexts2 = ax2.pie(
                df_obesity3,
                colors=custom_colors,
                autopct=lambda p: f"{p:.1f}%" if p > 0.1 else '',
                startangle=90,
                counterclock=False,
                pctdistance=0.75,
                wedgeprops=dict(width=0.38, edgecolor='white', linewidth=1.5)
            )
            ax2.set_title("Obesity Type III", fontweight="bold", fontsize=11)

            for autotext in autotexts1 + autotexts2:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)

            fig.suptitle("Transportation Method Disparities between Extreme Weight Groups", fontweight="bold", fontsize=12, y=0.98)

            fig.legend(
                wedges1,
                legend_labels,
                title="Transportation Method",
                loc="lower center",
                ncol=5,
                bbox_to_anchor=(0.5, -0.05),
                frameon=True
            )

            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")
            
        # -----------------------------
        # SECTION 4: CHARTS 15–20
        # -----------------------------
        with t4:
            # Chart 15: Split Violin Plot for Gender & CALC on BMI 
            st.subheader("15. Dual Effect of Gender & Alcohol Consumption (CALC) on BMI")
            fig, ax = plt.subplots(figsize=(9.5, 5.5))

            df_chart15 = df.copy()
            if 'BMI' not in df_chart15.columns:
                df_chart15['BMI'] = df_chart15['Weight'] / (df_chart15['Height'] ** 2)

            calc_order = ['no', 'Sometimes', 'Frequently', 'Always']
            gender_order = ['Female', 'Male']
            custom_palette = {'Female': '#D77256', 'Male': '#38A398'}

            sns.violinplot(
                data=df_chart15,
                x='CALC',
                y='BMI',
                hue='Gender',
                order=calc_order,
                hue_order=gender_order,
                split=True,
                inner='quartile',
                palette=custom_palette,
                linewidth=1.2,
                ax=ax
            )

            ax.set_title("Dual Effect of Gender & Alcohol Consumption (CALC) on BMI", fontweight="bold", fontsize=12)
            ax.set_xlabel("Alcohol Consumption Frequency (CALC)")
            ax.set_ylabel("BMI (kg/m²)")

            ax.legend(title="Gender", loc="upper right", frameon=True)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 16: Grouped Barplot for Obesity Tier by SMOKE 
            st.subheader("16. Obesity Tier Composition by Smoking Status (SMOKE)")
            fig, ax = plt.subplots(figsize=(10, 5.5))

            df_chart16 = df.copy()

            smoke_map = {'no': 'Non-Smoker', 'yes': 'Smoker'}
            df_chart16['SMOKE_label'] = df_chart16['SMOKE'].map(smoke_map)

            smoke_order = ['Non-Smoker', 'Smoker']
            tier_order = [
                'Insufficient_Weight',
                'Normal_Weight',
                'Obesity_Type_I',
                'Obesity_Type_II',
                'Obesity_Type_III',
                'Overweight_Level_I',
                'Overweight_Level_II'
            ]

            df_prop = (
                pd.crosstab(df_chart16['SMOKE_label'], df_chart16['NObeyesdad'], normalize='index') * 100
            ).reindex(index=smoke_order, columns=tier_order).reset_index()

            df_long = df_prop.melt(id_vars='SMOKE_label', var_name='NObeyesdad', value_name='Percentage')

            set2_colors = ['#66C2A5', '#FC8D62', '#8DA0CB', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']

            sns.barplot(
                data=df_long,
                x='SMOKE_label',
                y='Percentage',
                hue='NObeyesdad',
                hue_order=tier_order,
                palette=set2_colors,
                edgecolor='white',
                linewidth=1,
                ax=ax
            )

            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height) and height > 0:
                    ax.annotate(
                        f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3),
                        textcoords='offset points',
                        fontweight='bold',
                        fontsize=7.5
                    )

            ax.set_title("Obesity Tier Composition by Smoking Status (SMOKE)", fontweight="bold", fontsize=12)
            ax.set_xlabel("Smoking Status (SMOKE)")
            ax.set_ylabel("Percentage (%)")

            ax.set_ylim(0, 36)

            ax.legend(title="Obesity Tier", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 17: Interaction Effect Plot for Family History & FAVC on BMI
            st.subheader("17. Interaction Effect: Family History & High-Caloric Food (FAVC) on BMI")
            fig, ax = plt.subplots(figsize=(9.5, 5.5))

            df_chart17 = df.copy()
            if 'BMI' not in df_chart17.columns:
                df_chart17['BMI'] = df_chart17['Weight'] / (df_chart17['Height'] ** 2)

            fam_order = ['yes', 'no']
            favc_order = ['no', 'yes']
            custom_palette = {'no': '#2EC4B6', 'yes': '#E06D53'}

            sns.pointplot(
                data=df_chart17,
                x='family_history_with_overweight',
                y='BMI',
                hue='FAVC',
                order=fam_order,
                hue_order=favc_order,
                palette=custom_palette,
                markers=['o', 's'],
                linestyles=['-', '--'],
                errorbar=None,
                scale=1.1,
                ax=ax
            )

            ax.set_title("Interaction Effect: Family History & High-Caloric Food (FAVC) on BMI", fontweight="bold", fontsize=12)
            ax.set_xlabel("Family History with Overweight")
            ax.set_ylabel("Mean BMI (kg/m²)")

            ax.legend(title="High Caloric Food (FAVC)", loc="upper left", frameon=True)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 18: Cumulative Risk Score vs. Mean BMI 
            st.subheader("18. Cumulative Risk Factor Score vs. Average BMI")
            fig, ax = plt.subplots(figsize=(9.5, 5.5))

            df_chart18 = df.copy()
            if 'BMI' not in df_chart18.columns:
                df_chart18['BMI'] = df_chart18['Weight'] / (df_chart18['Height'] ** 2)

            f1 = (df_chart18['family_history_with_overweight'] == 'yes').astype(int)
            f2 = (df_chart18['FAVC'] == 'yes').astype(int)
            f3 = (df_chart18['FAF'] < 1).astype(int)
            df_chart18['Risk_Score'] = f1 + f2 + f3

            risk_summary = df_chart18.groupby('Risk_Score')['BMI'].mean().reset_index()

            custom_colors = ['#FFF5F0', '#FCAE91', '#D64740', '#580F18']

            bars = ax.bar(
                risk_summary['Risk_Score'].astype(str),
                risk_summary['BMI'],
                color=custom_colors[:len(risk_summary)],
                edgecolor='black',
                linewidth=0.8,
                width=0.8
            )

            for i, bar in enumerate(bars):
                height = bar.get_height()
                text_color = 'white' if i >= 2 else 'black'
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height / 2.,
                    f"{height:.2f}",
                    ha='center', 
                    va='center',
                    color=text_color,
                    fontweight='bold',
                    fontsize=10.5
                )

            ax.set_title("Cumulative Risk Factor Score vs. Average BMI", fontweight="bold", fontsize=12)
            ax.set_xlabel("Number of High-Risk Factors (Family History + FAVC + Low FAF)")
            ax.set_ylabel(r"Mean BMI ($kg/m^2$)")

            ax.set_ylim(0, 35)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 19: Detailed BMI Distribution Across Transportation Modes 
            st.subheader("19. Detailed BMI Distribution Across Transportation Modes (MTRANS)")
            fig, ax = plt.subplots(figsize=(9.5, 5.5))

            df_chart19 = df.copy()
            if 'BMI' not in df_chart19.columns:
                df_chart19['BMI'] = df_chart19['Weight'] / (df_chart19['Height'] ** 2)

            mtrans_order = ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike']
            custom_colors = {
                'Public_Transportation': '#8CBFC1',
                'Walking': '#FCF39B',
                'Automobile': '#BFBBD5',
                'Motorbike': '#E9887C',
                'Bike': '#7A9FB8'
            }

            sns.boxenplot(
                data=df_chart19,
                x='MTRANS',
                y='BMI',
                order=mtrans_order,
                palette=custom_colors,
                ax=ax
            )

            ax.set_title("Detailed BMI Distribution Across Transportation Modes (MTRANS)", fontweight="bold", fontsize=12)
            ax.set_xlabel("Transportation Mode (MTRANS)")
            ax.set_ylabel(r"BMI ($kg/m^2$)")

            plt.xticks(rotation=0)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            # Chart 20: Radar Chart for Lifestyle Profile
            st.subheader("20. Lifestyle Behavioral Profile: Normal Weight vs. Obesity Type III")
 
            features = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
            feature_labels = [
                'Veggie Intake\n(FCVC)',
                'Main Meals\n(NCP)',
                'Water Intake\n(CH2O)',
                'Physical Activity\n(FAF)',
                'Screen Time\n(TUE)'
            ]

            df_chart20 = df.copy()
            for col in features:
                min_val = df_chart20[col].min()
                max_val = df_chart20[col].max()
                df_chart20[col + '_norm'] = (df_chart20[col] - min_val) / (max_val - min_val) if max_val > min_val else 0

            norm_cols = [c + '_norm' for c in features]

            mean_normal = df_chart20[df_chart20['NObeyesdad'] == 'Normal_Weight'][norm_cols].mean().values
            mean_obese3 = df_chart20[df_chart20['NObeyesdad'] == 'Obesity_Type_III'][norm_cols].mean().values

            num_vars = len(features)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            mean_normal = np.concatenate((mean_normal, [mean_normal[0]]))
            mean_obese3 = np.concatenate((mean_obese3, [mean_obese3[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels, fontsize=10, color='#333333')

            color_normal = '#17C3B2'
            color_obese3 = '#EE6C4D'

            ax.plot(angles, mean_normal, color=color_normal, linewidth=2, label='Normal Weight')
            ax.fill(angles, mean_normal, color=color_normal, alpha=0.18)

            ax.plot(angles, mean_obese3, color=color_obese3, linewidth=2, label='Obesity Type III')
            ax.fill(angles, mean_obese3, color=color_obese3, alpha=0.18)

            ax.set_rlabel_position(35) 
            ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], color='#555555', fontsize=9)
            ax.set_ylim(0, 0.68)

            ax.grid(True, color='#D3D3D3', linestyle='-', linewidth=0.8)
            ax.spines['polar'].set_color('#D3D3D3')

            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.08), fontsize=10.5, frameon=True)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

            
elif page == "Extra":
    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.subheader("📊 Dataset")
    st.write(
        """
This project uses the **Estimation of Obesity Levels Based on Eating Habits
and Physical Condition** dataset from the UCI Machine Learning Repository,
containing 2,111 records and 17 attributes collected from individuals in
Mexico, Peru, and Colombia.

**Citation:**  
Palechor, F. M., & de la Hoz Manotas, A. (2019). Estimation of Obesity Levels
Based on Eating Habits and Physical Condition [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5H31Z
        """
    )

    st.subheader("🧠 Machine Learning Models")
    st.write(
        """
Three models were trained and compared:
- **Decision Tree** — 97.78% CV accuracy, 96.88% test accuracy *(selected as best model)*
- **Support Vector Machine (SVM)** — 96.40% CV accuracy, 96.88% test accuracy
- **K-Nearest Neighbors (KNN)** — 86.68% CV accuracy, 89.69% test accuracy

Decision Tree was selected for deployment due to its highest cross-validation
accuracy and strong interpretability through feature importance analysis.
        """
    )

    st.subheader("⚠️ Disclaimer")
    st.warning(
        """
This tool is developed for **academic and educational purposes only**.
Predictions are based on a machine learning model trained on a public dataset
and should **not** be used as a substitute for professional medical advice,
diagnosis, or treatment. Please consult a qualified healthcare provider for
any health-related concerns.
        """
    )

    st.subheader("🛠️ Built With")
    st.write("Python · Streamlit · scikit-learn · Pandas · Seaborn · Matplotlib")
