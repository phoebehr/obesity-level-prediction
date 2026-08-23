import math
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

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
    "Choose a page", ["Home", "Prediction", "Compare Models", "EDA", "About"]
)


if page == "Home":
    st.title("🏥 Obesity Prediction System")
    st.markdown("---")

    st.header("Project Overview")
    st.write(
        """
This system predicts a person's obesity level using Machine Learning, developed
to support healthcare organisations such as **KPJ Healthcare Berhad** in the
early identification of obesity risk through data-driven, preventive care.

The application allows users to:
- Predict obesity level based on physical habits and lifestyle attributes.
- Explore 20 Seaborn/Matplotlib visualizations matching exact Jupyter specifications.
- Compare three machine learning models.
- View model performance metrics and algorithm summaries.
"""
    )

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

            # -------------------------------------------------------------
            # 【修复位置】：强制转换类型为 float，允许接收 Scalar 的浮点数
            # -------------------------------------------------------------
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
    st.title("📊 Exploratory Data Analysis (Exact Jupyter Specifications)")
    st.write(
        "Explore interactive visualizations matching exact layout, titles, and plot types from the Notebook pipeline."
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

        # -----------------------------
        # SECTION 4: CHARTS 15–20
        # -----------------------------
        with t4:
            st.subheader("15–20. Advanced Risk Profiling")
            st.info("Additional advanced risk profile visualizations based on lifestyle and medical background.")

elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.subheader("📌 Project Background")
    st.write(
        """
This project was developed to demonstrate how machine learning can support
healthcare organisations, such as **KPJ Healthcare Berhad**, in the early
identification of obesity risk. By analysing demographic characteristics,
eating habits, and lifestyle behaviours, the system predicts an individual's
obesity level and can assist healthcare professionals in providing more
timely preventive care.
        """
    )

    st.subheader("🎯 Objective")
    st.write(
        """
To develop and compare machine learning classification models (KNN, Decision
Tree, and SVM) that predict obesity levels, and to deploy the best-performing
model (Decision Tree, 96.88% test accuracy) as an interactive prediction tool.
        """
    )

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