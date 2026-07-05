from pathlib import Path
import pickle
import joblib
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Wearable Stress Detection",
    page_icon="🧠",
    layout="wide"
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
APP_ARTIFACTS_DIR = PROJECT_ROOT / "app_artifacts"

MODEL_PATH = APP_ARTIFACTS_DIR / "stress_model.joblib"
FEATURE_COLUMNS_PATH = APP_ARTIFACTS_DIR / "feature_columns.json"
MODEL_METADATA_PATH = APP_ARTIFACTS_DIR / "model_metadata.json"
TEMPLATE_PATH = APP_ARTIFACTS_DIR / "feature_input_template.csv"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    # Load trained model only once
    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)
    return model


@st.cache_data
def load_feature_columns():
    # Load expected feature column names
    with open(FEATURE_COLUMNS_PATH, "r") as file:
        feature_columns = json.load(file)
    return feature_columns


@st.cache_data
def load_metadata():
    # Load model metadata for display
    with open(MODEL_METADATA_PATH, "r") as file:
        metadata = json.load(file)
    return metadata


def load_csv_if_exists(path):
    # Load a CSV file only if it exists
    if path.exists():
        return pd.read_csv(path)
    return None


def validate_uploaded_features(uploaded_df, required_columns):
    # Check whether uploaded CSV contains all required feature columns
    missing_columns = [
        column for column in required_columns
        if column not in uploaded_df.columns
    ]

    extra_columns = [
        column for column in uploaded_df.columns
        if column not in required_columns
    ]

    return missing_columns, extra_columns


def map_label_to_name(label):
    # Convert numeric model output to readable class name
    label_map = {
        1: "Neutral / Baseline",
        2: "Stress",
        3: "Amusement"
    }

    return label_map.get(int(label), "Unknown")


def show_metric_cards():
    # Display main project result metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Best Model", "Gradient Boosting")

    with col2:
        st.metric("LOSO Mean Accuracy", "0.687")

    with col3:
        st.metric("LOSO Mean Macro F1", "0.577")

    with col4:
        st.metric("Subjects", "15")


def plot_bar_chart(df, x_col, y_col, title, ylabel):
    # Create a simple Matplotlib bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Results Dashboard",
        "Batch Prediction",
        "Model Details",
        "About"
    ]
)


# ------------------------------------------------------------
# Load artifacts safely
# ------------------------------------------------------------
model_available = MODEL_PATH.exists()
features_available = FEATURE_COLUMNS_PATH.exists()
metadata_available = MODEL_METADATA_PATH.exists()

if metadata_available:
    metadata = load_metadata()
else:
    metadata = None


# ------------------------------------------------------------
# Page 1: Overview
# ------------------------------------------------------------
if page == "Overview":
    st.title("🧠 Wearable Stress Detection")
    st.subheader("Physiological Signal-Based Machine Learning and Deep Learning Project")

    st.write(
        """
        This application demonstrates a wearable stress detection pipeline using physiological
        signals from the WESAD dataset. The project classifies signal windows into:
        """
    )

    st.markdown(
        """
        - **Neutral / Baseline**
        - **Stress**
        - **Amusement**
        """
    )

    show_metric_cards()

    st.markdown("---")

    st.subheader("Project Pipeline")

    pipeline_steps = pd.DataFrame({
        "Step": [
            "1. Data Exploration",
            "2. Window Segmentation",
            "3. Feature Extraction",
            "4. Classical ML Models",
            "5. Deep Learning Models",
            "6. LOSO Evaluation",
            "7. Streamlit App"
        ],
        "Description": [
            "Loaded and visualized ECG, EDA, respiration, and temperature signals",
            "Created 30-second windows with 50% overlap",
            "Extracted 68 statistical and signal-change features",
            "Trained Logistic Regression, Random Forest, SVM, and Gradient Boosting",
            "Trained 1D CNN, CNN-GRU, and Feature-Based MLP",
            "Evaluated subject-independent generalization across 15 subjects",
            "Built interactive dashboard and batch prediction interface"
        ]
    })

    st.dataframe(pipeline_steps, use_container_width=True)

    st.markdown("---")

    st.subheader("Key Finding")

    st.success(
        """
        Gradient Boosting with handcrafted physiological features achieved the best
        subject-independent performance. This suggests that feature engineering is highly
        effective for small wearable physiological datasets.
        """
    )


# ------------------------------------------------------------
# Page 2: Results Dashboard
# ------------------------------------------------------------
elif page == "Results Dashboard":
    st.title("📊 Results Dashboard")

    show_metric_cards()

    st.markdown("---")

    st.subheader("All Model Comparison")

    all_model_path = RESULTS_DIR / "all_model_comparison.csv"
    all_model_df = load_csv_if_exists(all_model_path)

    if all_model_df is not None:
        st.dataframe(all_model_df, use_container_width=True)

        plot_bar_chart(
            all_model_df,
            x_col="model",
            y_col="macro_f1",
            title="Model Comparison by Macro F1-score",
            ylabel="Macro F1-score"
        )
    else:
        st.warning("all_model_comparison.csv not found in results folder.")

    st.markdown("---")

    st.subheader("Leave-One-Subject-Out Evaluation")

    loso_summary_path = RESULTS_DIR / "loso_model_summary.csv"
    loso_summary_df = load_csv_if_exists(loso_summary_path)

    if loso_summary_df is not None:
        st.dataframe(loso_summary_df, use_container_width=True)

        plot_bar_chart(
            loso_summary_df,
            x_col="model",
            y_col="mean_macro_f1",
            title="LOSO Model Comparison by Mean Macro F1-score",
            ylabel="Mean Macro F1-score"
        )
    else:
        st.warning("loso_model_summary.csv not found in results folder.")

    st.markdown("---")

    st.subheader("Result Figures")

    figure_paths = [
        RESULTS_DIR / "all_model_comparison_macro_f1.png",
        RESULTS_DIR / "loso_model_comparison_macro_f1.png",
        RESULTS_DIR / "loso_best_model_confusion_matrix.png",
    ]

    for figure_path in figure_paths:
        if figure_path.exists():
            st.image(str(figure_path), caption=figure_path.name, use_container_width=True)


# ------------------------------------------------------------
# Page 3: Batch Prediction
# ------------------------------------------------------------
elif page == "Batch Prediction":
    st.title("🔍 Batch Stress Prediction")

    st.write(
        """
        Upload a CSV file containing the extracted physiological features.
        The app will predict the stress class for each row.
        """
    )

    if not model_available or not features_available:
        st.error(
            """
            Model artifacts are missing. Please run:
            notebooks/09_export_final_model_for_streamlit.ipynb
            """
        )
    else:
        model = load_model()
        required_feature_columns = load_feature_columns()

        st.subheader("Expected Input Format")

        st.write(f"Required number of features: **{len(required_feature_columns)}**")

        with st.expander("Show required feature columns"):
            st.write(required_feature_columns)

        if TEMPLATE_PATH.exists():
            with open(TEMPLATE_PATH, "rb") as file:
                st.download_button(
                    label="Download Feature Input Template",
                    data=file,
                    file_name="feature_input_template.csv",
                    mime="text/csv"
                )

        uploaded_file = st.file_uploader(
            "Upload feature CSV file",
            type=["csv"]
        )

        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_df.head(), use_container_width=True)

            missing_columns, extra_columns = validate_uploaded_features(
                uploaded_df,
                required_feature_columns
            )

            if len(missing_columns) > 0:
                st.error("The uploaded file is missing required feature columns.")
                st.write(missing_columns)
            else:
                if len(extra_columns) > 0:
                    st.info(
                        "Extra columns were found. They will be ignored during prediction."
                    )

                # Keep columns in the exact order used during training
                X_input = uploaded_df[required_feature_columns].copy()

                # Fill missing values if any exist
                X_input = X_input.fillna(X_input.mean())

                # Predict classes
                predictions = model.predict(X_input)

                result_df = uploaded_df.copy()
                result_df["predicted_label"] = predictions
                result_df["predicted_class"] = [
                    map_label_to_name(label) for label in predictions
                ]

                # Add prediction probabilities if supported
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_input)

                    for index, class_label in enumerate(model.classes_):
                        class_name = map_label_to_name(class_label)
                        result_df[f"probability_{class_name}"] = probabilities[:, index]

                st.subheader("Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                st.subheader("Predicted Class Distribution")

                prediction_counts = result_df["predicted_class"].value_counts().reset_index()
                prediction_counts.columns = ["class", "count"]

                st.dataframe(prediction_counts, use_container_width=True)

                plot_bar_chart(
                    prediction_counts,
                    x_col="class",
                    y_col="count",
                    title="Predicted Class Distribution",
                    ylabel="Count"
                )

                csv_data = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_data,
                    file_name="stress_predictions.csv",
                    mime="text/csv"
                )


# ------------------------------------------------------------
# Page 4: Model Details
# ------------------------------------------------------------
elif page == "Model Details":
    st.title("🧪 Model Details")

    if metadata is not None:
        st.subheader("Model Metadata")

        st.json(metadata)

        st.markdown("---")

        st.subheader("Evaluation Summary")

        evaluation = metadata.get("evaluation", {})

        eval_df = pd.DataFrame([
            {
                "Metric": "LOSO Mean Accuracy",
                "Value": evaluation.get("loso_mean_accuracy")
            },
            {
                "Metric": "LOSO Std Accuracy",
                "Value": evaluation.get("loso_std_accuracy")
            },
            {
                "Metric": "LOSO Mean Macro F1",
                "Value": evaluation.get("loso_mean_macro_f1")
            },
            {
                "Metric": "LOSO Std Macro F1",
                "Value": evaluation.get("loso_std_macro_f1")
            }
        ])

        st.dataframe(eval_df, use_container_width=True)
    else:
        st.warning("Model metadata file not found.")

    st.markdown("---")

    st.subheader("Feature Columns")

    if features_available:
        feature_columns = load_feature_columns()
        feature_df = pd.DataFrame({
            "feature_index": range(1, len(feature_columns) + 1),
            "feature_name": feature_columns
        })

        st.dataframe(feature_df, use_container_width=True)
    else:
        st.warning("feature_columns.json not found.")


# ------------------------------------------------------------
# Page 5: About
# ------------------------------------------------------------
elif page == "About":
    st.title("ℹ️ About This App")

    st.write(
        """
        This Streamlit application was built as an interactive interface for a wearable
        stress detection machine learning project.
        """
    )

    st.subheader("Technologies Used")

    st.markdown(
        """
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - PyTorch
        - Matplotlib
        - Streamlit
        """
    )

    st.subheader("Author")

    st.write("**Lihini Karunarathne**")

    st.subheader("Project Summary")

    st.write(
        """
        The project compares classical machine learning and deep learning models for
        wearable stress detection using physiological signals. The best-performing model
        was Gradient Boosting under Leave-One-Subject-Out evaluation.
        """
    )