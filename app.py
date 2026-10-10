import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------- Page setup ----------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

DATA_FILE = "diabetes_prediction_dataset.csv"

# Exact column order used during training (after pd.get_dummies, drop_first=True)
FEATURE_COLUMNS = [
    "age", "hypertension", "heart_disease", "bmi", "HbA1c_level",
    "blood_glucose_level", "gender_Male", "gender_Other",
    "smoking_history_current", "smoking_history_ever",
    "smoking_history_former", "smoking_history_never",
    "smoking_history_not current",
]


# ---------- Load model & scaler (cached so they load only once) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()
MODEL_NAME = type(model).__name__
NICE_NAMES = {
    "RandomForestClassifier": "Random Forest",
    "LogisticRegression": "Logistic Regression",
    "KNeighborsClassifier": "KNN",
    "DecisionTreeClassifier": "Decision Tree",
    "SVC": "SVM",
}
MODEL_NICE = NICE_NAMES.get(MODEL_NAME, MODEL_NAME)


# ---------- Helpers ----------
def build_input(gender, age, hypertension, heart_disease,
                smoking_history, bmi, hba1c, glucose):
    """Turn the form values into the same 13 columns the model was trained on."""
    row = dict.fromkeys(FEATURE_COLUMNS, 0)
    row["age"] = age
    row["hypertension"] = int(hypertension == "Yes")
    row["heart_disease"] = int(heart_disease == "Yes")
    row["bmi"] = bmi
    row["HbA1c_level"] = hba1c
    row["blood_glucose_level"] = glucose

    # 'Female' and 'No Info' were dropped (drop_first=True) -> stay all zeros
    if gender == "Male":
        row["gender_Male"] = 1
    elif gender == "Other":
        row["gender_Other"] = 1
    if smoking_history != "No Info":
        row[f"smoking_history_{smoking_history}"] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def get_scores(clf, X):
    """Probability (or decision score) of the positive class, used for ROC/AUC."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


@st.cache_data(show_spinner="Preparing test data...")
def prepare_data(df):
    """Same split + encoding as the notebook (test_size=0.2, random_state=42)."""
    X = df.drop(columns=["diabetes"])
    y = df["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cat_cols = ["gender", "smoking_history"]
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    X_train = X_train.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    X_test = X_test.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Use the SAME scaler you saved from the notebook
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train.values, y_test.values


@st.cache_resource(show_spinner="Training all 5 models (SVM can take a few minutes)...")
def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(),
    }
    rows, curves = [], {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        scores = get_scores(clf, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        curves[name] = (fpr, tpr)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted"),
            "Recall": recall_score(y_test, pred, average="weighted"),
            "F1 Score": f1_score(y_test, pred, average="weighted"),
            "AUC": auc(fpr, tpr),
        })
    return pd.DataFrame(rows), curves


def load_dataset():
    """Look for the CSV next to app.py; otherwise ask the user to upload it."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    uploaded = st.file_uploader(
        f"Upload `{DATA_FILE}` (or put it in the app folder)", type="csv", key="dataset_uploader"
    )
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return None


# ---------- UI ----------
st.title("🩺 Diabetes Prediction App")
st.success(f"🏆 Best model in use: **{MODEL_NICE}**")

# Load the dataset ONCE and share it with both tabs below
df = load_dataset()

tab_predict, tab_perf, tab_compare = st.tabs(
    ["🔮 Predict", "📈 ROC & AUC (Best Model)", "🏆 Compare All Models"]
)

# ======================= TAB 1: PREDICT =======================
with tab_predict:
    st.write("Enter the patient details below and click **Predict**.")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age", min_value=1.0, max_value=120.0, value=40.0, step=1.0)
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)

    with col2:
        smoking_history = st.selectbox(
            "Smoking History",
            ["No Info", "never", "former", "current", "not current", "ever"],
        )
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
        glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=120, step=1)

    if st.button("Predict", type="primary", width="stretch"):
        input_df = build_input(gender, age, hypertension, heart_disease,
                               smoking_history, bmi, hba1c, glucose)
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]

        st.divider()
        if prediction == 1:
            st.error("⚠️ The model predicts: **Diabetic**")
        else:
            st.success("✅ The model predicts: **Not Diabetic**")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled)[0][1]
            st.metric("Probability of Diabetes", f"{prob * 100:.2f}%")
            st.progress(float(prob))

        with st.expander("See the data sent to the model"):
            st.dataframe(input_df)

# ======================= TAB 2: ROC & AUC =======================
with tab_perf:
    st.subheader(f"ROC Curve & AUC - {MODEL_NICE}")
    st.write(
        "ROC and AUC need the original dataset to build the test set "
        "(same 80/20 split as in your notebook)."
    )

    if df is None:
        st.info("Waiting for the dataset... (use the upload box at the top)")
    else:
        X_train_s, X_test_s, y_train, y_test = prepare_data(df)

        y_pred = model.predict(X_test_s)
        y_score = get_scores(model, X_test_s)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUC", f"{roc_auc:.4f}")
        m2.metric("Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
        m3.metric("Precision", f"{precision_score(y_test, y_pred) * 100:.1f}%")
        m4.metric("Recall", f"{recall_score(y_test, y_pred) * 100:.1f}%")
        m5.metric("F1", f"{f1_score(y_test, y_pred) * 100:.1f}%")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, linewidth=2, label=f"{MODEL_NICE} (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual: No Diabetes", "Actual: Diabetes"],
                columns=["Predicted: No Diabetes", "Predicted: Diabetes"],
            )
        )

# ======================= TAB 3: COMPARE ALL MODELS =======================
with tab_compare:
    st.subheader("Compare all 5 algorithms")
    st.write(
        "This retrains Logistic Regression, KNN, Decision Tree, Random Forest and SVM "
        "exactly like your notebook, then shows ROC curves, AUC and the best model."
    )
    if df is None:
        st.info("Waiting for the dataset... (use the upload box at the top)")
    else:
        if st.button("Run comparison", type="primary"):
            st.session_state["run_compare"] = True

        if st.session_state.get("run_compare"):
            X_train_s, X_test_s, y_train, y_test = prepare_data(df)
            results, curves = train_all_models(X_train_s, y_train, X_test_s, y_test)

            best_acc = results.loc[results["Accuracy"].idxmax()]
            best_auc = results.loc[results["AUC"].idxmax()]
            c1, c2 = st.columns(2)
            c1.success(f"🏆 Best by Accuracy: **{best_acc['Model']}** ({best_acc['Accuracy'] * 100:.2f}%)")
            c2.success(f"🏆 Best by AUC: **{best_auc['Model']}** ({best_auc['AUC']:.4f})")

            st.dataframe(
                results.set_index("Model")
                .style.format("{:.4f}")
                .highlight_max(axis=0, color="#1f6f3f"),
                width="stretch",
            )

            fig2, ax2 = plt.subplots(figsize=(7, 5.5))
            for name, (f, t) in curves.items():
                a = results.loc[results["Model"] == name, "AUC"].iloc[0]
                ax2.plot(f, t, linewidth=2, label=f"{name} (AUC = {a:.4f})")
            ax2.plot([0, 1], [0, 1], "k--", label="Random guess")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curves - All Models")
            ax2.legend(loc="lower right")
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots(figsize=(7, 4))
            ax3.bar(results["Model"], results["AUC"] * 100, color="#4c78a8")
            ax3.set_ylabel("AUC (%)")
            ax3.set_ylim(50, 100)
            ax3.set_title("AUC Comparison")
            plt.setp(ax3.get_xticklabels(), rotation=25)
            st.pyplot(fig3)

st.caption("⚠️ For educational purposes only. Not a substitute for medical advice.")