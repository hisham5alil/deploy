from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="Term Deposit Prediction", page_icon="🏦", layout="wide")

ARTIFACTS_PATH = Path(__file__).parent / "artifacts.pkl"
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"]

# Layout of the form: section title -> features (any feature not listed goes to "Other")
GROUPS = {
    "👤 Client profile": ["age", "job", "marital", "education", "default"],
    "💰 Financial status": ["balance", "housing", "loan"],
    "📞 Campaign details": ["contact", "day", "month", "duration",
                        "campaign", "pdays", "previous", "poutcome"],
}


@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACTS_PATH)


def category_options(feature, art):
    """Allowed values for a categorical feature, taken from the fitted encoders."""
    if feature == "month":
        options = list(art["month_encoder"].categories_[0])
        if all(m in MONTH_ORDER for m in options):
            options = sorted(options, key=MONTH_ORDER.index)
        return options
    return list(art["label_encoders"][feature].classes_)


def render_input(feature, art):
    """Draw the right widget for one feature and return the user's value."""
    label = feature.replace("_", " ").capitalize()

    if feature in art["num_meta"]:
        meta = art["num_meta"][feature]
        if meta["is_int"]:
            return st.number_input(label, min_value=int(meta["min"]), max_value=int(meta["max"]),
                                value=int(meta["median"]), step=1, key=feature)
        return st.number_input(label, min_value=meta["min"], max_value=meta["max"],
                            value=meta["median"], key=feature)

    return st.selectbox(label, category_options(feature, art), key=feature)


def preprocess(inputs, art):
    """Apply exactly the same steps as the notebook: clip -> scale -> encode."""
    row = pd.DataFrame([inputs])
    num_cols = art["num_cols"]
    row[num_cols] = row[num_cols].astype(float)

    # 1) IQR clipping (values outside the training range are capped, like in training)
    for col, (lo, hi) in art["clip_bounds"].items():
        row[col] = row[col].clip(lo, hi)

    # 2) MinMax scaling of the numeric columns
    row[num_cols] = art["scaler"].transform(row[num_cols])

    # 3) Encoding (categoricals via LabelEncoder, month via OrdinalEncoder)
    for col, enc in art["label_encoders"].items():
        if col in row.columns:
            row[col] = enc.transform(row[col])
    row["month"] = art["month_encoder"].transform(row[["month"]])

    return row[art["feature_order"]]


# ------------------------------------------------------------------ UI
st.title("🏦 Bank Term Deposit Prediction")
st.write("Fill in the client's information, then press **Predict** to see whether "
        "the client is likely to subscribe to a term deposit (SVM model).")

if not ARTIFACTS_PATH.exists():
    st.error("`artifacts.pkl` not found. Run the last cell of the notebook and "
            "put the generated file next to `app.py`.")
    st.stop()

art = load_artifacts()

if art.get("sklearn_version") != sklearn.__version__:
    st.warning(f"The model was trained with scikit-learn {art.get('sklearn_version')} but this "
            f"app runs {sklearn.__version__}. Pin the same version in requirements.txt.")

features = art["feature_order"]
grouped = {title: [f for f in feats if f in features] for title, feats in GROUPS.items()}
listed = {f for feats in grouped.values() for f in feats}
grouped["Other"] = [f for f in features if f not in listed]

with st.form("prediction_form"):
    inputs = {}
    for title, feats in grouped.items():
        if not feats:
            continue
        st.subheader(title)
        cols = st.columns(3)
        for i, feat in enumerate(feats):
            with cols[i % 3]:
                inputs[feat] = render_input(feat, art)
    submitted = st.form_submit_button("Predict", type="primary", width="stretch")

if submitted:
    X_row = preprocess(inputs, art)
    model = art["model"]
    pred = model.predict(X_row)[0]
    label = art["label_encoders"]["deposit"].inverse_transform([pred])[0]
    score = model.decision_function(X_row)[0]

    st.divider()
    if label == "yes":
        st.success("✅ Prediction: the client is **likely to subscribe** (deposit = yes)")
    else:
        st.error("❌ Prediction: the client is **unlikely to subscribe** (deposit = no)")

    st.metric("SVM decision score", f"{score:.3f}",
            help="Positive score → 'yes', negative → 'no'. The further from 0, "
                "the more confident the model is.")

    with st.expander("Show the preprocessed input sent to the model"):
        st.dataframe(X_row, width="stretch")