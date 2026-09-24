# ============================================================
# Paste this as the LAST cell of SVM.ipynb, add the one-line call
# at the bottom (see the comment), and run it.
# It replaces the old `pickle.dump(model, ...)` cell.
# ============================================================
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder


def save_artifacts(best_model, scale, ord_enc, X,
                    csv_path='bank.csv', out_path='artifacts.pkl'):
    """Save the fitted model + every preprocessing object the Streamlit app needs.

    best_model : grid.best_estimator_  (the tuned & FITTED SVC, not the empty SVC())
    scale      : the fitted MinMaxScaler
    ord_enc    : the fitted OrdinalEncoder used for 'month'
    X          : the feature DataFrame used for training (for column order)
    """
    # Re-read the raw data to rebuild the pieces the app needs
    raw = pd.read_csv(csv_path).dropna()

    num_cols = list(scale.feature_names_in_)   # numeric columns the scaler was fitted on
    cat_cols = [c for c in raw.select_dtypes(include='object').columns
                if c not in ('month', 'deposit')]

    # 1) IQR clipping bounds (same rule as the notebook; 'day' was not clipped)
    clip_bounds = {}
    for col in num_cols:
        if col == 'day':
            continue
        q1, q3 = raw[col].quantile(0.25), raw[col].quantile(0.75)
        iqr = q3 - q1
        clip_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # Sanity check: rebuilt bounds must match what the scaler saw during training
    for col, (lo, hi) in clip_bounds.items():
        i = num_cols.index(col)
        assert np.isclose(scale.data_max_[i], min(raw[col].max(), hi)), f'clip mismatch on {col}'
        assert np.isclose(scale.data_min_[i], max(raw[col].min(), lo)), f'clip mismatch on {col}'

    # 2) One LabelEncoder per categorical column (+ the target)
    #    (the notebook loop overwrote `le` each iteration, so we rebuild them here)
    label_encoders = {c: LabelEncoder().fit(raw[c]) for c in cat_cols + ['deposit']}

    # 3) Ranges / defaults used to build the input widgets in the app
    num_meta = {
        c: {
            'min': float(raw[c].min()),
            'max': float(raw[c].max()),
            'median': float(raw[c].median()),
            'is_int': bool((raw[c] % 1 == 0).all()),
        }
        for c in num_cols
    }

    artifacts = {
        'model': best_model,
        'scaler': scale,
        'month_encoder': ord_enc,
        'label_encoders': label_encoders,
        'clip_bounds': clip_bounds,
        'num_cols': num_cols,
        'num_meta': num_meta,
        'feature_order': list(X.columns),
        'sklearn_version': sklearn.__version__,
    }
    joblib.dump(artifacts, out_path)

    print(f'Saved {out_path}')
    print('Put this in requirements.txt ->  scikit-learn==' + sklearn.__version__)


# In the notebook, add this line right below the function and run the cell
# (these 4 names are the variables created by the earlier notebook cells):
#
# save_artifacts(best_model, scale, ord_enc, X)
#
# On Google Colab, then download the file to commit it to GitHub:
# from google.colab import files; files.download('artifacts.pkl')