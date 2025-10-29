# app.py
"""
Predictive Delivery Delay Optimizer
Single-file Streamlit app.
Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io
from pathlib import Path

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

st.set_page_config(page_title="Delivery Delay Optimizer", layout="wide")

st.title("📦 NexGen — Predictive Delivery Delay Optimizer")
st.markdown(
    "Load the provided CSV files, train a model to predict delayed deliveries, "
    "and get recommended corrective actions."
)

# --- Helper: expected filenames and quick-check ---
EXPECTED_FILES = {
    "orders": "orders.csv",
    "delivery": "delivery_performance.csv",
    "routes": "routes_distance.csv",
    "fleet": "vehicle_fleet.csv",
    "warehouse": "warehouse_inventory.csv",
    "feedback": "customer_feedback.csv",
    "costs": "cost_breakdown.csv",
}

st.sidebar.header("Dataset files & quick instructions")
st.sidebar.write("Place the CSVs in the project folder or upload them below.")
for k, v in EXPECTED_FILES.items():
    st.sidebar.write(f"- `{v}`")

st.sidebar.markdown("---")
st.sidebar.markdown("If column names differ from those used by your CSVs, see the 'File / Column checklist' at the bottom of the app.")

# --- File upload or local file usage ---
st.header("1) Load data")

use_upload = st.checkbox("Upload CSVs manually (otherwise app will try to read local files)", value=False)

data = {}
upload_errors = []

def load_local_if_exists(name, fname):
    p = Path(fname)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.error(f"Error reading {fname}: {e}")
            return None
    return None

if use_upload:
    st.info("Upload CSVs (use the exact logical file types). You may skip files you don't have.")
    for key, fname in EXPECTED_FILES.items():
        uploaded = st.file_uploader(f"Upload {fname}", type=["csv"], key=f"u_{key}")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                data[key] = df
                st.write(f"Loaded `{fname}` — rows: {len(df)}")
            except Exception as e:
                st.error(f"Failed to read uploaded {fname}: {e}")
else:
    st.info("Attempting to read the CSVs from the local working directory.")
    for key, fname in EXPECTED_FILES.items():
        df = load_local_if_exists(key, fname)
        if df is not None:
            data[key] = df
            st.write(f"Loaded `{fname}` — rows: {len(df)}")
        else:
            st.warning(f"`{fname}` not found locally.")

# Show which datasets loaded
st.write("---")
st.subheader("Loaded datasets summary")
for key in EXPECTED_FILES:
    status = "✓" if key in data else "✗"
    st.write(f"{status} {EXPECTED_FILES[key]}")

# --- Minimal required dataset check ---
if "delivery" not in data and "orders" not in data:
    st.error("You must provide at least `delivery_performance.csv` or `orders.csv` to proceed.")
    st.stop()

# --- Utility: try-get column with many possible names ---
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- Preprocess & Merge ---
st.header("2) Prepare & merge datasets")

@st.cache_data
def prepare_data(data):
    # make copies
    orders = data.get("orders", pd.DataFrame()).copy()
    delivery = data.get("delivery", pd.DataFrame()).copy()
    routes = data.get("routes", pd.DataFrame()).copy()
    fleet = data.get("fleet", pd.DataFrame()).copy()
    warehouse = data.get("warehouse", pd.DataFrame()).copy()
    feedback = data.get("feedback", pd.DataFrame()).copy()
    costs = data.get("costs", pd.DataFrame()).copy()

    # Normalize column names to lowercase
    for df in [orders, delivery, routes, fleet, warehouse, feedback, costs]:
        if df is not None and not df.empty:
            df.columns = [c.strip() for c in df.columns]

    # Try find order id column
    order_id_candidates = ["order_id", "orderid", "id", "order id"]
    o_col = None
    for df in [orders, delivery, routes]:
        if df is None or df.empty:
            continue
        o_col = pick_col(df, order_id_candidates)
        if o_col:
            break

    # If orders is empty but delivery exists, use delivery as base
    base = orders if not orders.empty else delivery

    # Ensure datetime parsing if possible
    def safe_parse_dates(df):
        if df is None or df.empty:
            return df
        for col in df.columns:
            if any(k in col.lower() for k in ["date", "time", "deliv", "ship"]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
        return df

    orders = safe_parse_dates(orders)
    delivery = safe_parse_dates(delivery)
    routes = safe_parse_dates(routes)
    fleet = safe_parse_dates(fleet)
    warehouse = safe_parse_dates(warehouse)
    feedback = safe_parse_dates(feedback)
    costs = safe_parse_dates(costs)

    # Merge logic:
    # prefer order_id as join key; otherwise merge on available keys (carrier, route_id, etc.)
    df = pd.DataFrame()
    if not orders.empty:
        df = orders.copy()
        # Try to merge delivery info
        if not delivery.empty:
            join_on = None
            if pick_col(delivery, order_id_candidates) and pick_col(df, order_id_candidates):
                join_on = pick_col(df, order_id_candidates)
                delivery_join_on = pick_col(delivery, order_id_candidates)
                df = df.merge(delivery, left_on=join_on, right_on=delivery_join_on, how="left", suffixes=("", "_del"))
            else:
                # fallback: merge on carrier + origin/destination + approximate date
                common = list(set(df.columns).intersection(set(delivery.columns)))
                if "carrier" in common:
                    df = df.merge(delivery, on="carrier", how="left", suffixes=("", "_del"))
        # add other merges
    else:
        # no orders; use delivery as base
        df = delivery.copy()

    # Merge routes by route id or order id
    if not routes.empty:
        # try route_id
        r_col = pick_col(routes, ["route_id", "routeid", "route id"])
        if r_col and r_col in df.columns:
            df = df.merge(routes, on=r_col, how="left")
        else:
            # try merge on order id
            oc = pick_col(df, order_id_candidates)
            rc = pick_col(routes, order_id_candidates)
            if oc and rc:
                df = df.merge(routes, left_on=oc, right_on=rc, how="left")
    # Merge fleet on vehicle id if exists
    if not fleet.empty:
        vf = pick_col(fleet, ["vehicle_id", "vehicleid", "vehicle id", "veh_id"])
        df_v = pick_col(df, ["vehicle_id", "vehicleid", "vehicle id", "veh_id"])
        if vf and df_v:
            df = df.merge(fleet, left_on=df_v, right_on=vf, how="left", suffixes=("", "_veh"))
    # Merge warehouse on warehouse id if exists
    if not warehouse.empty:
        wid = pick_col(warehouse, ["warehouse_id", "warehouseid", "warehouse id", "warehouse"])
        if wid and "warehouse_id" in df.columns:
            df = df.merge(warehouse, on=wid, how="left")

    # Create target: delayed = actual_delivery_time > promised_delivery_time (or status 'delayed')
    # Candidate columns
    promised_col = pick_col(df, ["promised_delivery_date", "promised_date", "promised_delivery", "promised"])
    actual_col = pick_col(df, ["actual_delivery_date", "actual_date", "delivered_date", "actual_delivery"])
    status_col = pick_col(df, ["delivery_status", "status", "delivery_status_classification"])

    if promised_col and actual_col:
        df["delay_days"] = (pd.to_datetime(df[actual_col], errors="coerce')") - pd.to_datetime(df[promised_col], errors="coerce")).dt.total_seconds() / 86400.0
        # If parsing failed produce NaN; correct expression above had a type; fix:
    # re-do safe calculation (due to potential quote issues)
    try:
        if promised_col and actual_col:
            df["__prom"] = pd.to_datetime(df[promised_col], errors="coerce")
            df["__act"] = pd.to_datetime(df[actual_col], errors="coerce")
            df["delay_days"] = (df["__act"] - df["__prom"]).dt.total_seconds() / 86400.0
            df.drop(columns=["__prom", "__act"], inplace=True)
    except Exception:
        pass

    # If delay_days not present, try to derive from promised/actual hours or status
    if "delay_days" not in df.columns or df["delay_days"].isna().all():
        if status_col:
            # treat status text containing 'delayed' or 'late' as delayed
            df["is_delayed"] = df[status_col].astype(str).str.lower().str.contains("delayed|late|overdue").fillna(False)
        else:
            df["is_delayed"] = False
    else:
        df["is_delayed"] = (df["delay_days"] > 0).fillna(False)

    # Basic feature engineering
    # Priority
    priority_col = pick_col(df, ["priority", "priority_level", "order_priority"])
    if priority_col:
        df["priority"] = df[priority_col].astype(str).str.lower()
    else:
        df["priority"] = "standard"

    # Distance: try common names
    dist_col = pick_col(df, ["distance", "distance_km", "distance_kms", "distance_traveled", "km"])
    if dist_col:
        df["distance_km"] = pd.to_numeric(df[dist_col], errors="coerce")
    else:
        # if routes contain start/stop lat/lon we could compute; skipping for now
        df["distance_km"] = np.nan

    # Carrier
    carrier_col = pick_col(df, ["carrier", "carrier_name", "assigned_carrier"])
    if carrier_col:
        df["carrier"] = df[carrier_col].astype(str)
    else:
        df["carrier"] = "unknown"

    # Vehicle type
    vtype_col = pick_col(df, ["vehicle_type", "veh_type", "vehiclecategory"])
    if vtype_col:
        df["vehicle_type"] = df[vtype_col].astype(str)
    else:
        df["vehicle_type"] = "unknown"

    # Customer rating numeric
    rating_col = pick_col(df, ["customer_rating", "rating", "customer_rating_stars"])
    if rating_col:
        df["customer_rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    else:
        df["customer_rating"] = np.nan

    # Delivery cost numeric
    cost_col = pick_col(df, ["delivery_cost", "cost", "deliverycost"])
    if cost_col:
        df["delivery_cost"] = pd.to_numeric(df[cost_col], errors="coerce")
    else:
        df["delivery_cost"] = np.nan

    # Keep relevant subset
    features = [
        "order_id", "priority", "distance_km", "carrier", "vehicle_type",
        "customer_rating", "delivery_cost", "is_delayed"
    ]
    # ensure order_id column name present
    oid = pick_col(df, order_id_candidates)
    if oid:
        df = df.rename(columns={oid: "order_id"})
    else:
        # create synthetic order id if none
        df = df.reset_index().rename(columns={"index": "order_id"})

    # Final safety: ensure at least priority and carrier exist
    df["priority"] = df["priority"].fillna("standard")
    df["carrier"] = df["carrier"].fillna("unknown")
    df["vehicle_type"] = df["vehicle_type"].fillna("unknown")

    return df

df = prepare_data(data)
st.write("Prepared merged dataset preview (first 5 rows):")
st.dataframe(df.head())

st.markdown("---")
st.header("3) Train delay prediction model")

# Select features/target
# By default use priority, distance_km, carrier, vehicle_type, customer_rating, delivery_cost
candidate_features = ["priority", "distance_km", "carrier", "vehicle_type", "customer_rating", "delivery_cost"]
available_features = [f for f in candidate_features if f in df.columns]
st.write("Available features detected:", available_features)

target_col = "is_delayed"
if target_col not in df.columns:
    st.error("Target column `is_delayed` not found after preprocessing. See earlier warnings.")
    st.stop()

# Option: allow user to drop or add features
chosen = st.multiselect("Choose features to train on", available_features, default=available_features)

min_rows = st.slider("Minimum rows with target & features required to train", min_value=50, max_value=2000, value=30)
train_button = st.button("Train model")

if train_button:
    # Prepare X and y
    train_df = df.dropna(subset=[target_col])  # ensure we have target
    # require at least one non-null feature row
    train_df = train_df[train_df[chosen].notna().any(axis=1)]
    st.write(f"Training rows available: {len(train_df)}")
    if len(train_df) < min_rows:
        st.error(f"Not enough rows ({len(train_df)}) to train. Need at least {min_rows}.")
    else:
        X = train_df[chosen].copy()
        y = train_df[target_col].astype(int)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Build preprocessing pipeline
        numeric_features = [c for c in chosen if pd.api.types.is_numeric_dtype(train_df[c]) or c in ["distance_km", "customer_rating", "delivery_cost"]]
        categorical_features = [c for c in chosen if c not in numeric_features]

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

        pre = ColumnTransformer([
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features)
        ], remainder="drop")

        model = Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ])

        with st.spinner("Training model..."):
            model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model.named_steps["clf"], "predict_proba") else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) > 1 else None

        st.subheader("Model performance (test set)")
        st.write(f"- Accuracy: **{acc:.3f}**")
        st.write(f"- Precision: **{prec:.3f}**")
        st.write(f"- Recall: **{rec:.3f}**")
        st.write(f"- F1 score: **{f1:.3f}**")
        if roc:
            st.write(f"- ROC AUC: **{roc:.3f}**")

        st.write("Confusion matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        # Save model to session state so user can use it for predictions
        st.session_state["trained_model"] = model
        st.success("Model trained and saved to session.")

st.markdown("---")
st.header("4) Predict & recommend corrective actions")

if "trained_model" not in st.session_state:
    st.info("Train a model in step 3, or upload a pre-trained model as a .pkl file.")
    upload_model = st.file_uploader("Or upload trained model (.pkl)", type=["pkl", "joblib"])
    if upload_model:
        try:
            model = joblib.load(upload_model)
            st.session_state["trained_model"] = model
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

if "trained_model" in st.session_state:
    model = st.session_state["trained_model"]

    st.write("You can predict delays for the whole dataset or for a selected order.")
    apply_all = st.button("Predict for all records in merged dataset")
    if apply_all:
        # Ensure chosen features available in df
        if "chosen" not in locals() or len(chosen) == 0:
            st.error("No features selected in training step. Go back to step 3.")
        else:
            to_pred = df[chosen].copy()
            # align missing columns if necessary
            # model expects same columns pipeline handles unknowns
            preds = model.predict(to_pred)
            proba = model.predict_proba(to_pred)[:, 1] if hasattr(model.named_steps["clf"], "predict_proba") else None
            df["predicted_delay"] = preds
            if proba is not None:
                df["pred_delay_proba"] = proba
            st.write("Predictions added to merged dataset. Preview:")
            st.dataframe(df[["order_id", "predicted_delay", "pred_delay_proba"]].head(50))
            # allow export
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv, file_name="predictions_with_delay.csv", mime="text/csv")

    st.write("---")
    st.subheader("Predict for a single order and get recommendations")
    order_select = st.selectbox("Select order_id (from merged dataset)", options=df["order_id"].astype(str).tolist())
    if st.button("Predict order & recommend", key="predict_one"):
        row = df[df["order_id"].astype(str) == str(order_select)].head(1)
        if row.empty:
            st.error("Order not found.")
        else:
            input_row = row[chosen].iloc[0:1]
            pred = model.predict(input_row)[0]
            proba = model.predict_proba(input_row)[:, 1][0] if hasattr(model.named_steps["clf"], "predict_proba") else None
            st.write(f"Predicted delayed: **{bool(pred)}** (prob: {proba:.2f})" if proba is not None else f"Predicted delayed: **{bool(pred)}**")

            # Simple rule-based recommendations
            recs = []
            # if priority is economy and predicted delayed -> recommend upgrade
            pr = row["priority"].iloc[0] if "priority" in row.columns else "standard"
            if pred:
                if "express" not in str(pr).lower():
                    recs.append("Consider upgrading priority to Express for time-critical orders.")
                # if distance large & assigned carrier is slow
                if row.get("distance_km", pd.Series([np.nan])).iloc[0] and (row.get("distance_km").iloc[0] > 200):
                    recs.append("Long route detected — consider reassigning to long-haul carrier or faster vehicle.")
                # vehicle type is unknown or bike but long distance
                vt = row.get("vehicle_type", pd.Series(["unknown"])).iloc[0]
                if "bike" in str(vt).lower() and row.get("distance_km", pd.Series([0])).iloc[0] > 50:
                    recs.append("Vehicle type mismatched for distance. Use van/truck instead of bike.")
                # if low rating maybe quality issue
                if row.get("customer_rating", pd.Series([np.nan])).iloc[0] < 3:
                    recs.append("Check delivery process for quality issues; customer rating is low.")
                # cost tradeoff suggestion
                recs.append("If business impact justifies, pay for expedited shipping or re-route through closer warehouse.")
            else:
                recs.append("No action needed now. Monitor the order.")

            st.markdown("**Recommended actions:**")
            for r in recs:
                st.write(f"- {r}")

st.markdown("---")
st.header("File / Column checklist (rename columns if your file uses different names)")
st.write("""
If your CSV files use different names, rename the columns quickly (in Excel or using a small Python script) to one of these expected names:

**Orders / Delivery file**
- `order_id` (or `orderid`, `id`)
- `promised_delivery_date` (or `promised_date`)
- `actual_delivery_date` (or `actual_date`, `delivered_date`)
- `delivery_status` (or `status`)
- `priority` (or `order_priority`)
- `carrier` (or `carrier_name`)
- `distance_km` (or `distance`, `distance_traveled`)
- `vehicle_type` (or `veh_type`)
- `customer_rating` (or `rating`)
- `delivery_cost` (or `cost`)

**Routes file**
- `route_id`, `order_id`, `distance_km` or `toll_charges`

**Fleet file**
- `vehicle_id`, `vehicle_type`, `fuel_efficiency`

**Warehouse file**
- `warehouse_id`, `stock_level`, `reorder_level`

**Customer feedback**
- `order_id`, `feedback_text`, `rating`, `feedback_date`

If you prefer, upload your files from the sidebar instead of placing them locally.
""")

st.markdown("### Quick troubleshooting tips")
st.write("""
- If the app says not enough rows to train: try lowering the slider or include more historical data rows.
- If predictions look wrong: check that `promised_delivery_date` and `actual_delivery_date` are parsed correctly (they should be full datetime strings).
- If your columns are named differently, either rename columns in CSVs or update files before training.
""")

st.markdown("---")
st.write("## Deliverables created by this app")
st.write("- `predictions_with_delay.csv` when you run predictions for all records")
st.write("- Optionally you can `joblib.dump(model, 'model.pkl')` to export the trained model (not automated in UI)")

st.write("Done. Good luck — train with historical data for best results!")