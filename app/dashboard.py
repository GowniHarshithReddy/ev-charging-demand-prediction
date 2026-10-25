import pandas as pd
import streamlit as st

st.set_page_config(page_title="EV Charging Dashboard", layout="wide")
st.title("EV Charging Demand Dashboard")


@st.cache_data
def load_data():
    return pd.read_csv("data/charging_features.csv", parse_dates=["StartDatetime"])


df = load_data()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

locations = st.sidebar.multiselect(
    "Location", sorted(df["Location"].unique()), default=sorted(df["Location"].unique())
)
charger_types = st.sidebar.multiselect(
    "Charger type", sorted(df["ChargerType"].unique()), default=sorted(df["ChargerType"].unique())
)
companies = st.sidebar.multiselect(
    "Charger company", sorted(df["ChargerCompany"].unique()), default=sorted(df["ChargerCompany"].unique())
)
min_date = df["StartDatetime"].min().date()
max_date = df["StartDatetime"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)

mask = (
    df["Location"].isin(locations)
    & df["ChargerType"].isin(charger_types)
    & df["ChargerCompany"].isin(companies)
)
if len(date_range) == 2:
    mask &= df["StartDatetime"].dt.date.between(date_range[0], date_range[1])
fdf = df[mask]

if fdf.empty:
    st.warning("No sessions match these filters.")
    st.stop()

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total sessions", f"{len(fdf):,}")
c2.metric("Average demand", f"{fdf['Demand'].mean():.1f}")
c3.metric("Locations", fdf["Location"].nunique())
c4.metric("Chargers", fdf["ChargerID"].nunique())

# ---------- Charts ----------
left, right = st.columns(2)

with left:
    st.subheader("Average demand by location")
    st.bar_chart(fdf.groupby("Location")["Demand"].mean().sort_values(ascending=False))

with right:
    st.subheader("Average demand by start hour")
    st.bar_chart(fdf.groupby("Hour")["Demand"].mean())

st.subheader("Sessions per day")
st.line_chart(fdf.groupby(fdf["StartDatetime"].dt.date).size())

st.subheader("Sessions by location")
st.bar_chart(fdf["Location"].value_counts())
# ---------- Prediction section ----------
import joblib
import numpy as np

st.divider()
st.header("Predict a charging session")


@st.cache_resource
def load_models():
    reg = joblib.load("models/xgb_regressor.pkl")
    reg_cols = joblib.load("models/regression_columns.pkl")
    clf = joblib.load("models/rf_classifier.pkl")
    clf_cols = joblib.load("models/classification_columns.pkl")
    thr = joblib.load("models/demand_threshold.pkl")
    return reg, reg_cols, clf, clf_cols, thr


reg_model, reg_cols, clf_model, clf_cols, threshold = load_models()
avg_hist = float(df.loc[df["UserPrevAvgDemand"].notna(), "UserPrevAvgDemand"].mean())


def time_of_day(hour):
    if hour <= 5:
        return "Night"
    if hour <= 11:
        return "Morning"
    if hour <= 16:
        return "Afternoon"
    if hour <= 21:
        return "Evening"
    return "LateEvening"


def build_row(columns, loc, ctype, comp, hour, dow, guest, prev_avg):
    row = {c: 0 for c in columns}
    row["ChargerCompany"] = comp
    row["ChargerType"] = ctype
    row["Hour"] = hour
    row["DayOfWeek"] = dow
    row["IsWeekend"] = int(dow >= 5)
    row["IsGuest"] = int(guest)
    row["HasHistory"] = 0 if guest else 1
    row["UserPrevSessions"] = 0 if guest else 20
    row["UserPrevAvgDemand"] = avg_hist if guest else prev_avg
    row["ChargerPrevSessions"] = 100
    for key in (f"Location_{loc}", f"TimeOfDay_{time_of_day(hour)}"):
        if key in row:
            row[key] = 1
    return pd.DataFrame([row], columns=columns)


f1, f2, f3 = st.columns(3)
p_loc = f1.selectbox("Location", sorted(df["Location"].unique()))
p_type = f2.selectbox("Charger type", [0, 1])
p_comp = f3.selectbox("Charger company", [0, 1])

f4, f5, f6 = st.columns(3)
p_hour = f4.slider("Start hour", 0, 23, 18)
p_dow = f5.selectbox(
    "Day of week", list(range(7)),
    format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
)
p_guest = f6.checkbox("Guest user (no history)", value=True)
p_prev = st.slider("User's past average demand", 1.0, 60.0, round(avg_hist, 1), disabled=p_guest)
if st.button("Predict"):
    reg_row = build_row(reg_cols, p_loc, p_type, p_comp, p_hour, p_dow, p_guest, p_prev)
    clf_row = build_row(clf_cols, p_loc, p_type, p_comp, p_hour, p_dow, p_guest, p_prev)
    pred = float(reg_model.predict(reg_row)[0])
    prob = float(clf_model.predict_proba(clf_row)[0][1])

    r1, r2 = st.columns(2)
    r1.metric("Predicted demand", f"{pred:.1f}")
    label = "High Demand" if prob >= 0.5 else "Normal Demand"
    r2.metric("Classification", label, f"{prob * 100:.0f}% chance of High (threshold {threshold:.1f})")

    # SHAP explanation for this prediction
    import shap
    import matplotlib.pyplot as plt

    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(reg_model)
    sv = explainer(reg_row)
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    fig_shap = plt.gcf()
    plt.tight_layout()
    st.pyplot(fig_shap)
    plt.close(fig_shap)
    st.caption("Red bars push the predicted demand up, blue bars push it down.")
    # ---------- Time-series forecast ----------
st.divider()
st.header("Daily demand forecast")

daily = (
    df.groupby(df["StartDatetime"].dt.date)
    .agg(Sessions=("Demand", "size"), TotalDemand=("Demand", "sum"))
)
daily.index = pd.to_datetime(daily.index)
daily = daily.asfreq("D").fillna(0)

metric = st.radio("Forecast", ["Sessions", "TotalDemand"], horizontal=True)
series = daily[metric]

# Same-weekday average of the previous 4 weeks (uses past data only)
seasonal = sum(series.shift(7 * k) for k in range(1, 5)) / 4
naive = series.shift(1)

test = series.index >= series.index[-60]
actual = series[test]
mae_seasonal = (actual - seasonal[test]).abs().mean()
mae_naive = (actual - naive[test]).abs().mean()

k1, k2 = st.columns(2)
k1.metric("MAE: 4-week same-weekday average", f"{mae_seasonal:.1f}")
k2.metric("MAE: yesterday baseline", f"{mae_naive:.1f}")

chart_df = pd.DataFrame({"Actual": actual, "Forecast": seasonal[test]})
st.line_chart(chart_df)

# Next 7 days
last_date = series.index[-1]
future = {}
history = series.copy()
for i in range(1, 8):
    day = last_date + pd.Timedelta(days=i)
    future[day] = sum(history.get(day - pd.Timedelta(days=7 * k), 0) for k in range(1, 5)) / 4
    history.loc[day] = future[day]
st.subheader("Forecast for the next 7 days")
forecast_df = pd.Series(future, name=f"Forecast {metric}").round(1).to_frame()
forecast_df.index = forecast_df.index.strftime("%Y-%m-%d (%a)")
st.dataframe(forecast_df)