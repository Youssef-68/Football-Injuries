import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sports Injury Analytics System",
    layout="wide"
)

# ======================================================
# AI INSIGHTS FUNCTIONS
# ======================================================

def get_trend_insight(data, x_col, y_col, title):
    if len(data) < 2:
        return "Not enough data points to identify trends."
    
    values = data[y_col].values
    first_val = values[0]
    last_val = values[-1]
    
    if last_val > first_val * 1.1:
        change = "increasing"
        direction = "upward"
    elif last_val < first_val * 0.9:
        change = "decreasing"
        direction = "downward"
    else:
        change = "stable"
        direction = "relatively stable"
    
    max_idx = values.argmax()
    min_idx = values.argmin()
    
    insight = f"The {title} shows a {change} trend with a {direction} direction. "
    insight += f"Peak value of {values[max_idx]:.1f} occurred, "
    insight += f"while lowest was {values[min_idx]:.1f}. "
    
    if change == "increasing":
        insight += f"This represents a {((last_val/first_val - 1) * 100):.1f}% increase over the period."
    elif change == "decreasing":
        insight += f"This represents a {((1 - last_val/first_val) * 100):.1f}% decrease over the period."
    
    return insight

def get_distribution_insight(data, column, title):
    values = data[column].dropna()
    
    if len(values) == 0:
        return "No data available for analysis."
    
    mean_val = values.mean()
    median_val = values.median()
    std_val = values.std()
    max_val = values.max()
    min_val = values.min()
    
    insight = f"The {title} distribution shows a mean of {mean_val:.1f} and median of {median_val:.1f}. "
    
    if abs(mean_val - median_val) / mean_val < 0.1:
        insight += "The distribution is relatively symmetric. "
    elif mean_val > median_val:
        insight += "The distribution is right-skewed, indicating some high values pulling the mean upward. "
    else:
        insight += "The distribution is left-skewed, indicating some low values pulling the mean downward. "
    
    insight += f"The standard deviation is {std_val:.1f}, with values ranging from {min_val:.1f} to {max_val:.1f}."
    
    return insight

def get_comparison_insight(data, category_col, value_col, title, top_n=5):
    if len(data) == 0:
        return "No data available for comparison."
    
    top_items = data.nlargest(top_n, value_col)
    bottom_items = data.nsmallest(top_n, value_col)
    
    insight = f"In the {title}, the highest values are: "
    for i, row in top_items.iterrows():
        insight += f"{row[category_col]} ({row[value_col]:.1f}), "
    
    insight = insight.rstrip(", ") + ". "
    
    insight += f"The lowest values are: "
    for i, row in bottom_items.iterrows():
        insight += f"{row[category_col]} ({row[value_col]:.1f}), "
    
    insight = insight.rstrip(", ") + ". "
    
    if len(top_items) > 0 and len(bottom_items) > 0:
        ratio = top_items[value_col].iloc[0] / bottom_items[value_col].iloc[0] if bottom_items[value_col].iloc[0] > 0 else float('inf')
        if ratio > 5:
            insight += f"The top category is {ratio:.1f} times higher than the bottom, showing significant variation."
        elif ratio > 2:
            insight += f"The top category is {ratio:.1f} times higher than the bottom, showing moderate variation."
    
    return insight

def get_monthly_pattern_insight(monthly_data, title):
    if len(monthly_data) == 0:
        return "No monthly data available."
    
    peak_month = monthly_data.loc[monthly_data["count"].idxmax(), "month_name"]
    low_month = monthly_data.loc[monthly_data["count"].idxmin(), "month_name"]
    peak_value = monthly_data["count"].max()
    low_value = monthly_data["count"].min()
    
    insight = f"The {title} shows seasonal patterns. "
    insight += f"The highest injury count occurs in {peak_month} ({peak_value:.0f} injuries), "
    insight += f"while the lowest occurs in {low_month} ({low_value:.0f} injuries). "
    
    summer_months = ["Jul", "Aug", "Sep"]
    winter_months = ["Dec", "Jan", "Feb"]
    
    summer_sum = monthly_data[monthly_data["month_name"].isin(summer_months)]["count"].sum()
    winter_sum = monthly_data[monthly_data["month_name"].isin(winter_months)]["count"].sum()
    
    if summer_sum > winter_sum:
        insight += "Injuries are more common in pre-season/early season months (July-September)."
    else:
        insight += "Injuries are more common in mid-season months (December-February)."
    
    return insight

# ======================================================
# DATA CLEANING
# ======================================================
@st.cache_data
def load_and_clean_data():
    
    df = pd.read_csv("player_injuries.csv")

    df["from_date"] = pd.to_datetime(df["from_date"], errors='coerce', dayfirst=True)
    df["end_date"] = pd.to_datetime(df["end_date"], errors='coerce', dayfirst=True)

    df["end_date"] = df["end_date"].fillna(
        df["from_date"] + pd.to_timedelta(df["days_missed"], unit='D')
    )

    df["from_date"] = df["from_date"].fillna(
        df["end_date"] - pd.to_timedelta(df["days_missed"], unit='D')
    )

    df["days_missed"] = df["days_missed"].fillna(
        (df["end_date"] - df["from_date"]).dt.days
    )

    df["season_name"] = df["season_name"].replace({
        'Dec-13': '12/13',
        '09-Oct': '09/10',
        '10-Nov': '10/11',
        '11-Dec': '11/12',
        '08-Sep': '08/09',
        '07-Aug': '07/08',
        '01-Feb': '01/02',
        '06-Jul': '06/07',
        '05-Jun': '05/06',
        '04-May': '04/05',
        '03-Apr': '03/04',
        '02-Mar': '02/03',
        '1910/11': '10/11',
        '1909/10': '09/10'
    })

    df["season_sort"] = df["season_name"].str[:2].astype(int)
    df["season_sort"] = df["season_sort"].apply(
        lambda x: 1900 + x if x >= 70 else 2000 + x
    )

    df = df[
        (df["season_sort"] >= 1989) & (df["season_sort"] <= 2024)
    ]

    def clean_injury(x):
        x = str(x).lower()
        
        if "unknown" in x:
            return "Unknown"
        
        if x in ["rest", "fitness", "quarantine"]:
            return "Non-injury"
        
        if any(i in x for i in ["flu", "ill", "corona", "fever", "cold", "virus", "infection"]):
            return "Illness"
        
        if "hamstring" in x:
            return "Hamstring"
        
        if "knee" in x:
            return "Knee"
        
        if "ankle" in x:
            return "Ankle"
        
        if "groin" in x or "adductor" in x:
            return "Groin/Adductor"
        
        if "back" in x:
            return "Back"
        
        if "shoulder" in x:
            return "Shoulder"
        
        if "head" in x or "concussion" in x:
            return "Head/Concussion"
        
        if "muscle" in x:
            return "Muscle"
        
        if "calf" in x:
            return "Calf"
        
        if "thigh" in x:
            return "Thigh"
        
        if "hip" in x:
            return "Hip"
        
        if "foot" in x:
            return "Foot"
        
        if "rib" in x:
            return "Rib"
        
        if "hand" in x or "wrist" in x:
            return "Hand/Wrist"
        
        if "arm" in x or "elbow" in x:
            return "Arm/Elbow"
        
        if "achilles" in x:
            return "Achilles"
        
        if "abdominal" in x or "oblique" in x or "abs" in x:
            return "Abdominal"
        
        if "quad" in x:
            return "Quadriceps"
        
        if "metatarsal" in x:
            return "Metatarsal"
        
        if any(i in x for i in ["nose", "jaw", "eye", "cheek", "facial", "face"]):
            return "Facial"
        
        if "ligament" in x and "ankle" in x:
            return "Ankle Ligament"
        
        if "ligament" in x and "knee" in x:
            return "Knee Ligament"
        
        if "cramp" in x:
            return "Cramp"
        
        if "bruise" in x or "contusion" in x:
            return "Bruise/Contusion"
        
        if "fracture" in x or "broken" in x:
            return "Fracture"
        
        if "tear" in x:
            return "Muscle Tear"
        
        if "strain" in x:
            return x.replace("strain", "Strain").title()
        
        if "sprain" in x:
            return x.replace("sprain", "Sprain").title()
        
        return x.title()
    
    df["injury_clean"] = df["injury_reason"].apply(clean_injury)
    
    df["duration"] = (df["end_date"] - df["from_date"]).dt.days
    df["month"] = df["from_date"].dt.month
    df["start_month"] = df["from_date"].dt.month
    df["end_month"] = df["end_date"].dt.month
    
    return df

df_cleaned = load_and_clean_data()

# ======================================================
# ML MODEL TRAINING
# ======================================================
@st.cache_resource
def train_models():
    
    df = df_cleaned.copy()
    
    df["injury_duration"] = (df["end_date"] - df["from_date"]).dt.days
    df["injury_month"] = df["from_date"].dt.month
    df["injury_dayofweek"] = df["from_date"].dt.dayofweek
    
    player_stats = df.groupby("player_id").agg({
        "games_missed": "mean",
        "injury_duration": "mean"
    }).rename(columns={
        "games_missed": "player_avg_games_missed",
        "injury_duration": "player_avg_injury_duration"
    })
    
    df = df.merge(player_stats, on="player_id", how="left")
    df["injury_count_player"] = df.groupby("player_id").cumcount()
    
    df = df.dropna(subset=["days_missed"])
    df = df[df["days_missed"] >= 0]
    
    cat_cols = ["injury_reason", "injury_clean"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    cols_to_drop = ["days_missed", "from_date", "end_date", "player_id", "duration", "games_missed", "season_name", "season_sort"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df["days_missed"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cols_to_clip = ["player_avg_injury_duration", "player_avg_games_missed"]
    for col in cols_to_clip:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(0).clip(0, 365)
            X_test[col] = X_test[col].fillna(0).clip(0, 365)
    
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    y_train = y_train.fillna(0)
    y_train = y_train.replace([np.inf, -np.inf], 0)
    y_train = y_train.clip(lower=0)
    
    y_train_log = np.log1p(y_train)
    
    if y_train_log.isna().any():
        valid_idx = ~y_train_log.isna()
        X_train = X_train[valid_idx]
        y_train_log = y_train_log[valid_idx]
        y_train = y_train[valid_idx]
    
    reg_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        min_child_weight=5,
        random_state=42
    )
    reg_model.fit(X_train, y_train_log)
    
    X_test_clean = X_test.copy()
    preds_log = reg_model.predict(X_test_clean)
    preds = np.expm1(preds_log)
    
    reg_mae = mean_absolute_error(y_test, preds)
    reg_rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    def injury_class(x):
        if x <= 7:
            return 0
        elif x <= 30:
            return 1
        elif x <= 90:
            return 2
        else:
            return 3
    
    y_train_cls = y_train.apply(injury_class)
    y_test_cls = y_test.apply(injury_class)
    
    cls_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    classes = np.unique(y_train_cls)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_cls)
    class_weights = dict(zip(classes, weights))
    sample_weights = y_train_cls.map(class_weights)
    
    cls_model.fit(X_train, y_train_cls, sample_weight=sample_weights)
    
    y_pred_cls = cls_model.predict(X_test)
    
    cls_report = classification_report(y_test_cls, y_pred_cls, output_dict=True)
    cls_matrix = confusion_matrix(y_test_cls, y_pred_cls)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': reg_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df_cleaned[col].astype(str))
        encoders[col] = le
    
    return {
        "reg_model": reg_model,
        "cls_model": cls_model,
        "reg_mae": reg_mae,
        "reg_rmse": reg_rmse,
        "cls_report": cls_report,
        "cls_matrix": cls_matrix,
        "feature_names": X_train.columns.tolist(),
        "feature_importance": feature_importance,
        "encoders": encoders
    }

models = train_models()

# ======================================================
# FILTERING FUNCTIONS
# ======================================================
def apply_filters(df, season_filter, injury_filter, start_date, end_date):
    filtered = df.copy()
    
    if season_filter:
        filtered = filtered[filtered["season_name"].isin(season_filter)]
    
    if injury_filter:
        filtered = filtered[filtered["injury_clean"].isin(injury_filter)]
    
    filtered = filtered[
        (filtered["from_date"] >= start_date) &
        (filtered["from_date"] <= end_date)
    ]
    
    return filtered

def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_no_outliers

# ======================================================
# SIDEBAR FILTERS
# ======================================================
st.sidebar.header("Filters")

def safe_multiselect(label, options):
    options = list(options)
    if len(options) == 0:
        st.sidebar.warning(f"No data available for {label}")
        return []
    
    selected = st.sidebar.multiselect(label, options, default=options)
    
    if len(selected) == 0:
        st.sidebar.warning(f"{label}: should select a choice")
    
    return selected

st.sidebar.subheader("Season Filter")
available_seasons = sorted(df_cleaned["season_name"].dropna().unique(), 
                          key=lambda x: int(x.split('/')[0]) if '/' in x else 0)
season_filter = safe_multiselect("Select Seasons", available_seasons)

st.sidebar.subheader("Injury Type Filter")
injury_options = sorted(df_cleaned[df_cleaned["injury_clean"] != "Unknown"]["injury_clean"].dropna().unique())
injury_filter = safe_multiselect("Select Injury Types", injury_options)

st.sidebar.subheader("Date Range Filter")

# Fixed date range: 01/07/1989 to 30/06/2025
fixed_start_date = pd.to_datetime("1989-07-01").date()
fixed_end_date = pd.to_datetime("2025-06-30").date()

# Get actual data date range for min/max constraints
actual_date_min = df_cleaned["from_date"].min().date()
actual_date_max = df_cleaned["from_date"].max().date()

# Use the fixed range as the limit, but don't exceed actual data bounds
min_allowed_date = max(fixed_start_date, actual_date_min)
max_allowed_date = min(fixed_end_date, actual_date_max)

date_range = st.sidebar.date_input(
    "Select Date Range (01/07/1989 to 30/06/2025)",
    value=(min_allowed_date, max_allowed_date),
    min_value=min_allowed_date,
    max_value=max_allowed_date,
    help="Select date range between 01 July 1989 and 30 June 2025"
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    
    # Validate that selected range is within allowed bounds
    if start_date.date() < fixed_start_date:
        st.sidebar.error(f"Start date cannot be before {fixed_start_date.strftime('%d/%m/%Y')}")
        start_date = pd.to_datetime(fixed_start_date)
    if end_date.date() > fixed_end_date:
        st.sidebar.error(f"End date cannot be after {fixed_end_date.strftime('%d/%m/%Y')}")
        end_date = pd.to_datetime(fixed_end_date)
else:
    st.sidebar.warning("Date Range: please select a valid range")
    start_date = pd.to_datetime(fixed_start_date)
    end_date = pd.to_datetime(fixed_end_date)

# Apply filters
filtered_df = apply_filters(df_cleaned, season_filter, injury_filter, start_date, end_date)
charts_df = filtered_df[filtered_df["injury_clean"] != "Unknown"].copy()

st.sidebar.markdown("---")
st.sidebar.subheader("Active Filters")

if season_filter:
    st.sidebar.write(f"Seasons: {len(season_filter)} selected")
else:
    st.sidebar.write("Seasons: all")

if injury_filter:
    st.sidebar.write(f"Injuries: {len(injury_filter)} selected")
else:
    st.sidebar.write("Injuries: all")

st.sidebar.write(f"Date: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
st.sidebar.markdown("---")
st.sidebar.metric("Total Records", len(filtered_df))
st.sidebar.metric("Records for Charts", len(charts_df))

if filtered_df.empty:
    st.error("No data matches your filters. Please adjust your selections.")
    st.stop()

st.success(f"Showing {len(filtered_df):,} records")
if len(filtered_df) > len(charts_df):
    st.info(f"Charts exclude {len(filtered_df) - len(charts_df):,} Unknown injury records")
    
    
# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Injuries Analysis",
    "Timeline Analysis", 
    "Days Missed",
    "Games Missed",
    "Injury Predictor"
])

# TAB 1: Injuries Analysis
with tab1:
    if charts_df.empty:
        st.warning("No data available for analysis after filtering")
    else:
        col1, col2 = st.columns(2)

        top_injuries = charts_df["injury_clean"].value_counts().reset_index().head(10)
        top_injuries.columns = ["Injury Type", "Count"]
        
        fig1 = px.bar(
            top_injuries,
            x="Injury Type",
            y="Count",
            title="Top 10 Injury Types",
            color="Count",
            color_continuous_scale="Viridis"
        )
        col1.plotly_chart(fig1, use_container_width=True)
        insight1 = get_comparison_insight(top_injuries, "Injury Type", "Count", "Top Injury Types")
        col1.info(f"Insight: {insight1}")

        season_counts = charts_df.groupby(["season_sort", "season_name"]).size().reset_index(name="count")
        
        fig2 = px.line(
            season_counts,
            x="season_name",
            y="count",
            markers=True,
            title="Injuries per Season",
            labels={"season_name": "Season", "count": "Injury Count"}
        )
        col2.plotly_chart(fig2, use_container_width=True)
        insight2 = get_trend_insight(season_counts, "season_name", "count", "injury trend across seasons")
        col2.info(f"Insight: {insight2}")

        trend = charts_df.groupby(["season_sort", "season_name", "injury_clean"]).size().reset_index(name="count")
        top_types = charts_df["injury_clean"].value_counts().head(8).index
        trend_filtered = trend[trend["injury_clean"].isin(top_types)]
        
        fig3 = px.line(
            trend_filtered,
            x="season_name",
            y="count",
            color="injury_clean",
            markers=True,
            title="Injury Trends Across Seasons (Top 8 Types)",
            labels={
                "season_name": "Season",
                "count": "Number of Injuries",
                "injury_clean": "Injury Category"
            }
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        if len(trend_filtered) > 0:
            volatility = trend_filtered.groupby("injury_clean")["count"].std().reset_index()
            volatile_type = volatility.loc[volatility["count"].idxmax(), "injury_clean"]
            insight3 = f"'{volatile_type}' shows the most variation across seasons."
            st.info(f"Insight: {insight3}")

# TAB 2: Timeline Analysis
with tab2:
    if charts_df.empty:
        st.warning("No data available for analysis after filtering")
    else:
        col1, col2 = st.columns(2)

        month_order = [7,8,9,10,11,12,1,2,3,4,5,6]
        month_labels = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]

        temp = charts_df.copy()
        temp["month"] = temp["from_date"].dt.month

        monthly = temp.groupby("month").size().reindex(month_order).fillna(0).reset_index(name="count")
        monthly["month_name"] = pd.Categorical(
            [month_labels[month_order.index(m)] for m in monthly["month"]],
            categories=month_labels,
            ordered=True
        )

        fig1 = px.line(
            monthly,
            x="month_name",
            y="count",
            markers=True,
            title="Monthly Injury Distribution",
            labels={"month_name": "Month", "count": "Injury Count"}
        )
        col1.plotly_chart(fig1, use_container_width=True)
        insight1 = get_monthly_pattern_insight(monthly, "monthly injury distribution")
        col1.info(f"Insight: {insight1}")

        temp["duration"] = (temp["end_date"] - temp["from_date"]).dt.days
        yearly_duration = temp.groupby(temp["from_date"].dt.year)["duration"].mean().reset_index()

        fig2 = px.line(
            yearly_duration,
            x="from_date",
            y="duration",
            markers=True,
            title="Average Injury Duration Over Years",
            labels={"from_date": "Year", "duration": "Duration (Days)"}
        )
        col2.plotly_chart(fig2, use_container_width=True)
        insight2 = get_trend_insight(yearly_duration, "from_date", "duration", "average injury duration")
        col2.info(f"Insight: {insight2}")

        temp["start_month"] = temp["from_date"].dt.month
        temp["end_month"] = temp["end_date"].dt.month

        start_counts = temp.groupby("start_month").size().reindex(month_order).fillna(0).reset_index(name="start")
        end_counts = temp.groupby("end_month").size().reindex(month_order).fillna(0).reset_index(name="end")

        month_comparison = pd.DataFrame({
            "month": month_labels,
            "Injury Start": start_counts["start"].values,
            "Injury End": end_counts["end"].values
        })

        fig3 = px.line(
            month_comparison,
            x="month",
            y=["Injury Start", "Injury End"],
            title="Injury Start vs End Months",
            labels={"month": "Month", "value": "Count"}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        start_peak = month_comparison.loc[month_comparison["Injury Start"].idxmax(), "month"]
        end_peak = month_comparison.loc[month_comparison["Injury End"].idxmax(), "month"]
        insight3 = f"Injuries most frequently start in {start_peak} but tend to end in {end_peak}."
        st.info(f"Insight: {insight3}")

        temp["injury_length"] = temp["duration"].apply(lambda x: "Long (>30 days)" if x > 30 else "Short (<=30 days)")
        temp["season_year"] = temp["from_date"].dt.year
        temp.loc[temp["from_date"].dt.month < 7, "season_year"] -= 1

        season_type = temp.groupby(["season_year", "injury_length"]).size().reset_index(name="count")

        fig4 = px.bar(
            season_type,
            x="season_year",
            y="count",
            color="injury_length",
            barmode="group",
            title="Short vs Long Injury Distribution",
            labels={"season_year": "Season", "count": "Number of Injuries"}
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        short_total = season_type[season_type["injury_length"] == "Short (<=30 days)"]["count"].sum()
        long_total = season_type[season_type["injury_length"] == "Long (>30 days)"]["count"].sum()
        ratio = short_total / long_total if long_total > 0 else float('inf')
        
        if ratio > 3:
            insight4 = f"Short-term injuries outnumber long-term injuries by a ratio of {ratio:.1f}:1."
        else:
            insight4 = f"Long-term injuries represent {long_total/(short_total+long_total)*100:.1f}% of all cases."
        
        st.info(f"Insight: {insight4}")

# TAB 3: Days Missed Analysis
with tab3:
    if filtered_df.empty:
        st.warning("No data available for analysis after filtering")
    else:
        days_df = remove_outliers(filtered_df, "days_missed")
        
        if days_df.empty:
            st.warning("No data available after removing outliers")
        else:
            col1, col2 = st.columns(2)

            fig1 = px.histogram(
                days_df,
                x="days_missed",
                nbins=30,
                title="Days Missed Distribution",
                labels={"days_missed": "Days Missed"},
                color_discrete_sequence=["#2E86AB"]
            )
            col1.plotly_chart(fig1, use_container_width=True)
            insight1 = get_distribution_insight(days_df, "days_missed", "days missed")
            col1.info(f"Insight: {insight1}")

            fig2 = px.box(
                days_df,
                y="days_missed",
                title="Days Missed Boxplot",
                labels={"days_missed": "Days Missed"},
                color_discrete_sequence=["#A23B72"]
            )
            col2.plotly_chart(fig2, use_container_width=True)
            
            q1 = days_df["days_missed"].quantile(0.25)
            q3 = days_df["days_missed"].quantile(0.75)
            median = days_df["days_missed"].median()
            insight2 = f"The middle 50% of injuries cause between {q1:.0f} and {q3:.0f} days missed, with a median of {median:.0f} days."
            col2.info(f"Insight: {insight2}")

            top_days = days_df.groupby("injury_clean")["days_missed"].mean().reset_index()
            top_days = top_days.sort_values("days_missed", ascending=False).head(10)

            fig3 = px.bar(
                top_days,
                x="injury_clean",
                y="days_missed",
                title="Average Days Missed by Injury Type (Top 10)",
                labels={"injury_clean": "Injury Type", "days_missed": "Avg Days Missed"},
                color="days_missed",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig3, use_container_width=True)
            insight3 = get_comparison_insight(top_days, "injury_clean", "days_missed", "average days missed by injury type", 3)
            st.info(f"Insight: {insight3}")

            season_avg = days_df.groupby("season_sort")["days_missed"].mean().reset_index()

            fig4 = px.line(
                season_avg,
                x="season_sort",
                y="days_missed",
                markers=True,
                title="Season Severity Trend (Average Days Missed)",
                labels={"season_sort": "Season", "days_missed": "Avg Days Missed"}
            )
            st.plotly_chart(fig4, use_container_width=True)
            insight4 = get_trend_insight(season_avg, "season_sort", "days_missed", "season severity trend")
            st.info(f"Insight: {insight4}")

# TAB 4: Games Missed Analysis
with tab4:
    if filtered_df.empty:
        st.warning("No data available for analysis after filtering")
    else:
        games_df = remove_outliers(filtered_df, "games_missed")
        
        if games_df.empty:
            st.warning("No data available after removing outliers")
        else:
            col1, col2 = st.columns(2)

            fig1 = px.histogram(
                games_df,
                x="games_missed",
                nbins=30,
                title="Games Missed Distribution",
                labels={"games_missed": "Games Missed"},
                color_discrete_sequence=["#2E86AB"]
            )
            col1.plotly_chart(fig1, use_container_width=True)
            insight1 = get_distribution_insight(games_df, "games_missed", "games missed")
            col1.info(f"Insight: {insight1}")

            fig2 = px.box(
                games_df,
                y="games_missed",
                title="Games Missed Boxplot",
                labels={"games_missed": "Games Missed"},
                color_discrete_sequence=["#A23B72"]
            )
            col2.plotly_chart(fig2, use_container_width=True)
            
            q1 = games_df["games_missed"].quantile(0.25)
            q3 = games_df["games_missed"].quantile(0.75)
            median = games_df["games_missed"].median()
            insight2 = f"The middle 50% of injuries cause between {q1:.0f} and {q3:.0f} games missed, with a median of {median:.0f} games."
            col2.info(f"Insight: {insight2}")

            top_games = games_df.groupby("injury_clean")["games_missed"].mean().reset_index()
            top_games = top_games.sort_values("games_missed", ascending=False).head(10)

            fig3 = px.bar(
                top_games,
                x="injury_clean",
                y="games_missed",
                title="Average Games Missed by Injury Type (Top 10)",
                labels={"injury_clean": "Injury Type", "games_missed": "Avg Games Missed"},
                color="games_missed",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig3, use_container_width=True)
            insight3 = get_comparison_insight(top_games, "injury_clean", "games_missed", "average games missed by injury type", 3)
            st.info(f"Insight: {insight3}")

            season_games = games_df.groupby("season_sort")["games_missed"].mean().reset_index()

            fig4 = px.line(
                season_games,
                x="season_sort",
                y="games_missed",
                markers=True,
                title="Season Games Missed Trend",
                labels={"season_sort": "Season", "games_missed": "Avg Games Missed"}
            )
            st.plotly_chart(fig4, use_container_width=True)
            insight4 = get_trend_insight(season_games, "season_sort", "games_missed", "season games missed trend")
            st.info(f"Insight: {insight4}")

# TAB 5: Injury Predictor
with tab5:
    st.header("Injury Duration Predictor")
    st.markdown("Enter injury details below to predict expected recovery time")
    
    st.info("Important: Player history is crucial for accurate predictions. Players with previous injuries typically have different recovery patterns.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Injury Information")
        
        injury_type = st.selectbox(
            "Injury Type *",
            options=sorted(df_cleaned["injury_clean"].unique()),
            help="Select the type of injury"
        )
        
        month_names = ["January", "February", "March", "April", "May", "June", 
                      "July", "August", "September", "October", "November", "December"]
        month_map = {name: i+1 for i, name in enumerate(month_names)}
        
        selected_month = st.selectbox(
            "Month of Injury *",
            options=month_names,
            help="When did the injury occur?"
        )
        injury_month = month_map[selected_month]
        
        day_of_week = st.selectbox(
            "Day of Week *",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            help="Day when injury occurred"
        )
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        injury_dayofweek = day_map[day_of_week]
    
    with col2:
        st.subheader("Player History (Required)")
        st.markdown("*These fields significantly impact prediction accuracy*")
        
        player_avg_games = st.number_input(
            "Player's Average Games Missed *",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="IMPORTANT: Average games missed by this player in previous injuries. Higher values indicate longer recovery patterns."
        )
        
        player_avg_duration = st.number_input(
            "Player's Average Injury Duration (Days) *",
            min_value=0.0,
            max_value=365.0,
            value=15.0,
            step=5.0,
            help="IMPORTANT: Average days missed by this player in previous injuries. Key factor for recovery prediction."
        )
        
        injury_count = st.number_input(
            "Number of Previous Injuries *",
            min_value=0,
            max_value=50,
            value=2,
            step=1,
            help="IMPORTANT: Total number of previous injuries. Players with multiple injuries often have different recovery trajectories."
        )
        
        last_injury_severity = st.select_slider(
            "Last Injury Severity",
            options=["Mild", "Moderate", "Severe", "Very Severe"],
            value="Moderate",
            help="Severity of player's most recent injury"
        )
        
        severity_map = {"Mild": 0, "Moderate": 1, "Severe": 2, "Very Severe": 3}
        last_severity_score = severity_map[last_injury_severity]
    
    st.markdown("---")
    
    # Show warning if player has high injury count
    if injury_count >= 5:
        st.warning(f"This player has had {injury_count} previous injuries. History shows higher risk for prolonged recovery.")
    elif injury_count >= 3:
        st.info(f"This player has had {injury_count} previous injuries. Recovery patterns may be established.")
    else:
        st.success(f"This player has had {injury_count} previous injuries. Limited historical data available.")
    
    # Show insight about player average
    if player_avg_duration > 30:
        st.warning(f"Player's historical average of {player_avg_duration:.0f} days suggests tendency for longer recoveries.")
    elif player_avg_duration > 14:
        st.info(f"Player's historical average of {player_avg_duration:.0f} days indicates moderate recovery patterns.")
    else:
        st.success(f"Player's historical average of {player_avg_duration:.0f} days suggests quick recovery tendency.")
    
    predict_button = st.button("Predict Injury Duration", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Validate inputs
            if player_avg_games <= 0 and injury_count > 0:
                st.warning("Note: Player has previous injuries but zero average games missed. Adjusting for prediction.")
            
            # Get all possible values from training data
            known_injuries = set(models['encoders']['injury_clean'].classes_)
            known_reasons = set(models['encoders']['injury_reason'].classes_)
            
            # Check if selected injury type is known
            if str(injury_type) not in known_injuries:
                st.warning(f"Warning: '{injury_type}' was not seen in training data. Using default encoding.")
                injury_encoded = 0
                injury_reason_encoded = 0
            else:
                injury_encoded = models['encoders']['injury_clean'].transform([str(injury_type)])[0]
                
                if str(injury_type) in known_reasons:
                    injury_reason_encoded = models['encoders']['injury_reason'].transform([str(injury_type)])[0]
                else:
                    injury_reason_encoded = injury_encoded
            
            # Adjust player averages based on injury count and severity
            adjusted_avg_duration = player_avg_duration
            adjusted_avg_games = player_avg_games
            
            # If player has multiple injuries, increase expected recovery
            if injury_count >= 5:
                adjusted_avg_duration = player_avg_duration * 1.2
                adjusted_avg_games = player_avg_games * 1.15
            elif injury_count >= 3:
                adjusted_avg_duration = player_avg_duration * 1.1
                adjusted_avg_games = player_avg_games * 1.08
            
            # Adjust based on last injury severity
            if last_severity_score >= 3:
                adjusted_avg_duration = adjusted_avg_duration * 1.15
            elif last_severity_score >= 2:
                adjusted_avg_duration = adjusted_avg_duration * 1.05
            
            input_data = pd.DataFrame({
                'injury_reason': [injury_reason_encoded],
                'injury_clean': [injury_encoded],
                'month': [injury_month],
                'start_month': [injury_month],
                'end_month': [injury_month],
                'injury_month': [injury_month],
                'injury_dayofweek': [injury_dayofweek],
                'injury_day': [injury_dayofweek],
                'type': [0],
                'player_avg_games_missed': [adjusted_avg_games],
                'player_avg_injury_duration': [adjusted_avg_duration],
                'injury_count_player': [injury_count],
                'injury_duration': [adjusted_avg_duration],
                'days_missed': [0]
            })
            
            # Add any missing columns with default value 0
            for col in models['feature_names']:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Ensure correct column order
            input_data = input_data[models['feature_names']]
            
            # Clean the data
            input_data = input_data.fillna(0)
            input_data = input_data.replace([np.inf, -np.inf], 0)
            
            # Clip values to reasonable ranges
            if 'player_avg_injury_duration' in input_data.columns:
                input_data['player_avg_injury_duration'] = input_data['player_avg_injury_duration'].clip(0, 365)
            if 'player_avg_games_missed' in input_data.columns:
                input_data['player_avg_games_missed'] = input_data['player_avg_games_missed'].clip(0, 365)
            
            # Make prediction
            pred_log = models['reg_model'].predict(input_data)[0]
            pred_days = np.expm1(pred_log)
            pred_days = max(0, min(365, pred_days))
            
            # Adjust prediction based on player history
            if injury_count >= 5:
                pred_days = pred_days * 1.15
            elif injury_count >= 3:
                pred_days = pred_days * 1.08
            
            # Cap at reasonable maximum
            pred_days = min(365, pred_days)
            
            # Classification
            if pred_days <= 7:
                severity = "Short (≤7 days)"
                severity_color = "green"
                recovery_expectation = "Expected to return within a week"
            elif pred_days <= 30:
                severity = "Medium (8-30 days)"
                severity_color = "orange"
                recovery_expectation = "Expected to return within 1-4 weeks"
            elif pred_days <= 90:
                severity = "Long (31-90 days)"
                severity_color = "red"
                recovery_expectation = "Expected to return within 1-3 months"
            else:
                severity = "Severe (>90 days)"
                severity_color = "darkred"
                recovery_expectation = "Expected to return after 3+ months"
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            
            with result_col1:
                st.metric("Predicted Days Missed", f"{pred_days:.0f} days")
            
            with result_col2:
                st.metric("Severity Classification", severity)
            
            with result_col3:
                confidence = min(95, max(50, 100 - (pred_days / 15)))
                st.metric("Prediction Confidence", f"{confidence:.0f}%")
            
            with result_col4:
                st.metric("Risk Factor", f"{injury_count} prev injuries")
            
            st.info(f"{recovery_expectation}")
            
            # Show player history impact
            st.markdown("---")
            st.subheader("Player History Impact Analysis")
            
            hist_col1, hist_col2, hist_col3 = st.columns(3)
            with hist_col1:
                st.metric("Previous Injuries", f"{injury_count}", delta="Key risk factor")
            with hist_col2:
                st.metric("Avg Recovery History", f"{player_avg_duration:.0f} days", delta="Historical pattern")
            with hist_col3:
                if injury_count >= 3:
                    st.metric("Recurrence Risk", "High", delta="Multiple injuries")
                else:
                    st.metric("Recurrence Risk", "Moderate", delta="Limited history")
            
            if pred_days > 30:
                st.warning("This is predicted to be a significant injury requiring extended recovery time")
            elif pred_days > 14:
                st.info("This is predicted to be a moderate injury")
            else:
                st.success("This is predicted to be a minor injury")
            
            # Show factor importance
            st.markdown("---")
            st.subheader("Factors Influencing This Prediction")
            
            factors = []
            if injury_count >= 3:
                factors.append(f"• Multiple previous injuries ({injury_count}): +{(pred_days/player_avg_duration - 1)*100:.0f}% recovery time")
            if player_avg_duration > 30:
                factors.append(f"• Historical long recovery pattern: +15% expected duration")
            if last_severity_score >= 2:
                factors.append(f"• Severe last injury: +10% expected recovery")
            if injury_month in [7, 8, 9]:
                factors.append(f"• Pre-season injury: May affect season start")
            if len(factors) == 0:
                factors.append("• First-time injury with average recovery pattern")
            
            for factor in factors:
                st.write(factor)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check all inputs and try again")