# =============================================================================
# BUILD-UP TIME PREDICTION SYSTEM
# Jeddah Cargo Export — Aviation Logistics Intelligence Platform
# =============================================================================
# Exact columns from code.ipynb (Cell 3 output):
#   Date, Year, Month, Season, Origin, Destination, Flight_Type,
#   Flight_ID, Aircraft_Type, Shipment_Count, Cargo_Weight_KG,
#   Cargo_Volume_CBM, ULD_Count, ULD_Type, Pallet_Count,
#   Nature_of_Goods, Cargo_Mix_Complexity, Manpower_Assigned,
#   Forklift_Count, Equipment_Availability, Shift, Weather_Condition,
#   Build_Up_Time_Minutes
#
# Model   : XGBoost Regressor  (build_up_time_model.pkl)
# Target  : Build_Up_Time_Minutes
# Dropped : Date, Flight_ID, Build_Up_Time_Minutes
# Encoding: LabelEncoder on: Season, Origin, Destination, Flight_Type,
#            Aircraft_Type, ULD_Type, Nature_of_Goods, Cargo_Mix_Complexity,
#            Equipment_Availability, Shift, Weather_Condition
# Selector: Flight_ID  (e.g. SV2587)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Build-Up Time Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    model           = joblib.load("models/build_up_time_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    label_encoders  = joblib.load("models/label_encoders.pkl")
    return model, feature_columns, label_encoders

@st.cache_data
def load_data():
    dataset   = pd.read_csv("dataset/JED_Cargo_Export_Dataset.csv")
    test_data = pd.read_csv("dataset/test_dataset.csv")
    return dataset, test_data

try:
    model, feature_columns, label_encoders = load_artifacts()
    dataset, test_data = load_data()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding-top: 1rem; padding-bottom: 0rem; }
.main            { background-color: #f5f7fb; }

/* ── Typography ── */
h1               { color: #0b1b5e; font-weight: 800; }
h2, h3           { color: #0b1b5e; }

/* ── Primary Button ── */
.stButton > button {
    background-color : #1565ff;
    color            : white;
    border-radius    : 10px;
    height           : 52px;
    width            : 100%;
    font-size        : 17px;
    font-weight      : bold;
    border           : none;
    margin-top       : 4px;
    transition       : background 0.2s;
}
.stButton > button:hover { background-color: #0d4ed8; }

/* ── KPI Metric Cards ── */
[data-testid="stMetric"] {
    background    : white;
    padding       : 20px 22px;
    border-radius : 18px;
    border-top    : 5px solid #1565ff;
    box-shadow    : 0px 2px 10px rgba(0,0,0,0.07);
    text-align    : left;
}
[data-testid="stMetricValue"] { font-size: 34px; font-weight: 700; color: #111827; }
[data-testid="stMetricLabel"] { font-size: 14px; color: #6B7280; }

/* ── Prediction Result Cards ── */
.pred-card-green {
    background    : linear-gradient(135deg, #dcfce7, #f0fdf4);
    padding       : 28px 32px;
    border-radius : 14px;
    border-left   : 6px solid #16a34a;
    margin-bottom : 16px;
}
.pred-card-yellow {
    background    : linear-gradient(135deg, #fef9c3, #fefce8);
    padding       : 28px 32px;
    border-radius : 14px;
    border-left   : 6px solid #ca8a04;
    margin-bottom : 16px;
}
.pred-card-red {
    background    : linear-gradient(135deg, #fee2e2, #fff1f2);
    padding       : 28px 32px;
    border-radius : 14px;
    border-left   : 6px solid #dc2626;
    margin-bottom : 16px;
}
.pred-title {
    font-size      : 13px;
    font-weight    : 600;
    color          : #6B7280;
    text-transform : uppercase;
    letter-spacing : 0.07em;
    margin-bottom  : 6px;
}
.pred-value {
    font-size   : 46px;
    font-weight : 800;
    color       : #111827;
    line-height : 1.1;
}
.pred-sub  { font-size: 18px; color: #374151; margin-top: 6px; }
.pred-risk { font-size: 20px; font-weight: 700; margin-top: 10px; }

/* ── Section Headers ── */
.section-header {
    font-size      : 18px;
    font-weight    : 700;
    color          : #0b1b5e;
    border-bottom  : 2px solid #1565ff;
    padding-bottom : 6px;
    margin-bottom  : 14px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    height        : 46px;
    padding       : 0 24px;
    border-radius : 8px 8px 0 0;
    font-size     : 15px;
    font-weight   : 600;
}
.stTabs [aria-selected="true"] { background-color: #1565ff; color: white; }

/* ── Table ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Inputs ── */
.stSelectbox div[data-baseweb="select"] > div { min-height: 42px; }
.stTextInput input { height: 42px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<h1 style='font-size:38px; color:#1d2340; font-weight:700; margin-bottom:2px;'>
Build-Up Time Prediction System
</h1>
<p style='color:#6B7280; font-size:16px; margin-bottom:0;'>
Jeddah Cargo Export — Aviation Logistics Intelligence Platform
</p>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3,tab4 = st.tabs([
    "Build-Up Time Prediction",
    "EDA",
    "Analytics Dashboard",
    "Bulk Build-Up Time Analysis"
])


# ═══════════════════════════════════════════════════════════
# TAB 1 — BUILD-UP TIME PREDICTION
# ═══════════════════════════════════════════════════════════

with tab1:

    # ── Flight Number Selector ────────────────────────────
    st.markdown("<div class='section-header'>Select Flight</div>", unsafe_allow_html=True)

    selected_flight = st.selectbox(
        "Flight Number",
        test_data["Flight_ID"].unique(),
        help="Select a Flight ID — all shipment fields auto-populate below"
    )

    # Fetch the matching row
    row = test_data[test_data["Flight_ID"] == selected_flight].iloc[0]

    # ── Shipment Details ──────────────────────────────────
    st.markdown("<div class='section-header'>Shipment Details</div>", unsafe_allow_html=True)

    left_col, right_col = st.columns(2)

    # ── LEFT: Shipment Information ────────────────────────
    with left_col:
        st.markdown("**Shipment Information**")

        st.text_input(
            "Date",
            value=str(row["Date"]),
            disabled=True
        )
        st.text_input(
            "Origin",
            value=str(row["Origin"]),
            disabled=True
        )
        st.text_input(
            "Destination",
            value=str(row["Destination"]),
            disabled=True
        )
        st.text_input(
            "Flight Type",
            value=str(row["Flight_Type"]),
            disabled=True
        )
        st.text_input(
            "Aircraft Type",
            value=str(row["Aircraft_Type"]),
            disabled=True
        )
        
        st.text_input(
            "Shipment Count",
            value=str(row["Shipment_Count"]),
            disabled=True
        )
        st.text_input(
            "Cargo Weight (KG)",
            value=str(row["Cargo_Weight_KG"]),
            disabled=True
        )

    # ── RIGHT: Operations Information ─────────────────────
    with right_col:
        st.markdown("**Operations Information**")



        st.text_input(
            "ULD Count",
            value=str(row["ULD_Count"]),
            disabled=True
        )
        st.text_input(
            "Pallet Count",
            value=str(row["Pallet_Count"]),
            disabled=True
        )
        
        st.text_input(
            "Manpower Assigned",
            value=str(row["Manpower_Assigned"]),
            disabled=True
        )
        
        st.text_input(
            "Equipment Count",
            value=str(row["Forklift_Count"]),
            disabled=True
        )

        st.text_input(
            "Shift",
            value=str(row["Shift"]),
            disabled=True
        )
        st.text_input(
            "Season",
            value=str(row["Season"]),
            disabled=True
        )
        st.text_input(
            "Weather Condition",
            value=str(row["Weather_Condition"]),
            disabled=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Button ────────────────────────────────────
    if st.button("Predict Build-Up Time", key="predict_single"):

        try:
            # 1. Build input dataframe from the selected row
            #    Drop columns excluded during training: Date, Flight_ID, Build_Up_Time_Minutes
            drop_train_cols = ["Date", "Flight_ID", "Build_Up_Time_Minutes"]
            input_df = test_data[test_data["Flight_ID"] == selected_flight].copy()
            input_df = input_df.drop(
                columns=[c for c in drop_train_cols if c in input_df.columns],
                errors="ignore"
            )

            # 2. Apply label encoders to categorical columns
            #    Categorical cols: Season, Origin, Destination, Flight_Type, Aircraft_Type,
            #    ULD_Type, Nature_of_Goods, Cargo_Mix_Complexity, Equipment_Availability,
            #    Shift, Weather_Condition
            for col, le in label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except ValueError:
                        input_df[col] = 0

            # 3. Align to training feature order
            final_input = input_df.reindex(columns=feature_columns, fill_value=0)

            # 4. Predict
            prediction = float(model.predict(final_input)[0])

            # 5. Convert minutes → hours + minutes
            hrs  = int(prediction // 60)
            mins = int(prediction  % 60)

            # 6. Risk classification
            if prediction <= 60:
                risk_label = "Low Processing Time"
                card_class = "pred-card-green"
            elif prediction <= 120:
                risk_label = "Medium Processing Time"
                card_class = "pred-card-yellow"
            else:
                risk_label = "High Processing Time"
                card_class = "pred-card-red"

            # ── Display Results ───────────────────────────
            result_col, factor_col = st.columns([1, 1])

            with result_col:
                st.markdown(
                    "<div class='section-header'>Prediction Result</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"""
                <div class='{card_class}'>
                    <div class='pred-title'>Predicted Build-Up Time</div>
                    <div class='pred-value'>{prediction:.0f}
                        <span style='font-size:22px; font-weight:400;'>Minutes</span>
                    </div>
                    <div class='pred-sub'>⏱ {hrs} Hours {mins:02d} Minutes</div>
                    <div class='pred-risk'>{risk_label}</div>
                </div>
                """, unsafe_allow_html=True)

            with factor_col:
                st.markdown(
                    "<div class='section-header'>Key Factors</div>",
                    unsafe_allow_html=True
                )
                key_factors = pd.DataFrame({
                    "Factor": [
                        "Cargo Weight (KG)",
                        "Cargo Volume (CBM)",
                        "ULD Type",
                        "ULD Count",
                        "Manpower Assigned",
                        "Forklift Count",
                        "Shift",
                        "Season"
                    ],
                    "Value": [
                        str(row["Cargo_Weight_KG"]),
                        str(row["Cargo_Volume_CBM"]),
                        str(row["ULD_Type"]),
                        str(row["ULD_Count"]),
                        str(row["Manpower_Assigned"]),
                        str(row["Forklift_Count"]),
                        str(row["Shift"]),
                        str(row["Season"])
                    ]
                })
                st.dataframe(
                    key_factors,
                    use_container_width=True,
                    hide_index=True
                )

            # Persist result in session state
            st.session_state["last_prediction"] = prediction

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.markdown(
            "<div class='section-header'>1. Build-Up Time Distribution</div>",
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(12,6))

        sns.histplot(
            dataset["Build_Up_Time_Minutes"],
            bins=30,
            kde=True,
            ax=ax
        )

        ax.set_title(
            "Distribution of Build-Up Time"
        )

        ax.set_xlabel(
            "Build-Up Time (Minutes)"
        )

        ax.set_ylabel(
            "Frequency"
        )

        st.pyplot(fig)

        
    with right_chart:
        st.markdown(
            "<div class='section-header'>2. Season Impact on Build-Up Time</div>",
            unsafe_allow_html=True
        )

        season_data = (
            dataset
            .groupby("Season")["Build_Up_Time_Minutes"]
            .mean()
            .reset_index()
        )

        season_data = season_data.sort_values(
            "Build_Up_Time_Minutes",
            ascending=False
        )

        fig = px.bar(
            season_data,
            x="Season",
            y="Build_Up_Time_Minutes",
            text_auto=".1f",
            color="Build_Up_Time_Minutes",
            color_continuous_scale="Blues",
        )

        fig.update_layout(
            template="plotly_white",
            height=350,
            coloraxis_showscale=False,
            xaxis_title="Season",
            yaxis_title="Average Build-Up Time (Minutes)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        
            
    st.markdown(
        "<div class='section-header'>3. Skewness Handling Analysis</div>",
        unsafe_allow_html=True
    )
    # Create transformed columns

    dataset["Cargo_Weight_KG_log"] = np.log1p(
        dataset["Cargo_Weight_KG"]
    )

    dataset["Cargo_Volume_CBM_log"] = np.log1p(
        dataset["Cargo_Volume_CBM"]
    )

    dataset["ULD_Count_log"] = np.log1p(
        dataset["ULD_Count"]
    )

    dataset["Pallet_Count_log"] = np.log1p(
        dataset["Pallet_Count"]
    )
    
    features = [
        "Cargo_Weight_KG",
        "Cargo_Volume_CBM",
        "ULD_Count",
        "Pallet_Count",
    ]

    for feature in features:

        st.markdown(
            f"### {feature}"
        )

        col1, col2 = st.columns(2)

        # Before Transformation

        with col1:

            st.markdown("#### Before Transformation")

            fig, ax = plt.subplots(
                figsize=(6,4)
            )

            sns.histplot(
                dataset[feature],
                kde=True,
                ax=ax
            )

            ax.set_title(
                f"Original {feature}"
            )

            st.pyplot(fig)

            st.metric(
                "Skewness",
                round(
                    dataset[feature].skew(),
                    2
                )
            )

        # After Transformation

        with col2:

            st.markdown("#### After Log Transformation")

            fig, ax = plt.subplots(
                figsize=(6,4)
            )

            sns.histplot(
                dataset[f"{feature}_log"],
                kde=True,
                ax=ax
            )

            ax.set_title(
                f"Log Transformed {feature}"
            )

            st.pyplot(fig)

            st.metric(
                "Skewness",
                round(
                    dataset[f"{feature}_log"].skew(),
                    2
                )
            )

        st.markdown("---")
        
    st.markdown(
        "<div class='section-header'>4. Categorical Feature Analysis</div>",
        unsafe_allow_html=True
    )

    categorical_cols = [
        "Season",
        "Flight_Type",
        "Aircraft_Type",
        "ULD_Type",
        "Nature_of_Goods",
        "Equipment_Availability",
        "Shift",
        "Weather_Condition"
    ]

    # Process 3 charts at a time
    for i in range(0, len(categorical_cols), 3):

        cols = st.columns(3)

        for j in range(3):

            if i + j < len(categorical_cols):

                feature = categorical_cols[i + j]

                with cols[j]:

                    st.markdown(
                        f"#### {feature.replace('_',' ')}"
                    )

                    fig, ax = plt.subplots(
                        figsize=(8,5)
                    )

                    sns.countplot(
                        data=dataset,
                        x=feature,
                        ax=ax
                    )
                    for container in ax.containers:
                        ax.bar_label(
                            container,
                            fmt='%d',
                            fontsize=9,
                            fontweight='bold',
                            padding=3
                        )

                    ax.set_xlabel("")
                    ax.set_ylabel("Count")

                    plt.xticks(
                        rotation=45,
                        ha="right"
                    )

                    plt.tight_layout()

                    st.pyplot(fig)

# ═══════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════

with tab3:

    st.markdown(
        "<div class='section-header'>Executive Dashboard</div>",
        unsafe_allow_html=True
    )

    # ── Prepare date column ───────────────────────────────
    dash_data = dataset.copy()
    dash_data["Date"] = pd.to_datetime(dash_data["Date"], errors="coerce")
    ds_start = dash_data["Date"].min()
    ds_end   = dash_data["Date"].max()

    # ── Filters ───────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        from_date = st.date_input(
            "From Date",
            value=ds_start.date(),
            min_value=ds_start.date(),
            max_value=ds_end.date()
        )
    with fc2:
        to_date = st.date_input(
            "To Date",
            value=ds_end.date(),
            min_value=ds_start.date(),
            max_value=ds_end.date()
        )
    with fc3:
        quick_filter = st.selectbox(
            "Quick Filter",
            ["All Time", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last 1 Year"]
        )

    # ── Apply quick filter ────────────────────────────────
    filtered = dash_data.copy()
    today_ts = filtered["Date"].max()

    if quick_filter == "Last 30 Days":
        from_date = (today_ts - pd.Timedelta(days=30)).date()
    elif quick_filter == "Last 90 Days":
        from_date = (today_ts - pd.Timedelta(days=90)).date()
    elif quick_filter == "Last 6 Months":
        from_date = (today_ts - pd.DateOffset(months=6)).date()
    elif quick_filter == "Last 1 Year":
        from_date = (today_ts - pd.DateOffset(years=1)).date()

    filtered = filtered[
        (filtered["Date"] >= pd.to_datetime(from_date)) &
        (filtered["Date"] <= pd.to_datetime(to_date))
    ]

    TARGET = "Build_Up_Time_Minutes"

    # ── KPI Cards ─────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric("Total Shipments",     f"{len(filtered):,}")
    with k2:
        st.metric("Avg Build-Up Time",   f"{filtered[TARGET].mean():.1f} min")
    with k3:
        st.metric("Max Build-Up Time",   f"{filtered[TARGET].max():.0f} min")
    with k4:
        st.metric("Min Build-Up Time",   f"{filtered[TARGET].min():.0f} min")
    with k5:
        st.metric("Avg ULD Utilization", f"{filtered['ULD_Count'].mean():.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart shared layout ────────────────────────────────
    CHART_LAYOUT = dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
        title_font=dict(size=20, color="#111827"),
        font=dict(color="#374151")
    )
    PRIMARY   = "#1565ff"
    BLUES     = ["#1d4ed8","#2563eb","#3b82f6","#60a5fa","#93c5fd","#bfdbfe","#dbeafe"]

    # ── ROW 1 ─────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2)

    # 1. Monthly Average Build-Up Time Trend
    with r1c1:
        monthly = (
            filtered
            .groupby(filtered["Date"].dt.to_period("M"))[TARGET]
            .mean()
            .reset_index()
        )
        monthly["Date"] = monthly["Date"].dt.to_timestamp()
        monthly[TARGET] = monthly[TARGET].round(1)

        fig1 = px.line(
            monthly, x="Date", y=TARGET,
            markers=True,
            title="1. Monthly Avg Build-Up Time Trend"
        )
        fig1.update_traces(
            mode="lines+markers+text",
            text=monthly[TARGET],
            textposition="top center",
            line=dict(color=PRIMARY, width=3),
            marker=dict(size=8, color=PRIMARY),
            textfont=dict(size=11, color="#111827")
        )
        fig1.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Month", showgrid=False, tickfont=dict(size=12)),
            yaxis=dict(title="Avg Build-Up Time (min)", gridcolor="#E5E7EB",
                       tickfont=dict(size=12))
        )
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    # 2. Commodity Type (Nature_of_Goods) vs Avg Build-Up Time
    with r1c2:
        comm_data = (
            filtered.groupby("Nature_of_Goods")[TARGET]
            .mean().reset_index()
            .sort_values(TARGET, ascending=False)
        )
        comm_data[TARGET] = comm_data[TARGET].round(1)

        fig2 = px.bar(
            comm_data, x="Nature_of_Goods", y=TARGET,
            text=TARGET,
            title="2. Commodity Type vs Avg Build-Up Time"
        )
        fig2.update_traces(
            marker_color=BLUES[:len(comm_data)],
            texttemplate="%{text:.1f}",
            textposition="outside"
        )
        fig2.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Commodity Type", tickfont=dict(size=12), tickangle=-20),
            yaxis=dict(title="Avg Build-Up Time (min)", gridcolor="#E5E7EB",
                       tickfont=dict(size=12),
                       range=[0, comm_data[TARGET].max() + 20]),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── ROW 2 ─────────────────────────────────────────────
    r2c1, r2c2, r2c3 = st.columns(3)

    # 3. ULD Type vs Avg Build-Up Time (Horizontal Bar)
    with r2c1:
        uld_data = (
            filtered.groupby("ULD_Type")[TARGET]
            .mean().reset_index()
            .sort_values(TARGET, ascending=True)
        )
        uld_data[TARGET] = uld_data[TARGET].round(1)

        fig3 = px.bar(
            uld_data, x=TARGET, y="ULD_Type",
            orientation="h", text=TARGET,
            color=TARGET,
            color_continuous_scale="Blues",
            title="3. ULD Type vs Avg Build-Up Time"
        )
        fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig3.update_layout(
            **CHART_LAYOUT,
            coloraxis_showscale=False,
            xaxis=dict(title="Avg Build-Up Time (min)", gridcolor="#E5E7EB",
                       tickfont=dict(size=12)),
            yaxis=dict(title="ULD Type", tickfont=dict(size=12))
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # 4. Shift-wise Build-Up Time Distribution (Pie)
    with r2c2:
        shift_data = (
            filtered.groupby("Shift")[TARGET]
            .mean().reset_index()
        )
        shift_data[TARGET] = shift_data[TARGET].round(1)

        fig4 = px.pie(
            shift_data, names="Shift", values=TARGET,
            hole=0.5,
            color_discrete_sequence=BLUES,
            title="4. Shift-wise Build-Up Time"
        )
        fig4.update_traces(
            textposition="outside",
            textinfo="percent+label",
            pull=[0.03] + [0] * (len(shift_data) - 1),
            marker=dict(line=dict(color="white", width=2))
        )
        fig4.update_layout(
            **CHART_LAYOUT,
            showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # 5. Cargo Season vs Avg Build-Up Time
    with r2c3:
        season_data = (
            filtered.groupby("Season")[TARGET]
            .mean().reset_index()
            .sort_values(TARGET, ascending=False)
        )
        season_data[TARGET] = season_data[TARGET].round(1)

        fig5 = px.bar(
            season_data, x="Season", y=TARGET,
            text=TARGET,
            title="5. Cargo Season vs Avg Build-Up Time",
            color=TARGET,
            color_continuous_scale="Blues"
        )
        fig5.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig5.update_layout(
            **CHART_LAYOUT,
            coloraxis_showscale=False,
            xaxis=dict(title="Cargo Season", tickfont=dict(size=12)),
            yaxis=dict(title="Avg Build-Up Time (min)", gridcolor="#E5E7EB",
                       tickfont=dict(size=12),
                       range=[0, season_data[TARGET].max() + 20])
        )
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════
# TAB 4 — BULK BUILD-UP TIME ANALYSIS
# ═══════════════════════════════════════════════════════════

with tab4:

    st.markdown(
        "<div class='section-header'>Bulk Build-Up Time Analysis</div>",
        unsafe_allow_html=True
    )

    if st.button("🚀 Run Bulk Prediction", key="bulk_predict"):

        try:
            # 1. Drop training-excluded columns
            drop_train = ["Date", "Flight_ID"]
            bulk_input = test_data.copy()
            bulk_raw   = test_data.copy()

            bulk_input = bulk_input.drop(
                columns=[c for c in drop_train if c in bulk_input.columns],
                errors="ignore"
            )

            # 2. Apply label encoders
            for col, le in label_encoders.items():
                if col in bulk_input.columns:
                    try:
                        bulk_input[col] = le.transform(bulk_input[col].astype(str))
                    except ValueError:
                        bulk_input[col] = 0

            # 3. Align to training feature order
            bulk_aligned = bulk_input.reindex(columns=feature_columns, fill_value=0)

            # 4. Predict
            predictions = model.predict(bulk_aligned)

            # 5. Build results table
            results = bulk_raw.copy()
            results["Predicted_Build_Up_Time"] = np.round(predictions, 1)

            def classify_risk(val):
                if val <= 150:
                    return "Low"
                elif val <= 200:
                    return "Medium"
                return "High"

            results["Risk_Category"] = results["Predicted_Build_Up_Time"].apply(classify_risk)

            st.session_state["bulk_results"] = results
            st.success(f"✅ Bulk prediction completed for {len(results):,} flights.")

        except Exception as e:
            st.error(f"Bulk prediction failed: {e}")

    # ── Show Results ──────────────────────────────────────
    if "bulk_results" in st.session_state:

        results = st.session_state["bulk_results"]

        # ── KPI Cards ──────────────────────────────────────
        bk1, bk2, bk3, bk4 = st.columns(4)

        with bk1:
            st.metric("Total Flights",         f"{len(results):,}")
        with bk2:
            st.metric("Avg Predicted Time",    f"{results['Predicted_Build_Up_Time'].mean():.1f} min")
        with bk3:
            st.metric("Max Predicted Time",    f"{results['Predicted_Build_Up_Time'].max():.0f} min")
        with bk4:
            st.metric("Min Predicted Time",    f"{results['Predicted_Build_Up_Time'].min():.0f} min")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk Filter ────────────────────────────────────
        risk_filter = st.selectbox(
            "Filter by Time Taking",
            ["All", "Low", "Medium", "High"],
            key="risk_filter"
        )

        display = (
            results if risk_filter == "All"
            else results[results["Risk_Category"] == risk_filter]
        )

        st.info(f"Showing **{len(display):,}** records — Risk: **{risk_filter}**")

        # ── Results Table ──────────────────────────────────
        display_cols = [
            "Flight_ID", "Destination", "ULD_Type", "ULD_Count",
            "Predicted_Build_Up_Time", "Risk_Category"
        ]
        # Keep only columns that actually exist in the dataframe
        display_cols = [c for c in display_cols if c in display.columns]

        st.dataframe(
            display[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        # ── Download ───────────────────────────────────────
        csv_data = display.to_csv(index=False)
        st.download_button(
            label="Download Build-Up Time Report",
            data=csv_data,
            file_name="build_up_time_predictions.csv",
            mime="text/csv"
        )