from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Care Transition Efficiency & Placement Outcome Analytics",
    page_icon="🧭",
    layout="wide",
)

st.title("Care Transition Efficiency & Placement Outcome Analytics")
st.caption(
    "Unified Mentor Internship Project 2 | UAC care-pipeline operational analytics"
)

DATA_PATH = Path(__file__).resolve().parent / "HHS_Care_Transition_Cleaned.csv"

# Use the supplied dataset by default. The sidebar also lets the user upload a replacement.
@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    raw = pd.read_csv(BytesIO(file_bytes))
    return clean_data(raw)

def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "Date",
        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Required column(s) not found: " + ", ".join(missing)
            + ". Check that you selected the expected HHS CSV file."
        )

    df = df[required].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = required[1:]
    for col in numeric_cols:
        # HHS Care may contain values like "2,484"; remove commas and whitespace.
        df[col] = (
            df[col].astype("string").str.replace(",", "", regex=False).str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows that are completely blank or have no usable date.
    df = df.dropna(subset=["Date"])
    # Remove rows where every metric is missing; retain partial rows for transparency.
    df = df.dropna(subset=numeric_cols, how="all")
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df = df.reset_index(drop=True)
    return df

# Load the project CSV from the same folder as app.py unless an upload is provided.
with st.sidebar:
    st.header("Data & Filters")
    uploaded_file = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    st.markdown("---")
    st.caption("Default file: HHS_Care_Transition_Cleaned.csv")

try:
    if uploaded_file is not None:
        df = load_csv(uploaded_file.getvalue())
        data_source = "Uploaded CSV"
    elif DATA_PATH.exists():
        df = load_csv(DATA_PATH.read_bytes())
        data_source = DATA_PATH.name
    else:
        st.error(
            "Dataset not found. Put HHS_Care_Transition_Cleaned.csv "
            "in the same folder as app.py, or upload a CSV in the sidebar."
        )
        st.stop()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if df.empty:
    st.error("No usable dated rows were found in this CSV.")
    st.stop()

numeric_cols = [
    "Children apprehended and placed in CBP custody*",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care",
]
apprehended_col = numeric_cols[0]
cbp_stock_col = numeric_cols[1]
transferred_col = numeric_cols[2]
hhs_stock_col = numeric_cols[3]
discharged_col = numeric_cols[4]

with st.sidebar:
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    selected_dates = st.date_input(
        "Reporting date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    st.markdown("---")
    st.subheader("Display options")
    show_rolling = st.checkbox("Show 7-report rolling average", value=True)
    ratio_mode = st.selectbox(
        "Efficiency ratio display",
        ["Percent", "Raw ratio"],
        index=0,
    )
    alert_transfer = st.slider(
        "Transfer ratio alert threshold (%)", 0, 100, 25, 5
    )
    alert_discharge = st.slider(
        "Discharge ratio alert threshold (%)", 0, 100, 5, 1
    )

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates

filtered = df[
    (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
].copy()
if filtered.empty:
    st.warning("No records exist in the selected date range. Change the date filter.")
    st.stop()

# Explicit definitions: ratios compare daily flows to same-day reported stock.
# They are operational indicators, not measured elapsed-time/speed metrics.
filtered["Transfer efficiency ratio"] = (
    filtered[transferred_col] / filtered[cbp_stock_col].replace(0, np.nan)
)
filtered["Discharge effectiveness ratio"] = (
    filtered[discharged_col] / filtered[hhs_stock_col].replace(0, np.nan)
)
# Throughput proxy compares completed HHS discharges with daily CBP apprehensions.
filtered["Pipeline throughput ratio"] = (
    filtered[discharged_col] / filtered[apprehended_col].replace(0, np.nan)
)
# Net change pressure is a flow balance, not a reconstructed count of all children in care.
filtered["CBP net flow"] = filtered[apprehended_col] - filtered[transferred_col]
filtered["HHS net flow"] = filtered[transferred_col] - filtered[discharged_col]
filtered["Pipeline net flow"] = filtered[apprehended_col] - filtered[discharged_col]
filtered["Transfer ratio (%)"] = filtered["Transfer efficiency ratio"] * 100
filtered["Discharge ratio (%)"] = filtered["Discharge effectiveness ratio"] * 100
filtered["Throughput ratio (%)"] = filtered["Pipeline throughput ratio"] * 100

st.info(
    f"Data source: **{data_source}** · {len(df):,} usable dated records after cleaning · "
    f"Showing {len(filtered):,} records from {start_date} to {end_date}."
)

# Quality checks
raw_rows = None
try:
    raw_rows = len(pd.read_csv(DATA_PATH)) if uploaded_file is None and DATA_PATH.exists() else None
except Exception:
    pass
missing_cells = int(filtered[numeric_cols].isna().sum().sum())
duplicate_dates = int(filtered["Date"].duplicated().sum())
negative_values = int((filtered[numeric_cols] < 0).sum().sum())

with st.expander("Data quality checks", expanded=False):
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows in selected period", f"{len(filtered):,}")
    q2.metric("Missing numeric cells", f"{missing_cells:,}")
    q3.metric("Duplicate dates after cleaning", f"{duplicate_dates:,}")
    q4.metric("Negative metric values", f"{negative_values:,}")
    st.write(
        "Cleaning performed: blank rows removed, dates parsed, comma-formatted numbers "
        "converted to numeric values, rows sorted by date, and duplicate dates reduced "
        "to the last row. Review source records if missing values or negative values appear."
    )

def safe_sum(series):
    return float(series.sum(min_count=1)) if series.notna().any() else np.nan

def safe_ratio(numerator, denominator):
    den = safe_sum(denominator)
    num = safe_sum(numerator)
    if pd.isna(den) or den == 0 or pd.isna(num):
        return np.nan
    return num / den

total_apprehended = safe_sum(filtered[apprehended_col])
total_transferred = safe_sum(filtered[transferred_col])
total_discharged = safe_sum(filtered[discharged_col])
latest = filtered.sort_values("Date").iloc[-1]

st.subheader("Executive overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Children apprehended (period total)", "—" if pd.isna(total_apprehended) else f"{total_apprehended:,.0f}")
c2.metric("Children transferred (period total)", "—" if pd.isna(total_transferred) else f"{total_transferred:,.0f}")
c3.metric("Children discharged (period total)", "—" if pd.isna(total_discharged) else f"{total_discharged:,.0f}")
c4.metric("Latest reported HHS care load", "—" if pd.isna(latest[hhs_stock_col]) else f"{latest[hhs_stock_col]:,.0f}")

transfer_period_ratio = safe_ratio(filtered[transferred_col], filtered[cbp_stock_col])
discharge_period_ratio = safe_ratio(filtered[discharged_col], filtered[hhs_stock_col])
throughput_period_ratio = safe_ratio(filtered[discharged_col], filtered[apprehended_col])

def ratio_text(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}" if ratio_mode == "Percent" else f"{value:.3f}x"

st.subheader("Efficiency indicators")
r1, r2, r3 = st.columns(3)
r1.metric("Transfer-to-CBP-stock indicator", ratio_text(transfer_period_ratio))
r2.metric("Discharge-to-HHS-stock indicator", ratio_text(discharge_period_ratio))
r3.metric("Discharges-to-apprehensions proxy", ratio_text(throughput_period_ratio))
st.caption(
    "Interpretation caution: these are aggregate flow-to-stock or flow-to-flow ratios. "
    "They do not measure the number of hours/days an individual child waited, and the "
    "period ratios above are ratios of period sums—not averages of daily percentages."
)

tabs = st.tabs([
    "Care pipeline",
    "Efficiency trends",
    "Backlog & bottlenecks",
    "Placement outcomes",
    "Data table",
])

with tabs[0]:
    st.subheader("Reported care pipeline")
    flow_values = [
        safe_sum(filtered[apprehended_col]),
        safe_sum(filtered[transferred_col]),
        safe_sum(filtered[discharged_col]),
    ]
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=25,
            thickness=22,
            line=dict(width=0.5),
            label=["Apprehended / CBP intake", "Transferred to HHS", "Discharged from HHS"],
        ),
        link=dict(
            source=[0, 1],
            target=[1, 2],
            value=[max(0, flow_values[1]) if not pd.isna(flow_values[1]) else 0,
                   max(0, flow_values[2]) if not pd.isna(flow_values[2]) else 0],
        ),
    ))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.warning(
        "The Sankey chart visualizes transfer and discharge volumes, not a reconciled "
        "one-to-one cohort flow. Children discharged in a period may have entered care earlier."
    )

    stock_cols = [cbp_stock_col, hhs_stock_col]
    stock_df = filtered[["Date"] + stock_cols].melt(
        id_vars="Date", var_name="Stage", value_name="Children reported"
    )
    stock_df["Stage"] = stock_df["Stage"].replace({
        cbp_stock_col: "CBP custody (reported stock)",
        hhs_stock_col: "HHS care (reported stock)",
    })
    fig_stock = px.line(
        stock_df, x="Date", y="Children reported", color="Stage",
        title="Reported active care load over time",
        markers=True,
    )
    fig_stock.update_layout(hovermode="x unified")
    st.plotly_chart(fig_stock, use_container_width=True)

with tabs[1]:
    st.subheader("Transfer and discharge indicators by reporting date")
    trend_df = filtered.sort_values("Date").copy()
    transfer_metric = "Transfer ratio (%)" if ratio_mode == "Percent" else "Transfer efficiency ratio"
    discharge_metric = "Discharge ratio (%)" if ratio_mode == "Percent" else "Discharge effectiveness ratio"
    plot_df = trend_df.melt(
        id_vars="Date",
        value_vars=[transfer_metric, discharge_metric],
        var_name="Indicator",
        value_name="Value",
    )
    fig = px.line(plot_df, x="Date", y="Value", color="Indicator", markers=True,
                  title="Daily flow-to-stock indicators")
    fig.add_hline(
        y=alert_transfer if ratio_mode == "Percent" else alert_transfer / 100,
        line_dash="dash", annotation_text="Transfer alert threshold"
    )
    fig.add_hline(
        y=alert_discharge if ratio_mode == "Percent" else alert_discharge / 100,
        line_dash="dot", annotation_text="Discharge alert threshold"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if show_rolling:
        roll = trend_df[["Date", "Transfer ratio (%)", "Discharge ratio (%)"]].copy()
        roll["Transfer 7-report average (%)"] = roll["Transfer ratio (%)"].rolling(7, min_periods=1).mean()
        roll["Discharge 7-report average (%)"] = roll["Discharge ratio (%)"].rolling(7, min_periods=1).mean()
        roll_plot = roll.melt(
            id_vars="Date",
            value_vars=["Transfer 7-report average (%)", "Discharge 7-report average (%)"],
            var_name="Indicator",
            value_name="7-report average (%)",
        )
        fig_roll = px.line(roll_plot, x="Date", y="7-report average (%)", color="Indicator",
                           title="Rolling average across 7 reported records")
        st.plotly_chart(fig_roll, use_container_width=True)
        st.caption("A 7-report average means seven available records, not necessarily seven consecutive calendar days.")

with tabs[2]:
    st.subheader("Flow balance and potential bottlenecks")
    net_cols = ["CBP net flow", "HHS net flow", "Pipeline net flow"]
    net_df = filtered[["Date"] + net_cols].melt(
        id_vars="Date", var_name="Flow balance", value_name="Net children per report"
    )
    fig_net = px.bar(net_df, x="Date", y="Net children per report", color="Flow balance",
                     barmode="group", title="Daily flow balance (inflow minus outflow)")
    fig_net.add_hline(y=0, line_color="gray")
    st.plotly_chart(fig_net, use_container_width=True)

    st.markdown(
        "- **CBP net flow = apprehensions − transfers.** Positive values indicate intake exceeded transfers in that report.\n"
        "- **HHS net flow = transfers − discharges.** Positive values indicate transfers exceeded discharges in that report.\n"
        "- **Pipeline net flow = apprehensions − discharges.** Positive values indicate intake exceeded exits during that report.\n\n"
        "These are flow-pressure indicators, not verified changes in total backlog. Daily stock changes may not reconcile because of reporting timing, other movements, or incomplete records."
    )
    flagged = filtered[
        (filtered["Transfer ratio (%)"] < alert_transfer)
        | (filtered["Discharge ratio (%)"] < alert_discharge)
    ].copy()
    st.write(f"**Reports below at least one selected threshold:** {len(flagged):,}")
    if not flagged.empty:
        st.dataframe(
            flagged[["Date", "Transfer ratio (%)", "Discharge ratio (%)", "CBP net flow", "HHS net flow"]]
            .sort_values("Date", ascending=False)
            .round(2),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No report in this date range fell below the selected ratio thresholds.")

with tabs[3]:
    st.subheader("Placement / discharge outcomes")
    daily = filtered.sort_values("Date").copy()
    fig_dis = px.line(daily, x="Date", y=discharged_col, markers=True,
                      title="Children discharged from HHS care by reporting date")
    st.plotly_chart(fig_dis, use_container_width=True)

    daily["Month"] = daily["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = daily.groupby("Month", as_index=False).agg(
        Apprehensions=(apprehended_col, "sum"),
        Transfers=(transferred_col, "sum"),
        Discharges=(discharged_col, "sum"),
        Mean_HHS_care=(hhs_stock_col, "mean"),
        Mean_CBP_custody=(cbp_stock_col, "mean"),
    )
    monthly_long = monthly.melt(
        id_vars="Month",
        value_vars=["Apprehensions", "Transfers", "Discharges"],
        var_name="Measure",
        value_name="Children",
    )
    fig_month = px.line(monthly_long, x="Month", y="Children", color="Measure", markers=True,
                        title="Monthly intake, transfers and discharges")
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("**Monthly summary table**")
    st.dataframe(monthly.round(2), use_container_width=True, hide_index=True)

    valid_daily = daily["Discharge ratio (%)"].dropna()
    if len(valid_daily) >= 2:
        mean_ratio = valid_daily.mean()
        std_ratio = valid_daily.std()
        cv = std_ratio / mean_ratio if mean_ratio != 0 else np.nan
        st.metric("Daily discharge-ratio coefficient of variation", "N/A" if pd.isna(cv) else f"{cv:.2f}")
        st.caption(
            "Lower variability can indicate more consistent ratios, but does not prove better "
            "reunification outcomes. This dataset records aggregate discharges, not case-level outcomes."
        )
    else:
        st.info("Not enough valid records to calculate discharge-ratio variability.")

with tabs[4]:
    st.subheader("Cleaned records")
    display_df = filtered.copy()
    display_df["Transfer ratio (%)"] = display_df["Transfer ratio (%)"].round(2)
    display_df["Discharge ratio (%)"] = display_df["Discharge ratio (%)"].round(2)
    display_df["Throughput ratio (%)"] = display_df["Throughput ratio (%)"].round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_bytes,
        file_name="care_transition_filtered_data.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Responsible-use note: this dashboard supports aggregate operational analysis. "
    "It does not assess individual children, predict individual placement, or establish causation. "
    "Use verified official definitions and reporting notes before making policy decisions."
)
