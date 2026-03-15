import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os

st.set_page_config(
    page_title="DataCo Supply Chain Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.kpi-card {
    background: linear-gradient(135deg, #1a1f35, #212840);
    border: 1px solid #2d3555; border-radius: 14px;
    padding: 18px 22px; text-align: center;
}
.kpi-val  { font-size: 1.9rem; font-weight: 700; color: #60a5fa; }
.kpi-lbl  { font-size: 0.78rem; color: #94a3b8; margin-top: 4px;
             letter-spacing: 0.04em; text-transform: uppercase; }
.sec-hdr  { font-size: 1rem; font-weight: 600; color: #e2e8f0;
             border-left: 3px solid #3b82f6; padding-left: 10px; margin-bottom: 12px; }
.pred-box { border-radius: 12px; padding: 20px; text-align: center; margin-top: 8px; }
.pred-risk   { background: linear-gradient(135deg,#3b1a1a,#4a2020); border:1px solid #ef4444; }
.pred-safe   { background: linear-gradient(135deg,#1a3b1a,#204a20); border:1px solid #22c55e; }
.pred-demand { background: linear-gradient(135deg,#1a2a3b,#1e3550); border:1px solid #3b82f6; }
.pred-val { font-size: 2rem; font-weight: 700; }
.pred-lbl { font-size: 0.82rem; color: #94a3b8; margin-top: 6px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 18px;
    background: #1e2336; color: #94a3b8; }
.stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }
.rec-card { border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; }
.rec-critical { background: linear-gradient(135deg,#3b1212,#4a1a1a); border-left: 4px solid #ef4444; }
.rec-warning  { background: linear-gradient(135deg,#3b2a12,#4a3218); border-left: 4px solid #f59e0b; }
.rec-success  { background: linear-gradient(135deg,#12301a,#183d22); border-left: 4px solid #22c55e; }
.rec-info     { background: linear-gradient(135deg,#12203b,#182844); border-left: 4px solid #3b82f6; }
.rec-title    { font-size: 1rem; font-weight: 700; margin-bottom: 6px; }
.rec-body     { font-size: 0.85rem; color: #cbd5e1; line-height: 1.6; }
.rec-tag      { display: inline-block; border-radius: 6px; padding: 2px 10px;
                font-size: 0.72rem; font-weight: 600; margin-right: 6px; margin-bottom: 4px; }
.tag-red   { background: #7f1d1d; color: #fca5a5; }
.tag-amber { background: #78350f; color: #fcd34d; }
.tag-green { background: #14532d; color: #86efac; }
.tag-blue  { background: #1e3a5f; color: #93c5fd; }
.quadrant-box { border-radius: 14px; padding: 20px; text-align: center; }
.q-rr { background: linear-gradient(135deg,#3b1a1a,#4a2020); border: 2px solid #ef4444; }
.q-ro { background: linear-gradient(135deg,#3b2a12,#4a3218); border: 2px solid #f59e0b; }
.q-gr { background: linear-gradient(135deg,#182844,#1e3550); border: 2px solid #3b82f6; }
.q-gg { background: linear-gradient(135deg,#12301a,#183d22); border: 2px solid #22c55e; }
.q-label { font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; }
.q-sub   { font-size: 0.82rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Static label-encoding maps (alphabetical = sklearn LabelEncoder default) ──
SHIPPING_MODE_ENC = {'First Class': 0, 'Same Day': 1, 'Second Class': 2, 'Standard Class': 3}
MARKET_ENC        = {'Africa': 0, 'Europe': 1, 'LATAM': 2, 'Pacific Asia': 3, 'USCA': 4}
SEGMENT_ENC       = {'Consumer': 0, 'Corporate': 1, 'Home Office': 2}
TYPE_ENC          = {'CASH': 0, 'DEBIT': 1, 'PAYMENT': 2, 'TRANSFER': 3}
ORDER_STATUS_ENC  = {
    'CANCELED': 0, 'CLOSED': 1, 'COMPLETE': 2, 'ON_HOLD': 3,
    'PAYMENT_REVIEW': 4, 'PENDING': 5, 'PENDING_PAYMENT': 6,
    'PROCESSING': 7, 'SUSPECTED_FRAUD': 8
}
DEPARTMENT_ENC = {
    'Apparel': 0, 'Book Shop': 1, 'Discs Shop': 2, 'Fan Shop': 3,
    'Fitness': 4, 'Footwear': 5, 'Golf': 6, 'Health and Beauty': 7,
    'Outdoors': 8, 'Pet Shop': 9, 'Technology': 10
}
REGION_ENC = {
    'Caribbean': 0, 'Central Africa': 1, 'Central America': 2, 'Central Asia': 3,
    'East Africa': 4, 'East Asia': 5, 'Eastern Europe': 6, 'North Africa': 7,
    'North America': 8, 'Oceania': 9, 'South America': 10, 'South Asia': 11,
    'Southeast Asia': 12, 'Southern Africa': 13, 'US Center': 14, 'US East': 15,
    'US South': 16, 'US West': 17, 'West Africa': 18, 'West Asia': 19, 'Western Europe': 20
}

# Pre-computed per-shipping-mode stats from training data distribution
SHIPPING_STATS = {
    'First Class':    {'ShippingMode_Avg_Real_Days': 2.18, 'ShippingMode_Late_Delivery_Rate': 0.064, 'ShippingMode_Avg_Scheduled_Days': 2.0},
    'Same Day':       {'ShippingMode_Avg_Real_Days': 1.57, 'ShippingMode_Late_Delivery_Rate': 0.001, 'ShippingMode_Avg_Scheduled_Days': 0.0},
    'Second Class':   {'ShippingMode_Avg_Real_Days': 4.77, 'ShippingMode_Late_Delivery_Rate': 0.529, 'ShippingMode_Avg_Scheduled_Days': 2.0},
    'Standard Class': {'ShippingMode_Avg_Real_Days': 5.44, 'ShippingMode_Late_Delivery_Rate': 0.063, 'ShippingMode_Avg_Scheduled_Days': 4.0},
}
SEGMENT_STATS = {
    'Consumer':    {'Segment_Late_Delivery_Rate': 0.55, 'Segment_Avg_Quantity': 2.5, 'Segment_Avg_Sales': 210.0},
    'Corporate':   {'Segment_Late_Delivery_Rate': 0.54, 'Segment_Avg_Quantity': 2.6, 'Segment_Avg_Sales': 220.0},
    'Home Office': {'Segment_Late_Delivery_Rate': 0.54, 'Segment_Avg_Quantity': 2.4, 'Segment_Avg_Sales': 200.0},
}

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "delivery_model.pkl"), "rb") as f:
        delivery = pickle.load(f)
    with open(os.path.join(base, "demand_model.pkl"), "rb") as f:
        demand = pickle.load(f)
    return delivery, demand

delivery_model, demand_model = load_models()
DELIVERY_FEATURES = list(delivery_model.feature_names_in_)
DEMAND_FEATURES   = list(demand_model.feature_names_in_)

# ── Encode DataFrame for batch predictions using static maps ──────────────────
def encode_df_for_model(df, features):
    """
    Use static alphabetical label encoding for known categoricals.
    High-cardinality cols (Product, Category, Country, City) use pd.factorize
    so they at least vary consistently across rows.
    """
    tmp = df.copy()
    static_maps = {
        'Shipping Mode':    SHIPPING_MODE_ENC,
        'Market':           MARKET_ENC,
        'Customer Segment': SEGMENT_ENC,
        'Type':             TYPE_ENC,
        'Order Status':     ORDER_STATUS_ENC,
        'Department Name':  DEPARTMENT_ENC,
        'Order Region':     REGION_ENC,
    }
    high_card = ['Product Name', 'Category Name', 'Order Country', 'Order City']

    for col in features:
        if col not in tmp.columns:
            tmp[col] = 0
            continue
        if col in static_maps:
            tmp[col] = tmp[col].map(static_maps[col]).fillna(0).astype(int)
        elif col in high_card and tmp[col].dtype == object:
            tmp[col] = pd.factorize(tmp[col])[0]

    return tmp[features].fillna(0)


def run_delivery_predictions(df):
    enc = encode_df_for_model(df, DELIVERY_FEATURES)
    return delivery_model.predict(enc), delivery_model.predict_proba(enc)[:, 1]


def run_demand_predictions(df):
    enc = encode_df_for_model(df, DEMAND_FEATURES)
    return demand_model.predict(enc)


# ── Build pre-encoded single-row dicts for live predictor ────────────────────
def build_delivery_row(inp):
    ss   = SHIPPING_STATS[inp['shipping_mode']]
    segs = SEGMENT_STATS[inp['customer_segment']]
    return pd.DataFrame([{
        'Product Name':                   0,
        'Category Name':                  0,
        'Department Name':                DEPARTMENT_ENC.get(inp['department'], 3),
        'Order Region':                   REGION_ENC.get(inp['region'], 8),
        'Order Country':                  0,
        'Order City':                     0,
        'Customer Segment':               SEGMENT_ENC[inp['customer_segment']],
        'Shipping Mode':                  SHIPPING_MODE_ENC[inp['shipping_mode']],
        'Days for shipment (scheduled)':  inp['days_scheduled'],
        'Days for shipping (real)':       inp['days_real'],
        'Type':                           TYPE_ENC.get(inp['order_type'], 1),
        'Order Status':                   ORDER_STATUS_ENC.get(inp['order_status'], 2),
        'Order Item Quantity':            inp['quantity'],
        'Sales':                          inp['sales'],
        'Product Price':                  inp['product_price'],
        'Demand_Spike_Flag':              inp['demand_spike'],
        'High_Demand_Risk_Flag':          inp['high_demand_risk'],
        'Logistics_Load_Score':           inp['logistics_load'],
        'Product_Peak_Hour':              12,
        'Category_Peak_Hour':             12,
        'Hourly_Access_Count':            inp['hourly_access'],
        'Product_Total_Accesses':         50,
        'Category_Total_Accesses':        200,
        'Region_Avg_Shipping_Days':       ss['ShippingMode_Avg_Real_Days'],
        'Region_Order_Count':             500,
        'ShippingMode_Avg_Real_Days':     ss['ShippingMode_Avg_Real_Days'],
        'ShippingMode_Late_Delivery_Rate':ss['ShippingMode_Late_Delivery_Rate'],
        'ShippingMode_Avg_Scheduled_Days':ss['ShippingMode_Avg_Scheduled_Days'],
        'Segment_Late_Delivery_Rate':     segs['Segment_Late_Delivery_Rate'],
        'Market':                         MARKET_ENC.get(inp['market'], 4),
        'Benefit per order':              inp['benefit'],
        'Order Item Profit Ratio':        inp['profit_ratio'],
    }])


def build_demand_row(inp):
    ss   = SHIPPING_STATS[inp['shipping_mode']]
    segs = SEGMENT_STATS[inp['customer_segment']]
    return pd.DataFrame([{
        'Product Name':                   0,
        'Category Name':                  0,
        'Department Name':                DEPARTMENT_ENC.get(inp['department'], 3),
        'Order Region':                   REGION_ENC.get(inp['region'], 8),
        'Order Country':                  0,
        'Order City':                     0,
        'Customer Segment':               SEGMENT_ENC[inp['customer_segment']],
        'Product_Total_Accesses':         50,
        'Product_Avg_Hour':               12,
        'Product_Peak_Hour':              12,
        'Product_Demand_CV':              inp['demand_cv'],
        'Product_Active_Days':            inp['active_days'],
        'Category_Total_Accesses':        200,
        'Category_Avg_Hour':              12,
        'Category_Peak_Hour':             12,
        'Dept_Total_Accesses':            500,
        'Hourly_Access_Count':            inp['hourly_access'],
        'Region_Total_Quantity':          1000,
        'Region_Avg_Quantity_Per_Order':  inp['region_avg_qty'],
        'Region_Total_Sales':             50000,
        'Region_Order_Count':             500,
        'Segment_Avg_Quantity':           segs['Segment_Avg_Quantity'],
        'Segment_Avg_Sales':              segs['Segment_Avg_Sales'],
        'Sales':                          inp['sales'],
        'Shipping Mode':                  SHIPPING_MODE_ENC[inp['shipping_mode']],
        'Product Price':                  inp['product_price'],
        'Order Item Discount':            inp['discount'],
        'Order Item Discount Rate':       inp['discount_rate'],
        'Demand_Spike_Flag':              inp['demand_spike'],
        'Logistics_Load_Score':           inp['logistics_load'],
        'High_Demand_Risk_Flag':          inp['high_demand_risk'],
        'Product_Popularity_Percentile':  inp['popularity_pct'],
        'Category_Demand_Percentile':     50.0,
        'Market':                         MARKET_ENC.get(inp['market'], 4),
    }])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚚 Supply Chain AI")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload DataCo CSV", type=["csv"])
    st.markdown("---")

st.markdown("# 📦 DataCo Supply Chain Dashboard")

if not uploaded_file:
    st.info("👈 Upload your **DataCo Supply Chain CSV** to get started.")
    st.stop()


@st.cache_data
def load_data(f):
    df = pd.read_csv(f, encoding="latin1")
    for col in ["order date (DateOrders)", "shipping date (DateOrders)"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

df = load_data(uploaded_file)

with st.sidebar:
    st.markdown("### Filters")
    def uniq_vals(col):
        return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []
    sel_market   = st.selectbox("Market",   ["All"] + uniq_vals("Market"))
    sel_category = st.selectbox("Category", ["All"] + uniq_vals("Category Name"))
    sel_segment  = st.selectbox("Customer Segment", ["All"] + uniq_vals("Customer Segment"))

fdf = df.copy()
if sel_market   != "All" and "Market" in fdf.columns:        fdf = fdf[fdf["Market"] == sel_market]
if sel_category != "All" and "Category Name" in fdf.columns: fdf = fdf[fdf["Category Name"] == sel_category]
if sel_segment  != "All" and "Customer Segment" in fdf.columns: fdf = fdf[fdf["Customer Segment"] == sel_segment]

st.markdown(f"**{len(fdf):,} orders** after filters")
st.markdown("---")

tabs = st.tabs(["📊 Overview", "🚦 Delivery Risk", "📈 Demand Forecast", "🎛️ Live Predictor", "🧠 Decision Engine"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    k1,k2,k3,k4,k5 = st.columns(5)
    total_sales    = fdf["Sales"].sum() if "Sales" in fdf.columns else 0
    total_profit   = fdf["Order Profit Per Order"].sum() if "Order Profit Per Order" in fdf.columns else 0
    avg_discount   = fdf["Order Item Discount Rate"].mean()*100 if "Order Item Discount Rate" in fdf.columns else 0
    late_pct       = fdf["Late_delivery_risk"].mean()*100 if "Late_delivery_risk" in fdf.columns else 0
    uniq_customers = fdf["Customer Id"].nunique() if "Customer Id" in fdf.columns else 0

    with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-val">${total_sales/1e6:.1f}M</div><div class="kpi-lbl">Total Sales</div></div>', unsafe_allow_html=True)
    with k2:
        c = "#4ade80" if total_profit >= 0 else "#f87171"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{c}">${total_profit/1e6:.1f}M</div><div class="kpi-lbl">Total Profit</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_discount:.1f}%</div><div class="kpi-lbl">Avg Discount Rate</div></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#f87171">{late_pct:.1f}%</div><div class="kpi-lbl">Late Delivery Risk</div></div>', unsafe_allow_html=True)
    with k5: st.markdown(f'<div class="kpi-card"><div class="kpi-val">{uniq_customers:,}</div><div class="kpi-lbl">Unique Customers</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns([2,1])
    with c1:
        st.markdown('<p class="sec-hdr">Sales Over Time</p>', unsafe_allow_html=True)
        if "order date (DateOrders)" in fdf.columns and "Sales" in fdf.columns:
            ts = fdf.groupby(fdf["order date (DateOrders)"].dt.to_period("M"))["Sales"].sum().reset_index()
            ts["order date (DateOrders)"] = ts["order date (DateOrders)"].astype(str)
            fig = px.area(ts, x="order date (DateOrders)", y="Sales", color_discrete_sequence=["#3b82f6"], template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=270)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<p class="sec-hdr">Delivery Status</p>', unsafe_allow_html=True)
        if "Delivery Status" in fdf.columns:
            ds = fdf["Delivery Status"].value_counts().reset_index()
            ds.columns = ["Status","Count"]
            fig = px.pie(ds, names="Status", values="Count", color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.5, template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=270, legend=dict(font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<p class="sec-hdr">Profit by Category</p>', unsafe_allow_html=True)
        if "Category Name" in fdf.columns and "Order Profit Per Order" in fdf.columns:
            cat = fdf.groupby("Category Name")["Order Profit Per Order"].sum().sort_values().reset_index()
            fig = px.bar(cat, x="Order Profit Per Order", y="Category Name", orientation="h",
                         color="Order Profit Per Order", color_continuous_scale="Blues", template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown('<p class="sec-hdr">Sales by Market</p>', unsafe_allow_html=True)
        if "Market" in fdf.columns and "Sales" in fdf.columns:
            mkt = fdf.groupby("Market")["Sales"].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(mkt, x="Market", y="Sales", color="Sales", color_continuous_scale="Blues", template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    c5,c6 = st.columns(2)
    with c5:
        st.markdown('<p class="sec-hdr">Orders by Shipping Mode</p>', unsafe_allow_html=True)
        if "Shipping Mode" in fdf.columns:
            sm_c = fdf["Shipping Mode"].value_counts().reset_index()
            sm_c.columns = ["Mode","Count"]
            fig = px.bar(sm_c, x="Mode", y="Count", color="Count", color_continuous_scale="Blues", template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=280, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with c6:
        st.markdown('<p class="sec-hdr">Sales vs Profit</p>', unsafe_allow_html=True)
        if "Sales" in fdf.columns and "Order Profit Per Order" in fdf.columns:
            samp = fdf.sample(min(2000,len(fdf)), random_state=42)
            fig = px.scatter(samp, x="Sales", y="Order Profit Per Order",
                             color="Market" if "Market" in fdf.columns else None,
                             opacity=0.5, template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=280)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="sec-hdr">Top 10 Products by Sales</p>', unsafe_allow_html=True)
    if "Product Name" in fdf.columns and "Sales" in fdf.columns:
        top = fdf.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top, x="Sales", y="Product Name", orientation="h",
                     color="Sales", color_continuous_scale="Blues", template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0,r=0,t=10,b=0), height=320,
                          coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Raw Data"):
        st.dataframe(fdf.head(500), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — DELIVERY RISK
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🚦 Delivery Risk Model")
    st.caption("RandomForestClassifier · predicts late delivery risk (1 = Late, 0 = On Time)")
    st.markdown("---")

    with st.spinner("Running delivery predictions…"):
        preds_d, probs_d = run_delivery_predictions(fdf)

    fdf_d = fdf.copy()
    fdf_d["Predicted_Late_Risk"]   = preds_d
    fdf_d["Late_Risk_Probability"] = probs_d

    k1,k2,k3,k4 = st.columns(4)
    n_late = int(preds_d.sum()); total = len(preds_d)
    with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#f87171">{n_late:,}</div><div class="kpi-lbl">Predicted Late</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_late/total*100:.1f}%</div><div class="kpi-lbl">Late Rate</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#fb923c">{probs_d.mean()*100:.1f}%</div><div class="kpi-lbl">Avg Risk Prob</div></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#f87171">{(probs_d>=0.7).sum():,}</div><div class="kpi-lbl">High Risk (≥70%)</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec-hdr">Risk Probability Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(x=probs_d, nbins=40, color_discrete_sequence=["#3b82f6"], template="plotly_dark", labels={"x":"Late Delivery Probability"})
        fig.add_vline(x=0.5, line_dash="dash", line_color="#f87171", annotation_text="Threshold")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<p class="sec-hdr">Late Risk by Shipping Mode</p>', unsafe_allow_html=True)
        if "Shipping Mode" in fdf_d.columns:
            grp = fdf_d.groupby("Shipping Mode")["Late_Risk_Probability"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(grp, x="Shipping Mode", y="Late_Risk_Probability",
                         color="Late_Risk_Probability", color_continuous_scale="RdYlGn_r",
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<p class="sec-hdr">Late Risk by Market</p>', unsafe_allow_html=True)
        if "Market" in fdf_d.columns:
            grp = fdf_d.groupby("Market")["Late_Risk_Probability"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(grp, x="Market", y="Late_Risk_Probability",
                         color="Late_Risk_Probability", color_continuous_scale="RdYlGn_r",
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown('<p class="sec-hdr">Risk Over Time</p>', unsafe_allow_html=True)
        if "order date (DateOrders)" in fdf_d.columns:
            ts = fdf_d.groupby(fdf_d["order date (DateOrders)"].dt.to_period("M"))["Late_Risk_Probability"].mean().reset_index()
            ts["order date (DateOrders)"] = ts["order date (DateOrders)"].astype(str)
            fig = px.line(ts, x="order date (DateOrders)", y="Late_Risk_Probability", color_discrete_sequence=["#f87171"], template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="sec-hdr">Feature Importance — Delivery Model</p>', unsafe_allow_html=True)
    fi = pd.DataFrame({"Feature": DELIVERY_FEATURES, "Importance": delivery_model.feature_importances_})
    fi = fi.sort_values("Importance", ascending=True).tail(15)
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Orders with Predictions"):
        show = [c for c in ["Order Id","Product Name","Shipping Mode","Market","Late_Risk_Probability","Predicted_Late_Risk"] if c in fdf_d.columns]
        st.dataframe(fdf_d[show].sort_values("Late_Risk_Probability", ascending=False).head(300), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — DEMAND FORECAST
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 📈 Demand Forecast Model")
    st.caption("RandomForestRegressor · predicts order item quantity")
    st.markdown("---")

    with st.spinner("Running demand predictions…"):
        preds_q = run_demand_predictions(fdf)

    fdf_q = fdf.copy()
    fdf_q["Predicted_Quantity"] = preds_q
    actual_col = "Order Item Quantity" in fdf_q.columns

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#60a5fa">{preds_q.mean():.2f}</div><div class="kpi-lbl">Avg Predicted Qty</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi-card"><div class="kpi-val">{preds_q.max():.1f}</div><div class="kpi-lbl">Max Predicted Qty</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi-card"><div class="kpi-val">{preds_q.sum():,.0f}</div><div class="kpi-lbl">Total Predicted Demand</div></div>', unsafe_allow_html=True)
    with k4:
        if actual_col:
            mae = np.abs(preds_q - fdf_q["Order Item Quantity"]).mean()
            st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#fb923c">{mae:.2f}</div><div class="kpi-lbl">Mean Abs Error</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{preds_q.min():.1f}</div><div class="kpi-lbl">Min Predicted Qty</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec-hdr">Predicted Quantity Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(x=preds_q, nbins=30, color_discrete_sequence=["#3b82f6"], template="plotly_dark", labels={"x":"Predicted Quantity"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if actual_col:
            st.markdown('<p class="sec-hdr">Actual vs Predicted</p>', unsafe_allow_html=True)
            idx = np.random.choice(len(fdf_q), min(1000, len(fdf_q)), replace=False)
            fig = px.scatter(x=fdf_q["Order Item Quantity"].iloc[idx], y=preds_q[idx],
                             opacity=0.4, color_discrete_sequence=["#3b82f6"], template="plotly_dark",
                             labels={"x":"Actual","y":"Predicted"})
            mn = min(fdf_q["Order Item Quantity"].min(), preds_q.min())
            mx = max(fdf_q["Order Item Quantity"].max(), preds_q.max())
            fig.add_shape(type="line", x0=mn,y0=mn,x1=mx,y1=mx, line=dict(color="#f87171",dash="dash"))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<p class="sec-hdr">Avg Demand by Category</p>', unsafe_allow_html=True)
        if "Category Name" in fdf_q.columns:
            grp = fdf_q.groupby("Category Name")["Predicted_Quantity"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(grp, x="Category Name", y="Predicted_Quantity", color="Predicted_Quantity",
                         color_continuous_scale="Blues", template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown('<p class="sec-hdr">Avg Demand by Market</p>', unsafe_allow_html=True)
        if "Market" in fdf_q.columns:
            grp = fdf_q.groupby("Market")["Predicted_Quantity"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(grp, x="Market", y="Predicted_Quantity", color="Predicted_Quantity",
                         color_continuous_scale="Blues", template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="sec-hdr">Predicted Demand Over Time</p>', unsafe_allow_html=True)
    if "order date (DateOrders)" in fdf_q.columns:
        ts = fdf_q.groupby(fdf_q["order date (DateOrders)"].dt.to_period("M"))["Predicted_Quantity"].sum().reset_index()
        ts["order date (DateOrders)"] = ts["order date (DateOrders)"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["order date (DateOrders)"], y=ts["Predicted_Quantity"], name="Predicted", line=dict(color="#3b82f6")))
        if actual_col:
            act = fdf_q.groupby(fdf_q["order date (DateOrders)"].dt.to_period("M"))["Order Item Quantity"].sum().reset_index()
            act["order date (DateOrders)"] = act["order date (DateOrders)"].astype(str)
            fig.add_trace(go.Scatter(x=act["order date (DateOrders)"], y=act["Order Item Quantity"], name="Actual", line=dict(color="#4ade80", dash="dash")))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="sec-hdr">Feature Importance — Demand Model</p>', unsafe_allow_html=True)
    fi2 = pd.DataFrame({"Feature": DEMAND_FEATURES, "Importance": demand_model.feature_importances_})
    fi2 = fi2.sort_values("Importance", ascending=True).tail(15)
    fig = px.bar(fi2, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Orders with Demand Predictions"):
        show2 = [c for c in ["Order Id","Product Name","Category Name","Market","Order Item Quantity","Predicted_Quantity"] if c in fdf_q.columns]
        st.dataframe(fdf_q[show2].head(300), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🎛️ Live Order Predictor")
    st.caption("Predictions update live from the model. Change any field and hit Predict.")
    st.markdown("---")

    # ── DELIVERY RISK INPUTS ──────────────────────────────────────
    st.markdown("### 🚚 Delivery Risk Model")
    st.caption("Top 3 drivers: **Days for Shipping (Real)**, **Shipping Mode**, **Days Scheduled**")

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        shipping_mode = st.selectbox("Shipping Mode ⭐", list(SHIPPING_MODE_ENC.keys()))
    with dc2:
        # Use mode-specific scheduled days as default so user sees meaningful starting point
        default_sched = int(SHIPPING_STATS[shipping_mode]['ShippingMode_Avg_Scheduled_Days'])
        days_scheduled = st.slider("Days Scheduled ⭐", 0, 10, default_sched)
    with dc3:
        days_real = st.slider("Days for Shipping (Real) ⭐", 0, 15, default_sched + 1)

    dc4, dc5, dc6 = st.columns(3)
    with dc4:
        order_status = st.selectbox("Order Status", list(ORDER_STATUS_ENC.keys()), index=2)
    with dc5:
        order_type = st.selectbox("Payment Type", list(TYPE_ENC.keys()), index=1)
    with dc6:
        benefit = st.number_input("Benefit per Order ($)", value=30.0, step=5.0)

    dc7, dc8 = st.columns(2)
    with dc7:
        profit_ratio = st.slider("Profit Ratio", -1.0, 1.0, 0.1, step=0.01)
    with dc8:
        customer_segment = st.selectbox("Customer Segment", list(SEGMENT_ENC.keys()))

    st.markdown("---")

    # ── DEMAND FORECAST INPUTS ─────────────────────────────────────
    st.markdown("### 📈 Demand Forecast Model")
    st.caption("Top drivers: **Demand Spike Flag**, **Sales / Product Price ratio** (= implied quantity), **Discount**")

    # Show the Sales/Price ratio live so user understands what they're controlling
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        demand_spike = st.selectbox("Demand Spike Flag ⭐", [0, 1],
                                     format_func=lambda x: "🔥 Yes — High Demand" if x else "📉 No — Normal")
    with qc2:
        product_price = st.number_input("Product Price ($) ⭐", value=40.0, step=5.0,
                                         min_value=1.0, max_value=500.0)
    with qc3:
        sales = st.number_input("Sales ($) ⭐", value=160.0, step=10.0,
                                 min_value=1.0, max_value=2000.0)

    # Show implied ratio
    ratio = sales / product_price if product_price > 0 else 0
    ratio_color = "#4ade80" if ratio >= 3 else "#fb923c" if ratio >= 2 else "#f87171"
    st.markdown(
        f"**Implied Sales/Price ratio: "
        f"<span style='color:{ratio_color};font-size:1.1rem'>{ratio:.2f}x</span>** "
        f"— the demand model uses this ratio as the core signal for quantity prediction.",
        unsafe_allow_html=True
    )

    qc4, qc5, qc6 = st.columns(3)
    with qc4:
        discount_rate = st.slider("Discount Rate", 0.0, 0.225, 0.05, step=0.005,
                                   help="Model splits on rates up to 0.225")
    with qc5:
        discount = st.number_input("Order Item Discount ($)", value=round(sales * discount_rate, 2),
                                    step=0.5, min_value=0.0)
    with qc6:
        logistics_load = st.slider("Logistics Load Score", 0.36, 0.97, 0.60, step=0.01,
                                    help="Model splits between 0.36–0.97")

    with st.expander("⚙️ Extra inputs (minor impact)"):
        ex1, ex2 = st.columns(2)
        with ex1:
            market        = st.selectbox("Market", list(MARKET_ENC.keys()))
            department    = st.selectbox("Department", list(DEPARTMENT_ENC.keys()))
            region        = st.selectbox("Order Region", list(REGION_ENC.keys()))
            high_demand_risk = st.selectbox("High Demand Risk Flag", [0, 1])
        with ex2:
            quantity       = st.number_input("Order Item Quantity", value=2, min_value=1, step=1)
            hourly_access  = st.number_input("Hourly Access Count", value=5000, step=500,
                                              help="Model splits between 2219–30650")
            region_avg_qty = st.number_input("Region Avg Qty Per Order", value=2.05, step=0.01,
                                              help="Model splits between 1.97–2.26")
            popularity_pct = st.number_input("Product Popularity Percentile", value=5.0, step=1.0,
                                              help="Model splits between 2.4–28.9")
            demand_cv      = st.number_input("Product Demand CV", value=0.54, step=0.01,
                                              help="Model splits between 0.53–0.55")
            active_days    = st.number_input("Product Active Days", value=3000, step=500,
                                              help="Model splits between 1236–13238")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict", use_container_width=True, type="primary"):
        inp = dict(
            shipping_mode=shipping_mode, days_real=days_real, days_scheduled=days_scheduled,
            order_status=order_status, order_type=order_type, benefit=benefit,
            profit_ratio=profit_ratio, demand_spike=demand_spike, product_price=product_price,
            sales=sales, discount_rate=discount_rate, discount=discount,
            logistics_load=logistics_load, high_demand_risk=high_demand_risk,
            customer_segment=customer_segment, market=market, department=department,
            region=region, quantity=quantity, hourly_access=hourly_access,
            region_avg_qty=region_avg_qty, popularity_pct=popularity_pct,
            demand_cv=demand_cv, active_days=int(active_days),
        )

        # ── Run both models ──────────────────────────────────────
        d_row  = build_delivery_row(inp)
        d_pred = delivery_model.predict(d_row)[0]
        d_prob = delivery_model.predict_proba(d_row)[0][1]

        q_row  = build_demand_row(inp)
        q_pred = demand_model.predict(q_row)[0]

        # ── Results ──────────────────────────────────────────────
        st.markdown("### 🎯 Prediction Results")
        r1, r2 = st.columns(2)
        with r1:
            box_cls = "pred-risk" if d_pred == 1 else "pred-safe"
            icon    = "⚠️" if d_pred == 1 else "✅"
            label   = "LATE DELIVERY RISK" if d_pred == 1 else "ON TIME"
            col_hex = "#f87171" if d_pred == 1 else "#4ade80"
            ss = SHIPPING_STATS[shipping_mode]
            st.markdown(f"""<div class="pred-box {box_cls}">
                <div style="font-size:2.5rem">{icon}</div>
                <div class="pred-val" style="color:{col_hex}">{label}</div>
                <div class="pred-lbl">Late Risk Probability: <strong style="color:{col_hex}">{d_prob*100:.1f}%</strong></div>
                <div class="pred-lbl" style="margin-top:10px">
                    {shipping_mode} &nbsp;·&nbsp; Real: {days_real}d &nbsp;·&nbsp; Scheduled: {days_scheduled}d<br>
                    Mode late rate: {ss['ShippingMode_Late_Delivery_Rate']*100:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            qty_label = "🔥 High" if q_pred >= 3.5 else ("📊 Medium" if q_pred >= 2 else "📉 Low")
            qty_color = "#4ade80" if q_pred >= 3.5 else ("#fb923c" if q_pred >= 2 else "#94a3b8")
            st.markdown(f"""<div class="pred-box pred-demand">
                <div style="font-size:2.5rem">📦</div>
                <div class="pred-val" style="color:#60a5fa">{q_pred:.2f} units</div>
                <div class="pred-lbl">demand_model.predict() output</div>
                <div style="margin-top:12px;display:flex;justify-content:center;gap:16px;flex-wrap:wrap">
                    <div style="text-align:center">
                        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">Demand Level</div>
                        <div style="font-size:1rem;font-weight:700;color:{qty_color}">{qty_label}</div>
                    </div>
                    <div style="text-align:center">
                        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">Spike Flag</div>
                        <div style="font-size:1rem;font-weight:700;color:{'#f87171' if demand_spike else '#94a3b8'}">{"🔥 Active" if demand_spike else "Inactive"}</div>
                    </div>
                    <div style="text-align:center">
                        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">Model Range</div>
                        <div style="font-size:1rem;font-weight:700;color:#e2e8f0">1.0 – 5.0</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Gauges ───────────────────────────────────────────────
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=d_prob * 100,
                title={"text": "Late Delivery Risk %", "font": {"color": "#e2e8f0"}},
                number={"suffix": "%", "font": {"color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar":  {"color": "#3b82f6"},
                    "steps": [
                        {"range": [0, 40],   "color": "#166534"},
                        {"range": [40, 70],  "color": "#854d0e"},
                        {"range": [70, 100], "color": "#7f1d1d"},
                    ],
                    "threshold": {"line": {"color": "#f87171", "width": 4}, "thickness": 0.75, "value": 50}
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                              height=250, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(q_pred),
                title={"text": "Predicted Demand (units)", "font": {"color": "#e2e8f0"}},
                number={"font": {"color": "#60a5fa"}},
                gauge={
                    "axis": {"range": [0, 5], "tickcolor": "#94a3b8"},
                    "bar":  {"color": "#3b82f6"},
                    "steps": [
                        {"range": [0, 2],   "color": "#1e3a5f"},
                        {"range": [2, 3.5], "color": "#1e4d8c"},
                        {"range": [3.5, 5], "color": "#1d6fa4"},
                    ],
                }
            ))
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                               height=250, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Exact feature values fed to each model ────────────────
        st.markdown("#### 🔬 Exact values passed to each model")
        t1, t2 = st.columns(2)
        ss = SHIPPING_STATS[shipping_mode]
        segs = SEGMENT_STATS[customer_segment]
        with t1:
            st.markdown("**Delivery Risk Model inputs:**")
            st.dataframe(pd.DataFrame({
                "Feature": [
                    "Shipping Mode (encoded)", "Days for Shipping (Real)", "Days Scheduled",
                    "ShippingMode_Late_Delivery_Rate", "ShippingMode_Avg_Real_Days",
                    "ShippingMode_Avg_Scheduled_Days", "Segment_Late_Delivery_Rate",
                    "Order Status (encoded)", "Benefit per Order", "Profit Ratio",
                ],
                "Value": [
                    f"{shipping_mode} → {SHIPPING_MODE_ENC[shipping_mode]}",
                    days_real, days_scheduled,
                    ss['ShippingMode_Late_Delivery_Rate'],
                    ss['ShippingMode_Avg_Real_Days'],
                    ss['ShippingMode_Avg_Scheduled_Days'],
                    segs['Segment_Late_Delivery_Rate'],
                    f"{order_status} → {ORDER_STATUS_ENC[order_status]}",
                    benefit, profit_ratio,
                ]
            }), hide_index=True, use_container_width=True)
        with t2:
            st.markdown("**Demand Forecast Model inputs:**")
            st.dataframe(pd.DataFrame({
                "Feature": [
                    "Demand_Spike_Flag", "Sales", "Product Price",
                    "Sales / Price (implied qty)", "Order Item Discount",
                    "Order Item Discount Rate", "Logistics_Load_Score",
                    "Region_Avg_Quantity_Per_Order", "Hourly_Access_Count",
                    "Customer Segment (encoded)",
                ],
                "Value": [
                    demand_spike, sales, product_price,
                    f"{ratio:.2f}x",
                    discount, discount_rate, logistics_load,
                    region_avg_qty, hourly_access,
                    f"{customer_segment} → {SEGMENT_ENC[customer_segment]}",
                ]
            }), hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🧠 Decision Engine")
    st.caption("Actionable recommendations derived from both model outputs across your entire filtered dataset.")
    st.markdown("---")

    # ── Run both models on full filtered dataset ─────────────────
    with st.spinner("Running both models on filtered data…"):
        preds_d_eng, probs_d_eng = run_delivery_predictions(fdf)
        preds_q_eng              = run_demand_predictions(fdf)

    fdf_eng = fdf.copy()
    fdf_eng["Late_Risk_Prob"]    = probs_d_eng
    fdf_eng["Late_Risk_Pred"]    = preds_d_eng
    fdf_eng["Predicted_Qty"]     = preds_q_eng
    fdf_eng["Sales_Price_Ratio"] = (
        fdf_eng["Sales"] / fdf_eng["Product Price"].replace(0, np.nan)
    ).fillna(0) if "Sales" in fdf_eng.columns and "Product Price" in fdf_eng.columns else 1.0

    # ── Decision thresholds ──────────────────────────────────────
    LATE_HIGH   = 0.60   # ≥ 60% = high delivery risk
    LATE_MED    = 0.40   # 40–60% = moderate risk
    QTY_HIGH    = 3.5    # ≥ 3.5 = high demand
    QTY_LOW     = 2.0    # ≤ 2.0 = low demand

    # ── Segment each order into a quadrant ───────────────────────
    def get_quadrant(prob, qty):
        risk = "HIGH" if prob >= LATE_HIGH else ("MED" if prob >= LATE_MED else "LOW")
        demand = "HIGH" if qty >= QTY_HIGH else ("MED" if qty > QTY_LOW else "LOW")
        return risk, demand

    fdf_eng["Risk_Level"]   = fdf_eng["Late_Risk_Prob"].apply(
        lambda p: "HIGH" if p >= LATE_HIGH else ("MED" if p >= LATE_MED else "LOW"))
    fdf_eng["Demand_Level"] = fdf_eng["Predicted_Qty"].apply(
        lambda q: "HIGH" if q >= QTY_HIGH else ("MED" if q > QTY_LOW else "LOW"))
    fdf_eng["Quadrant"]     = fdf_eng["Risk_Level"] + "_" + fdf_eng["Demand_Level"]

    # ── Aggregate KPIs ────────────────────────────────────────────
    n_total       = len(fdf_eng)
    n_high_risk   = (fdf_eng["Risk_Level"] == "HIGH").sum()
    n_med_risk    = (fdf_eng["Risk_Level"] == "MED").sum()
    n_low_risk    = (fdf_eng["Risk_Level"] == "LOW").sum()
    n_high_demand = (fdf_eng["Demand_Level"] == "HIGH").sum()
    n_low_demand  = (fdf_eng["Demand_Level"] == "LOW").sum()
    avg_risk_prob = probs_d_eng.mean() * 100
    avg_pred_qty  = preds_q_eng.mean()

    # ── Top KPI strip ─────────────────────────────────────────────
    ka, kb, kc, kd, ke = st.columns(5)
    with ka: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#f87171">{n_high_risk:,}</div><div class="kpi-lbl">High Risk Orders</div></div>', unsafe_allow_html=True)
    with kb: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#fb923c">{n_med_risk:,}</div><div class="kpi-lbl">Medium Risk Orders</div></div>', unsafe_allow_html=True)
    with kc: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#4ade80">{n_low_risk:,}</div><div class="kpi-lbl">Low Risk Orders</div></div>', unsafe_allow_html=True)
    with kd: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#60a5fa">{n_high_demand:,}</div><div class="kpi-lbl">High Demand Orders</div></div>', unsafe_allow_html=True)
    with ke: st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_pred_qty:.2f}</div><div class="kpi-lbl">Avg Predicted Qty</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 1 — RISK × DEMAND QUADRANT MATRIX
    # ══════════════════════════════════════════════════════════════
    st.markdown('<p class="sec-hdr">📊 Risk × Demand Quadrant Matrix</p>', unsafe_allow_html=True)
    st.caption("Each quadrant = a distinct operational situation with its own action plan.")

    q1, q2 = st.columns(2)
    q3, q4 = st.columns(2)

    n_hr_hd = len(fdf_eng[fdf_eng["Quadrant"].isin(["HIGH_HIGH","HIGH_MED"])])
    n_hr_ld = len(fdf_eng[fdf_eng["Quadrant"].isin(["HIGH_LOW"])])
    n_lr_hd = len(fdf_eng[fdf_eng["Quadrant"].isin(["LOW_HIGH","MED_HIGH"])])
    n_lr_ld = len(fdf_eng[fdf_eng["Quadrant"].isin(["LOW_LOW","MED_LOW","LOW_MED","MED_MED"])])

    with q1:
        pct = n_hr_hd / n_total * 100
        st.markdown(f"""<div class="quadrant-box q-rr">
            <div class="q-label" style="color:#f87171">⚠️ HIGH RISK · HIGH DEMAND</div>
            <div style="font-size:2rem;font-weight:700;color:#fca5a5">{n_hr_hd:,} <span style="font-size:1rem">orders ({pct:.1f}%)</span></div>
            <div class="q-sub" style="margin-top:8px">🔴 Urgent action needed<br>Late delivery + stockout risk combined</div>
        </div>""", unsafe_allow_html=True)
    with q2:
        pct = n_hr_ld / n_total * 100
        st.markdown(f"""<div class="quadrant-box q-ro">
            <div class="q-label" style="color:#fbbf24">⚡ HIGH RISK · LOW DEMAND</div>
            <div style="font-size:2rem;font-weight:700;color:#fcd34d">{n_hr_ld:,} <span style="font-size:1rem">orders ({pct:.1f}%)</span></div>
            <div class="q-sub" style="margin-top:8px">🟡 Logistics review needed<br>Delivery problems on low-priority items</div>
        </div>""", unsafe_allow_html=True)
    with q3:
        pct = n_lr_hd / n_total * 100
        st.markdown(f"""<div class="quadrant-box q-gr">
            <div class="q-label" style="color:#60a5fa">📦 LOW RISK · HIGH DEMAND</div>
            <div style="font-size:2rem;font-weight:700;color:#93c5fd">{n_lr_hd:,} <span style="font-size:1rem">orders ({pct:.1f}%)</span></div>
            <div class="q-sub" style="margin-top:8px">🔵 Inventory focus needed<br>Deliveries safe but demand is high</div>
        </div>""", unsafe_allow_html=True)
    with q4:
        pct = n_lr_ld / n_total * 100
        st.markdown(f"""<div class="quadrant-box q-gg">
            <div class="q-label" style="color:#4ade80">✅ LOW RISK · LOW DEMAND</div>
            <div style="font-size:2rem;font-weight:700;color:#86efac">{n_lr_ld:,} <span style="font-size:1rem">orders ({pct:.1f}%)</span></div>
            <div class="q-sub" style="margin-top:8px">🟢 Stable — monitor only<br>Normal operations</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 2 — DECISION RULES ENGINE
    # ══════════════════════════════════════════════════════════════
    st.markdown('<p class="sec-hdr">🎯 Decision Rules — What Action to Take</p>', unsafe_allow_html=True)

    def make_rec(title, body, level, tags):
        tag_html = "".join([f'<span class="rec-tag tag-{t[0]}">{t[1]}</span>' for t in tags])
        return f"""<div class="rec-card rec-{level}">
            <div class="rec-title">{title}</div>
            {tag_html}
            <div class="rec-body" style="margin-top:8px">{body}</div>
        </div>"""

    # ── Build recommendations from actual model outputs ───────────
    recommendations = []

    # --- Rule 1: Fleet / Shipping capacity ---
    if n_high_risk / n_total > 0.4:
        recommendations.append(make_rec(
            "🚨 Critical: High Late Delivery Rate Across Fleet",
            f"{n_high_risk:,} orders ({n_high_risk/n_total*100:.1f}%) have &gt;60% late delivery probability. "
            f"Immediately audit shipping partners and carrier SLAs. Consider shifting volume to Same Day or First Class for high-value orders.",
            "critical",
            [("red","URGENT"), ("red","Logistics"), ("red","Carrier SLA")]
        ))
    elif n_high_risk / n_total > 0.2:
        recommendations.append(make_rec(
            "⚠️ Elevated Delivery Risk — Proactive Rerouting Recommended",
            f"{n_high_risk:,} orders ({n_high_risk/n_total*100:.1f}%) flagged as high risk. "
            f"Pre-emptively upgrade {min(n_high_risk, int(n_high_risk*0.5)):,} high-value orders to faster shipping modes before dispatch.",
            "warning",
            [("amber","MODERATE"), ("amber","Shipping Mode"), ("amber","Rerouting")]
        ))
    else:
        recommendations.append(make_rec(
            "✅ Delivery Risk Under Control",
            f"Only {n_high_risk/n_total*100:.1f}% of orders are high risk. Maintain current carrier mix and monitor weekly.",
            "success",
            [("green","STABLE"), ("green","Logistics")]
        ))

    # --- Rule 2: Second Class shipping is the problem ---
    if "Shipping Mode" in fdf_eng.columns:
        sc_risk = fdf_eng[fdf_eng["Shipping Mode"] == "Second Class"]["Late_Risk_Prob"].mean() if "Second Class" in fdf_eng["Shipping Mode"].values else 0
        sc_count = (fdf_eng["Shipping Mode"] == "Second Class").sum()
        if sc_count > 0 and sc_risk > 0.5:
            recommendations.append(make_rec(
                "📦 Second Class Shipping is Your Biggest Risk Driver",
                f"{sc_count:,} orders use Second Class shipping with avg late risk of {sc_risk*100:.1f}%. "
                f"The model learned Second Class has a 52.9% historical late delivery rate. "
                f"Recommend: cap Second Class usage for orders with &gt;$200 value, offer upgrade prompts at checkout.",
                "critical",
                [("red","Second Class"), ("red","High Late Rate"), ("amber","Mode Switch")]
            ))

    # --- Rule 3: Demand-side inventory ---
    if n_high_demand / n_total > 0.35:
        recommendations.append(make_rec(
            "📈 High Demand Spike — Increase Safety Stock Immediately",
            f"{n_high_demand:,} orders ({n_high_demand/n_total*100:.1f}%) have predicted quantity ≥ 3.5 units. "
            f"Avg predicted demand is {avg_pred_qty:.2f} units/order. "
            f"Trigger reorder points 20–30% earlier for top-selling SKUs. "
            f"Coordinate with suppliers for expedited replenishment on spike-flagged products.",
            "critical",
            [("red","STOCKOUT RISK"), ("red","Inventory"), ("amber","Reorder Now")]
        ))
    elif n_high_demand / n_total > 0.15:
        recommendations.append(make_rec(
            "📊 Moderate Demand Uplift — Review Reorder Points",
            f"{n_high_demand:,} orders show elevated demand. "
            f"Review safety stock levels for categories with avg predicted qty &gt; 3. "
            f"Prioritise replenishment for spike-flagged SKUs before end of week.",
            "warning",
            [("amber","MODERATE"), ("amber","Inventory"), ("blue","Reorder Review")]
        ))
    else:
        recommendations.append(make_rec(
            "✅ Demand Levels Normal — Standard Replenishment Applies",
            f"Only {n_high_demand/n_total*100:.1f}% of orders show high demand. "
            f"Maintain standard reorder cadence. Monitor Demand_Spike_Flag daily.",
            "success",
            [("green","STABLE"), ("green","Inventory")]
        ))

    # --- Rule 4: The Danger Zone — HIGH risk + HIGH demand ---
    danger = fdf_eng[(fdf_eng["Risk_Level"] == "HIGH") & (fdf_eng["Demand_Level"] == "HIGH")]
    if len(danger) > 0:
        d_sales = danger["Sales"].sum() if "Sales" in danger.columns else 0
        recommendations.append(make_rec(
            f"🔥 Danger Zone: {len(danger):,} Orders Are Both Late Risk AND High Demand",
            f"These {len(danger):,} orders represent ${d_sales:,.0f} in sales at simultaneous risk of late delivery AND stockout. "
            f"Escalate to ops team immediately. Actions: (1) Express-upgrade shipping, "
            f"(2) Allocate dedicated stock buffer, (3) Set customer expectation alerts, "
            f"(4) Flag for daily tracking dashboard.",
            "critical",
            [("red","HIGHEST PRIORITY"), ("red","Dual Risk"), ("red","Escalate Now"), ("amber",f"${d_sales:,.0f} at risk")]
        ))

    # --- Rule 5: Low demand + High risk = wasteful shipping ---
    waste = fdf_eng[(fdf_eng["Risk_Level"] == "HIGH") & (fdf_eng["Demand_Level"] == "LOW")]
    if len(waste) > 0:
        recommendations.append(make_rec(
            f"💸 {len(waste):,} Orders Are High Risk But Low Priority — Review Shipping Spend",
            f"These orders have high late delivery risk but low predicted demand (≤ 2 units). "
            f"Downgrade shipping mode to reduce cost — late delivery on low-demand items has lower customer impact. "
            f"Redirect shipping budget to high-demand orders instead.",
            "warning",
            [("amber","COST OPTIMISATION"), ("amber","Low Priority"), ("blue","Downgrade Shipping")]
        ))

    # --- Rule 6: Market-level risk analysis ---
    if "Market" in fdf_eng.columns:
        mkt_risk = fdf_eng.groupby("Market")["Late_Risk_Prob"].mean().sort_values(ascending=False)
        worst_mkt = mkt_risk.index[0]
        worst_prob = mkt_risk.iloc[0]
        best_mkt = mkt_risk.index[-1]
        best_prob = mkt_risk.iloc[-1]
        if worst_prob - best_prob > 0.1:
            recommendations.append(make_rec(
                f"🌍 Market Disparity — {worst_mkt} is Significantly Riskier Than {best_mkt}",
                f"{worst_mkt} has avg late risk of {worst_prob*100:.1f}% vs {best_mkt} at {best_prob*100:.1f}%. "
                f"Investigate regional carrier performance, customs delays, and last-mile infrastructure in {worst_mkt}. "
                f"Consider local fulfillment partners or additional DC placement.",
                "warning",
                [("amber","MARKET RISK"), ("amber",worst_mkt), ("blue","Regional Ops")]
            ))

    # --- Rule 7: Demand Spike correlation ---
    spike_col = "Demand_Spike_Flag" if "Demand_Spike_Flag" in fdf_eng.columns else None
    if spike_col:
        n_spike = fdf_eng[spike_col].sum() if fdf_eng[spike_col].dtype in [int, float] else 0
        if n_spike > 0:
            spike_avg_qty = preds_q_eng[fdf_eng[spike_col] == 1].mean() if (fdf_eng[spike_col] == 1).any() else 0
            recommendations.append(make_rec(
                f"⚡ {int(n_spike):,} Orders Have Active Demand Spike Flag",
                f"These orders average {spike_avg_qty:.2f} predicted units — significantly above normal. "
                f"Demand spikes are the #1 driver of the demand model (importance: 77%). "
                f"Trigger automatic stock alerts for spike-flagged SKUs and pre-position inventory at regional DCs.",
                "info",
                [("blue","DEMAND SPIKE"), ("blue","Inventory Alert"), ("blue","77% Feature Importance")]
            ))

    # --- Rule 8: Low demand → consider bundling / promotions ---
    if n_low_demand / n_total > 0.4:
        recommendations.append(make_rec(
            "📉 High Share of Low-Demand Orders — Consider Bundling or Promotions",
            f"{n_low_demand:,} orders ({n_low_demand/n_total*100:.1f}%) predict only 1–2 units. "
            f"These may represent missed upsell opportunities. "
            f"Test bundle offers, volume discounts, or product recommendations to lift order quantities on these SKUs.",
            "info",
            [("blue","REVENUE UPLIFT"), ("blue","Bundling"), ("blue","Promotions")]
        ))

    # ── Render all recommendations ────────────────────────────────
    for rec in recommendations:
        st.markdown(rec, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 3 — SHIPPING MODE ACTION TABLE
    # ══════════════════════════════════════════════════════════════
    st.markdown('<p class="sec-hdr">🚚 Shipping Mode Action Table</p>', unsafe_allow_html=True)

    if "Shipping Mode" in fdf_eng.columns:
        sm_summary = fdf_eng.groupby("Shipping Mode").agg(
            Orders=("Late_Risk_Prob","count"),
            Avg_Late_Risk=("Late_Risk_Prob","mean"),
            High_Risk_Count=("Late_Risk_Pred","sum"),
            Avg_Pred_Qty=("Predicted_Qty","mean"),
        ).reset_index()

        sm_summary["Late_Risk_%"]     = (sm_summary["Avg_Late_Risk"] * 100).round(1)
        sm_summary["High_Risk_%"]     = (sm_summary["High_Risk_Count"] / sm_summary["Orders"] * 100).round(1)
        sm_summary["Avg_Pred_Qty"]    = sm_summary["Avg_Pred_Qty"].round(2)

        def action_for_mode(row):
            if row["Late_Risk_%"] > 50:
                return "🔴 Audit immediately — high late rate"
            elif row["Late_Risk_%"] > 30:
                return "🟡 Monitor — moderate late rate"
            else:
                return "🟢 Healthy — no action needed"

        sm_summary["Recommended Action"] = sm_summary.apply(action_for_mode, axis=1)
        sm_summary = sm_summary[["Shipping Mode","Orders","Late_Risk_%","High_Risk_Count","Avg_Pred_Qty","Recommended Action"]]
        st.dataframe(sm_summary, hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 4 — ORDER-LEVEL PRIORITY TABLE
    # ══════════════════════════════════════════════════════════════
    st.markdown('<p class="sec-hdr">🗂️ High-Priority Orders Requiring Immediate Action</p>', unsafe_allow_html=True)
    st.caption("Orders with high late risk AND high demand — sorted by urgency score.")

    priority = fdf_eng[(fdf_eng["Late_Risk_Prob"] >= LATE_MED) | (fdf_eng["Demand_Level"] == "HIGH")].copy()
    priority["Urgency_Score"] = (priority["Late_Risk_Prob"] * 0.6 + (priority["Predicted_Qty"] / 5) * 0.4).round(3)
    priority["Action"] = priority.apply(lambda r:
        "🔥 Escalate — dual risk" if r["Risk_Level"] == "HIGH" and r["Demand_Level"] == "HIGH"
        else ("⚠️ Expedite shipping" if r["Risk_Level"] == "HIGH"
        else ("📦 Pre-stock inventory" if r["Demand_Level"] == "HIGH"
        else "👀 Monitor")), axis=1)

    display_cols = [c for c in ["Order Id","Product Name","Category Name","Market","Shipping Mode",
                                "Late_Risk_Prob","Predicted_Qty","Risk_Level","Demand_Level",
                                "Urgency_Score","Action"] if c in priority.columns]
    priority["Late_Risk_Prob"] = priority["Late_Risk_Prob"].round(3)
    priority["Predicted_Qty"]  = priority["Predicted_Qty"].round(2)

    st.dataframe(
        priority[display_cols].sort_values("Urgency_Score", ascending=False).head(200),
        hide_index=True, use_container_width=True
    )

    # ══════════════════════════════════════════════════════════════
    # SECTION 5 — SCATTER: Risk vs Demand (all orders)
    # ══════════════════════════════════════════════════════════════
    st.markdown('<p class="sec-hdr">🔭 Risk vs Demand — All Orders</p>', unsafe_allow_html=True)
    sample_eng = fdf_eng.sample(min(3000, len(fdf_eng)), random_state=42)
    color_col  = "Market" if "Market" in sample_eng.columns else "Risk_Level"
    fig = px.scatter(
        sample_eng, x="Late_Risk_Prob", y="Predicted_Qty",
        color=color_col, opacity=0.5,
        labels={"Late_Risk_Prob": "Late Delivery Risk Probability", "Predicted_Qty": "Predicted Order Quantity"},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    # Draw quadrant lines
    fig.add_vline(x=LATE_HIGH, line_dash="dash", line_color="#f87171", annotation_text="High Risk")
    fig.add_vline(x=LATE_MED,  line_dash="dot",  line_color="#f59e0b", annotation_text="Med Risk")
    fig.add_hline(y=QTY_HIGH,  line_dash="dash", line_color="#3b82f6", annotation_text="High Demand")
    fig.add_hline(y=QTY_LOW,   line_dash="dot",  line_color="#94a3b8", annotation_text="Low Demand")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0,r=0,t=20,b=0), height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ── Footer summary ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1f35,#212840);border-radius:12px;padding:20px;border:1px solid #2d3555;">
        <div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:12px;">📋 Decision Engine Summary</div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:2">
            📦 Total orders analysed: <strong style="color:#e2e8f0">{n_total:,}</strong> &nbsp;·&nbsp;
            🔴 High risk: <strong style="color:#f87171">{n_high_risk:,} ({n_high_risk/n_total*100:.1f}%)</strong> &nbsp;·&nbsp;
            🟡 Med risk: <strong style="color:#fbbf24">{n_med_risk:,} ({n_med_risk/n_total*100:.1f}%)</strong> &nbsp;·&nbsp;
            🟢 Low risk: <strong style="color:#4ade80">{n_low_risk:,} ({n_low_risk/n_total*100:.1f}%)</strong><br>
            📈 High demand: <strong style="color:#60a5fa">{n_high_demand:,} ({n_high_demand/n_total*100:.1f}%)</strong> &nbsp;·&nbsp;
            📉 Low demand: <strong style="color:#94a3b8">{n_low_demand:,} ({n_low_demand/n_total*100:.1f}%)</strong> &nbsp;·&nbsp;
            🔥 Danger zone (dual risk): <strong style="color:#f87171">{len(danger):,}</strong> &nbsp;·&nbsp;
            💡 Total recommendations: <strong style="color:#e2e8f0">{len(recommendations)}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
