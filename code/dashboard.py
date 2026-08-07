"""
Multi-Modal Evidence Review - Streamlit Dashboard
------------------------------------------------
Interactive web application for inspecting claims verification results,
visual evidence images, status distributions, severity breakdowns, and risk flags.

Usage:
  streamlit run code/dashboard.py
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Multi-Modal Claims Review Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Custom CSS for Sleek Dark Aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1E222D;
        border: 1px solid #2E3440;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ECEFF4;
    }
    .metric-label {
        font-size: 14px;
        color: #D8DEE9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-supported {
        background-color: #2E7D32;
        color: #E8F5E9;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 13px;
    }
    .badge-contradicted {
        background-color: #C62828;
        color: #FFEBEE;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 13px;
    }
    .badge-info {
        background-color: #F57F17;
        color: #FFFDE7;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 13px;
    }
    .badge-risk {
        background-color: #D84315;
        color: #FBE9E7;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 12px;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


def resolve_img(rel_path: str) -> Path | None:
    rel = Path(rel_path)
    if rel.is_absolute() and rel.exists():
        return rel

    candidates = [
        BASE_DIR / rel,
        BASE_DIR / "dataset" / rel,
        BASE_DIR / "images" / rel,
        BASE_DIR / "hackerrank-orchestrate-june26-main" / rel,
        BASE_DIR / "hackerrank-orchestrate-june26-main" / "dataset" / rel,
    ]
    if rel.parts and rel.parts[0] in ["dataset", "images"]:
        candidates.append(BASE_DIR / "images" / Path(*rel.parts[1:]))

    for c in candidates:
        if c.exists():
            return c
    return None


@st.cache_data
def load_data():
    output_path = BASE_DIR / "output.csv"
    if not output_path.exists():
        output_path = BASE_DIR / "dataset" / "test_predictions.csv"
    if not output_path.exists():
        return None
    return pd.read_csv(output_path)


def main():
    st.title("🔍 Multi-Modal Evidence Review Dashboard")
    st.markdown("### Damage Claims Verification & Visual Evidence Inspector")

    df = load_data()
    if df is None or df.empty:
        st.error("No `output.csv` file found. Please run `python code/pipeline.py` first to generate predictions.")
        return

    # Sidebar Controls
    st.sidebar.header("📊 Filter Claims")

    objects = ["All"] + sorted(df["claim_object"].astype(str).unique().tolist())
    selected_object = st.sidebar.selectbox("Claim Object", objects)

    statuses = ["All"] + sorted(df["claim_status"].astype(str).unique().tolist())
    selected_status = st.sidebar.selectbox("Claim Status", statuses)

    severities = ["All"] + sorted(df["severity"].astype(str).unique().tolist())
    selected_severity = st.sidebar.selectbox("Severity", severities)

    search_query = st.sidebar.text_input("Search User ID or Claim Text", "")

    # Filter Logic
    filtered_df = df.copy()
    if selected_object != "All":
        filtered_df = filtered_df[filtered_df["claim_object"] == selected_object]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["claim_status"] == selected_status]
    if selected_severity != "All":
        filtered_df = filtered_df[filtered_df["severity"] == selected_severity]
    if search_query:
        query = search_query.lower()
        filtered_df = filtered_df[
            filtered_df["user_id"].astype(str).str.lower().str.contains(query) |
            filtered_df["user_claim"].astype(str).str.lower().str.contains(query)
        ]

    # Metrics Summary Bar
    total_claims = len(df)
    supported_cnt = (df["claim_status"] == "supported").sum()
    contradicted_cnt = (df["claim_status"] == "contradicted").sum()
    info_cnt = (df["claim_status"] == "not_enough_information").sum()
    risk_cnt = (df["risk_flags"] != "none").sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_claims}</div><div class="metric-label">Total Claims</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #4CAF50;">{supported_cnt} ({supported_cnt/total_claims*100:.0f}%)</div><div class="metric-label">Supported</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #F44336;">{contradicted_cnt} ({contradicted_cnt/total_claims*100:.0f}%)</div><div class="metric-label">Contradicted</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #FFC107;">{info_cnt} ({info_cnt/total_claims*100:.0f}%)</div><div class="metric-label">Needs Info</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #FF5722;">{risk_cnt} ({risk_cnt/total_claims*100:.0f}%)</div><div class="metric-label">Risk Flagged</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Analytics Section (Charts)
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.subheader("📌 Status Distribution")
        status_counts = df["claim_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        
        status_chart = alt.Chart(status_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Status:N', title='', sort='-y', axis=alt.Axis(labelAngle=-45, labelColor='#D8DEE9')),
            y=alt.Y('Count:Q', title='', axis=alt.Axis(gridColor='#2E3440', labelColor='#D8DEE9')),
            color=alt.Color('Status:N', scale=alt.Scale(
                domain=['supported', 'contradicted', 'not_enough_information'],
                range=['#2E7D32', '#C62828', '#F57F17']
            ), legend=None),
            tooltip=['Status', 'Count']
        ).properties(height=300).configure_view(strokeWidth=0).configure_axis(domain=False)
        st.altair_chart(status_chart, use_container_width=True)

    with chart_col2:
        st.subheader("🔥 Severity Breakdown")
        sev_counts = df["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        
        sev_chart = alt.Chart(sev_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Severity:N', title='', sort='-y', axis=alt.Axis(labelAngle=-45, labelColor='#D8DEE9')),
            y=alt.Y('Count:Q', title='', axis=alt.Axis(gridColor='#2E3440', labelColor='#D8DEE9')),
            color=alt.Color('Severity:N', scale=alt.Scale(scheme='warmgreys'), legend=None),
            tooltip=['Severity', 'Count']
        ).properties(height=300).configure_view(strokeWidth=0).configure_axis(domain=False)
        st.altair_chart(sev_chart, use_container_width=True)

    with chart_col3:
        st.subheader("⚠️ Risk Flags Frequency")
        all_flags = []
        for f_str in df["risk_flags"].dropna():
            for f in str(f_str).split(";"):
                f_clean = f.strip()
                if f_clean and f_clean != "none":
                    all_flags.append(f_clean)
        if all_flags:
            flag_df = pd.Series(all_flags).value_counts().reset_index()
            flag_df.columns = ["Flag", "Count"]
            
            flag_chart = alt.Chart(flag_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X('Flag:N', title='', sort='-y', axis=alt.Axis(labelAngle=-45, labelColor='#D8DEE9')),
                y=alt.Y('Count:Q', title='', axis=alt.Axis(gridColor='#2E3440', labelColor='#D8DEE9')),
                color=alt.Color('Flag:N', scale=alt.Scale(scheme='orangered'), legend=None),
                tooltip=['Flag', 'Count']
            ).properties(height=300).configure_view(strokeWidth=0).configure_axis(domain=False)
            st.altair_chart(flag_chart, use_container_width=True)
        else:
            st.info("No risk flags detected in current dataset.")

    st.markdown("---")

    # Expandable Claims List
    st.subheader(f"📋 Claims Details ({len(filtered_df)} shown)")

    for idx, row in filtered_df.iterrows():
        status = row["claim_status"]
        status_class = (
            "badge-supported" if status == "supported" else
            "badge-contradicted" if status == "contradicted" else
            "badge-info"
        )

        title = f"Case #{idx+1} | User: {row['user_id']} | Object: {str(row['claim_object']).upper()} | Status: {status.upper()}"
        
        with st.expander(title, expanded=False):
            exp_col1, exp_col2 = st.columns([1, 1])

            with exp_col1:
                st.markdown(f"**User ID:** `{row['user_id']}`")
                st.markdown(f"**Claim Object:** `{row['claim_object']}`")
                st.markdown(f"**Status:** <span class='{status_class}'>{status.upper()}</span>", unsafe_allow_html=True)
                st.markdown(f"**Issue Type:** `{row['issue_type']}` | **Object Part:** `{row['object_part']}`")
                st.markdown(f"**Severity:** `{row['severity']}`")

                st.markdown("#### 💬 User Transcript")
                st.info(row["user_claim"])

                st.markdown("#### 📝 Justification")
                st.write(row["claim_status_justification"])

                st.markdown(f"**Supporting Image IDs:** `{row['supporting_image_ids']}`")
                st.markdown(f"**Evidence Standard Met:** `{row['evidence_standard_met']}` ({row['evidence_standard_met_reason']})")
                st.markdown(f"**Valid Image:** `{row['valid_image']}`")

                # Risk Flags Badges
                r_flags = str(row['risk_flags']).split(';')
                st.markdown("**Risk Flags:**")
                flag_html = "".join([f"<span class='badge-risk'>{f.strip()}</span>" for f in r_flags if f.strip()])
                st.markdown(flag_html, unsafe_allow_html=True)

            with exp_col2:
                st.markdown("#### 🖼️ Submitted Images")
                img_paths = [p.strip() for p in str(row["image_paths"]).split(";") if p.strip()]
                if img_paths:
                    img_cols = st.columns(min(len(img_paths), 3))
                    for i_idx, p in enumerate(img_paths):
                        resolved = resolve_img(p)
                        with img_cols[i_idx % 3]:
                            if resolved:
                                try:
                                    img = Image.open(resolved)
                                    st.image(img, caption=f"ID: {Path(p).stem}", use_container_width=True)
                                except Exception as err:
                                    st.error(f"Cannot load image {p}: {err}")
                            else:
                                st.warning(f"Image missing: {p}")
                else:
                    st.write("No image paths attached.")


if __name__ == "__main__":
    main()
