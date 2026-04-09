import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.knowledge_base import PRESETS
from src.analysis import (
    build_company_profile,
    capability_scores,
    competitor_match,
    gap_analysis,
    recommendation_engine,
    rationale_text,
    executive_report,
)

st.set_page_config(page_title="StrategyPilot Elite", layout="wide")

st.markdown(
    '''
    <style>
    .block-container {padding-top: 1.1rem; max-width: 1320px;}
    .hero {
      padding: 1.5rem 1.7rem;
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(3,105,161,0.90), rgba(22,163,74,0.84));
      border: 1px solid rgba(255,255,255,0.10);
      box-shadow: 0 24px 60px rgba(2,6,23,0.32);
      margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin: 0 0 0.25rem 0; font-size: 2.45rem;}
    .hero p {color: #e2e8f0; margin: 0; line-height: 1.75;}
    .glass {
      padding: 1rem 1rem; border-radius: 22px; background: rgba(15,23,42,0.62);
      border: 1px solid rgba(255,255,255,0.08); margin-bottom: 0.9rem;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="hero">
      <h1>StrategyPilot Elite</h1>
      <p>AI Copilot for Competitive Intelligence, Capability Gaps, and Strategic Product Decisions — upgraded with stronger logic, rationale, and executive-style outputs.</p>
    </div>
    ''',
    unsafe_allow_html=True
)

preset_name = st.selectbox("Quick preset", ["Custom"] + list(PRESETS.keys()))
if preset_name != "Custom":
    preset = PRESETS[preset_name]
    default_company = preset_name
    default_sector = preset["sector"]
    default_website = preset["website"]
    default_context = preset["context"]
else:
    default_company = ""
    default_sector = ""
    default_website = ""
    default_context = ""

left, right = st.columns([1.05, 0.95])

with left:
    company_name = st.text_input("Company name", value=default_company)
    sector = st.text_input("Sector / Industry", value=default_sector)
    website_url = st.text_input("Website URL (optional for presentation only)", value=default_website)
    context = st.text_area("Business / product context", value=default_context, height=140)
    analyze = st.button("Analyze Company Strategy", use_container_width=True)

with right:
    st.markdown(
        '''
        <div class="glass">
        <b>What this elite version adds</b><br><br>
        • better capability extraction<br>
        • semantic competitor matching<br>
        • clearer rationale for findings<br>
        • stronger recommendation logic<br>
        • downloadable executive brief
        </div>
        ''',
        unsafe_allow_html=True
    )

if analyze and company_name and sector:
    profile = build_company_profile(company_name, sector, context)
    target_scores, matched_terms = capability_scores(profile)
    competitors_df = competitor_match(company_name, sector, profile, top_k=5)
    gap_df = gap_analysis(company_name, target_scores, competitors_df)
    rec_df = recommendation_engine(company_name, gap_df)
    rationale = rationale_text(company_name, sector, competitors_df, gap_df)
    report = executive_report(company_name, sector, profile, competitors_df, gap_df, rec_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Competitors Found", len(competitors_df))
    m2.metric("Gap Areas", int((gap_df["status"] == "Gap").sum()))
    m3.metric("Lead Areas", int((gap_df["status"] == "Leading").sum()))
    m4.metric("Recommendations", len(rec_df))

    col1, col2 = st.columns([1.15, 0.85])

    with col2:
        st.markdown("### Company Profile")
        st.info(profile)
        st.markdown("### Capability Signals")
        caps_df = pd.DataFrame({
            "feature": list(target_scores.keys()),
            "score": list(target_scores.values()),
            "matched_terms": [", ".join(matched_terms[k]) if matched_terms[k] else "-" for k in target_scores.keys()]
        })
        st.dataframe(caps_df, use_container_width=True, hide_index=True)

    with col1:
        st.markdown("### Top Competitors")
        st.dataframe(competitors_df, use_container_width=True, hide_index=True)
        st.markdown("### Gap Analysis")
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

    st.markdown("### Why these recommendations?")
    st.success(rationale)

    st.markdown("### Strategic Recommendations")
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        gap_plot = gap_df.sort_values("gap", ascending=False).head(8)
        fig = plt.figure(figsize=(8, 4.5))
        plt.barh(gap_plot["feature"], gap_plot["gap"])
        plt.gca().invert_yaxis()
        plt.xlabel("Competitor Advantage")
        plt.title("Largest Strategic Gaps")
        st.pyplot(fig)

    with b:
        status_counts = gap_df["status"].value_counts()
        fig2 = plt.figure(figsize=(7, 4.5))
        plt.bar(status_counts.index, status_counts.values)
        plt.ylabel("Feature Count")
        plt.title("Strategic Position Status")
        st.pyplot(fig2)

    st.download_button(
        "Download Executive Brief",
        data=report,
        file_name=f"{company_name.lower().replace(' ', '_')}_strategy_brief.txt",
        mime="text/plain",
        use_container_width=True
    )
else:
    st.info("Choose a preset or enter a company name and sector, then click Analyze Company Strategy.")
