import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.knowledge_base import COMPANIES, FEATURES

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\-\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_company_profile(company_name: str, sector: str, context: str):
    profile = f"{company_name} operates in {sector}. {context}"
    return profile.strip()

def capability_scores(text: str):
    t = normalize_text(text)
    scores = {}
    matched_terms = {}
    for feature, terms in FEATURES.items():
        count = sum(t.count(term.lower()) for term in terms)
        score = min(1.0, count / 3.0)
        scores[feature] = round(score, 3)
        matched_terms[feature] = [term for term in terms if term.lower() in t][:5]
    return scores, matched_terms

def competitor_match(company_name: str, sector: str, profile: str, top_k=5):
    docs = [profile]
    meta = [{"name": company_name, "sector": sector, "description": profile}]
    for c in COMPANIES:
        docs.append(f"{c['name']} operates in {c['sector']}. {c['description']}")
        meta.append(c)

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2500)
    X = vec.fit_transform(docs)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()

    rows = []
    for i, score in enumerate(sims):
        c = meta[i+1]
        if c["name"].lower() == company_name.lower():
            continue
        sector_bonus = 0.10 if sector.lower().split("/")[0].strip() in c["sector"].lower() else 0.0
        final_score = float(score + sector_bonus)
        rows.append({
            "competitor_company": c["name"],
            "competitor_sector": c["sector"],
            "similarity_score": round(final_score, 4),
            "competitor_description": c["description"]
        })
    out = pd.DataFrame(rows).sort_values("similarity_score", ascending=False).head(top_k).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out[["rank", "competitor_company", "competitor_sector", "similarity_score", "competitor_description"]]

def gap_analysis(company_name: str, target_scores: dict, competitors_df: pd.DataFrame):
    rows = []
    comp_feature_scores = []
    for _, r in competitors_df.iterrows():
        scores, _ = capability_scores(r["competitor_description"])
        comp_feature_scores.append(scores)

    for feature in FEATURES.keys():
        target = target_scores[feature]
        comp_avg = sum(s[feature] for s in comp_feature_scores) / max(len(comp_feature_scores), 1)
        gap = round(comp_avg - target, 3)
        if gap > 0.15:
            status = "Gap"
        elif gap < -0.15:
            status = "Leading"
        else:
            status = "Parity"
        rows.append({
            "company_name": company_name,
            "feature": feature,
            "target_score": round(target, 3),
            "competitor_avg": round(comp_avg, 3),
            "gap": gap,
            "status": status
        })
    return pd.DataFrame(rows).sort_values(["status", "gap"], ascending=[True, False]).reset_index(drop=True)

def recommendation_engine(company_name: str, gap_df: pd.DataFrame):
    mapping = {
        "AI Capability": "Introduce stronger AI-led product experiences to increase strategic differentiation and long-term defensibility.",
        "Automation": "Productize repetitive workflows into automation modules to improve operating leverage and customer value.",
        "Analytics": "Expand insight-generation and predictive analytics to become more decision-oriented, not just service-oriented.",
        "Geospatial Intelligence": "Double down on geospatial intelligence as a core differentiator and package it more explicitly.",
        "Enterprise Readiness": "Strengthen platform and enterprise deployment messaging to win larger and more complex accounts.",
        "Real-Time Operations": "Add real-time visibility or operational intelligence modules to improve daily workflow relevance.",
        "Self-Service Experience": "Develop self-serve experiences to reduce friction and make adoption easier.",
        "Integrations": "Invest in deeper integrations so the product fits naturally into customer ecosystems.",
        "Security & Compliance": "Improve governance and compliance positioning to unlock enterprise and regulated opportunities.",
        "Consulting Depth": "Package consulting and managed-service strength into fixed offerings with clearer repeatability.",
        "Public Sector Focus": "Sharpen public-sector-specific messaging and product motions if this is a target market."
    }
    rows = []
    for _, r in gap_df[gap_df["status"] == "Gap"].sort_values("gap", ascending=False).head(6).iterrows():
        rows.append({
            "company_name": company_name,
            "feature": r["feature"],
            "priority": "High" if r["gap"] >= 0.35 else "Medium",
            "recommendation": mapping.get(r["feature"], f"Improve {r['feature']} to close strategic gaps.")
        })
    if not rows:
        rows.append({
            "company_name": company_name,
            "feature": "Overall Positioning",
            "priority": "Medium",
            "recommendation": "The company is close to parity. Focus on sharper product packaging, positioning, and narrative differentiation."
        })
    return pd.DataFrame(rows)

def rationale_text(company_name: str, sector: str, competitors_df: pd.DataFrame, gap_df: pd.DataFrame):
    top_competitors = ", ".join(competitors_df["competitor_company"].head(3).tolist())
    top_gaps = ", ".join(gap_df[gap_df["status"] == "Gap"]["feature"].head(3).tolist())
    if not top_gaps:
        top_gaps = "no major structural gaps"
    return (
        f"{company_name} was compared against the most semantically similar companies in the {sector} landscape. "
        f"The strongest competitor cluster includes {top_competitors}. "
        f"The most important areas to improve are {top_gaps}. "
        f"This suggests the company should focus on more productized differentiation rather than relying only on generic positioning."
    )

def executive_report(company_name: str, sector: str, profile: str, competitors_df: pd.DataFrame, gap_df: pd.DataFrame, rec_df: pd.DataFrame):
    return f"""StrategyPilot Elite Executive Brief

Company:
{company_name}

Sector:
{sector}

Profile Summary:
{profile[:1200]}

Top Competitors:
{chr(10).join([f"- {r['competitor_company']} ({r['competitor_sector']}) | similarity={r['similarity_score']}" for _, r in competitors_df.iterrows()])}

Largest Gaps:
{chr(10).join([f"- {r['feature']} | target={r['target_score']} competitor_avg={r['competitor_avg']} gap={r['gap']}" for _, r in gap_df[gap_df['status']=='Gap'].head(5).iterrows()])}

Priority Recommendations:
{chr(10).join([f"- [{r['priority']}] {r['recommendation']}" for _, r in rec_df.iterrows()])}

End of report.
"""
