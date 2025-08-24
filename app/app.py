# -*- coding: utf-8 -*-
# MVP: Asthma Response Scoring + Slope Charts + PDF Export
# UI language: English; Privacy note: we do not store data.

import io
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

st.set_page_config(page_title="Biologic Response MVP", layout="wide")

# -------------------------
# UI: Header
# -------------------------
st.title("Biologic Response BARS (MVP)")
st.info("🔒 We do **not** store your data. Everything happens in your browser session.")

# Feedback link (edit to your real form)
FEEDBACK_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdbHHtazLwS3LLaFwR3O--6Ew3vIX5NkFjBDboCPW0OU45gWQ/viewform?usp=header"
# -------------------------
# Disclaimer + Guides
# -------------------------
st.info("⚠️**Disclaimer:** This tool is an early prototype. It is intended for research and exploratory purposes only, not for clinical decision-making.")

st.markdown(
    """
    📖 Detailed guides are available here:  
    - 🌍 English: [User Guide (Google Doc)](https://docs.google.com/document/d/1Q_cmL5kw4DrvHhLetdx9YNdkxBtujdwZ9wTDkRfxVmw/edit?pli=1&tab=t.0#heading=h.stp0wv5r313tK)  
    - 🇩🇪 Deutsch: [Benutzerhandbuch (Google Doc)](https://docs.google.com/document/d/10OtFdVOHlg203ageKtMVAOsUKLbUHUcV71KepEatlS0/edit?usp=sharing)  
    """,
    unsafe_allow_html=True
)


# -------------------------
# Parameter descriptions 
# -------------------------
with st.expander("Parameter descriptions"):
    st.markdown("""
This calculation method was proposed in the publication: [doi: 10.1055/a-2014-4350](https://www.thieme-connect.com/products/ejournals/abstract/10.1055/a-2014-4350)
                
**Columns required (case-sensitive):**

- `Patient ID` — unique identifier per patient  
- `OCS_BL`, `OCS_FU` — oral corticosteroid dose at **Baseline** and **Follow-up**  
- `ACT_BL`, `ACT_FU` — Asthma Control Test score at **Baseline** and **Follow-up**  
- `Exacerbation_BL`, `Exacerbation_FU` — number/rate of exacerbations during the last 12 months at **Baseline** and **Follow-up**  
- `Treatment` — biologic/therapy group label (used to compute group means)

    """)

# -------------------------
# Limitations & Advantages (English)
# -------------------------
with st.expander("Advantages & Limitations"):
    st.markdown("""
**Advantages**
- **Standardized assessment:** Provides a reproducible, structured way to evaluate response to biologic therapy in severe asthma.
- **Single-page workflow:** paste/upload → calculate → charts → PDF.
- **Transparency:** Transparent rules based on published algorithm of baseline/follow-up metrics.
- **Standardized assessment:** Provides a reproducible, structured way to evaluate response to biologic therapy in severe asthma.
- **Research utility:** Facilitates uniform reporting of treatment response across centers, supporting clinical studies and multi-center comparisons.
- **Extensible:** Designed as an open tool that can be expanded with additional response criteria (ACT, OCS tapering, exacerbations, biomarkers, etc.) and future machine-learning models.

**Limitations**
- The BARS algorithm is based on expert consensus; it is not yet included in international asthma guidelines (e.g., GINA) and should be considered investigational.
- This is an MVP intended for exploratory use, not clinical decision-making.
- Input validation is minimal; please sanity-check units and outliers.
- Accuracy of results depends entirely on accurate and complete input of patient data.
- Group means are unadjusted (no covariate control).
- PDF layout is simple; heavy branding/tables may require refinement.
""")


# -------------------------
# Input table / Upload
# -------------------------
st.subheader("Paste or upload your table")

template_cols = [
    "Patient ID","OCS_BL","ACT_BL","Exacerbation_BL",
    "Treatment","OCS_FU","ACT_FU","Exacerbation_FU"
]
template_df = pd.DataFrame(columns=template_cols)

tab1, tab2 = st.tabs(["Paste/Edit table", "Upload Excel/CSV"])
with tab1:
    st.write("Paste from Excel (Ctrl/Cmd+V) or type:")
    data_edit = st.data_editor(
        template_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="editor"
    )

with tab2:
    up = st.file_uploader("Drop an Excel (.xlsx) or CSV", type=["xlsx","csv"])
    st.markdown(
        "👉[Example](https://docs.google.com/spreadsheets/d/1TJ2y93y5zY5uar0b_7wC4dXkjEJffSuUHC_bG19T474/edit?usp=sharing)",
        unsafe_allow_html=True
    )

    if up is not None:
        try:
            if up.name.lower().endswith(".xlsx"):
                data_edit = pd.read_excel(up)
            else:
                data_edit = pd.read_csv(up)
            st.success(f"Loaded: {up.name}")
            st.dataframe(data_edit, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read file: {e}")

# Validate minimal columns
def has_required_columns(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in template_cols)

# -------------------------
# Scoring (vectorized)
# -------------------------
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cast numerics when possible
    num_cols = ["OCS_BL","ACT_BL","Exacerbation_BL","OCS_FU","ACT_FU","Exacerbation_FU"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # OCS_score
    with np.errstate(divide="ignore", invalid="ignore"):
        ocs_ratio = (df["OCS_BL"] - df["OCS_FU"]) / df["OCS_BL"]
    ocs_score = np.select(
        [
            (df["OCS_FU"] == 0),
            (df["OCS_BL"] > 0) & (ocs_ratio >= 0.75),
            (df["OCS_BL"] > 0) & (ocs_ratio < 0.5),
            (df["OCS_BL"] == 0) & (df["OCS_FU"] != 0),
        ],
        [2, 2, 0, 0],
        default=1
    )

    # ACT_score
    act_delta = df["ACT_FU"] - df["ACT_BL"]
    act_score = np.select(
        [
            df["ACT_FU"] >= 20,
            act_delta >= 6,
            act_delta < 3,
        ],
        [2, 2, 0],
        default=1
    )

    # Exacerbation_score
    ex_fu_zero = (df["Exacerbation_FU"] == 0)
    ex_bl_pos = (df["Exacerbation_BL"] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ex_ratio = (df["Exacerbation_BL"] - df["Exacerbation_FU"]) / df["Exacerbation_BL"]
    ex_score = np.select(
        [
            (df["Exacerbation_FU"] == 0),
            (df["Exacerbation_BL"] > 0) & (ex_ratio >= 0.75),
            (df["Exacerbation_BL"] > 0) & (ex_ratio < 0.5),
            (df["Exacerbation_BL"] == 0) & (df["Exacerbation_FU"] != 0),
        ],
        [2, 2, 0, 0],
        default=1
    )

    df["OCS_score"] = ocs_score
    df["ACT_score"] = act_score
    df["Exacerbation_score"] = ex_score
    df["Response_mean"] = df[["OCS_score","ACT_score","Exacerbation_score"]].mean(axis=1)
    df["Response_score"] = np.select(
        [df["Response_mean"] >= 1.5, df["Response_mean"] < 0.5],
        [2, 0],
        default=1
    ).astype(int)

    return df

# -------------------------
# Distribution
# -------------------------
def plot_distributions(data):
    figs = {}

    for variable in ["OCS", "ACT", "Exacerbation"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data=data[(data["Type"] == variable) & (data["Time"] == "BL")]["Value"],
            bins=10, kde=False, ax=ax, color="dodgerblue", alpha=0.7, label=f"{variable}.BL"
        )
        sns.histplot(
            data=data[(data["Type"] == variable) & (data["Time"] == "FU")]["Value"],
            bins=10, kde=False, ax=ax, color="limegreen", alpha=0.7, label=f"{variable}.FU"
        )
        ax.legend()
        ax.set_title(f"{variable} Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        figs[variable] = fig

    return figs

# -------------------------
# Boxplot
# -------------------------
def plot_boxplots(data):
    sns.set(style="whitegrid")
    custom_palette = {"BL": "dodgerblue", "FU": "limegreen"}
    figs = {}

    for variable in ["OCS", "ACT", "Exacerbation"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=data[data["Type"] == variable],
            x="Time", y="Value", ax=ax, palette=custom_palette
        )
        ax.set_title(f"{variable} Box Plot")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        figs[variable] = fig

    return figs

# -------------------------
# Slope charts (green up, red down)
# -------------------------
def slope_chart_means(df_means, left_title, right_title, title):
    labels = df_means.index.astype(str).to_numpy()
    y_left = df_means["mean_before"].to_numpy()
    y_right = df_means["mean_after"].to_numpy()
    x_left, x_right = 1.0, 3.0

    segs = np.stack([
        np.column_stack([np.full_like(y_left, x_left), y_left]),
        np.column_stack([np.full_like(y_right, x_right), y_right])
    ], axis=1)

    delta = y_right - y_left
    up = delta >= 0

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    # Color per segment: green (up), red (down)
    colors_list = np.where(up, "green", "red")
    lc = LineCollection(segs, colors=colors_list, linewidths=2)
    ax.add_collection(lc)

    ax.scatter(np.full_like(y_left, x_left), y_left, s=30, c=colors_list)
    ax.scatter(np.full_like(y_right, x_right), y_right, s=30, c=colors_list)

    for i in range(len(labels)):
        ax.text(x_left-0.05, y_left[i], f"{labels[i]}, {y_left[i]:.2f}", ha="right", va="center", fontsize=9)
        ax.text(x_right+0.05, y_right[i], f"{labels[i]}, {y_right[i]:.2f}", ha="left", va="center", fontsize=9)

    y_all = np.concatenate([y_left, y_right])
    pad = max(1.0, 0.08 * (y_all.max() - y_all.min() if np.isfinite(y_all).all() else 1.0))
    ax.set_ylim(y_all.min()-pad, y_all.max()+pad)
    ax.set_xlim(0, 4)

    ax.vlines([x_left, x_right], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
              colors="lightgray", linestyles="dotted", linewidth=1)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels([left_title, right_title])
    ax.set_title(title)
    for s in ["top","right","left","bottom"]:
        ax.spines[s].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig

def means_by_treatment(df, bl_col, fu_col):
    sub = df[[bl_col, fu_col, "Treatment"]].dropna()
    g = sub.groupby("Treatment")[[bl_col, fu_col]].mean(numeric_only=True)
    g = g.rename(columns={bl_col:"mean_before", fu_col:"mean_after"}).sort_values("mean_before")
    return g

# -------------------------
# Actions: Calculate & Show charts
# -------------------------

calc = st.button("Calculate", type="primary")
charts = {}

if calc:
    if data_edit is None or len(data_edit) == 0 or not has_required_columns(data_edit):
        st.error("Please provide a table with all required columns.")
    else:
        df_scored = compute_scores(data_edit)
        st.success("Calculated.")
        st.dataframe(df_scored, use_container_width=True)
        
        plot_data = pd.melt(
            df_scored,
            id_vars=["Patient ID"],
            value_vars=["OCS_BL", "OCS_FU", "ACT_BL", "ACT_FU", "Exacerbation_BL", "Exacerbation_FU"],
            var_name="Variable",
            value_name="Value"
        )
        plot_data["Type"] = plot_data["Variable"].str.split("_").str[0]
        plot_data["Time"] = plot_data["Variable"].str.split("_").str[1]

        # -------------------------
        # 3. Distributions
        # -------------------------
        st.subheader("Distributions")
        fig_dist = plot_distributions(plot_data)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(fig_dist["OCS"])
            charts[f"OCS_dist.png"] = fig_dist["OCS"]
        with col2:
            st.pyplot(fig_dist["ACT"])
            charts[f"ACT_dist.png"] = fig_dist["ACT"]
        with col1:
            st.pyplot(fig_dist["Exacerbation"])
            charts[f"Exacerbation_dist.png"] = fig_dist["Exacerbation"]
              
        # -------------------------
        # 4. Boxplots 
        # -------------------------
        st.subheader("Boxplots (Baseline vs Follow-up)")
        fig_box = plot_boxplots(plot_data)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(fig_box["OCS"])
            charts[f"OCS_boxplot.png"] = fig_box["OCS"]
        with col2:
            st.pyplot(fig_box["ACT"])
            charts[f"ACT_boxplot.png"] = fig_box["ACT"]
        with col3:
            st.pyplot(fig_box["Exacerbation"])
            charts[f"Exacerbation_boxplot.png"] = fig_box["Exacerbation"]
        
        

        # -------------------------
        # 5. Slope charts
        # -------------------------
        st.subheader("Slope charts (means by Biologic)")

        act_means = means_by_treatment(df_scored, "ACT_BL", "ACT_FU")
        ocs_means = means_by_treatment(df_scored, "OCS_BL", "OCS_FU")
        exa_means = means_by_treatment(df_scored, "Exacerbation_BL", "Exacerbation_FU")        


        col1, col2, col3 = st.columns(3)
        with col1:
            fig1 = slope_chart_means(act_means, "ACT_BL", "ACT_FU",
                                     "Slope Chart of Means by Biologic: ACT")
            st.pyplot(fig1, use_container_width=True)
            charts["act.png"] = fig1
        with col2:
            fig2 = slope_chart_means(ocs_means, "OCS_BL", "OCS_FU",
                                     "Slope Chart of Means by Biologic: OCS")
            st.pyplot(fig2, use_container_width=True)
            charts["ocs.png"] = fig2
        with col3:
            fig3 = slope_chart_means(exa_means, "Exacerbation_BL", "Exacerbation_FU",
                                     "Slope Chart of Means by Biologic: Exacerbations")
            st.pyplot(fig3, use_container_width=True)
            charts["exa.png"] = fig3

        st.session_state["df_scored"] = df_scored
        st.session_state["charts"] = charts


# -------------------------
# PDF Export
# -------------------------
if st.button("Build PDF"):
    df_scored = st.session_state.get("df_scored")
    charts = st.session_state.get("charts", {})
    if df_scored is None:
        st.warning("Please run Calculate first.")
    else:
        # Save chart figures to PNG bytes
        img_buffers = {}
        for name, fig in charts.items():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_buffers[name] = buf

        # Build PDF
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, title="Biologic Response Report")
        styles = getSampleStyleSheet()
        flow = []

        flow.append(Paragraph("<b>Biologic Response Report</b>", styles["Title"]))
        flow.append(Paragraph(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Normal"]))
        flow.append(Spacer(1, 8))
        flow.append(Paragraph("We do not store your data. Generated locally for your session.", styles["Italic"]))
        flow.append(Spacer(1, 12))

        # Summary table (first N rows)
        show_cols = ["Patient ID","Treatment",
                     "OCS_BL","OCS_FU",
                     "ACT_BL","ACT_FU",
                     "Exacerbation_BL","Exacerbation_FU",
                     "Response_score"]
        summary = df_scored[show_cols].copy()
        # Cap to a reasonable number for PDF
        max_rows = 60
        if len(summary) > max_rows:
            summary = summary.head(max_rows)

        tbl_data = [summary.columns.tolist()] + summary.astype(str).values.tolist()
        tbl = Table(tbl_data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
            ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
            ("FONTSIZE",(0,0),(-1,-1), 8),
            ("ALIGN",(0,0),(-1,0),"CENTER"),
        ]))
        flow.append(tbl)
        flow.append(Spacer(1, 12))

        # Charts
        flow.append(Paragraph("<b>Charts</b>", styles["Heading2"]))
        for name in ["act.png","ocs.png","exa.png","OCS_dist.png", "ACT_dist.png", "Exacerbation_dist.png", "/ocs_boxplot.png", "ACT_boxplot.png", "Exacerbation_boxplot.png"]:
            if name in img_buffers:
                img = RLImage(img_buffers[name], width=500, height=320)
                flow.append(img)
                flow.append(Spacer(1, 10))

        # Feedback
        flow.append(Spacer(1, 12))
        flow.append(Paragraph(f'Feedback form: <a href="{FEEDBACK_URL}">Click here</a>', styles["Normal"]))

        doc.build(flow)
        pdf_bytes = pdf_buffer.getvalue()
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="biologic_response_report.pdf",
            mime="application/pdf"
        )
        st.success("PDF ready.")

# -------------------------
# Call to Action 
# -------------------------
st.markdown(
    """
    <div style="background-color:#e6f2ff; padding:15px; border-radius:10px; border:1px solid #3399ff;">
        <h3 style="color:#0059b3;">Do you find this tool useful?</h3>
        <p style="font-size:15px; color:#000;">
            If yes, and you would like to <b>participate in validation</b> 
            and become a <b>co-author in a future publication</b>, 
            please fill in this short questionnaire 👉 
            <a href="https://docs.google.com/forms/d/e/1FAIpQLSdbHHtazLwS3LLaFwR3O--6Ew3vIX5NkFjBDboCPW0OU45gWQ/viewform?usp=header" target="_blank"><b>Questionnaire Link</b></a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()
st.markdown("**Privacy:** We do not store your data on the server. Close the tab to clear your session.")

