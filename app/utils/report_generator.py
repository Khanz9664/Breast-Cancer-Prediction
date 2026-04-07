from fpdf import FPDF
from datetime import datetime
import numpy as np
import pandas as pd

# -- Colour palette -------------------------------------------------------------
_RED = (220,  53,  69)
_GREEN = (40, 167,  69)
_ORANGE = (255, 140,   0)
_BLUE = (102, 126, 234)
_DARK = (33,  37,  41)
_LIGHT = (248, 249, 250)
_MID = (206, 212, 218)
_WHITE = (255, 255, 255)
_GREY = (108, 117, 125)


def _rgb(pdf, colour):
    pdf.set_text_color(*colour)


def _fill(pdf, colour):
    pdf.set_fill_color(*colour)


def _draw(pdf, colour):
    pdf.set_draw_color(*colour)


# -- FPDF subclass --------------------------------------------------------------
class PatientReport(FPDF):
    _title_text = "Breast Cancer Diagnostic - Clinical Risk Report"

    def header(self):
        # coloured banner
        _fill(self, _BLUE)
        # Use self.w to cover entire page width safely
        self.rect(0, 0, self.w, 18, "F")
        self.set_font("Helvetica", "B", 13)
        _rgb(self, _WHITE)
        self.set_y(4)
        # Use 0 and align="C" with margins set to center properly
        self.cell(0, 10, self._title_text, align="C")
        self.ln(16)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        _rgb(self, _GREY)
        self.cell(0, 6,
                  f"Page {self.page_no()} | AI Diagnostics v2.2 - STRICTLY CONFIDENTIAL",
                  align="C")


# -- Helpers --------------------------------------------------------------------
def _section_title(pdf, text):
    """Bold blue-left-border section heading."""
    _fill(pdf, _BLUE)
    # Background bar
    pdf.rect(pdf.l_margin, pdf.get_y(), 2, 8, "F")
    pdf.set_x(pdf.l_margin + 4)
    pdf.set_font("Helvetica", "B", 13)
    _rgb(pdf, _DARK)
    # Explicit width for safety
    epw = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.cell(epw - 4, 8, text, ln=True)
    pdf.ln(2)


def _divider(pdf):
    _draw(pdf, _MID)
    # Use epw + left margin to end exactly at right margin
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)


def _kv_row(pdf, label, value, bold_value=False):
    pdf.set_font("Helvetica", "B", 10)
    _rgb(pdf, _GREY)
    label_w = 70
    pdf.cell(label_w, 7, label, ln=False)
    pdf.set_font("Helvetica", "B" if bold_value else "", 10)
    _rgb(pdf, _DARK)
    # Explicitly calculate remaining width to avoid 'Not enough horizontal space'
    avail_w = pdf.w - pdf.r_margin - pdf.get_x()
    pdf.multi_cell(max(10, avail_w), 7, str(value))


def _prob_bar(pdf, label, pct, colour):
    """Draw a labelled horizontal bar for a probability."""
    LABEL_W = 50   # fixed label column width (mm)
    BAR_W = 100    # Reduced from 110 for better margin safety
    BAR_H = 10
    # Start relative to left margin
    bar_x = pdf.l_margin + LABEL_W + 5
    y = pdf.get_y()

    # bar background
    _fill(pdf, _MID)
    _draw(pdf, _MID)
    pdf.rect(bar_x, y, BAR_W, BAR_H, "FD")

    # filled portion
    filled = max(2, BAR_W * pct)
    _fill(pdf, colour)
    _draw(pdf, colour)
    pdf.rect(bar_x, y, filled, BAR_H, "FD")

    # label text (left column)
    pdf.set_font("Helvetica", "B", 9)
    _rgb(pdf, _DARK)
    pdf.set_xy(pdf.l_margin, y)
    pdf.cell(LABEL_W, BAR_H, label)

    # percentage text (right of bar)
    pdf.set_xy(bar_x + BAR_W + 3, y)
    _rgb(pdf, colour)
    # Explicit width for percentage label
    pdf.cell(15, BAR_H, f"{pct:.1%}")
    pdf.ln(BAR_H + 3)


def _table_header(pdf, cols, widths, fill=_BLUE):
    """Render a coloured table header row."""
    _fill(pdf, fill)
    _draw(pdf, fill)
    pdf.set_font("Helvetica", "B", 9)
    _rgb(pdf, _WHITE)
    for col, w in zip(cols, widths):
        pdf.cell(w, 8, f" {col}", border=1, fill=True)
    pdf.ln()


def _table_row(pdf, cells, widths, colours=None, alternating=False, row_idx=0):
    """Render a single table row, optional per-cell text colours."""
    if alternating and row_idx % 2 == 0:
        _fill(pdf, _LIGHT)
    else:
        _fill(pdf, _WHITE)
    pdf.set_font("Helvetica", "", 9)
    for i, (cell, w) in enumerate(zip(cells, widths)):
        if colours and i < len(colours) and colours[i] is not None:
            _rgb(pdf, colours[i])
        else:
            _rgb(pdf, _DARK)
        pdf.cell(w, 7, f" {cell}", border=1, fill=True)
    pdf.ln()
    _rgb(pdf, _DARK)


# ==============================================================================
#  Public API
# ==============================================================================
def generate_patient_report(
    prediction_text, confidence, proba,
    feature_names, input_df, current_shap,
    insights, threshold, robustness=None
):
    """
    Generates a comprehensive multi-section clinical PDF report.

    Parameters
    ----------
    prediction_text : str          "Malignant" | "Benign"
    confidence      : float        max(proba)
    proba           : array-like   [p_benign, p_malignant]
    feature_names   : list[str]
    input_df        : pd.DataFrame single-row feature input
    current_shap    : array-like | None
    insights        : list[dict]   from get_clinical_insights()
    threshold       : float        decision threshold
    robustness      : dict | None  from calculate_model_robustness()
    """
    is_malignant = (prediction_text == "Malignant")
    result_colour = _RED if is_malignant else _GREEN

    pdf = PatientReport()
    # Safer margins for CI environments
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Calculate effective page width manually for compatibility
    epw = pdf.w - pdf.l_margin - pdf.r_margin

    # -- 1. Report Metadata -----------------------------------------------------
    _section_title(pdf, "1. Report Information")
    _kv_row(pdf, "Generation Date / Time",
            datetime.now().strftime("%Y-%m-%d  %H:%M:%S  (UTC+5:30)"))
    _kv_row(pdf, "System Version", "v2.2 - Clinical Decision Support System")
    _kv_row(pdf, "Decision Threshold", f"{threshold:.2f}  (user-defined)")
    _kv_row(pdf, "Model Type",
            "Random Forest Ensemble (SHAP-Explainable)")
    _divider(pdf)

    # -- 2. Diagnostic Conclusion -----------------------------------------------
    _section_title(pdf, "2. Diagnostic Conclusion")

    # Large result box
    _fill(pdf, result_colour)
    _draw(pdf, result_colour)
    # Use epw for full width box
    pdf.rect(pdf.l_margin, pdf.get_y(), epw, 22, "FD")
    pdf.set_font("Helvetica", "B", 22)
    _rgb(pdf, _WHITE)
    pdf.cell(epw, 22, f"  RESULT:  {prediction_text.upper()}", ln=True)

    pdf.ln(3)
    _kv_row(pdf, "AI Confidence Score",
            f"{confidence:.2%}", bold_value=True)
    _kv_row(pdf, "Malignant Probability",
            f"{proba[1]:.2%}", bold_value=True)
    _kv_row(pdf, "Benign Probability",
            f"{proba[0]:.2%}", bold_value=True)
    pdf.ln(3)
    _divider(pdf)

    # -- 3. Probability Visualisation ------------------------------------------
    _section_title(pdf, "3. Probability Distribution (Visual)")
    pdf.set_font("Helvetica", "I", 9)
    _rgb(pdf, _GREY)
    pdf.cell(0, 6,
             "The bars below visualise the model's predicted class probabilities.",
             ln=True)
    pdf.ln(2)

    pdf.set_x(pdf.l_margin + 4)
    _prob_bar(pdf, "Malignant Risk", proba[1], _RED)
    pdf.set_x(pdf.l_margin + 4)
    _prob_bar(pdf, "Benign Probability", proba[0], _GREEN)

    pdf.ln(2)
    _divider(pdf)

    # -- 4. Diagnostic Robustness -----------------------------------------------
    _section_title(pdf, "4. Diagnostic Robustness")
    if robustness:
        level = robustness.get("stability_level", "N/A")
        score = robustness.get("stability_score", 0.0)
        ci = robustness.get("ci", (0.0, 1.0))

        level_colour = _GREEN if "Highly" in level else (_ORANGE if "Stable" in level else _RED)

        _kv_row(pdf, "Stability Level", level)
        _kv_row(pdf, "Ensemble Agreement", f"{score:.0f} / 100")
        _kv_row(pdf, "95% Confidence Interval",
                f"[{ci[0]:.2%}  -  {ci[1]:.2%}]")

        # mini bar for stability score
        pdf.ln(2)
        pdf.set_x(14)
        pdf.set_font("Helvetica", "B", 9)
        _rgb(pdf, _DARK)
        pdf.cell(0, 6, "Stability Score Gauge:", ln=True)
        pdf.set_x(14)
        _prob_bar(pdf, "Stability", score / 100.0, level_colour)
    else:
        pdf.set_font("Helvetica", "I", 9)
        _rgb(pdf, _GREY)
        pdf.cell(0, 7, "Robustness data not available for this run.", ln=True)
    pdf.ln(2)
    _divider(pdf)

    # -- 5. Patient Input Measurements -----------------------------------------
    _section_title(pdf, "5. Patient Input Measurements (All Features)")
    pdf.set_font("Helvetica", "I", 9)
    _rgb(pdf, _GREY)
    pdf.cell(epw, 6,
             "Full set of tumour morphology values submitted for this analysis.",
             ln=True)
    pdf.ln(2)

    cols_feat = ["Feature", "Value"]
    widths_feat = [130, 60]
    _table_header(pdf, cols_feat, widths_feat)

    for i, feat in enumerate(feature_names):
        val = input_df[feat].iloc[0] if feat in input_df.columns else "N/A"
        val_str = f"{val:.5f}" if isinstance(val, float) else str(val)
        label = feat.replace("_", " ").title()
        _table_row(pdf, [label, val_str], widths_feat,
                   alternating=True, row_idx=i)
    pdf.ln(4)
    _divider(pdf)

    # -- 6. SHAP Feature Contributions -----------------------------------------
    pdf.add_page()
    _section_title(pdf, "6. Key Feature Contributions (SHAP Analysis)")
    pdf.set_font("Helvetica", "I", 9)
    _rgb(pdf, _GREY)
    pdf.multi_cell(
        0, 5,
        "SHAP (SHapley Additive exPlanations) values quantify the contribution of each "
        "tumour characteristic toward the final prediction. Positive values push the "
        "result toward Malignant; negative values push toward Benign.",
    )
    pdf.ln(3)

    if current_shap is not None:
        shap_flat = np.array(current_shap).flatten()
    else:
        shap_flat = np.zeros(len(feature_names))

    min_len = min(len(feature_names), len(shap_flat))
    imp_df = pd.DataFrame({
        "Feature":   [f.replace("_", " ").title() for f in feature_names[:min_len]],
        "Influence": shap_flat[:min_len],
    })
    imp_df["AbsInf"] = imp_df["Influence"].abs()
    top10 = imp_df.sort_values("AbsInf", ascending=False).head(10)

    cols_shap = ["Rank", "Tumour Characteristic", "SHAP Score", "Direction"]
    widths_shap = [15, 90, 35, 50]
    _table_header(pdf, cols_shap, widths_shap)

    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        direction = "-> Malignant" if row["Influence"] > 0 else "-> Benign"
        dir_colour = _RED if row["Influence"] > 0 else _GREEN
        _table_row(
            pdf,
            [str(rank), row["Feature"], f"{row['Influence']:+.4f}", direction],
            widths_shap,
            colours=[None, None, None, dir_colour],
            alternating=True, row_idx=rank,
        )

    pdf.ln(4)
    _divider(pdf)

    # -- 7. Clinical Observations -----------------------------------------------
    _section_title(pdf, "7. Clinical Observations (Rule-Based Flags)")
    pdf.set_font("Helvetica", "I", 9)
    _rgb(pdf, _GREY)
    pdf.multi_cell(
        epw, 5,
        "Observations derived from evidence-based morphological thresholds.",
    )
    pdf.ln(3)

    severity_map = {"high": _RED, "medium": _ORANGE, "low": _GREEN}

    for ins in insights:
        feat = ins.get("feature", "")
        text = ins.get("insight", "")
        sev = ins.get("severity", "low")
        sev_col = severity_map.get(sev, _GREY)
        sev_label = sev.upper()

        # severity badge box
        y0 = pdf.get_y()
        _fill(pdf, sev_col)
        _draw(pdf, sev_col)
        pdf.set_font("Helvetica", "B", 8)
        _rgb(pdf, _WHITE)
        pdf.set_xy(14, y0)
        pdf.cell(22, 6, f" {sev_label}", fill=True)

        pdf.set_font("Helvetica", "B", 9)
        _rgb(pdf, _DARK)
        pdf.set_xy(pdf.l_margin + 28, y0)
        pdf.cell(epw - 28, 6, feat, ln=True)

        pdf.set_font("Helvetica", "", 9)
        _rgb(pdf, _GREY)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(epw, 5, text)
        pdf.ln(2)

    pdf.ln(3)
    _divider(pdf)

    # -- 8. Patient Action Plan -------------------------------------------------
    _section_title(pdf, "8. Patient Action Plan & Recommendations")

    if is_malignant:
        _fill(pdf, (255, 235, 238))  # light red bg
        _draw(pdf, _RED)
        pdf.rect(pdf.l_margin, pdf.get_y(), epw, 10, "FD")
        pdf.set_font("Helvetica", "B", 10)
        _rgb(pdf, _RED)
        pdf.cell(epw, 10,
                 "  IMPORTANT: The AI system has flagged a HIGH MALIGNANCY RISK.",
                 ln=True)
        pdf.ln(3)

        steps = [
            ("Immediate Specialist Referral",
             "Contact an oncologist or breast surgeon within 48-72 hours. This AI result "
             "should be communicated to your healthcare provider immediately."),
            ("Confirmatory Imaging",
             "Request diagnostic mammography and/or breast ultrasound if not already done. "
             "3D tomosynthesis (DBT) is preferred where available."),
            ("Tissue Biopsy",
             "A core needle biopsy or vacuum-assisted biopsy is essential to confirm the "
             "diagnosis and determine tumour type, grade, and receptor status."),
            ("Multidisciplinary Assessment",
             "Ensure evaluation by a tumour board including radiology, pathology, oncology, "
             "and surgery. This is standard of care for any suspected malignancy."),
            ("Genetic Counselling",
             "Discuss family history with your clinician. Testing for BRCA1/BRCA2 mutations "
             "may be recommended if strong familial patterns are present."),
            ("Mental Health & Support",
             "Cancer diagnosis is stressful. Ask your care team about counselling, patient "
             "support groups, and charitable organisations (e.g., Susan G. Komen, Breastcancer.org)."),
            ("Do NOT Delay",
             "Early-stage breast cancer has a >90% five-year survival rate. Acting quickly "
             "on this result significantly improves prognosis."),
        ]
    else:
        _fill(pdf, (232, 255, 237))  # light green bg
        _draw(pdf, _GREEN)
        pdf.rect(pdf.l_margin, pdf.get_y(), epw, 10, "FD")
        pdf.set_font("Helvetica", "B", 10)
        _rgb(pdf, _GREEN)
        pdf.cell(epw, 10,
                 "  The AI system has classified the tumour profile as LIKELY BENIGN.",
                 ln=True)
        pdf.ln(3)

        steps = [
            ("Confirm with Your Clinician",
             "A benign AI result is reassuring, but must be reviewed by your doctor alongside "
             "your clinical examination, imaging, and personal history."),
            ("Routine Follow-Up Imaging",
             "Schedule a clinical breast examination every 6-12 months. Annual mammography "
             "screening is recommended for women aged 40+ (consult your physician)."),
            ("Self-Examination",
             "Perform monthly breast self-checks. Seek medical attention immediately if you "
             "notice new lumps, nipple discharge, skin changes, or persistent pain."),
            ("Healthy Lifestyle",
             "Maintain a balanced diet, exercise regularly (>=150 min/week moderate activity), "
             "avoid smoking, and limit alcohol to reduce long-term risk."),
            ("Know Your Risk Factors",
             "Discuss your full risk profile (family history, density, HRT use) with your GP. "
             "High-risk individuals may benefit from supplemental MRI screening."),
            ("Stay Informed",
             "AI-assisted tools improve but are not infallible. Always combine AI output with "
             "professional clinical judgement and established screening guidelines."),
        ]

    for step_num, (title, body) in enumerate(steps, 1):
        pdf.set_font("Helvetica", "B", 10)
        _rgb(pdf, _DARK)
        pdf.set_x(pdf.l_margin + 4)
        pdf.cell(epw - 4, 7, f"  {step_num}. {title}", ln=True)
        pdf.set_font("Helvetica", "", 9)
        _rgb(pdf, _GREY)
        pdf.set_x(pdf.l_margin + 8)
        pdf.multi_cell(epw - 8, 5, body)
        pdf.ln(2)

    pdf.ln(3)
    _divider(pdf)

    # -- 9. Disclaimer ---------------------------------------------------------
    _section_title(pdf, "9. Important Disclaimer")
    _fill(pdf, (255, 243, 205))
    _draw(pdf, _ORANGE)
    pdf.rect(pdf.l_margin, pdf.get_y(), epw, 38, "FD")
    pdf.set_font("Helvetica", "B", 9)
    _rgb(pdf, (133, 77, 0))
    pdf.set_x(pdf.l_margin + 4)
    pdf.multi_cell(
        epw - 8, 6,
        "DISCLAIMER\n\n"
        "This report is generated by an Artificial Intelligence system for research, "
        "educational, and clinical decision-support purposes ONLY. "
        "It MUST NOT be used as a standalone diagnostic tool, and does not constitute "
        "a formal medical diagnosis. All clinical decisions must be made by qualified, "
        "registered healthcare professionals using this report as one of many inputs.\n\n"
        "Accuracy is not guaranteed. The model may be subject to distributional shift "
        "and should be periodically re-validated against the target population.",
    )

    # -- Output -----------------------------------------------------------------
    try:
        import fpdf as _fpdf_mod
        if hasattr(_fpdf_mod, "__version__") and _fpdf_mod.__version__.startswith("2"):
            out = pdf.output()
        else:
            out = pdf.output(dest="S")
    except Exception:
        try:
            out = pdf.output(dest="S")
        except Exception:
            out = pdf.output()

    if isinstance(out, (bytearray, bytes)):
        return bytes(out)
    elif isinstance(out, str):
        return out.encode("latin-1")
    return bytes()
