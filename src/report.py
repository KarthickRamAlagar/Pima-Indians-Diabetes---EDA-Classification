"""
report.py
IN:  the cleaned DataFrame, the data-quality DataFrame, a dict of matplotlib
     Figure objects (from visualize.py), and the model comparison DataFrame
OUT: a PDF report as raw bytes (in-memory, nothing written to disk) --
     styled to match the Streamlit app's dark theme, one visualization per
     page with its own subtitle.
"""

import io
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
)

PAGE_W, PAGE_H = A4
MARGIN = 0.8 * inch
MAX_IMG_WIDTH = PAGE_W - 2 * MARGIN

# Colors matched to the Streamlit dark theme used in the app
BG = colors.HexColor("#0E1117")
SURFACE = colors.HexColor("#1C1F26")
SURFACE_ALT = colors.HexColor("#12151C")
ACCENT = colors.HexColor("#7F77DD")
ACCENT_DARK = colors.HexColor("#534AB7")
TEXT = colors.HexColor("#FAFAFA")
TEXT_MUTED = colors.HexColor("#B4B2A9")
GRID = colors.HexColor("#3A3D46")

TITLE_STYLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=24,
                              textColor=TEXT, spaceAfter=6, alignment=TA_LEFT)
SUBTITLE_STYLE = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=13,
                                 textColor=ACCENT, spaceAfter=14)
SECTION_STYLE = ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=17,
                                textColor=ACCENT, spaceAfter=12)
PAGE_SUBTITLE_STYLE = ParagraphStyle("page_sub", fontName="Helvetica-Bold", fontSize=15,
                                      textColor=TEXT, spaceAfter=14)
BODY_STYLE = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5,
                             textColor=TEXT_MUTED, spaceAfter=6)
CELL_STYLE = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, textColor=TEXT_MUTED)
CELL_HEADER_STYLE = ParagraphStyle("cell_head", fontName="Helvetica-Bold", fontSize=9, textColor=TEXT)


def _dark_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.restoreState()


def _fig_to_image(fig, max_width=MAX_IMG_WIDTH, max_height=PAGE_H - 3.2 * inch):
    """Convert a matplotlib Figure to a reportlab Image, fit to the page."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    w_in, h_in = fig.get_size_inches()
    aspect = h_in / w_in
    width = max_width
    height = width * aspect
    if height > max_height:
        height = max_height
        width = height / aspect
    return Image(buf, width=width, height=height)


def _df_to_table(df: pd.DataFrame, col_widths=None, include_index=True):
    header = ([Paragraph("", CELL_HEADER_STYLE)] if include_index else []) + \
             [Paragraph(str(c), CELL_HEADER_STYLE) for c in df.columns]
    data = [header]
    for idx, row in df.iterrows():
        line = ([Paragraph(str(idx), CELL_STYLE)] if include_index else []) + \
               [Paragraph(str(v), CELL_STYLE) for v in row.tolist()]
        data.append(line)

    table = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT_DARK),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]
    for r in range(1, len(data)):
        style.append(("BACKGROUND", (0, r), (-1, r), SURFACE if r % 2 else SURFACE_ALT))
    table.setStyle(TableStyle(style))
    return table


def generate_pdf(df: pd.DataFrame, quality_df: pd.DataFrame, figures: dict,
                  metrics_df: pd.DataFrame, knn_k_df: pd.DataFrame = None,
                  knn_k_fig=None, best_compare_df: pd.DataFrame = None,
                  best_compare_fig=None) -> bytes:
    """
    IN:
      df              - cleaned dataset (row/column counts on the cover)
      quality_df      - dtype + cardinality table
      figures         - dict of {title: matplotlib Figure}, one per page
      metrics_df      - KNN(default) vs Naive Bayes comparison table
      knn_k_df        - optional: KNN metrics across k values (table)
      knn_k_fig       - optional: matplotlib Figure, metrics-vs-k line chart
      best_compare_df - optional: best-tuned KNN vs Naive Bayes table
      best_compare_fig- optional: matplotlib Figure, grouped bar chart
    OUT: PDF file content as bytes
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             topMargin=MARGIN, bottomMargin=MARGIN,
                             leftMargin=MARGIN, rightMargin=MARGIN)
    story = []

    # Cover
    story.append(Spacer(1, 1.6 * inch))
    story.append(Paragraph("Assignment 1", TITLE_STYLE))
    story.append(Paragraph("CDC Diabetes Health Indicators &mdash; EDA &amp; Classification (26DS601)",
                            SUBTITLE_STYLE))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Dataset shape: {df.shape[0]} rows &times; {df.shape[1]} columns "
                            f"(source: CDC BRFSS 2015, UCI ML Repository id=891). "
                            f"KNN/Naive Bayes trained on a stratified 15,000-row sample.",
                            BODY_STYLE))
    story.append(PageBreak())

    # Data quality
    story.append(Paragraph("Data Quality Report", SECTION_STYLE))
    story.append(_df_to_table(quality_df, col_widths=[150, 140, 140], include_index=True))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Missing values were imputed using class-conditional medians "
                            "before this report was generated.", BODY_STYLE))
    story.append(PageBreak())

    # One visualization per page, each with its own subtitle
    for title, fig in figures.items():
        story.append(Paragraph("Visualizations", SECTION_STYLE))
        story.append(Paragraph(title, PAGE_SUBTITLE_STYLE))
        story.append(_fig_to_image(fig))
        story.append(PageBreak())

    # Model results
    story.append(Paragraph("Model Results", SECTION_STYLE))
    story.append(Paragraph("KNN vs Naive Bayes", PAGE_SUBTITLE_STYLE))
    story.append(_df_to_table(metrics_df, include_index=False))

    # KNN k-comparison (table + chart, each own page)
    if knn_k_df is not None and not knn_k_df.empty:
        story.append(PageBreak())
        story.append(Paragraph("KNN: k Comparison", SECTION_STYLE))
        story.append(Paragraph("Evaluation metrics by k", PAGE_SUBTITLE_STYLE))
        story.append(_df_to_table(knn_k_df, include_index=False))
        if knn_k_fig is not None:
            story.append(PageBreak())
            story.append(Paragraph("KNN: k Comparison", SECTION_STYLE))
            story.append(Paragraph("Metrics vs k (chart)", PAGE_SUBTITLE_STYLE))
            story.append(_fig_to_image(knn_k_fig))

    # Best model comparison (table + chart, each own page)
    if best_compare_df is not None and not best_compare_df.empty:
        story.append(PageBreak())
        story.append(Paragraph("Best Model Comparison", SECTION_STYLE))
        story.append(Paragraph("Best-tuned KNN vs Naive Bayes", PAGE_SUBTITLE_STYLE))
        story.append(_df_to_table(best_compare_df, include_index=False))
        if best_compare_fig is not None:
            story.append(PageBreak())
            story.append(Paragraph("Best Model Comparison", SECTION_STYLE))
            story.append(Paragraph("Metric-by-metric chart", PAGE_SUBTITLE_STYLE))
            story.append(_fig_to_image(best_compare_fig))

    doc.build(story, onFirstPage=_dark_page, onLaterPages=_dark_page)
    buf.seek(0)
    return buf.getvalue()
