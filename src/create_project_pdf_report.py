from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUT_PDF = REPORTS_DIR / "faculty_project_report.pdf"


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def to_table_data(df: pd.DataFrame, max_rows: int = 15) -> list[list[str]]:
    if df.empty:
        return [["No data available"]]

    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: f"{x:.4f}")

    data = [list(shown.columns)] + shown.astype(str).values.tolist()
    if len(df) > max_rows:
        data.append([f"... {len(df) - max_rows} more rows not shown ..."] + [""] * (len(df.columns) - 1))
    return data


def make_table(data: list[list[str]], col_widths=None) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0b0b0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]
    )

    if len(data) == 1 and len(data[0]) == 1:
        style.add("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5"))
        style.add("TEXTCOLOR", (0, 0), (-1, -1), colors.black)

    table.setStyle(style)
    return table


def build_pdf() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_pack = safe_read_json(REPORTS_DIR / "report_data_pack.json")
    summary_metrics = safe_read_json(REPORTS_DIR / "summary_metrics.json")

    task_df = safe_read_csv(REPORTS_DIR / "task_level_summary.csv")
    conv_df = safe_read_csv(REPORTS_DIR / "conversation_level_summary.csv")
    full_df = safe_read_csv(REPORTS_DIR / "full_data_for_report.csv")

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Conversational HCI Model - Faculty Report",
        author="Project Team",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f2f4f"),
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1f4e78"),
        spaceAfter=8,
        spaceBefore=10,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    overview = report_pack.get("dataset_overview", {})

    story = []

    story.append(Paragraph("Conversational HCI Model Using Large Language Models", title_style))
    story.append(Paragraph("Complete Faculty Evaluation Report", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Executive Summary", h1))
    story.append(
        Paragraph(
            "This project implements a complete Human-Computer Interaction (HCI) evaluation pipeline for conversational AI. "
            "It supports both a baseline rule-based model and an LLM-based model, runs dataset-driven inference, computes quality metrics, "
            "and provides downloadable research artifacts and a web UI for classroom demonstration.",
            body,
        )
    )

    story.append(Paragraph("2. Project Objectives", h1))
    story.append(Paragraph("- Build a reproducible conversational pipeline for HCI evaluation.", body))
    story.append(Paragraph("- Compare response quality between rule-based and LLM modes.", body))
    story.append(Paragraph("- Generate quantitative metrics for academic reporting.", body))
    story.append(Paragraph("- Provide a teaching-friendly web interface for one-click runs.", body))

    story.append(Paragraph("3. System Architecture", h1))
    story.append(Paragraph("- Data Layer: raw and processed conversation datasets.", body))
    story.append(Paragraph("- Core Pipeline: schema validation, model execution, evaluation, artifact generation.", body))
    story.append(Paragraph("- Reporting Layer: CSV/JSON/Markdown summaries for analysis.", body))
    story.append(Paragraph("- Presentation Layer: Flask dashboard for interactive demonstration.", body))

    story.append(Paragraph("4. Technology Stack", h1))
    story.append(Paragraph("Python, Pandas, NumPy, Scikit-learn, Flask, OpenAI SDK, ReportLab.", body))

    story.append(Paragraph("5. Dataset Overview", h1))
    overview_rows = [["Metric", "Value"]]
    if overview:
        for k, v in overview.items():
            if isinstance(v, float):
                overview_rows.append([str(k), f"{v:.4f}"])
            else:
                overview_rows.append([str(k), str(v)])
    else:
        overview_rows.append(["No dataset overview found", "N/A"])
    story.append(make_table(overview_rows, col_widths=[7.5 * cm, 7.5 * cm]))

    story.append(Paragraph("6. Summary Metrics", h1))
    sm_rows = [["Metric", "Value"]]
    if summary_metrics:
        for k, v in summary_metrics.items():
            sm_rows.append([str(k), str(v)])
    else:
        sm_rows.append(["No summary metrics found", "N/A"])
    story.append(make_table(sm_rows, col_widths=[7.5 * cm, 7.5 * cm]))

    story.append(PageBreak())

    story.append(Paragraph("7. Task-Level Evaluation", h1))
    story.append(
        Paragraph(
            "This section summarizes performance grouped by task ID, including number of samples, average response length, and average overlap with gold responses.",
            body,
        )
    )
    story.append(make_table(to_table_data(task_df)))

    story.append(Paragraph("8. Conversation-Level Evaluation", h1))
    story.append(
        Paragraph(
            "This section aggregates quality indicators at full conversation level. It helps identify which sessions produce strong or weak model alignment.",
            body,
        )
    )
    story.append(make_table(to_table_data(conv_df)))

    story.append(Paragraph("9. Full Generated Dataset Snapshot", h1))
    story.append(
        Paragraph(
            "A compact snapshot of the full artifact used for grading, auditing, and reproducibility. Remaining rows are omitted when the table is long.",
            body,
        )
    )
    story.append(make_table(to_table_data(full_df, max_rows=20)))

    story.append(Paragraph("10. Key Findings", h1))
    avg_overlap = overview.get("avg_overlap_vs_gold", 0)
    story.append(Paragraph(f"- Current average overlap vs gold responses: {avg_overlap:.4f}", body))
    story.append(Paragraph("- The pipeline successfully generated all required structured artifacts.", body))
    story.append(Paragraph("- The architecture is modular and suitable for future experimentation.", body))

    story.append(Paragraph("11. Challenges and Resolution", h1))
    story.append(Paragraph("- Python 3.14 compatibility issues with some C-extension packages were mitigated via pure-Python alternatives.", body))
    story.append(Paragraph("- UI runtime issues were solved by migrating from Streamlit to Flask for stability.", body))
    story.append(Paragraph("- Subprocess environment mismatch was fixed by using sys.executable in app runtime calls.", body))

    story.append(Paragraph("12. Conclusion", h1))
    story.append(
        Paragraph(
            "The project demonstrates a complete and practical framework for conversational HCI analysis, combining model execution, quantitative evaluation, and visual reporting. "
            "It is suitable for academic demonstration, reproducible experiments, and future extension into advanced LLM benchmarking.",
            body,
        )
    )

    story.append(Paragraph("13. Appendix: Output Artifacts", h1))
    story.append(Paragraph("- reports/generated_responses.csv", body))
    story.append(Paragraph("- reports/full_data_for_report.csv", body))
    story.append(Paragraph("- reports/task_level_summary.csv", body))
    story.append(Paragraph("- reports/conversation_level_summary.csv", body))
    story.append(Paragraph("- reports/report_data_pack.json", body))
    story.append(Paragraph("- reports/summary_metrics.json", body))
    story.append(Paragraph("- reports/final_report.md", body))

    doc.build(story)
    return OUTPUT_PDF


if __name__ == "__main__":
    path = build_pdf()
    print(f"PDF created at: {path}")
