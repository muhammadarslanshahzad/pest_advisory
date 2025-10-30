"""
Streamlit dashboard for the Pest Advisory System.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def find_project_root(start: Path) -> Path:
    """Locate repository root that contains src/ and outputs/."""
    current = start
    while True:
        if (current / "src").is_dir() and (current / "outputs").is_dir():
            return current
        if current.parent == current:
            return start  # Fallback to starting directory
        current = current.parent


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.config.config import (  # noqa: E402
    CHARTS_OUTPUT_DIR,
    ENCODER_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
)
from src.utils import load_json, sanitize_filename  # noqa: E402


@dataclass
class EncoderRecord:
    label: str
    path: Path
    data: Dict


def list_encoder_results(limit: Optional[int] = None) -> List[EncoderRecord]:
    """Discover encoder results on disk."""
    output_dir = PROJECT_ROOT / ENCODER_OUTPUT_DIR
    if not output_dir.exists():
        return []

    files = sorted(
        [p for p in output_dir.glob("*.json") if "batch_summary" not in p.name],
        key=os.path.getmtime,
        reverse=True,
    )
    if limit:
        files = files[:limit]

    records: List[EncoderRecord] = []
    for file_path in files:
        try:
            data = load_json(file_path)
        except json.JSONDecodeError:
            st.warning(f"Unable to parse {file_path.name}; skipping.")
            continue

        tehsil = data.get("tehsil", "Unknown")
        timeframe = data.get("timeframe", "Unknown")
        label = f"{tehsil} | {timeframe} | {file_path.name}"
        records.append(EncoderRecord(label=label, path=file_path, data=data))

    return records


def display_context_summary(context: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    """Show top-level metrics about the survey."""
    st.subheader("Survey Snapshot")
    metric_cols = st.columns(4)

    metric_cols[0].metric("Tehsil", context.get("tehsil", "N/A"))
    metric_cols[1].metric("Timeframe", context.get("timeframe", "N/A"))
    metric_cols[2].metric(
        "Severity Score",
        f"{context.get('severity_score', 0):.1f}/30",
        context.get("severity_level", "N/A"),
    )
    metric_cols[3].metric("Pest Pressure / Acre", context.get("pest_pressure_per_acre", 0))

    st.write(
        f"**Action Urgency:** {analysis.get('action_urgency', 'N/A')}  |  "
        f"**Risk Level:** {analysis.get('risk_level', 'N/A')}  |  "
        f"**Economic Impact:** {analysis.get('economic_impact', 'N/A')}"
    )


def display_pest_tables(context: Dict[str, Any]) -> None:
    """Render pest and disease tables."""
    pest_df = (
        pd.DataFrame(context.get("pest_data", {}))
        .T.reset_index()
        .rename(columns={"index": "Pest"})
    )
    if not pest_df.empty:
        st.subheader("Pest Populations vs ETL")
        pest_df = pest_df[
            ["Pest", "count", "threshold", "status", "percentage_above"]
        ].rename(
            columns={
                "count": "Avg Count",
                "threshold": "ETL",
                "status": "Status",
                "percentage_above": "% Above ETL",
            }
        )
        st.dataframe(pest_df, use_container_width=True)

    disease_df = pd.DataFrame(context.get("disease_data", {})).T.reset_index()
    if not disease_df.empty:
        st.subheader("Disease Infection Rates (%)")
        disease_df = disease_df.rename(
            columns={
                "index": "Disease",
                "spot_infection": "Spot Infection %",
                "area_infection": "Area Infection %",
            }
        )
        st.dataframe(disease_df, use_container_width=True)


def display_threats(analysis: Dict[str, Any]) -> None:
    """Highlight primary and secondary threats."""
    st.subheader("Threat Assessment")
    col1, col2 = st.columns(2)
    primary = analysis.get("primary_threats", [])
    secondary = analysis.get("secondary_concerns", [])

    col1.markdown("**Primary Threats**")
    if primary:
        col1.write("\n".join(f"- {item}" for item in primary))
    else:
        col1.info("No critical primary threats identified.")

    col2.markdown("**Secondary Concerns**")
    if secondary:
        col2.write("\n".join(f"- {item}" for item in secondary))
    else:
        col2.info("No secondary concerns flagged.")


def display_insights(analysis: Dict[str, Any]) -> None:
    """Show key insights and farmer guidance."""
    st.subheader("Key Insights")
    insights = analysis.get("key_insights", [])
    if insights:
        st.write("\n".join(f"- {insight}" for insight in insights))
    else:
        st.info("No insights available.")

    st.subheader("Farmer Advisory (English)")
    advice = analysis.get("farmer_advice", [])
    if advice:
        st.write("\n".join(f"- {item}" for item in advice))
    else:
        st.info("Advice not available in this record. Re-run encoder to populate guidance.")


def _normalize_chart_paths(charts: Dict[str, str]) -> Dict[str, Path]:
    """Convert chart entries to project-root-relative Paths."""
    normalized: Dict[str, Path] = {}
    for chart_name, chart_path in charts.items():
        path_obj = Path(chart_path)
        if not path_obj.is_absolute():
            path_obj = PROJECT_ROOT / path_obj
        normalized[chart_name] = path_obj
    return normalized


def _discover_charts(context: Dict[str, Any], existing: Dict[str, Path]) -> Dict[str, Path]:
    """Augment chart mapping by scanning disk for matching tehsil."""
    tehsil_slug = sanitize_filename(context.get("tehsil", "") or "")
    if not tehsil_slug:
        return existing

    chart_dir = PROJECT_ROOT / CHARTS_OUTPUT_DIR
    if not chart_dir.exists():
        return existing

    matches = sorted(
        chart_dir.glob(f"*_{tehsil_slug}_*.png"),
        key=os.path.getmtime,
        reverse=True,
    )

    enriched = dict(existing)
    for path in matches:
        chart_key = path.name.split(f"_{tehsil_slug}_")[0]
        if chart_key not in enriched:
            enriched[chart_key] = path
    return enriched


def display_charts(context: Dict[str, Any], charts: Dict[str, str]) -> None:
    """Display generated charts if available."""
    normalized = _normalize_chart_paths(charts or {})
    normalized = _discover_charts(context, normalized)

    if not normalized:
        st.info("No charts found for this run.")
        return

    st.subheader("Visualization Gallery")
    for chart_name, abs_path in normalized.items():
        if not abs_path.exists():
            st.warning(f"{chart_name}: {abs_path} not found on disk.")
            continue
        st.image(str(abs_path), caption=chart_name.replace("_", " ").title(), use_column_width=True)


def _discover_report(context: Dict[str, Any]) -> Optional[Path]:
    tehsil_slug = sanitize_filename(context.get("tehsil", "") or "")
    if not tehsil_slug:
        return None

    reports_dir = PROJECT_ROOT / REPORT_OUTPUT_DIR
    if not reports_dir.exists():
        return None

    matches = sorted(
        reports_dir.glob(f"pest_report_{tehsil_slug}_*.pdf"),
        key=os.path.getmtime,
        reverse=True,
    )

    return matches[0] if matches else None


def display_report_link(context: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Provide a link to the generated PDF report."""
    report_path = result.get("report_path")
    st.subheader("PDF Report")
    if report_path:
        abs_path = Path(report_path)
        if not abs_path.is_absolute():
            abs_path = PROJECT_ROOT / report_path
        if abs_path.exists():
            rel_path = abs_path.relative_to(PROJECT_ROOT)
            st.success(f"Report available: `{rel_path}`")
            with open(abs_path, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF",
                    data=pdf_file,
                    file_name=abs_path.name,
                    mime="application/pdf",
                )
        else:
            st.warning(f"Report metadata found but file missing: {report_path}")
    else:
        fallback_report = _discover_report(context)
        if fallback_report:
            rel_path = fallback_report.relative_to(PROJECT_ROOT)
            st.success(f"Closest report found: `{rel_path}`")
            with open(fallback_report, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF",
                    data=pdf_file,
                    file_name=fallback_report.name,
                    mime="application/pdf",
                )
        else:
            st.info("Report not generated for this run.")


def main() -> None:
    st.set_page_config(
        page_title="Pest Advisory Dashboard",
        layout="wide",
        page_icon="🪲",
    )
    st.title("Pest Advisory System")
    st.caption("Visualize encoder outputs, charts, and farmer advisories from pest survey data.")

    records = list_encoder_results()
    if not records:
        st.warning(
            "No encoder results found. Run the encoder pipeline to generate analysis files "
            f"in `{ENCODER_OUTPUT_DIR}`."
        )
        return

    selected_label = st.selectbox(
        "Select encoder run",
        [record.label for record in records],
    )

    record = next(r for r in records if r.label == selected_label)
    context = record.data.get("context", {})
    analysis = record.data.get("encoder_analysis", {})

    display_context_summary(context, analysis)
    display_threats(analysis)
    display_pest_tables(context)
    display_insights(analysis)

    charts = record.data.get("charts", {})
    display_charts(context, charts)
    display_report_link(context, record.data)

    st.divider()
    st.caption(
        f"Loaded from `{record.path.relative_to(PROJECT_ROOT)}` | Source charts: `{CHARTS_OUTPUT_DIR}` | "
        f"Reports: `{REPORT_OUTPUT_DIR}`"
    )


if __name__ == "__main__":
    main()
