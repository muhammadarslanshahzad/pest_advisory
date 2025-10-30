"""
PDF report generation utilities for the Pest Advisory System.
"""

import os
import textwrap
from typing import Dict, Any, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.config.config import REPORT_OUTPUT_DIR, REPORT_FONT_SIZES
from src.utils import sanitize_filename, get_timestamp, build_default_farmer_advice


class PestReportGenerator:
    """
    Creates PDF reports combining encoder insights and visualization assets.
    """

    def __init__(self, output_dir: str = REPORT_OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.font_sizes = REPORT_FONT_SIZES

    def generate_report(
        self,
        encoder_result: Dict[str, Any],
        chart_paths: Dict[str, str],
    ) -> str:
        """
        Create a multi-page PDF report for a single tehsil.

        Args:
            encoder_result: Output from `PestDataEncoder.analyze`.
            chart_paths: Mapping of chart name to image path produced by `PestVisualizer`.

        Returns:
            Path to the generated PDF file.
        """
        context = encoder_result.get("context", {})
        analysis = encoder_result.get("encoder_analysis", {})

        tehsil = context.get("tehsil", "unknown_tehsil")
        timeframe = context.get("timeframe", "unknown_period")
        filename = f"pest_report_{sanitize_filename(tehsil)}_{get_timestamp()}.pdf"
        pdf_path = os.path.join(self.output_dir, filename)

        with PdfPages(pdf_path) as pdf:
            self._add_overview_page(pdf, tehsil, timeframe, context, analysis)
            self._add_insights_page(pdf, context, analysis)
            self._add_farmer_advice_page(pdf, context, analysis)
            self._add_chart_pages(pdf, chart_paths)

        print(f"📄 Report ready: {pdf_path}")
        return pdf_path

    def _add_overview_page(
        self,
        pdf: PdfPages,
        tehsil: str,
        timeframe: str,
        context: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> None:
        """Add summary overview page with key metrics."""
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
        fig.patch.set_facecolor("white")

        title = f"Pest Advisory Report\n{tehsil}"
        subtitle = f"Timeframe: {timeframe}"

        fig.text(
            0.5,
            0.92,
            title,
            ha="center",
            va="center",
            fontsize=self.font_sizes["title"],
            weight="bold",
        )
        fig.text(
            0.5,
            0.87,
            subtitle,
            ha="center",
            va="center",
            fontsize=self.font_sizes["section"],
            weight="bold",
        )

        metrics = [
            (
                "Overall Severity",
                f"{context.get('severity_level', 'N/A')} ({context.get('severity_score', 'N/A')}/30)",
            ),
            ("Risk Level", analysis.get("risk_level", "N/A")),
            (
                "Action Urgency",
                str(analysis.get("action_urgency", "N/A")).replace("_", " "),
            ),
            ("Economic Impact", analysis.get("economic_impact", "N/A")),
            ("Total Spots Surveyed", str(context.get("total_spots", "N/A"))),
            ("Total Area Covered (acres)", f"{context.get('total_area', 'N/A')}"),
            ("Pest Pressure per Acre", f"{context.get('pest_pressure_per_acre', 'N/A')}"),
            (
                "Threat Pests",
                ", ".join(str(p.get("name", "Unknown")) for p in context.get("threat_pests", [])[:4]) or "None",
            ),
        ]

        table_y = 0.75
        line_height = 0.045

        fig.text(
            0.08,
            table_y + 0.02,
            "Survey Summary",
            fontsize=self.font_sizes["section"],
            weight="bold",
        )

        for label, value in metrics:
            fig.text(
                0.1,
                table_y,
                f"{label}:",
                fontsize=self.font_sizes["body"],
                weight="bold",
            )
            fig.text(
                0.45,
                table_y,
                str(value),
                fontsize=self.font_sizes["body"],
            )
            table_y -= line_height

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _add_insights_page(
        self,
        pdf: PdfPages,
        context: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> None:
        """Add page summarizing key insights and threat pests."""
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")

        fig.text(
            0.5,
            0.92,
            "Key Insights & Threats",
            ha="center",
            va="center",
            fontsize=self.font_sizes["section"],
            weight="bold",
        )

        threat_section_y = 0.82
        threat_pests = context.get("threat_pests", [])

        fig.text(
            0.08,
            threat_section_y,
            "Critical Threat Pests",
            fontsize=self.font_sizes["subsection"],
            weight="bold",
        )

        if threat_pests:
            y_cursor = threat_section_y - 0.05
            for pest in threat_pests[:6]:
                pest_line = (
                    f"- {pest.get('name', 'Unknown')} | "
                    f"{pest.get('multiplier', 'N/A')}x ETL | "
                    f"{pest.get('severity', 'N/A')} | "
                    f"Count: {pest.get('count', 'N/A')} (ETL {pest.get('threshold', 'N/A')})"
                )
                fig.text(
                    0.1,
                    y_cursor,
                    pest_line,
                    fontsize=self.font_sizes["body"],
                )
                y_cursor -= 0.035
        else:
            fig.text(
                0.1,
                threat_section_y - 0.05,
                "No pests above critical thresholds detected.",
                fontsize=self.font_sizes["body"],
            )

        insights_section_y = 0.55
        insights = analysis.get("key_insights", [])

        fig.text(
            0.08,
            insights_section_y,
            "Key Insights",
            fontsize=self.font_sizes["subsection"],
            weight="bold",
        )

        if insights:
            y_cursor = insights_section_y - 0.05
            for idx, insight in enumerate(insights[:6], start=1):
                wrapped = textwrap.fill(str(insight), width=90)
                fig.text(
                    0.1,
                    y_cursor,
                    f"{idx}. {wrapped}",
                    fontsize=self.font_sizes["body"],
                )
                y_cursor -= 0.045
        else:
            fig.text(
                0.1,
                insights_section_y - 0.05,
                "No insights available from encoder.",
                fontsize=self.font_sizes["body"],
            )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _add_farmer_advice_page(
        self,
        pdf: PdfPages,
        context: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> None:
        """Add expert advisory page focused on farmer actions."""
        advice_points = self._prepare_farmer_advice(context, analysis)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")

        fig.text(
            0.5,
            0.92,
            "Expert Advisory for Farmers",
            ha="center",
            va="center",
            fontsize=self.font_sizes["section"],
            weight="bold",
        )

        intro = "Follow these practical steps to safeguard the crop based on current field conditions:"
        fig.text(
            0.08,
            0.85,
            textwrap.fill(intro, width=100),
            fontsize=self.font_sizes["body"],
        )

        line_spacing = 0.045

        guidance_cursor = 0.78
        fig.text(
            0.08,
            guidance_cursor,
            "Actionable Guidance (English)",
            fontsize=self.font_sizes["subsection"],
            weight="bold",
        )
        guidance_cursor -= 0.05

        if advice_points:
            for point in advice_points[:12]:
                wrapped = textwrap.wrap(point, width=100) or [point]
                fig.text(
                    0.08,
                    guidance_cursor,
                    f"• {wrapped[0]}",
                    fontsize=self.font_sizes["body"],
                )
                guidance_cursor -= line_spacing

                for continuation in wrapped[1:]:
                    fig.text(
                        0.11,
                        guidance_cursor,
                        continuation,
                        fontsize=self.font_sizes["body"],
                    )
                    guidance_cursor -= line_spacing

                if guidance_cursor < 0.1:
                    break
        else:
            fig.text(
                0.08,
                guidance_cursor,
                "No farmer guidance available.",
                fontsize=self.font_sizes["body"],
            )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _add_chart_pages(self, pdf: PdfPages, chart_paths: Dict[str, str]) -> None:
        """Add one page per chart image, if available."""
        if not chart_paths:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            fig.text(
                0.5,
                0.5,
                "No charts available for this report.",
                ha="center",
                fontsize=self.font_sizes["body"],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return

        chart_titles = {
            "pest_comparison": "Pest Comparison vs ETL Thresholds",
            "severity_gauge": "Severity Gauge",
            "threat_pests": "Critical Threat Pests",
            "disease": "Disease Infection Rates",
            "dashboard": "Pest Advisory Dashboard Overview",
        }

        for chart_key, chart_path in chart_paths.items():
            if not os.path.exists(chart_path):
                continue

            image = plt.imread(chart_path)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.imshow(image)
            ax.axis("off")

            title = chart_titles.get(chart_key, chart_key.replace("_", " ").title())
            fig.suptitle(
                title,
                fontsize=self.font_sizes["subsection"],
                weight="bold",
                y=0.98,
            )
            fig.text(
                0.5,
                0.02,
                os.path.basename(chart_path),
                ha="center",
                fontsize=self.font_sizes["caption"],
                alpha=0.7,
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    def _prepare_farmer_advice(
        self,
        context: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> List[str]:
        """Return farmer advice, preferring LLM output with fallback."""
        advice = self._ensure_list(analysis.get("farmer_advice"))

        if not advice:
            fallback_context = dict(context)
            fallback_context["action_urgency"] = analysis.get("action_urgency")
            advice = build_default_farmer_advice(fallback_context)

        return advice

    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        """Ensure value is represented as a list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, (list, tuple, set)):
            items: List[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    items.append(text)
            return items
        text = str(value).strip()
        return [text] if text else []
