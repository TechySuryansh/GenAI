"""
PDF Export Extension for the AI Retention Strategy Assistant.
Generates a downloadable PDF containing the customer's risk summary,
retention plan, sources, and ethical disclaimer.
"""

import re
from datetime import datetime
from fpdf import FPDF


class RetentionReportPDF(FPDF):
    """Custom PDF class with branded header/footer for retention reports."""

    def header(self):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(44, 62, 80)
        # Use explicit width to avoid "0" width space issues
        full_width = self.w - self.l_margin - self.r_margin
        self.cell(full_width, 12, "ChurnGuard AI", ln=True, align="C")

        self.set_font("Helvetica", "", 9)
        self.set_text_color(127, 140, 141)
        self.cell(full_width, 6, "Customer Retention Action Plan", ln=True, align="C")

        # Draw line
        self.line(
            self.l_margin, self.get_y() + 2, self.w - self.r_margin, self.get_y() + 2
        )
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(149, 165, 166)
        full_width = self.w - self.l_margin - self.r_margin
        self.cell(
            full_width,
            10,
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  |  "
            f"Page {self.page_no()}  |  CONFIDENTIAL",
            align="C",
        )

    def section_title(self, title):
        self.ln(4)
        self.set_x(self.l_margin)  # Reset X to margin
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(41, 128, 185)

        full_width = self.w - self.l_margin - self.r_margin
        self.cell(full_width, 10, title, ln=True)
        self.set_draw_color(41, 128, 185)
        self.line(self.l_margin, self.get_y(), 80, self.get_y())
        self.ln(3)

    def _sanitize(self, text):
        """Remove characters that fpdf's latin-1 encoding cannot handle."""
        if not text:
            return ""
        # Strip markdown bold/heading markers
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"#{1,4}\s*", "", text)
        # Replace common unicode with ASCII equivalents
        replacements = {
            "\u2022": "-",
            "\u2013": "-",
            "\u2014": "--",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2122": "(TM)",
            "\u00ae": "(R)",
            "\u00a9": "(C)",
            "\u2026": "...",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        # Strip any remaining non-latin1 characters (emojis, etc.)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def safe_multi_cell(self, w, h, txt, align="J", **kwargs):
        """Robust wrapper for multi_cell to previz 'Not enough horizontal space' crash."""
        if not txt:
            return

        # Ensure a minimum effective width for rendering
        min_w = 15.0
        safe_w = max(w, min_w)

        # Check if we are too far right to reasonably fit the cell
        if self.get_x() + safe_w > self.w - self.r_margin + 2:
            self.set_x(self.l_margin)

        try:
            # In fpdf2, multi_cell supports align, ln, etc.
            self.multi_cell(safe_w, h, txt, align=align, **kwargs)
        except Exception:
            # Absolute fallback to write() to prevent application crash
            self.write(h, txt + "\n")

    def _sanitize(self, text):
        """Remove characters that fpdf's latin-1 encoding cannot handle."""
        if not text:
            return ""
        # Strip markdown bold/heading markers
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"#{1,4}\s*", "", text)
        # Replace common unicode with ASCII equivalents
        replacements = {
            "\u2022": "-",
            "\u2013": "-",
            "\u2014": "--",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2122": "(TM)",
            "\u00ae": "(R)",
            "\u00a9": "(C)",
            "\u2026": "...",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        # Strip any remaining non-latin1 characters (emojis, etc.)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def body_text(self, text):
        """
        Robust body text rendering that handles markdown-like bullets and long words.
        """
        if not text:
            return

        self.set_font("Helvetica", "", 10)
        self.set_text_color(52, 73, 94)

        margin_left = self.l_margin
        margin_right = self.r_margin
        page_width = self.w

        clean = self._sanitize(str(text))

        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                self.ln(3)
                continue

            # Identify bullets
            indent = 0
            if line.startswith(("- ", "* ")):
                indent = 6
                display_line = f"  {line}"
            else:
                display_line = line

            # Start position and available width
            start_x = margin_left + indent
            eff_width = page_width - start_x - margin_right

            # Defensive Word-Breaking for extremely narrow spaces
            if eff_width < 40:  # If space is tight, break long words
                words = display_line.split(" ")
                processed_words = []
                for word in words:
                    if self.get_string_width(word) > eff_width:
                        # Break word into manageable chunks
                        chunk = ""
                        for char in word:
                            if self.get_string_width(chunk + char) < (eff_width - 5):
                                chunk += char
                            else:
                                processed_words.append(chunk)
                                chunk = char
                        processed_words.append(chunk)
                    else:
                        processed_words.append(word)
                display_line = " ".join(processed_words)

            self.set_x(start_x)
            if line.startswith("|"):  # Special case for table-like lines
                self.set_font("Courier", "", 9)
                self.safe_multi_cell(eff_width, 5, display_line)
                self.set_font("Helvetica", "", 10)
            else:
                self.safe_multi_cell(eff_width, 6, display_line)

            self.set_x(margin_left)


def generate_retention_pdf(
    risk_level: str = "N/A",
    churn_probability: float = 0.0,
    contract: str = "N/A",
    monthly_charges: float = 0.0,
    tenure: int = 0,
    risk_summary: str = "",
    recommendations: str = "",
    sources: list = None,
    disclaimer: str = "",
) -> bytes:
    """
    Generate a Retention Action Plan PDF and return as bytes.
    """
    pdf = RetentionReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    margin_left = pdf.l_margin
    eff_width = pdf.w - margin_left - pdf.r_margin

    # Customer Info Box
    pdf.set_fill_color(240, 244, 248)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(44, 62, 80)

    info_line_1 = f"  Risk Level: {risk_level}  |  Churn Probability: {churn_probability*100:.1f}%"
    info_line_2 = f"  Contract: {contract}  |  Monthly: ${monthly_charges:.2f}  |  Tenure: {tenure} months"

    pdf.cell(eff_width, 10, info_line_1, ln=True, fill=True)
    pdf.cell(eff_width, 8, info_line_2, ln=True, fill=True)
    pdf.ln(4)

    if risk_summary:
        pdf.section_title("Risk Summary")
        pdf.body_text(risk_summary)

    if recommendations:
        pdf.section_title("Retention Action Plan")
        pdf.body_text(recommendations)

    if sources:
        pdf.section_title("Sources & Best Practices")
        for i, src in enumerate(sources, 1):
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(52, 73, 94)
            safe_src = pdf._sanitize(src)
            # Use safe wrapper even here
            pdf.set_x(margin_left + 5)
            pdf.safe_multi_cell(eff_width - 8, 6, f"{i}. {safe_src}")
            pdf.ln(1)

    if disclaimer:
        pdf.section_title("Disclaimer")
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        safe_disclaimer = pdf._sanitize(disclaimer)
        pdf.set_x(margin_left)
        pdf.safe_multi_cell(eff_width, 5, safe_disclaimer)

    return bytes(pdf.output(dest="S"))
