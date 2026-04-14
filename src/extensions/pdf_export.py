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
        self.cell(0, 12, "ChurnGuard AI", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(127, 140, 141)
        self.cell(0, 6, "Customer Retention Action Plan", ln=True, align="C")
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(149, 165, 166)
        self.cell(
            0, 10,
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  |  "
            f"Page {self.page_no()}  |  CONFIDENTIAL",
            align="C",
        )

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(41, 128, 185)
        self.cell(0, 10, title, ln=True)
        self.set_draw_color(41, 128, 185)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(52, 73, 94)
        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        clean = re.sub(r"#{1,4}\s*", "", clean)
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                self.ln(3)
                continue
            if line.startswith(("- ", "* ", "• ")):
                self.set_x(15)
                self.multi_cell(0, 6, f"  {chr(8226)} {line[2:]}")
            elif line.startswith("|"):
                self.set_font("Courier", "", 9)
                self.multi_cell(0, 5, line)
                self.set_font("Helvetica", "", 10)
            else:
                self.multi_cell(0, 6, line)


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

    # Customer Info Box
    pdf.set_fill_color(236, 240, 241)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, f"  Risk Level: {risk_level}  |  Churn Probability: {churn_probability*100:.1f}%", ln=True, fill=True)
    pdf.cell(0, 8, f"  Contract: {contract}  |  Monthly: ${monthly_charges:.2f}  |  Tenure: {tenure} months", ln=True, fill=True)
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
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(52, 73, 94)
            pdf.set_x(15)
            pdf.multi_cell(0, 6, f"{i}. {src}")
            pdf.ln(1)

    if disclaimer:
        pdf.section_title("Disclaimer")
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        clean_disclaimer = re.sub(r"\*\*(.*?)\*\*", r"\1", disclaimer)
        clean_disclaimer = clean_disclaimer.replace(chr(9878) + chr(65039), "").strip()
        pdf.multi_cell(0, 5, clean_disclaimer)

    return pdf.output()
