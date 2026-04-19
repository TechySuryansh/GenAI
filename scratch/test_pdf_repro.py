import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from extensions.pdf_export import generate_retention_pdf


def test_pdf():
    try:
        # Sample data that might cause issues
        risk_summary = "This is a test risk summary with a very long word: " + "A" * 500
        recommendations = "Plan: \n- Item 1\n- Item 2\n" + "B" * 200
        sources = ["Source 1", "Source 2"]
        disclaimer = "Ethics disclaimer."

        pdf_bytes = generate_retention_pdf(
            risk_level="HIGH",
            churn_probability=0.85,
            contract="Month-to-month",
            monthly_charges=100.0,
            tenure=5,
            risk_summary=risk_summary,
            recommendations=recommendations,
            sources=sources,
            disclaimer=disclaimer,
        )
        print(f"Successfully generated PDF: {len(pdf_bytes)} bytes")
    except Exception as e:
        print(f"Failed to generate PDF: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_pdf()
