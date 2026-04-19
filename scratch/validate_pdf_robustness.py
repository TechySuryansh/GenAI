import sys
import os

# Set up paths
sys.path.append(os.path.join(os.getcwd(), "src"))

from extensions.pdf_export import generate_retention_pdf


def validate_robustness():
    print("🚀 Starting PDF robustness validation...")

    test_cases = [
        {
            "name": "Extremely long word",
            "risk_summary": "Summary" * 20 + "LONG" * 50,
            "recommendations": "Recs",
        },
        {"name": "Empty strings", "risk_summary": "", "recommendations": ""},
        {
            "name": "Special characters and emojis",
            "risk_summary": "Risk with emojis 🚀🔥 and unicode \u2022 \u2122",
            "recommendations": "Recs",
        },
        {
            "name": "Very long list of sources",
            "risk_summary": "Summary",
            "recommendations": "Recs",
            "sources": ["Source " + str(i) + " " + "A" * 100 for i in range(20)],
        },
    ]

    for case in test_cases:
        print(f"Testing case: {case['name']}...")
        try:
            pdf_bytes = generate_retention_pdf(
                risk_level="HIGH",
                churn_probability=0.85,
                contract="Month-to-month",
                monthly_charges=120.0,
                tenure=2,
                risk_summary=case.get("risk_summary", ""),
                recommendations=case.get("recommendations", ""),
                sources=case.get("sources", []),
                disclaimer="Generic disclaimer content that is somewhat long but should fit.",
            )
            print(f"✅ Success! Generated {len(pdf_bytes)} bytes.")
        except Exception as e:
            print(f"❌ Failed case {case['name']}: {e}")
            import traceback

            traceback.print_exc()
            return False

    print("\n✨ All robustness tests passed!")
    return True


if __name__ == "__main__":
    validate_robustness()
