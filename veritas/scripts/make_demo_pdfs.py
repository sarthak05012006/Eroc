"""Generate a small, realistic demo set (needs reportlab: pip install reportlab)."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent.parent / "demo_pdfs"
OUT.mkdir(exist_ok=True)
S = getSampleStyleSheet()
H, B = S["Heading2"], S["BodyText"]


def build(name, parts):
    doc = SimpleDocTemplate(str(OUT / name), pagesize=A4, title=name, leftMargin=56, rightMargin=56, topMargin=56, bottomMargin=56)
    flow = []
    for kind, val in parts:
        if kind == "h":
            flow.append(Paragraph(val, H))
        elif kind == "p":
            flow += [Paragraph(val, B), Spacer(1, 8)]
        elif kind == "br":
            flow.append(PageBreak())
        elif kind == "t":
            t = Table(val, hAlign="LEFT", colWidths=[230, 140])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
            flow += [t, Spacer(1, 12)]
    doc.build(flow)


build("Loan_Agreement_2023.pdf", [
    ("h", "PERSONAL LOAN AGREEMENT"),
    ("p", "Dated: 1 January 2023. Effective from: 1 January 2023. Between Sunrise Bank Ltd. (the Lender) and the Borrower named in the Schedule."),
    ("h", "1. Loan Amount and Tenure"),
    ("p", "The Lender shall advance a personal loan of up to Rs. 10,00,000 (Rupees Ten Lakh). The tenure of the loan shall not exceed 60 months from the date of disbursement."),
    ("h", "2. Interest"),
    ("p", "The Borrower shall pay interest at the rate of 10.5% per annum, calculated on the reducing balance and payable monthly with each instalment."),
    ("h", "3. Eligibility"),
    ("p", "The Borrower must be between 21 and 60 years of age and must have a minimum monthly income of Rs. 25,000 to be eligible for this loan."),
    ("h", "4. Processing Fee"),
    ("p", "A one-time processing fee is payable on disbursement, as per the fee table below, and is non-refundable."),
    ("t", [["Loan Amount", "Processing Fee"], ["Up to Rs. 2,00,000", "1.0% of loan amount"], ["Rs. 2,00,001 to Rs. 5,00,000", "1.5% of loan amount"], ["Above Rs. 5,00,000", "2.0% of loan amount"]]),
    ("h", "5. Instalments"),
    ("p", "The loan shall be repaid in equal monthly instalments (EMI) due on the 5th day of every month. Each instalment comprises principal and interest."),
    ("br", None),
    ("h", "6. Late Payment"),
    ("p", "If the Borrower fails to pay any instalment on its due date, a late payment charge of 2% per month shall be levied on the overdue amount. A grace period of 7 days shall be allowed after the due date before the charge is applied."),
    ("p", "The late payment charge shall accrue from the due date until the date of actual payment and shall be calculated on the overdue instalment only, not on the outstanding principal."),
    ("br", None),
    ("h", "7. Prepayment"),
    ("p", "The Borrower may prepay the loan in full or in part after 6 months from disbursement on payment of a prepayment charge of 2% of the principal amount prepaid."),
    ("h", "8. Governing Law"),
    ("p", "This Agreement shall be governed by the laws of India and subject to the jurisdiction of the courts at Raipur."),
    ("h", "Schedule A - Summary of Key Terms"),
    ("t", [["Item", "Term"], ["Interest rate", "10.5% per annum"], ["Maximum tenure", "60 months"], ["Grace period for late payment", "5 days after the due date"]]),
])

build("Amendment_1_2024.pdf", [
    ("h", "AMENDMENT NO. 1 TO THE PERSONAL LOAN AGREEMENT DATED 1 JANUARY 2023"),
    ("p", "Dated: 15 June 2024. Effective from: 1 July 2024. This Amendment modifies the Personal Loan Agreement dated 1 January 2023 between Sunrise Bank Ltd. and the Borrower (the Agreement)."),
    ("h", "1. Amendment to Clause 6 (Late Payment)"),
    ("p", "Clause 6 of the Agreement is amended. The late payment charge shall be 3% per month on the overdue amount, in place of the earlier rate of 2% per month."),
    ("h", "2. Amendment to Clause 3 (Eligibility)"),
    ("p", "Clause 3 of the Agreement is amended. The Borrower must have a minimum monthly income of Rs. 30,000 to be eligible for this loan."),
    ("h", "3. Other Terms"),
    ("p", "All other terms and conditions of the Agreement remain unchanged and in full force."),
])

build("Amendment_2_2025.pdf", [
    ("h", "AMENDMENT NO. 2 TO THE PERSONAL LOAN AGREEMENT DATED 1 JANUARY 2023 (AS AMENDED BY AMENDMENT NO. 1)"),
    ("p", "Dated: 10 January 2025. Effective from: 1 February 2025. This Amendment further modifies the Personal Loan Agreement dated 1 January 2023, as amended by Amendment No. 1 dated 15 June 2024."),
    ("h", "1. Amendment to Clause 6 (Late Payment)"),
    ("p", "Clause 6 of the Agreement, as amended by Amendment No. 1, is further amended. The late payment charge shall be 2.5% per month on the overdue amount."),
    ("h", "2. Other Terms"),
    ("p", "All other terms of the Agreement and Amendment No. 1 remain unchanged."),
])

build("Amendment_3_2027.pdf", [
    ("h", "AMENDMENT NO. 3 TO THE PERSONAL LOAN AGREEMENT DATED 1 JANUARY 2023"),
    ("p", "Dated: 1 October 2026. Effective from: 1 April 2027. This Amendment modifies the Personal Loan Agreement dated 1 January 2023 as previously amended by Amendment No. 1 and Amendment No. 2."),
    ("h", "1. Amendment to Clause 7 (Prepayment)"),
    ("p", "Clause 7 of the Agreement is amended. No prepayment charge shall be payable on any principal prepaid; the prepayment charge shall be 0% with effect from the date stated above."),
])

build("Product_Brochure_2025.pdf", [
    ("h", "Sunrise Personal Loan - Product Brochure"),
    ("p", "Issued: 5 March 2025. This brochure is for general information and is a marketing document."),
    ("h", "Why choose Sunrise Personal Loan?"),
    ("p", "Borrow up to Rs. 10,00,000 with flexible tenure of up to 84 months. Processing fee: only 0.5% of the loan amount on every loan, whatever the amount."),
    ("p", "Eligibility: salaried and self-employed applicants aged 21 to 60 years with a minimum monthly income of Rs. 20,000."),
    ("p", "Instant approval, minimal documentation and doorstep service. Terms and conditions apply."),
])
print("Wrote:", *sorted(p.name for p in OUT.glob("*.pdf")), sep="\n  ")
