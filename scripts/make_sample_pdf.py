"""Generate the synthetic sample PDF used by the demo and the evaluation set.

The document is **fictional demo data**. Northwind Capital is an invented
company; no real customer, pricing or regulatory information is used.

Page breaks are explicit so that the expected source page of every evaluation
question is deterministic.

Usage::

    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "sample" / "northwind_lending_policy.pdf"

PAGE_WIDTH, PAGE_HEIGHT = fitz.paper_size("a4")
MARGIN_X, MARGIN_Y = 56, 64
BODY_TOP = 150
BODY_FONT = "helv"
BODY_SIZE = 10.5
LINE_HEIGHT = 14.5
BULLET_INDENT = "    -   "

DISCLAIMER = (
    "SYNTHETIC SAMPLE DOCUMENT - FICTIONAL DEMO CONTENT. Northwind Capital is "
    "an invented company created for demonstration purposes. This document "
    "does not describe any real product, customer or regulation."
)


class Block(NamedTuple):
    """A renderable body block: a paragraph or a bulleted list."""

    text: str
    is_bulleted: bool = False


def para(*sentences: str) -> Block:
    """A paragraph block. Multiple strings are joined with blank lines."""
    return Block(text="\n\n".join(sentences))


def bullets(*items: str) -> Block:
    """A bulleted list block."""
    return Block(text="\n".join(f"{BULLET_INDENT}{item}" for item in items), is_bulleted=True)


def page(title: str, *blocks: Block) -> tuple[str, list[Block]]:
    """Group blocks under a page title."""
    return title, list(blocks)


# One entry per page, in order. Page numbers in evaluation/questions.json refer
# to these positions.
PAGES: list[tuple[str, list[Block]]] = [
    page(
        "Consumer Lending Policy",
        para(DISCLAIMER),
        bullets(
            "Document control: NW-LP-2024-07",
            "Document owner: Retail Credit Policy Committee",
            "Version: 4.2 (supersedes version 4.1)",
            "Effective date: 1 March 2024",
            "Next review date: 1 March 2025",
            "Classification: Internal - demo use only",
        ),
        para(
            "This policy defines the eligibility criteria, required "
            "documentation, pricing and servicing standards for consumer "
            "lending products offered by Northwind Capital. It applies to all "
            "retail branches and to the direct channel. Where this policy "
            "conflicts with a signed customer agreement, the customer "
            "agreement prevails."
        ),
    ),
    page(
        "1. Loan Eligibility",
        para(
            "Applicants must satisfy all of the following criteria before a "
            "loan application can be approved."
        ),
        bullets(
            "Minimum applicant age: 21 years.",
            "Maximum applicant age at maturity: 65 years.",
            "Minimum monthly net income: 2,500 units.",
            "Maximum loan tenure: 20 years.",
            "Minimum loan tenure: 1 year.",
            "Maximum loan amount: 250,000 units.",
            "Minimum loan amount: 1,000 units.",
        ),
        para(
            "Applicants must have held a continuous employment contract for "
            "at least 12 months, or have operated a registered business for "
            "at least 24 months. Self-employed applicants are assessed on "
            "their last two submitted annual accounts."
        ),
        para(
            "Applicants with an active delinquency on any retail product in "
            "the last 6 months are declined. A delinquency is defined as a "
            "payment that is 30 or more days overdue."
        ),
    ),
    page(
        "2. Required Documentation",
        para(
            "The following documents must be submitted with every new loan "
            "application. Incomplete applications are held for 10 business "
            "days and then closed."
        ),
        bullets(
            "Proof of identity: national identity card or passport.",
            "Proof of address: utility bill issued in the last 90 days.",
            "Proof of income: 3 months of payslips or 2 filed tax returns.",
            "Proof of employment: signed employment contract.",
            "Bank statements for the last 6 months.",
            "Signed credit agreement and direct debit mandate.",
        ),
        para(
            "Self-employed applicants must additionally provide business "
            "registration certificates and audited accounts for the most "
            "recent financial year. Additional documents may be requested for "
            "loan amounts above 100,000 units, at the discretion of the credit "
            "officer."
        ),
    ),
    page(
        "3. Interest, Fees and Repayment",
        para(
            "Interest is calculated on the reducing balance using a fixed "
            "nominal annual rate. The standard annual percentage rate for "
            "unsecured loans is 11.9 percent."
        ),
        para(
            "A one-off processing fee of 1.5 percent of the approved principal "
            "is charged at disbursement. The fee is deducted from the amount "
            "transferred to the customer, and no other origination charge may "
            "be applied."
        ),
        para(
            "Repayment is made through direct debit on the 5th of each month. "
            "Customers may switch to a quarterly repayment schedule by written "
            "request, and this request must be submitted at least 15 days "
            "before the next instalment date."
        ),
        para(
            "Prepayment is permitted at any time without penalty. Early "
            "settlement reduces the total amount payable by the interest that "
            "would have accrued on the early settled balance."
        ),
    ),
    page(
        "4. Insurance and Security",
        para(
            "All loans secured against a property must be covered by a "
            "property insurance policy for the full duration of the loan. The "
            "policy must name the lender as loss payee and must be produced "
            "before disbursement."
        ),
        para(
            "Unsecured loans above 50,000 units require a credit life "
            "insurance policy that covers the outstanding principal in the "
            "event of the death or permanent disability of the borrower."
        ),
        para(
            "Collateral must be independently valued by an approved valuer. "
            "The valuation report is valid for 6 months from the date of "
            "inspection."
        ),
    ),
    page(
        "5. Service Levels and Escalation",
        para(
            "Northwind Capital commits to the following service levels for "
            "retail credit requests."
        ),
        bullets(
            "A credit decision is communicated within 3 business days of "
            "receiving a complete application.",
            "Disbursement occurs within 2 business days of the applicant "
            "accepting the offer.",
            "Complaints receive a first written response within 5 business days.",
        ),
        para(
            "A customer who is dissatisfied with a complaint outcome may "
            "escalate it to the Head of Retail Credit within 30 days of the "
            "first response. The escalation is concluded within 10 business "
            "days."
        ),
    ),
    page(
        "6. Data Retention and Privacy",
        para(
            "Application records and supporting documents are retained for 7 "
            "years from the date of the final settlement of the loan. Documents "
            "submitted with declined applications are destroyed after 12 months."
        ),
        para(
            "Customer data is processed only for credit assessment, servicing "
            "and regulatory reporting. Data is never sold to third parties. "
            "Credit information is shared with registered credit bureaus "
            "monthly."
        ),
        para(
            "Customers may request a copy of the personal data held about "
            "them at any time, free of charge, and may ask for inaccurate data "
            "to be corrected."
        ),
    ),
]


def _render_heading(page_obj: fitz.Page, title: str, page_number: int, page_total: int) -> None:
    """Draw the running header, the page title and the synthetic-data banner."""
    page_obj.insert_text(
        (MARGIN_X, MARGIN_Y),
        "Northwind Capital - Consumer Lending Policy (SYNTHETIC DEMO DATA)",
        fontname="helv",
        fontsize=7.5,
        color=(0.45, 0.45, 0.45),
    )
    page_obj.insert_text(
        (MARGIN_X, MARGIN_Y + 16),
        title,
        fontname="hebo",
        fontsize=15,
        color=(0.08, 0.16, 0.35),
    )
    page_obj.insert_text(
        (PAGE_WIDTH - MARGIN_X - 60, MARGIN_Y),
        f"Page {page_number} of {page_total}",
        fontname="helv",
        fontsize=7.5,
        color=(0.45, 0.45, 0.45),
    )
    page_obj.draw_line(
        (MARGIN_X, MARGIN_Y + 26),
        (PAGE_WIDTH - MARGIN_X, MARGIN_Y + 26),
        color=(0.75, 0.78, 0.85),
        width=0.8,
    )


def _render_block(page_obj: fitz.Page, block: Block, cursor: float) -> float:
    """Draw one body block and return the updated vertical cursor."""
    remaining_height = PAGE_HEIGHT - cursor - MARGIN_Y
    overflow = page_obj.insert_textbox(
        fitz.Rect(MARGIN_X, cursor, PAGE_WIDTH - MARGIN_X, cursor + remaining_height),
        block.text,
        fontname=BODY_FONT,
        fontsize=BODY_SIZE,
        lineheight=1.35,
        align=fitz.TEXT_ALIGN_LEFT if block.is_bulleted else fitz.TEXT_ALIGN_JUSTIFY,
    )
    if overflow < 0:
        raise RuntimeError(
            f"Body block overflows the page by {-overflow:.0f} points: "
            f"{block.text[:60]!r}..."
        )
    # insert_textbox does not report the consumed height, so it is recomputed
    # from the wrapped line count at the configured line height.
    line_count = _estimate_line_count(block)
    return cursor + line_count * LINE_HEIGHT + (10 if block.is_bulleted else 12)


def _estimate_line_count(block: Block) -> int:
    """Estimate the number of rendered lines, accounting for text wrapping."""
    usable_width = PAGE_WIDTH - 2 * MARGIN_X
    chars_per_line = max(20, int(usable_width / (BODY_SIZE * 0.5)))
    total = 0
    for line in block.text.split("\n"):
        total += max(1, -(-len(line) // chars_per_line))
    return total


def build_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    """Create the synthetic sample PDF and write it to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    total = len(PAGES)

    for index, (title, blocks) in enumerate(PAGES, start=1):
        page_obj = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        _render_heading(page_obj, title, index, total)
        cursor = BODY_TOP
        for block in blocks:
            cursor = _render_block(page_obj, block, cursor)

    document.set_metadata(
        {
            "title": "Northwind Capital - Consumer Lending Policy (synthetic demo)",
            "author": "Enterprise RAG Assistant - sample data generator",
            "subject": "Fictional demo document for RAG evaluation",
            "keywords": "synthetic, demo, RAG, sample",
        }
    )
    document.save(output_path, garbage=3, deflate=True)
    document.close()
    return output_path


def main() -> int:
    """Generate the sample PDF and report its size."""
    path = build_pdf()
    size_kb = path.stat().st_size / 1024
    print(f"Wrote {path} ({size_kb:.1f} KB, {len(PAGES)} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
