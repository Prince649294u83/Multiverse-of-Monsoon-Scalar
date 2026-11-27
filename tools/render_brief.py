"""Render the two-page design brief from brief_source.md."""
from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


def main() -> None:
    source = Path(__file__).resolve().parents[1] / "brief_source.md"
    output = source.with_name("brief.pdf")
    sections = source.read_text(encoding="utf-8").split("## Page 2")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="BriefTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=17, leading=21, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="BriefHeading", parent=styles["Heading2"],
        fontSize=12, leading=15, spaceBefore=6, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BriefBody", parent=styles["BodyText"],
        fontSize=9.3, leading=12, spaceAfter=6,
    ))

    story = []
    for page_index, section in enumerate(sections):
        for line in section.splitlines():
            text = line.strip()
            if not text:
                story.append(Spacer(1, 3))
            elif text.startswith("# "):
                story.append(Paragraph(text[2:], styles["BriefTitle"]))
            elif text.startswith("## "):
                story.append(Paragraph(text[3:], styles["BriefHeading"]))
            else:
                escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(escaped, styles["BriefBody"]))
        if page_index == 0:
            story.append(PageBreak())

    document = SimpleDocTemplate(
        str(output), pagesize=A4,
        rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
    )
    document.build(story)
    print(output)


if __name__ == "__main__":
    main()
