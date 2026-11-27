"""Render the evidence-led two-page Track 2 design brief."""
from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "brief.pdf"
PAGE_W, PAGE_H = A4
BG = colors.HexColor("#F3F7F4")
INK = colors.HexColor("#18352E")
MUTED = colors.HexColor("#5F7770")
GREEN = colors.HexColor("#177456")
MINT = colors.HexColor("#BEE9D5")
PALE = colors.HexColor("#E7F2EC")
LINE = colors.HexColor("#D5E5DC")
PURPLE = colors.HexColor("#8955C7")
BLUE = colors.HexColor("#397FB9")
AMBER = colors.HexColor("#E2A84A")
WHITE = colors.white


def top_rect(c: canvas.Canvas, x: float, y: float, w: float, h: float,
             fill: colors.Color, radius: float = 10, stroke: colors.Color | None = None) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.roundRect(x, PAGE_H - y - h, w, h, radius, fill=1, stroke=1 if stroke else 0)


def top_line(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float,
             color: colors.Color = LINE, width: float = 1) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x1, PAGE_H - y1, x2, PAGE_H - y2)


def txt(c: canvas.Canvas, x: float, y: float, value: str, size: float = 9,
        color: colors.Color = INK, font: str = "Helvetica") -> None:
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, PAGE_H - y - size, value)


def wrap_lines(value: str, font: str, size: float, width: float) -> list[str]:
    words = re.sub(r"\s+", " ", value.strip()).split(" ")
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and stringWidth(candidate, font, size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def paragraph(c: canvas.Canvas, x: float, y: float, width: float, value: str,
              size: float = 8.4, leading: float = 12, color: colors.Color = MUTED,
              font: str = "Helvetica", max_lines: int | None = None) -> float:
    lines = wrap_lines(value, font, size, width)
    if max_lines:
        lines = lines[:max_lines]
    for idx, line in enumerate(lines):
        txt(c, x, y + idx * leading, line, size, color, font)
    return y + len(lines) * leading


def section(c: canvas.Canvas, x: float, y: float, number: str, title: str) -> None:
    txt(c, x, y, number, 8, GREEN, "Helvetica-Bold")
    txt(c, x + 31, y - 1, title, 14, INK, "Helvetica-Bold")


def header(c: canvas.Canvas, page: int, title: str, subtitle: str) -> None:
    c.setFillColor(INK)
    c.rect(0, PAGE_H - 132, PAGE_W, 132, fill=1, stroke=0)
    # restrained field-mark motif
    for i, color in enumerate((MINT, GREEN, PURPLE)):
        c.setFillColor(color)
        c.circle(39 + i * 12, PAGE_H - 36, 3.3, fill=1, stroke=0)
    txt(c, 34, 48, "MULTIVERSE OF MONSOON", 8, MINT, "Helvetica-Bold")
    txt(c, 34, 65, title, 23, WHITE, "Helvetica-Bold")
    txt(c, 34, 101, subtitle, 9, colors.HexColor("#D6E6DE"))
    top_rect(c, PAGE_W - 110, 42, 76, 48, colors.HexColor("#25483E"), 12)
    txt(c, PAGE_W - 99, 50, "INDIA", 8, MINT, "Helvetica-Bold")
    txt(c, PAGE_W - 99, 65, "32 × 36", 10, WHITE, "Helvetica-Bold")
    top_line(c, 34, PAGE_H - 22, PAGE_W - 34, PAGE_H - 22, LINE, .8)
    txt(c, 34, PAGE_H - 18, "MULTIVERSE OF MONSOON  ·  DESIGN BRIEF", 7, MUTED, "Helvetica-Bold")
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 7)
    c.drawRightString(PAGE_W - 34, 18, f"{page:02d}  /  02")


def card(c: canvas.Canvas, x: float, y: float, w: float, h: float,
         title: str, body: str, tag: str | None = None,
         accent: colors.Color = GREEN) -> None:
    top_rect(c, x, y, w, h, WHITE, 10, LINE)
    c.setFillColor(accent)
    c.roundRect(x, PAGE_H - y - h, 3, h, 1.5, fill=1, stroke=0)
    title_y = y + 12
    if tag:
        txt(c, x + 14, title_y, tag.upper(), 6.5, accent, "Helvetica-Bold")
        title_y += 12
    txt(c, x + 14, title_y, title, 9.2, INK, "Helvetica-Bold")
    paragraph(c, x + 14, title_y + 17, w - 27, body, 7.5, 10.2)


def field_chip(c: canvas.Canvas, x: float, y: float, w: float,
               code: str, label: str, unit: str, accent: colors.Color) -> None:
    top_rect(c, x, y, w, 34, WHITE, 8, LINE)
    top_rect(c, x + 8, y + 7, 38, 20, accent, 6)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(x + 27, PAGE_H - y - 21, code)
    txt(c, x + 53, y + 6, label, 7.5, INK, "Helvetica-Bold")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 6.8)
    c.drawRightString(x + w - 9, PAGE_H - y - 21, unit)


def draw_page_one(c: canvas.Canvas) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    header(c, 1, "A climate model that can branch.", "Three supplied states in. A physically bounded, local 10+ day rollout out.")

    section(c, 34, 151, "01", "Project & model design")
    paragraph(c, 34, 176, PAGE_W - 68,
              "An offline, CPU-only climate digital twin trained on the supplied 2005–2021 ERA5 data. It receives three daily physical-unit states and evolves them autoregressively; a localized perturbation changes the final input state to create a live alternate branch.",
              8.2, 11.4, INK)

    # three-step model flow
    top_rect(c, 34, 224, PAGE_W - 68, 105, WHITE, 12, LINE)
    flow = [
        (48, "01  INPUT", "3 daily states", "Six coupled fields + day-of-year", PALE, GREEN),
        (221, "02  EVOLVE", "Gated local transport", "Climatology-anchored residual model", colors.HexColor("#E9E2F5"), PURPLE),
        (394, "03  ROLLOUT", "10+ physical days", "Prime or perturbed branch", colors.HexColor("#DDECF4"), BLUE),
    ]
    for x, kicker, title, body, tint, accent in flow:
        top_rect(c, x, 239, 151, 74, tint, 9)
        txt(c, x + 11, 248, kicker, 6.7, accent, "Helvetica-Bold")
        txt(c, x + 11, 262, title, 9.2, INK, "Helvetica-Bold")
        paragraph(c, x + 11, 279, 130, body, 7, 9, MUTED)
    for x in (204, 377):
        top_line(c, x, 276, x + 12, 276, MUTED, 1.2)
        top_line(c, x + 8, 272, x + 12, 276, MUTED, 1.2)
        top_line(c, x + 8, 280, x + 12, 276, MUTED, 1.2)

    section(c, 34, 348, "02", "Six coupled atmospheric fields")
    fields = [
        ("T2M", "2 m temperature", "K", GREEN), ("TP", "Precipitation", "mm/day", BLUE),
        ("U10", "Eastward wind", "m/s", PURPLE), ("V10", "Northward wind", "m/s", PURPLE),
        ("MSL", "Sea-level pressure", "Pa", GREEN), ("TCWV", "Column water vapour", "kg/m²", BLUE),
    ]
    chip_w, gap = 169, 10
    for i, (code, label, unit, accent) in enumerate(fields):
        row, col = divmod(i, 3)
        field_chip(c, 34 + col * (chip_w + gap), 375 + row * 41, chip_w, code, label, unit, accent)

    section(c, 34, 472, "03", "Drift, blur & shock safeguards")
    card(c, 34, 499, 255, 91, "Limit drift", "Residual anomaly retention, seasonal climatology anchoring, multi-step training, bounded transport, and physical output guards.", "Stability")
    card(c, 305, 499, 255, 91, "Preserve structure", "Rain-aware objectives plus spatial-variance and gradient terms target missed rain and over-smoothed long leads.", "Anti-blur", BLUE)
    card(c, 34, 601, 255, 91, "Keep shocks local first", "Local convolutions and bounded wind transport constrain early response; autoregressive steps allow gradual outward spread.", "Causality", PURPLE)
    card(c, 305, 601, 255, 91, "Deterministic physical output", "CPU inference, exact null-shock path, nonnegative precipitation, and contract-range temperature clamps.", "Validity", GREEN)

    top_rect(c, 34, 709, PAGE_W - 68, 77, colors.HexColor("#E8EFEA"), 10)
    txt(c, 48, 721, "KNOWN FAILURE MODES", 7, GREEN, "Helvetica-Bold")
    paragraph(c, 48, 738, PAGE_W - 96,
              "One-degree resolution cannot resolve local convection. Climatology anchoring can suppress rare extremes, and autoregressive uncertainty grows with lead time. A valid, smooth rollout is not proof of forecast skill. This is a scenario tool—not operational NWP or an official warning.",
              7.5, 10.2, INK)
    c.showPage()


def mini_grid(c: canvas.Canvas, x: float, y: float, w: float, h: float, variant: int) -> None:
    cols, rows = 10, 5
    cw, ch = (w - 20) / cols, (h - 22) / rows
    palette = [PALE, MINT, colors.HexColor("#70CDB8"), colors.HexColor("#5594D0"), colors.HexColor("#263878")]
    for row in range(rows):
        for col in range(cols):
            active = (col > 1 and col < 8 and row > 0) or (variant == 1 and col < 3 and row > 2)
            idx = (row + col + variant * 2) % len(palette) if active else 0
            if variant == 2 and (row + col) % 3:
                fill = colors.HexColor("#E9E2F5")
            else:
                fill = palette[idx]
            c.setFillColor(fill)
            c.rect(x + 10 + col * cw, PAGE_H - (y + 10 + (row + 1) * ch), cw - 1, ch - 1, fill=1, stroke=0)


def metric_tile(c: canvas.Canvas, x: float, y: float, w: float,
                value: str, label: str, detail: str, accent: colors.Color) -> None:
    top_rect(c, x, y, w, 61, WHITE, 9, LINE)
    c.setFillColor(accent)
    c.circle(x + 12, PAGE_H - y - 15, 3, fill=1, stroke=0)
    txt(c, x + 21, y + 8, value, 13.5, INK, "Helvetica-Bold")
    txt(c, x + 11, y + 29, label, 7.1, INK, "Helvetica-Bold")
    txt(c, x + 11, y + 43, detail, 6.4, MUTED)


def draw_page_two(c: canvas.Canvas) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    header(c, 2, "See the branch. Read the impact.", "A live dashboard for controlled scenario exploration—not a claim of observed truth.")

    section(c, 34, 149, "04", "What the local validation says")
    txt(c, 34, 171, "2020–21 stress proxy · 62 windows · not organizer hidden cases", 7.2, MUTED)
    tile_gap, tile_w = 8, (PAGE_W - 68 - 3 * 8) / 4
    tiles = [
        ("0.0955", "Mean RMSE skill", "target 0.12", GREEN),
        ("0.0919", "Heavy-rain CSI skill", "target 0.10", BLUE),
        ("10 / 10", "Anti-blur", "proxy points", PURPLE),
        ("2 / 5", "Full shock regimes", "null check passes", AMBER),
    ]
    for i, (value, label, detail, accent) in enumerate(tiles):
        metric_tile(c, 34 + i * (tile_w + tile_gap), 191, tile_w, value, label, detail, accent)
    txt(c, 34, 260, "62/62 outputs were physically valid; day-4 persistence was beaten on 60/62 windows.", 7.2, MUTED)

    section(c, 34, 282, "05", "Live visualization: Prime / Branch / Difference")
    top_rect(c, 34, 310, PAGE_W - 68, 138, WHITE, 11, LINE)
    pane_w, pane_gap, pane_y = 151, 20, 321
    panes = [
        ("TIMELINE PRIME", "Baseline rollout", 0, GREEN),
        ("SHOCK BRANCH", "Same model · altered input", 1, PURPLE),
        ("FIELD DIFFERENCE", "Exact Branch − Prime", 2, BLUE),
    ]
    for i, (title, body, variant, accent) in enumerate(panes):
        x = 46 + i * (pane_w + pane_gap)
        top_rect(c, x, pane_y, pane_w, 99, BG, 8)
        txt(c, x + 9, pane_y + 7, title, 6.3, accent, "Helvetica-Bold")
        txt(c, x + 9, pane_y + 18, body, 6.2, MUTED)
        mini_grid(c, x + 8, pane_y + 35, pane_w - 16, 55, variant)
    # Short arrows communicate sequence without implying spatial propagation speed.
    for x in (210, 381):
        top_line(c, x, 363, x + 9, 363, MUTED, 1)
        top_line(c, x + 5, 360, x + 9, 363, MUTED, 1)
        top_line(c, x + 5, 366, x + 9, 363, MUTED, 1)
    txt(c, 46, 429, "Schematic interface diagram · maps use the live 32 × 36 model field and bundled country geometry.", 6.5, MUTED)

    section(c, 34, 467, "06", "Impact use case · monsoon continuity & rainfall extremes")
    metric_w, metric_gap = 169, 10
    impact = [
        ("Period total", "Accumulated forecast rain", GREEN),
        ("Low-rain run", "Longest run below 2.5 mm/day", BLUE),
        ("Benchmark days", "Days ≥ 10 mm/day", PURPLE),
        ("3-day peak", "Maximum rolling rainfall total", AMBER),
        ("Seasonal departure", "Cumulative difference from climatology", GREEN),
        ("Cell trace", "Daily rainfall at a selected grid cell", BLUE),
    ]
    for i, (title, body, accent) in enumerate(impact):
        row, col = divmod(i, 3)
        card(c, 34 + col * (metric_w + metric_gap), 495 + row * 76, metric_w, 66,
             title, body, None, accent)

    top_rect(c, 34, 656, PAGE_W - 68, 133, colors.HexColor("#E8EFEA"), 10)
    txt(c, 48, 671, "INTERPRETATION & SAFEGUARDS", 7, GREEN, "Helvetica-Bold")
    paragraph(c, 48, 689, PAGE_W - 96,
              "Prime, Branch, and Branch-minus-Prime views are computed from the live /simulate rollouts. India summaries use cosine-latitude and land-fraction weighting; pinning a cell reveals its daily rainfall trace. The Shock Lab includes warm-ocean, monsoon-stall, null, and custom physical-delta controls, with per-channel response summaries.",
              7.4, 10, INK)
    paragraph(c, 48, 738, PAGE_W - 96,
              "The 10 mm/day cutoff is a model benchmark, not an IMD warning. Low-rain runs are not drought diagnoses. Flood probability and crop-water stress are not modeled: crop stage, evapotranspiration, soil water, drainage, and local vulnerability are absent. Scenario differences show model sensitivity—not likelihood, uncertainty, or observed future weather.",
              7.1, 9.3, MUTED)
    c.showPage()


def main() -> None:
    doc = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=1)
    doc.setTitle("Multiverse of Monsoon — Track 2 Design Brief")
    doc.setAuthor("Multiverse of Monsoon Team")
    doc.setSubject("Model design, validation limits, live dashboard, and impact extension")
    draw_page_one(doc)
    draw_page_two(doc)
    doc.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
