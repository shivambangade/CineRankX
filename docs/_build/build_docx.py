"""Build CineRankX_Complete_Project_Study_and_Viva_Guide.docx from the markdown chapters.

Markdown subset: #..#### headings, paragraphs, '- ' bullets (two-space indent = level 2), '1. ' numbered
items, ``` code blocks, | tables, '> ' callouts, $$ display math $$ (one line), ![caption](figs/x.png),
<<<pagebreak>>>, and inline **bold**, *italic*, `code`.
"""

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BUILD = Path(__file__).parent
CHAPTERS = sorted((BUILD / "chapters").glob("*.md"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else BUILD.parent / "CineRankX_Complete_Project_Study_and_Viva_Guide.docx"
SLATE, ORANGE, BLUE = RGBColor(0x0F, 0x17, 0x2A), RGBColor(0xEA, 0x58, 0x0C), RGBColor(0x25, 0x63, 0xEB)
BODY_FONT, CODE_FONT = "Calibri", "Consolas"


def shade(element, fill: str) -> None:
    pr = element.get_or_add_pPr() if hasattr(element, "get_or_add_pPr") else element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    pr.append(shd)


def border(par, side="left", color="EA580C", size=18, space=6) -> None:
    pPr = par._p.get_or_add_pPr()
    borders = pPr.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr"); pPr.append(borders)
    el = OxmlElement(f"w:{side}")
    for k, v in (("val", "single"), ("sz", str(size)), ("space", str(space)), ("color", color)):
        el.set(qn(f"w:{k}"), v)
    borders.append(el)


def set_font(run, name=BODY_FONT, size=None, bold=None, italic=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size: run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if italic is not None: run.italic = italic
    if color is not None: run.font.color.rgb = color


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\s][^*]*?\*)")


def add_inline(par, text: str, size=None, color=None, base_bold=False) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            r = par.add_run(part[2:-2].replace("`", "")); set_font(r, size=size, bold=True, color=color)
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); set_font(r, CODE_FONT, (size or 10.5) - 1, color=RGBColor(0xB4, 0x30, 0x0A))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = par.add_run(part[1:-1]); set_font(r, size=size, italic=True, color=color)
        else:
            r = par.add_run(part); set_font(r, size=size, bold=base_bold or None, color=color)


def setup(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.0)
    st = doc.styles
    n = st["Normal"]; n.font.name = BODY_FONT; n.font.size = Pt(10.5)
    n.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    n.paragraph_format.space_after = Pt(6); n.paragraph_format.line_spacing = 1.12
    for name, size, before, after, color in (("Heading 1", 21, 0, 10, SLATE), ("Heading 2", 15, 16, 6, ORANGE),
                                              ("Heading 3", 12.5, 12, 4, BLUE), ("Heading 4", 11, 8, 3, SLATE)):
        h = st[name]; h.font.name = BODY_FONT; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = color
        h.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT); h.element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
        h.element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
        h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(before), Pt(after)
        h.paragraph_format.keep_with_next = True
    for name in ("List Bullet", "List Bullet 2"):
        st[name].font.name = BODY_FONT; st[name].font.size = Pt(10.5)
        st[name].paragraph_format.space_after = Pt(2)


def field(par, instr: str) -> None:
    for kind, text in (("begin", None), (None, instr), ("separate", None), ("text", "1"), ("end", None)):
        r = par.add_run()
        if kind in ("begin", "separate", "end"):
            c = OxmlElement("w:fldChar"); c.set(qn("w:fldCharType"), kind); r._r.append(c)
        elif kind == "text":
            r.text = text
        else:
            t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = text; r._r.append(t)


def header_footer(doc: Document) -> None:
    sec = doc.sections[0]
    sec.different_first_page_header_footer = True
    hp = sec.header.paragraphs[0]; hp.text = ""
    r = hp.add_run("CineRankX - Complete Project Study and Viva Guide"); set_font(r, size=8.5, color=RGBColor(0x64, 0x74, 0x8B))
    border(hp, "bottom", "E2E8F0", 6, 4)
    fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("Page "); set_font(r, size=8.5, color=RGBColor(0x64, 0x74, 0x8B))
    field(fp, "PAGE")


def cover(doc: Document, meta: dict) -> None:
    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph(); r = p.add_run("CineRankX"); set_font(r, size=44, bold=True, color=SLATE)
    p = doc.add_paragraph(); r = p.add_run("Complete Project Study and Viva Guide"); set_font(r, size=24, bold=True, color=ORANGE)
    border(p, "bottom", "EA580C", 24, 8)
    p = doc.add_paragraph(); r = p.add_run("An end-to-end technical learning resource: history, mathematics, code walkthrough, evaluation, "
                                           "demonstration script and 150+ viva questions"); set_font(r, size=12, color=RGBColor(0x47, 0x55, 0x69))
    for _ in range(3):
        doc.add_paragraph()
    for line in (meta["dept"], meta["college"], meta["year"], "", meta["built"], meta["verify"]):
        p = doc.add_paragraph(); r = p.add_run(line); set_font(r, size=11, color=SLATE)
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("Every statement in this guide is grounded in the repository, its git history and its saved evaluation artifacts. "
                  "Where something could not be verified it is marked UNVERIFIED."); set_font(r, size=9.5, italic=True, color=RGBColor(0x64, 0x74, 0x8B))
    doc.add_page_break()
    p = doc.add_paragraph(); r = p.add_run("Table of Contents"); set_font(r, size=21, bold=True, color=SLATE)
    p = doc.add_paragraph(); field(p, 'TOC \\o "1-2" \\h \\z \\u')
    p = doc.add_paragraph(); r = p.add_run("(If page numbers are missing, right-click the table in Word and choose Update Field.)")
    set_font(r, size=8.5, italic=True, color=RGBColor(0x94, 0xA3, 0xB8))
    doc.add_page_break()


def add_code(doc: Document, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        p = doc.add_paragraph(); pf = p.paragraph_format
        pf.space_after = Pt(0); pf.space_before = Pt(0); pf.left_indent = Cm(0.3); pf.line_spacing = 1.0
        shade(p._p, "F1F5F9")
        r = p.add_run(line if line else " "); set_font(r, CODE_FONT, 8.3, color=SLATE)
        if i == 0:
            pf.space_before = Pt(3)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_table(doc: Document, rows: list[list[str]]) -> None:
    ncols = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncols); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(ncols):
            cell = t.cell(i, j); cell.text = ""
            par = cell.paragraphs[0]; par.paragraph_format.space_after = Pt(1)
            add_inline(par, row[j] if j < len(row) else "", size=8.8, base_bold=(i == 0))
            if i == 0:
                shade(cell._tc, "FFEDD5")
            elif i % 2 == 0:
                shade(cell._tc, "F8FAFC")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def parse_table(block: list[str]) -> list[list[str]]:
    rows = []
    for line in block:
        if re.match(r"^\s*\|?\s*:?-{2,}", line):
            continue
        cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
        rows.append([c.strip().replace("\\|", "|") for c in cells])
    return rows


def render(doc: Document, text: str, first: bool) -> None:
    lines = text.splitlines()
    i, para = 0, []

    def flush():
        nonlocal para
        if para:
            p = doc.add_paragraph(); add_inline(p, " ".join(s.strip() for s in para)); para = []

    while i < len(lines):
        line = lines[i]
        if line.strip() == "<<<pagebreak>>>":
            flush(); doc.add_page_break(); i += 1; continue
        if line.startswith("```"):
            flush(); i += 1; code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i].rstrip("\n")); i += 1
            add_code(doc, code); i += 1; continue
        if line.startswith("|"):
            flush(); block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            add_table(doc, parse_table(block)); continue
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush(); level = len(m.group(1))
            if level == 1 and not first:
                doc.add_page_break()
            first = False
            doc.add_heading(re.sub(r"[`*]", "", m.group(2)).strip(), level=level); i += 1; continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)", line)
        if m:
            flush(); path = BUILD / m.group(2)
            if path.exists():
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(path), width=Cm(16.0)); p.paragraph_format.keep_with_next = True
                c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = c.add_run(m.group(1)); set_font(r, size=9, italic=True, color=RGBColor(0x47, 0x55, 0x69))
            i += 1; continue
        if line.startswith("$$"):
            flush(); eq = line.strip().strip("$").strip()
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; shade(p._p, "F8FAFC")
            r = p.add_run(eq); set_font(r, "Cambria Math", 11, italic=True, color=SLATE); i += 1; continue
        if line.startswith(">"):
            flush(); p = doc.add_paragraph(); shade(p._p, "FFF7ED"); border(p, "left")
            p.paragraph_format.left_indent = Cm(0.4)
            add_inline(p, line.lstrip("> ").strip(), size=10); i += 1; continue
        m = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if m:
            flush(); style = "List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet"
            p = doc.add_paragraph(style=style); add_inline(p, m.group(2)); i += 1; continue
        m = re.match(r"^(\d+)\.\s+(.*)", line)
        if m:
            flush(); p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.9); p.paragraph_format.first_line_indent = Cm(-0.6)
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(f"{m.group(1)}.  "); set_font(r, bold=True, color=ORANGE); add_inline(p, m.group(2)); i += 1; continue
        if re.match(r"^\*\*[A-S]\d+ \[.*\*\*$", line.strip()):
            flush(); p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(7); p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.keep_with_next = True; add_inline(p, line.strip(), size=10.5, color=SLATE); i += 1; continue
        if not line.strip():
            flush(); i += 1; continue
        para.append(line); i += 1
    flush()


def main() -> None:
    doc = Document(); setup(doc); header_footer(doc)
    meta = {"dept": "Department of Artificial Intelligence and Machine Learning",
            "college": "Walchand College of Engineering, Sangli", "year": "Mini Project, Academic Year 2026-27",
            "built": f"Document built from repository commit {Path('docs/_build/commit.txt').read_text().strip() if Path('docs/_build/commit.txt').exists() else 'working tree'}",
            "verify": "Metrics verified against saved artifacts in data/processed/ and reports/"}
    cover(doc, meta)
    for k, chapter in enumerate(CHAPTERS):
        render(doc, chapter.read_text(encoding="utf-8"), first=(k == 0))
    doc.core_properties.title = "CineRankX - Complete Project Study and Viva Guide"
    doc.core_properties.author = "CineRankX project team"
    doc.save(OUT)
    print("saved", OUT, f"({len(CHAPTERS)} chapters)")


if __name__ == "__main__":
    main()
