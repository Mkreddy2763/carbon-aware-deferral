"""Build the PCDS 2026 Word submission from word/blocks.json using the official
Proceeding-Template (converted to template.docx). Only the template's own styles are
used; section/table/figure/reference numbering is left to the template's automatic
numbering, which follows document order (matching the LaTeX numbering).

  python paper/word/tex2blocks.py && python paper/word/build_docx.py
"""
import copy
import json
import os
import re

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
AUTHORS = [  # name, email (order as in the paper)
    ("Bilakanti Mukesh Reddy", "se24uari026@mahindrauniversity.edu.in"),
    ("Manda Siddharth", "se24uari039@mahindrauniversity.edu.in"),
    ("Akhil Varun Varma Kalidindi", "se24uari028@mahindrauniversity.edu.in"),
    ("Rishi Kamishetty", "se24uari025@mahindrauniversity.edu.in"),
    ("Lakshmi Pavan Rangisetti", "se24uari015@mahindrauniversity.edu.in"),
    ("Veeraiah Talagondapati", "veeraiah.talagondapati@mahindrauniversity.edu.in"),
]
DEPT = "Dept. of Computer Science and Engineering"
ORG = "Mahindra University"
CITY = "Hyderabad, Telangana, India"
COL_W = Inches(3.5)
# column widths (inches, sum ~3.45) so labels do not wrap inside the 3.5" column
WIDTHS = {
    "tab:main": [0.62] + [0.405] * 7,
    "tab:util": [1.17] + [0.38] * 6,
    "tab:forecast": [0.5, 0.48] + [0.353] * 7,
}


def clear_body(doc):
    body = doc.element.body
    final = body.find(qn("w:sectPr"))
    for el in list(body):
        if el is not final:
            body.remove(el)
    return final


def sect_break(doc, template_sectpr, cols, space_twips, start="continuous"):
    """Append an empty paragraph that ends the current section with the given columns."""
    p = doc.add_paragraph()
    ppr = p._p.get_or_add_pPr()
    sp = copy.deepcopy(template_sectpr)
    for tag in ("w:headerReference", "w:footerReference", "w:titlePg"):
        for el in sp.findall(qn(tag)):
            sp.remove(el)
    t = sp.find(qn("w:type"))
    if t is None:
        t = OxmlElement("w:type")
        sp.insert(0, t)
    t.set(qn("w:val"), start)
    c = sp.find(qn("w:cols"))
    if c is None:
        c = OxmlElement("w:cols")
        sp.append(c)
    for k in list(c.attrib):
        del c.attrib[k]
    c.set(qn("w:num"), str(cols))
    c.set(qn("w:space"), str(space_twips))
    ppr.append(sp)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    return p


TOKEN = re.compile(r"(⟦_|⟦\^|⟧|\^†)")


def emit(p, t, bold=None, italic=None, size=None):
    """Add text with ⟦_sub⟧ / ⟦^sup⟧ markup (nesting allowed) and ^† as superscript."""
    stack = []
    for tok in TOKEN.split(t):
        if not tok:
            continue
        if tok in ("⟦_", "⟦^"):
            stack.append(tok[1])
            continue
        if tok == "⟧":
            if stack:
                stack.pop()
            continue
        if tok == "^†":
            r = p.add_run("†")
            r.font.superscript = True
        else:
            r = p.add_run(tok)
            if stack:
                if stack[-1] == "_":
                    r.font.subscript = True
                else:
                    r.font.superscript = True
        if bold is not None:
            r.bold = bold
        if italic:
            r.italic = True
        if size is not None:
            r.font.size = size
    return p


def add_runs(p, runs, base_bold=None):
    for t, b, i in runs:
        emit(p, t, bold=True if b else base_bold, italic=i)
    return p


def cell_text(cell, txt, style, size):
    p = cell.paragraphs[0]
    p.style = style
    emit(p, txt, size=size)
    return p


def set_cell_border(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, val in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), "6" if val != "nil" else "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)


def add_table(doc, blk):
    cap = doc.add_paragraph(style="table head")
    emit(cap, blk["caption"])
    rows = blk["rows"]
    ncols = max(sum(c["span"] for c in r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    hdr = blk["header_rows"]
    font = Pt(7) if ncols >= 8 else Pt(7.5)
    t.autofit = False
    # tight cell margins (default 0.08" per side wraps short headers in narrow columns)
    tblPr = t._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), "30")
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)
    widths = WIDTHS.get(blk["label"])
    if widths:
        for row in t.rows:
            for k, cell in enumerate(row.cells):
                cell.width = Inches(widths[k])
        for k, col in enumerate(t.columns):
            col.width = Inches(widths[k])
    for ri, row in enumerate(rows):
        ci = 0
        for c in row:
            cell = t.cell(ri, ci)
            if c["span"] > 1:
                cell = cell.merge(t.cell(ri, ci + c["span"] - 1))
            style = "table col head" if ri < hdr else "table copy"
            p = cell_text(cell, c["text"], style, font)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            ci += c["span"]
        for cell in t.rows[ri].cells:
            edges = {}
            if ri == 0:
                edges["top"] = "single"
            if ri == hdr - 1:
                edges["bottom"] = "single"
            if ri == len(rows) - 1:
                edges["bottom"] = "single"
            if edges:
                set_cell_border(cell, **edges)
    # small gap after the table
    gap = doc.add_paragraph(style="Body Text")
    gap.paragraph_format.space_after = Pt(2)
    return t


def add_figure(doc, blk):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(blk["path"], width=COL_W)
    cap = doc.add_paragraph(style="figure caption")
    emit(cap, blk["caption"])


def add_equation(doc, blk):
    p = doc.add_paragraph(style="equation")
    p.add_run("\t")
    p.add_run().add_picture(blk["path"], width=Inches(3.0))
    p.add_run(f"\t({blk['number']})")


def main():
    blocks = json.load(open(os.path.join(HERE, "blocks.json"), encoding="utf-8"))
    doc = docx.Document(os.path.join(HERE, "template.docx"))
    final_sectpr = clear_body(doc)
    tmpl = copy.deepcopy(final_sectpr)

    for blk in blocks:
        typ = blk["type"]
        if typ == "title":
            p = doc.add_paragraph(style="paper title")
            p.add_run(blk["text"])
            sect_break(doc, tmpl, 1, 720, start="nextPage")
            # authors: 3 across, then 2 across (template layout)
            for group, ncol in ((AUTHORS[:3], 3), (AUTHORS[3:], 2)):
                for k, (name, mail) in enumerate(group):
                    a = doc.add_paragraph(style="Author")
                    a.add_run(name).font.size = Pt(11)
                    for line in (DEPT, ORG, CITY, mail):
                        a.add_run().add_break()
                        r = a.add_run(line)
                        r.italic = line == DEPT
                        r.font.size = Pt(8.5) if line == mail else Pt(10)
                    if k < len(group) - 1:
                        a.add_run().add_break(WD_BREAK.COLUMN)
                sect_break(doc, tmpl, ncol, 216)
        elif typ == "authors":
            continue
        elif typ == "abstract":
            p = doc.add_paragraph(style="Abstract")
            r = p.add_run("Abstract—")
            r.italic = True
            r.bold = True
            add_runs(p, blk["runs"], base_bold=True)
        elif typ == "keywords":
            p = doc.add_paragraph(style="Keywords")
            r = p.add_run("Keywords—")
            r.italic = True
            r.bold = True
            add_runs(p, blk["runs"], base_bold=True)
        elif typ == "heading":
            style = "Heading 1" if blk["number"] else "Heading 5"
            p = doc.add_paragraph(style=style)
            p.add_run(blk["text"])
        elif typ == "para":
            p = doc.add_paragraph(style="Body Text")
            add_runs(p, blk["runs"])
        elif typ == "bullet":
            p = doc.add_paragraph(style="bullet list")
            add_runs(p, blk["runs"])
        elif typ == "table":
            add_table(doc, blk)
        elif typ == "figure":
            add_figure(doc, blk)
        elif typ == "equation":
            add_equation(doc, blk)
        elif typ == "references":
            h = doc.add_paragraph(style="Heading 5")
            h.add_run("References")
            for e in blk["entries"]:
                p = doc.add_paragraph(style="references")
                runs = [[e["runs"][0][0].lstrip(), *e["runs"][0][1:]]] + e["runs"][1:]
                add_runs(p, runs)

    # the body's final section: two columns, 0.25" gutter, continuous
    fin = doc.element.body.find(qn("w:sectPr"))
    t = fin.find(qn("w:type"))
    if t is None:
        t = OxmlElement("w:type")
        fin.insert(0, t)
    t.set(qn("w:val"), "continuous")
    c = fin.find(qn("w:cols"))
    for k in list(c.attrib):
        del c.attrib[k]
    c.set(qn("w:num"), "2")
    c.set(qn("w:space"), "360")
    # move the final sectPr to the end of the body (python-docx appends before it already)
    out = os.path.join(HERE, "CarbonSched_PCDS2026.docx")
    doc.core_properties.title = blocks[0]["text"]
    doc.core_properties.author = ", ".join(a for a, _ in AUTHORS)
    doc.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
