"""Convert the LaTeX paper into a list of structured blocks for the Word build.

Resolves \\res{..}{..}{..}{..} from numbers.tex, \\cite from main.aux (IEEE numeric,
ranges compressed like the cite package), \\ref/\\eqref from main.aux, converts inline
math to Unicode, renders the display equation to PNG via pdflatex, and parses the
generated LaTeX tables and the .bbl bibliography. Output: word/blocks.json.
Every number therefore still comes from analysis/make_tables.py.
"""
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)


def read(p):
    return open(os.path.join(PAPER, p), encoding="utf-8").read()


# ------------------------------------------------------------------ lookups
MACROS = dict(re.findall(r"\\csname r@(.*?)\\endcsname\{(.*?)\}\n", read("numbers.tex")))
AUX = read("main.aux")
CITES = dict(re.findall(r"\\bibcite\{(.*?)\}\{(\d+)\}", AUX))
LABELS = {k: v for k, v in re.findall(r"\\newlabel\{(.*?)\}\{\{(.*?)\}", AUX)}


def res(m):
    key = "@".join(m.groups())
    v = MACROS.get(key)
    if v is None:
        raise SystemExit(f"unresolved macro {key}")
    return v.replace("{,}", ",")


def cite(m):
    nums = sorted(int(CITES[k.strip()]) for k in m.group(1).split(","))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"[{nums[i]}]" if j == i else (f"[{nums[i]}], [{nums[j]}]" if j == i + 1
                                                   else f"[{nums[i]}]–[{nums[j]}]"))
        i = j + 1
    return ", ".join(out)


GREEK = {r"\alpha": "α", r"\lambda": "λ", r"\iota": "ι", r"\rho": "ρ", r"\le": "≤", r"\ge": "≥",
         r"\wedge": "∧", r"\mid": " | ", r"\cdot": "·", r"\Pr": "Pr", r"\times": "×", r"\dagger": "†",
         r"\pm": "±", r"\approx": "≈"}
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def math(s):
    s = s.replace("{=}", "=").replace(r"\,", " ").replace(r"\!", "")
    s = re.sub(r"\\(mathrm|text|operatorname)\{(.*?)\}", r"\2", s)
    s = s.replace(r"\int", "∫").replace(r"\Pr", "Pr")
    for k, v in sorted(GREEK.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(k, v)
    s = re.sub(r"\^\{?\\?dagger\}?", "^†", s)
    # sub/superscripts -> markup ⟦_..⟧ / ⟦^..⟧ rendered by the Word builder;
    # braced groups first (innermost), then single characters
    for _ in range(3):
        s = re.sub(r"([_^])\{([^{}]*)\}", lambda m: "⟦" + m.group(1) + m.group(2) + "⟧", s)
    s = re.sub(r"(?<!⟦)_([A-Za-z0-9])", r"⟦_\1⟧", s)
    s = re.sub(r"(?<!⟦)\^([A-Za-z0-9])", r"⟦^\1⟧", s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    return s


def text(s):
    """Inline LaTeX -> list of runs [text, bold, italic]."""
    s = re.sub(r"\\res\{(.*?)\}\{(.*?)\}\{(.*?)\}\{(.*?)\}", res, s)
    s = re.sub(r"~?\\cite\{(.*?)\}", lambda m: " " + cite(m), s)
    s = re.sub(r"Eq\.~\\(?:eq)?ref\{(.*?)\}", lambda m: f"Eq. ({LABELS[m.group(1)]})", s)
    s = re.sub(r"\\eqref\{(.*?)\}", lambda m: f"({LABELS[m.group(1)]})", s)
    s = re.sub(r"\\ref\{(.*?)\}", lambda m: LABELS[m.group(1)], s)
    s = re.sub(r"\$(.*?)\$", lambda m: math(m.group(1)), s, flags=re.S)
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_").replace("\\#", "#")
    s = s.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”")
    s = s.replace("\\,", " ").replace("\\ ", " ").replace("~", " ").replace("\\\\", "\n")
    s = s.replace("{}", "")
    s = re.sub(r"\s*\n\s*", " ", s) if "\n" not in s.strip("\n") or True else s
    runs = []
    pos = 0
    for m in re.finditer(r"\\(textbf|emph|textit)\{((?:[^{}]|\{[^{}]*\})*)\}", s):
        if m.start() > pos:
            runs.append([s[pos:m.start()], False, False])
        runs.append([m.group(2), m.group(1) == "textbf", m.group(1) != "textbf"])
        pos = m.end()
    runs.append([s[pos:], False, False])
    clean = []
    for t, b, i in runs:
        t = re.sub(r"\\[a-zA-Z]+\*?", "", t).replace("{", "").replace("}", "")
        t = re.sub(r"[ \t]+", " ", t)
        if t:
            clean.append([t, b, i])
    return clean


# ------------------------------------------------------------------ tables
def parse_table(path):
    src = read(path)
    cap = re.search(r"\\caption\{(.*?)\}\n\\label", src, re.S).group(1)
    label = re.search(r"\\label\{(.*?)\}", src).group(1)
    body = src.split("\\toprule", 1)[1].split("\\bottomrule", 1)[0]
    rows, header_rows, seen_mid = [], 0, False
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("\\cmidrule"):
            continue
        if line.startswith("\\midrule"):
            seen_mid = True
            if not header_rows:
                header_rows = len(rows)
            continue
        line = line.rstrip("\\").strip()
        cells = []
        for c in line.split("&"):
            c = c.strip()
            m = re.match(r"\\multicolumn\{(\d+)\}\{.*?\}\{(.*)\}$", c)
            if m:
                cells.append({"text": "".join(r[0] for r in text(m.group(2))), "span": int(m.group(1))})
            else:
                cells.append({"text": "".join(r[0] for r in text(c)).strip(), "span": 1})
        rows.append(cells)
    return {"type": "table", "label": label, "number": LABELS[label],
            "caption": "".join(r[0] for r in text(cap)), "rows": rows, "header_rows": header_rows}


# ------------------------------------------------------------------ equation
def render_equation(src, out_png):
    tex = ("\\documentclass[border=2pt,varwidth=9cm]{standalone}\n\\usepackage{amsmath}\n"
           "\\begin{document}\n$\\displaystyle " + src + "$\n\\end{document}\n")
    d = os.path.join(HERE, "eq")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "eq.tex"), "w").write(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "eq.tex"], cwd=d, capture_output=True)
    subprocess.run(["pdftoppm", "-png", "-r", "600", "-singlefile", "eq.pdf", "eq"], cwd=d, check=True)
    os.replace(os.path.join(d, "eq.png"), out_png)


# ------------------------------------------------------------------ bibliography
def bib_entries():
    src = read("main.bbl")
    out = []
    for m in re.finditer(r"\\bibitem\{(.*?)\}\n(.*?)(?=\n\\bibitem|\n\\end\{thebibliography\})", src, re.S):
        s = m.group(2)
        s = re.sub(r"\\hskip.*?\\relax", " ", s)
        s = re.sub(r"\\BIBentry[A-Za-z]*", "", s)
        s = s.replace("\\newblock", " ")
        s = re.sub(r"\\url\{(.*?)\}", r"\1", s)
        s = re.sub(r"\{\\'(\w)\}", lambda k: k.group(1) + "\u0301", s)
        s = re.sub(r"\{\\\"(\w)\}", lambda k: k.group(1) + "\u0308", s)
        s = re.sub(r"\\'\{?(\w)\}?", lambda k: k.group(1) + "\u0301", s)
        s = re.sub(r"\\\"\{?(\w)\}?", lambda k: k.group(1) + "\u0308", s)
        s = re.sub(r"\{\\`(\w)\}", lambda k: k.group(1) + "\u0300", s)
        s = re.sub(r"\\`\{?(\w)\}?", lambda k: k.group(1) + "\u0300", s)
        s = re.sub(r"\\c\{(\w)\}", r"\1̧", s)
        s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_").replace("\\#", "#")
        import unicodedata
        runs = [[unicodedata.normalize("NFC", r[0]), r[1], r[2]] for r in text(s)]
        out.append({"key": m.group(1), "number": int(CITES[m.group(1)]), "runs": runs})
    return sorted(out, key=lambda e: e["number"])


# ------------------------------------------------------------------ main walk
def section_blocks(path, blocks, figdir):
    src = read(path)
    src = re.sub(r"(?m)^%.*$", "", src)
    # tables pulled in with \input
    for tpath in re.findall(r"\\input\{(tables/.*?)\}", src):
        blocks.append(parse_table(tpath if tpath.endswith(".tex") else tpath + ".tex"))
    src = re.sub(r"\\input\{tables/.*?\}", "", src)
    tokens = re.split(r"(\\section\*?\{.*?\}(?:\\label\{.*?\})?|\\begin\{figure\}.*?\\end\{figure\}|"
                      r"\\begin\{itemize\}.*?\\end\{itemize\}|\\begin\{equation\}.*?\\end\{equation\}|"
                      r"\\begin\{abstract\}.*?\\end\{abstract\}|\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\})",
                      src, flags=re.S)
    for tok in tokens:
        if not tok.strip():
            continue
        if tok.startswith("\\section"):
            m = re.match(r"\\section(\*?)\{(.*?)\}(?:\\label\{(.*?)\})?", tok)
            num = LABELS.get(m.group(3), "") if m.group(3) else ""
            blocks.append({"type": "heading", "number": num, "text": m.group(2)})
        elif tok.startswith("\\begin{abstract}"):
            body = tok[len("\\begin{abstract}"):-len("\\end{abstract}")]
            blocks.append({"type": "abstract", "runs": text(body.strip())})
        elif tok.startswith("\\begin{IEEEkeywords}"):
            body = tok[len("\\begin{IEEEkeywords}"):-len("\\end{IEEEkeywords}")]
            blocks.append({"type": "keywords", "runs": text(body.strip())})
        elif tok.startswith("\\begin{figure}"):
            f = re.search(r"\\includegraphics.*?\{figs/(.*?)\.pdf\}", tok).group(1)
            cap = re.search(r"\\caption\{(.*?)\}\n\\label", tok, re.S).group(1)
            lab = re.search(r"\\label\{(.*?)\}", tok).group(1)
            blocks.append({"type": "figure", "path": os.path.join(figdir, f + ".png"),
                           "number": LABELS[lab], "caption": "".join(r[0] for r in text(cap))})
        elif tok.startswith("\\begin{itemize}"):
            for item in re.split(r"\\item\s+", tok)[1:]:
                item = item.replace("\\end{itemize}", "").strip()
                blocks.append({"type": "bullet", "runs": text(item)})
        elif tok.startswith("\\begin{equation}"):
            body = re.sub(r"\\label\{.*?\}", "", tok[len("\\begin{equation}"):-len("\\end{equation}")]).strip()
            png = os.path.join(HERE, "eq_decomp.png")
            render_equation(body, png)
            blocks.append({"type": "equation", "path": png, "number": LABELS["eq:decomp"]})
        else:
            for para in re.split(r"\n\s*\n", tok):
                if para.strip():
                    blocks.append({"type": "para", "runs": text(para.strip())})


def main():
    main_src = read("main.tex")
    figdir = os.path.join(HERE, "figs")
    title = re.search(r"\\title\{(.*?)\}\n", main_src, re.S).group(1).replace("\\\\", " ")
    names = re.search(r"\\IEEEauthorblockN\{(.*?)\}\n", main_src).group(1)
    aff = re.search(r"\\IEEEauthorblockA\{(.*?)\}\}", main_src, re.S).group(1)
    aff_lines = [a.replace("\\{", "{").replace("\\}", "}").strip() for a in aff.split("\\\\")]
    blocks = [{"type": "title", "text": title},
              {"type": "authors", "names": names, "affiliation": aff_lines}]
    for sec in re.findall(r"\\input\{(sections/.*?)\}", main_src):
        section_blocks(sec if sec.endswith(".tex") else sec + ".tex", blocks, figdir)
    ack = re.search(r"\\section\*\{Acknowledgment\}\n(.*?)\n\n", main_src, re.S).group(1)
    blocks.append({"type": "heading", "number": "", "text": "Acknowledgment"})
    blocks.append({"type": "para", "runs": text(ack)})
    blocks.append({"type": "references", "entries": bib_entries()})
    # sanity: nothing unresolved, no stray LaTeX
    dump = json.dumps(blocks, ensure_ascii=False)
    for bad in ("\\res", "??", "\\cite", "\\ref", "$"):
        if bad in dump:
            raise SystemExit(f"leftover {bad!r} in blocks")
    with open(os.path.join(HERE, "blocks.json"), "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=1)
    print(f"{len(blocks)} blocks -> word/blocks.json")


if __name__ == "__main__":
    main()
