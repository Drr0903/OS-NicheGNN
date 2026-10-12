"""Numbered display equations in the LaTeX and Word outputs of a Markdown manuscript.

LaTeX: pandoc writes display maths as \\[ … \\]; each becomes an equation environment, numbered by LaTeX.
Word: pandoc writes display maths as a paragraph holding one m:oMathPara; each becomes a borderless three-column table
(empty | centred equation | right-aligned number), the usual way to number equations in Word, which keeps the equation in
display style and renders the same in Word and LibreOffice.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path

DISPLAY_TEX = re.compile(r"\\\[(.*?)\\\]", re.S)
PARA_MATH = re.compile(r"<w:p>(?:<w:pPr>(?:(?!</w:pPr>).)*</w:pPr>)?(<m:oMathPara>.*?</m:oMathPara>)</w:p>", re.S)


def number_tex(tex: str) -> tuple[str, int]:
    n = len(DISPLAY_TEX.findall(tex))
    return DISPLAY_TEX.sub(lambda m: "\\begin{equation}\n" + m.group(1).strip() + "\n\\end{equation}", tex), n


def _text_width(xml: str) -> int:
    w = re.search(r'<w:pgSz[^>]*w:w="(\d+)"', xml); m = re.search(r'<w:pgMar[^>]*w:left="(\d+)"[^>]*w:right="(\d+)"', xml) or re.search(r'<w:pgMar[^>]*w:right="(\d+)"[^>]*w:left="(\d+)"', xml)
    page = int(w.group(1)) if w else 11906
    return page - (int(m.group(1)) + int(m.group(2)) if m else 2880)


def _cell(width: int, body: str, jc: str) -> str:
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:spacing w:before="60" w:after="60"/><w:ind w:left="0" w:firstLine="0"/><w:jc w:val="{jc}"/></w:pPr>{body}</w:p></w:tc>')


def _table(math: str, n: int, tw: int) -> str:
    side = tw // 10; mid = tw - 2 * side
    none = "".join(f'<w:{b} w:val="nil"/>' for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
    return (f'<w:tbl><w:tblPr><w:tblW w:w="{tw}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>{none}</w:tblBorders>'
            f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{side}"/><w:gridCol w:w="{mid}"/><w:gridCol w:w="{side}"/></w:tblGrid><w:tr>'
            + _cell(side, "", "left") + _cell(mid, math, "center") + _cell(side, f"<w:r><w:t>({n})</w:t></w:r>", "right") + "</w:tr></w:tbl>")


LETTER_RUN = re.compile(r'(<m:r><m:rPr>((?:(?!</m:rPr>).)*<m:sty m:val="p" />(?:(?!</m:rPr>).)*)</m:rPr><m:t>[A-Za-z]+</m:t></m:r>)'
                        r'((?:<m:r><m:rPr>\2</m:rPr><m:t>[A-Za-z]+</m:t></m:r>)+)', re.S)


def _merge_words(xml: str) -> str:
    """Upright words in equations (softmax, mean, rec, LN …) arrive as one run per letter; merge them into one run so that every
    editor shows a word rather than spaced letters."""
    def sub(m):
        letters = "".join(re.findall(r"<m:t>([A-Za-z]+)</m:t>", m.group(0)))
        return f"<m:r><m:rPr>{m.group(2)}</m:rPr><m:t>{letters}</m:t></m:r>"
    prev = None
    while prev != xml:
        prev, xml = xml, LETTER_RUN.sub(sub, xml)
    return xml


def number_docx(path: Path) -> int:
    """Rewrite every display equation of word/document.xml as a numbered equation table and merge upright words; returns the
    number of display equations."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8"); items = {i: z.read(i) for i in z.namelist()}
    tw = _text_width(xml); k = [0]

    def sub(m):
        k[0] += 1; return _table(m.group(1), k[0], tw)
    xml = _merge_words(PARA_MATH.sub(sub, xml))
    if True:
        tmp = Path(tempfile.mkstemp(suffix=".docx")[1])
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in items.items():
                z.writestr(name, xml.encode("utf-8") if name == "word/document.xml" else data)
        shutil.move(tmp, path)
    return k[0]
