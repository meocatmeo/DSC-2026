"""Vietnamese legal document structure: reflow, hierarchy, article chunking.

Pure stdlib. Every field is read out of the supplied corpus text. Nothing is
generated, no source URL is followed and no external lexicon is used.

The corpus passages come from an HTML to text conversion that hard wraps lines
mid sentence, so the text is reflowed first; the legal hierarchy (Chương / Mục /
Điều) is then recovered and each article becomes one retrieval unit carrying a
breadcrumb of its parents. Documents with no article sequence fall back to Roman
numbered sections and then to sliding word windows, so every document is indexed.
"""
import re
import unicodedata

UPPER = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠ-Ỹ"

# ---------------------------------------------------------------- normalisation

def nfc(text):
    return unicodedata.normalize("NFC", text or "")


def reflow(text):
    """Undo the hard line wrapping of the supplied HTML-to-text conversion.

    Visual lines are separated by blank lines and frequently break mid sentence.
    A line is joined to the previous one unless the previous line ends a sentence
    or the next line opens a structural heading. No text is added or removed.
    """
    text = nfc(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(x.split()) for x in text.split("\n")]
    lines = [x for x in lines if x]
    out = []
    for line in lines:
        if out and not _breaks_before(line) and not _ends_block(out[-1]):
            out[-1] = out[-1] + " " + line
        else:
            out.append(line)
    return "\n".join(out)


_HEADING_START = re.compile(
    rf"^(Điều\s+\d+[a-zA-Z]?\s*[.:]|Chương\s+[IVXLCDM\d]+|Mục\s+\d+|Phần\s+[IVXLCDM\d]|"
    rf"PHẦN\s+[IVXLCDM\d]|CHƯƠNG\s+[IVXLCDM\d]|MỤC\s+\d+|\d+[.)]\s|[a-zđ][.)]\s|"
    rf"[IVXLCDM]{{1,4}}[.)]\s|Số\s*:)"
)


def _breaks_before(line):
    return bool(_HEADING_START.match(line))


def _ends_block(line):
    return bool(re.search(r"[.;:!?…]$", line)) or line.isupper()


# ------------------------------------------------------------------- doc number

DOC_NUMBER = re.compile(r"(?m)^Số\s*:\s*(\S+)")
NAME_NUMBER = re.compile(r"(\d+)-(\d{4})-([A-Za-z\-]+)")

DOC_KINDS = (
    "Nghị định", "Nghị quyết", "Thông tư liên tịch", "Thông tư", "Quyết định",
    "Luật", "Bộ luật", "Pháp lệnh", "Chỉ thị", "Công văn", "Kế hoạch",
    "Hiến pháp", "Văn bản hợp nhất", "Lệnh", "Thông báo", "Báo cáo", "Hướng dẫn",
    "Quy chuẩn", "Tiêu chuẩn", "Sắc lệnh", "Điều lệ", "Quy định", "Quy chế",
)


def document_number(text):
    match = DOC_NUMBER.search(text)
    return match.group(1).strip().rstrip(".") if match else ""


def slug_words(name):
    """Turn the supplied ASCII slug title into spaced words."""
    return " ".join(x for x in re.split(r"[-_]+", nfc(name or "")) if x)


BOILERPLATE = (
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT", "Độc lập - Tự do - Hạnh phúc",
    "ĐỘC LẬP - TỰ DO - HẠNH PHÚC",
)


def _upper_word(word):
    letters = [c for c in word if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def subject_line(lines, head_lines=16, limit=60):
    """Longest run of uppercase words in the document header.

    In this corpus the accented document kind and subject are printed in capitals
    between the date and the first 'Căn cứ' clause, so the longest capitalised run
    of the header recovers it without guessing at layout.
    """
    text = " ".join(lines[:head_lines])
    for phrase in BOILERPLATE:
        text = text.replace(phrase, " | ")
    words = text.split()
    best, current = [], []
    for word in words:
        if _upper_word(word) or (current and not any(c.isalpha() for c in word)):
            current.append(word)
        else:
            if len(current) > len(best):
                best = current
            current = []
    if len(current) > len(best):
        best = current
    while best and not any(c.isalpha() for c in best[-1]):
        best.pop()
    return " ".join(best[:limit]) if len(best) >= 2 else ""


# -------------------------------------------------------------------- hierarchy

ARTICLE = re.compile(rf"(?m)^Điều\s+(\d+[a-zA-Z]?)\s*[.:]\s*(.*)$")
CHAPTER = re.compile(rf"(?mi)^(Chương|PHẦN|Phần)\s+([IVXLCDM\d]+)\s*[.:]?\s*(.*)$")
SECTION = re.compile(rf"(?mi)^(Mục)\s+(\d+)\s*[.:]?\s*(.*)$")
ROMAN_SECTION = re.compile(rf"(?m)^([IVXLCDM]{{1,5}})\s*[.)]\s+([{UPPER}][^\n]*)$")


def _roman(value):
    table = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, previous = 0, 0
    for char in reversed(value.upper()):
        current = table.get(char, 0)
        total += current if current >= previous else -current
        previous = max(previous, current)
    return total


def _monotone(items):
    """Longest non-decreasing run by number, to reject in-text cross references."""
    if not items:
        return []
    best = []
    lengths = [1] * len(items)
    previous = [-1] * len(items)
    for i in range(len(items)):
        for j in range(i):
            if items[j][0] <= items[i][0] and lengths[j] + 1 > lengths[i]:
                lengths[i], previous[i] = lengths[j] + 1, j
    index = max(range(len(items)), key=lambda i: lengths[i])
    while index >= 0:
        best.append(items[index])
        index = previous[index]
    return best[::-1]


def parse_units(text):
    """Split reflowed text into (breadcrumb, body) legal units in document order.

    Articles are preferred. Documents without an article sequence fall back to
    Roman numbered sections, then to the whole body.
    """
    articles = [(int(re.match(r"\d+", m.group(1)).group()), m) for m in ARTICLE.finditer(text)]
    articles = _monotone(articles)
    if len(articles) >= 2:
        return _units_from(text, [m for _, m in articles], "Điều")
    sections = [(_roman(m.group(1)), m) for m in ROMAN_SECTION.finditer(text)]
    sections = _monotone(sections)
    if len(sections) >= 2:
        return _units_from(text, [m for _, m in sections], "roman")
    return [("", text.strip())]


def _units_from(text, matches, kind):
    parents = _parent_map(text)
    units = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start():end].strip()
        crumb = " > ".join(x for x in parents_at(parents, match.start()) if x)
        units.append((crumb, body))
    return units


def _parent_map(text):
    events = []
    for match in CHAPTER.finditer(text):
        label = " ".join(x for x in (match.group(1).title() + " " + match.group(2), match.group(3)) if x).strip()
        events.append((match.start(), 0, label))
    for match in SECTION.finditer(text):
        label = ("Mục " + match.group(2) + " " + (match.group(3) or "")).strip()
        events.append((match.start(), 1, label))
    events.sort()
    return events


def parents_at(events, position):
    current = {}
    for start, level, label in events:
        if start > position:
            break
        current[level] = label
        for deeper in [k for k in current if k > level]:
            del current[deeper]
    return [current[k] for k in sorted(current)]


# --------------------------------------------------------------------- chunking

def split_long(body, size, overlap):
    words = body.split()
    if len(words) <= size:
        return [body]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def document_chunks(name, passage, size=320, overlap=64, header_words=220):
    """Breadcrumb-prefixed legal chunks for one document.

    Yields dicts with the chunk text plus the retrieval fields it came from, so a
    field weighted lexical model and a cross encoder can both use the same units.
    """
    text = reflow(passage)
    lines = text.split("\n")
    title = slug_words(name)
    number = document_number(text)
    subject = subject_line(lines)
    header = " ".join(text.split()[:header_words])
    label = " | ".join(x for x in (title, number, subject) if x)

    out = []
    # A document level unit so questions about the document as a whole can match.
    out.append({"unit": "header", "crumb": "", "body": header, "label": label,
                "title": title, "number": number, "subject": subject})
    for crumb, body in parse_units(text):
        if not body.strip():
            continue
        for piece in split_long(body, size, overlap):
            out.append({"unit": "article", "crumb": crumb, "body": piece, "label": label,
                        "title": title, "number": number, "subject": subject})
    return out


def chunk_text(chunk):
    """The single string given to the embedder / reranker for this chunk."""
    head = " | ".join(x for x in (chunk["label"], chunk["crumb"]) if x)
    return (head + "\n" + chunk["body"]).strip() if head else chunk["body"]
