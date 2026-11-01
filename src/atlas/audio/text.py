"""Everything about the words of a clip: cleaning titles and summaries for a speech engine, the script of a paper clip, the intros and the
outro, sentence splitting and the UTF-16 offsets of the manifest. Pure functions, no audio, no network.

The text of a clip is exactly what the voice says: the player shows it as the transcript and highlights the spoken word, so the cleaning
happens here, once, before synthesis, and never afterwards."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

# ------------------------------------------------------------------ UTF-16 offsets (the unit of Swift's NSString/String.utf16)


def utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def utf16_offsets(text: str) -> list[int]:
    """offsets[i] = UTF-16 code unit index of text[i]; offsets[len(text)] = UTF-16 length. Same as len(text[:i].encode('utf-16-le'))//2."""
    out = [0] * (len(text) + 1)
    n = 0
    for i, ch in enumerate(text):
        out[i] = n
        n += 2 if ord(ch) > 0xFFFF else 1
    out[len(text)] = n
    return out


@dataclass(frozen=True)
class Word:
    text: str
    char_start: int
    char_end: int
    u16_start: int
    u16_end: int


_WORD = re.compile(r"\S+")


def split_words(text: str) -> list[Word]:
    """The words of a script: maximal runs of non-whitespace, with character and UTF-16 spans. The manifest has one entry per word."""
    off = utf16_offsets(text)
    return [Word(m.group(), m.start(), m.end(), off[m.start()], off[m.end()]) for m in _WORD.finditer(text)]


# ------------------------------------------------------------------ LaTeX and odd characters

_GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta", "epsilon": "epsilon", "varepsilon": "epsilon", "zeta": "zeta",
    "eta": "eta", "theta": "theta", "vartheta": "theta", "iota": "iota", "kappa": "kappa", "lambda": "lambda", "mu": "mu", "nu": "nu",
    "xi": "xi", "pi": "pi", "varpi": "pi", "rho": "rho", "varrho": "rho", "sigma": "sigma", "varsigma": "sigma", "tau": "tau",
    "upsilon": "upsilon", "phi": "phi", "varphi": "phi", "chi": "chi", "psi": "psi", "omega": "omega",
    "Gamma": "Gamma", "Delta": "Delta", "Theta": "Theta", "Lambda": "Lambda", "Xi": "Xi", "Pi": "Pi", "Sigma": "Sigma", "Phi": "Phi",
    "Psi": "Psi", "Omega": "Omega",
}
_GREEK_CHARS = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi", "Π": "Pi",
    "Σ": "Sigma", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega",
}
_MATH_WORDS = {
    "times": "times", "cdot": "", "to": "to", "rightarrow": "to", "longrightarrow": "to", "mapsto": "to", "leftarrow": "from",
    "leq": "at most", "le": "at most", "geq": "at least", "ge": "at least", "neq": "not equal to", "ne": "not equal to",
    "approx": "about", "sim": "about", "infty": "infinity", "pm": "plus or minus", "ell": "l", "partial": "partial", "nabla": "nabla",
    "sum": "sum", "prod": "product", "int": "integral", "in": "in", "log": "log", "ln": "ln", "exp": "exp", "min": "min", "max": "max",
    "ldots": "", "dots": "", "cdots": "", "ll": "much less than", "gg": "much greater than", "star": "star", "ast": "",
}
# "\textbf{x}" -> x for the usual text-mode wrappers
_WRAPPERS = re.compile(r"\\(?:textbf|textit|textsc|textrm|textsf|texttt|emph|mathrm|mathbf|mathit|mathcal|mathbb|mathsf|mathtt|text|mbox|boldsymbol|bm|operatorname|underline|overline|hat|tilde|bar|vec)\s*\{([^{}]*)\}")
_ACCENT = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302", "~": "\u0303", "=": "\u0304", ".": "\u0307"}
_ACCENT_CMD = re.compile(r"\\([`'\"^~=.])\s*\{?([A-Za-z])\}?")
_SIMPLE_CMD = {r"\o": "ø", r"\O": "Ø", r"\ss": "ß", r"\ae": "æ", r"\AE": "Æ", r"\aa": "å", r"\AA": "Å", r"\l": "ł", r"\L": "Ł", r"\i": "i"}
_INLINE_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)", re.S)
_CONTROL = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")


def _math_to_words(m: str) -> str:
    """Inline math -> something speakable: Greek names, a few operators, exponents. Unknown commands are dropped, braces removed."""
    m = _WRAPPERS.sub(r"\1", m)
    m = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1 over \2", m)
    m = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"square root of \1", m)

    def cmd(match: re.Match[str]) -> str:
        name = match.group(1)
        return " " + (_GREEK.get(name) or _MATH_WORDS.get(name, "")) + " "

    m = re.sub(r"\\([A-Za-z]+)", cmd, m)
    m = re.sub(r"\^\s*\{?\s*2\s*\}?", " squared ", m)
    m = re.sub(r"\^\s*\{?\s*3\s*\}?", " cubed ", m)
    m = re.sub(r"\^\s*\{([^{}]*)\}", r" to the \1 ", m)
    m = re.sub(r"\^\s*([A-Za-z0-9])", r" to the \1 ", m)
    m = re.sub(r"_\s*\{([^{}]*)\}", r" \1 ", m)      # subscripts are read as a trailing letter or number: L_p -> "L p"
    m = re.sub(r"_\s*([A-Za-z0-9])", r" \1 ", m)
    m = m.replace("\\", " ").replace("{", "").replace("}", "")
    m = m.replace("<=", " at most ").replace(">=", " at least ").replace("->", " to ").replace("~", " ")
    return re.sub(r"\s+", " ", m).strip()


def clean_latex(text: str) -> str:
    """Titles and abstracts from arXiv carry TeX: $\\alpha$-divergence, \\textbf{x}, G\\\"odel, \\&, braces used for capitalisation."""
    t = text
    t = _INLINE_MATH.sub(lambda m: _math_to_words(next(g for g in m.groups() if g is not None)), t)
    for cmd, rep in _SIMPLE_CMD.items():
        t = re.sub(re.escape(cmd) + r"(?![A-Za-z])", rep, t)

    def accent(m: re.Match[str]) -> str:
        return unicodedata.normalize("NFC", m.group(2) + _ACCENT[m.group(1)])

    t = _ACCENT_CMD.sub(accent, t)
    for _ in range(3):  # nested wrappers
        t = _WRAPPERS.sub(r"\1", t)
    t = t.replace(r"\&", " and ").replace(r"\%", " percent").replace(r"\_", " ").replace(r"\#", " number ").replace(r"\$", " dollars ")
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)       # any command left over
    t = t.replace("{", "").replace("}", "").replace("\\", " ")
    return t


# ------------------------------------------------------------------ speech normaliser

_ABBREV = [   # (pattern, replacement); applied case-insensitively on word boundaries
    (r"\be\.g\.,?", "for example,"), (r"\bi\.e\.,?", "that is,"), (r"\bw\.r\.t\.?", "with respect to"), (r"\bvs\.?(?=\s)", "versus"),
    (r"\bet\s+al\.?", "and colleagues"), (r"\betc\.", "and so on"), (r"\bcf\.", "compare"), (r"\bapprox\.", "approximately"),
    (r"\bFig\.\s*(?=\d)", "figure "), (r"\bEq\.\s*(?=\d)", "equation "), (r"\bSec\.\s*(?=\d)", "section "), (r"\bw/o\b", "without"),
    (r"\bw/(?=\s)", "with"),
]
_TOKEN_FIXES = [  # things a TTS model reads badly
    (r"\bC\+\+", "C plus plus"), (r"\bC#", "C sharp"), (r"(\d)\s*%", r"\1 percent"), (r"%", " percent"),
    (r"(?<=\d)\s*[x×](?![A-Za-z0-9])", " times"), (r"×", " times "), (r"\s&\s", " and "), (r"&", " and "),
    (r"≥", " at least "), (r"≤", " at most "), (r"→", " to "), (r"←", " from "), (r"≈", " about "), (r"±", " plus or minus "), (r"∞", " infinity "),
    (r"°", " degrees"), (r"…", "..."),
]
_URL = re.compile(r"https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|org|net|io|ai|github\.io)/\S*", re.I)
_ARXIV_REF = re.compile(r"\barXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?", re.I)
_CITATION = re.compile(r"\s*\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]|\s*\((?:[A-Z][A-Za-z'’-]+(?: et al\.?| and [A-Z][A-Za-z'’-]+)?,? (?:19|20)\d\d[a-z]?(?:[;,] ?)?)+\)")
_DASH_SPACED = re.compile(r"\s+[–—―‒-]{1,3}\s+|\s*[—―]\s*|\s*-{2,3}\s*")
_NUM_RANGE = re.compile(r"(?<=\d)\s*[–—]\s*(?=\d)")
_EMOJI_ETC = re.compile(r"[\U0001F000-\U0001FFFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
_SLASH = re.compile(r"(?<=[A-Za-z0-9])/(?=[A-Za-z0-9])")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u201a": ",", "\u201e": '"', "\u2032": "'", "\u00a0": " ",
                         "\u2009": " ", "\u202f": " ", "\u2003": " ", "\u2002": " "})
_CJK = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\u3040-\u30ff]")


def _unspeakable(ch: str) -> bool:
    """A letter of a script the voice cannot read (it speaks English; names in CJK or Cyrillic would come out as noise)."""
    if not ch.isalpha():
        return False
    try:
        return not unicodedata.name(ch).startswith("LATIN")
    except ValueError:
        return True


def normalise_for_speech(text: str) -> str:
    """One paragraph a speech model reads well: TeX resolved, URLs and citation marks gone, symbols spelled out, odd characters dropped,
    dashes turned into pauses. Letters with accents (é, ü) stay: the model and the transcript both handle them."""
    t = unicodedata.normalize("NFKC", text or "")
    t = _CONTROL.sub("", t)
    t = clean_latex(t)
    t = t.translate(_QUOTES)
    t = _URL.sub(" ", t)
    t = _ARXIV_REF.sub(" ", t)
    t = _CITATION.sub(" ", t)
    t = _EMOJI_ETC.sub(" ", t)
    for pat, rep in _ABBREV:
        t = re.sub(pat, rep, t, flags=re.I)
    for pat, rep in _TOKEN_FIXES:
        t = re.sub(pat, rep, t)
    t = "".join(_GREEK_CHARS.get(ch, ch) if ch in _GREEK_CHARS else (" " if _unspeakable(ch) else ch) for ch in t)
    t = _NUM_RANGE.sub(" to ", t)
    t = t.replace("--", " – ")
    t = _DASH_SPACED.sub(", ", t)
    t = _SLASH.sub(" slash ", t)
    t = re.sub(r"\s+([,;:.!?])", r"\1", t)
    t = re.sub(r"([,;:])\1+", r"\1", t)
    t = re.sub(r"\(\s*\)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    # no token without a letter or digit may survive (a lone dash or bullet would be a "word" with no sound): fold punctuation into the previous word
    out: list[str] = []
    for tok in t.split(" "):
        if tok and not any(c.isalnum() for c in tok):
            if out and tok[0] in ",;:.!?":
                out[-1] = out[-1].rstrip(",;:") + tok[0]
            continue
        out.append(tok)
    t = " ".join(out)
    t = re.sub(r"^[,;:\s]+", "", t)
    return t


def clean_title(title: str) -> str:
    """A paper title as it is spoken: normalised, no trailing full stop (the clip adds its own sentence end)."""
    t = normalise_for_speech(title)
    return t.rstrip(" .,;:")


# ------------------------------------------------------------------ the script of a paper clip


def clean_name(name: str) -> str:
    n = normalise_for_speech(name)
    return n.strip(" ,;")


def clean_affiliation(aff: str | None) -> str | None:
    """A short institution name from OpenAlex or arXiv's free text; None when it is not something one would say aloud."""
    if not aff:
        return None
    a = normalise_for_speech(aff.split(";")[0])
    a = re.sub(r"\s*\([^)]*\)", "", a).strip(" ,.;")
    if not a or len(a) > 70 or "@" in a or re.search(r"\d{3,}", a):
        return None
    return a


def byline(first_author: str, n_authors: int, affiliation: str | None) -> str:
    """'From Deyuan Liu and colleagues at Inclusion AI.' - one author: no 'and colleagues'; unknown first author: 'From the authors.'"""
    name = clean_name(first_author) if first_author else ""
    who = name if name and any(c.isalpha() for c in name) else ""
    if not who:
        return "From the authors."
    if n_authors > 1:
        who += " and colleagues"
    aff = clean_affiliation(affiliation)
    return f"From {who} at {aff}." if aff else f"From {who}."


_ENDS_SENTENCE = re.compile(r"""[.!?\u2026]["')\]]*$""")


def ensure_sentence_end(s: str) -> str:
    s = s.strip()
    return s if not s or _ENDS_SENTENCE.search(s) else s + "."


def paper_clip_text(title: str, first_author: str, n_authors: int, affiliation: str | None, summary: str) -> str:
    """`{title}. From {first author}{ and colleagues}{ at {affiliation}}. {spoken summary}`  (a title that ends in ? or ! keeps its mark)."""
    head = ensure_sentence_end(clean_title(title))
    body = ensure_sentence_end(normalise_for_speech(summary))
    return " ".join(p for p in (head, byline(first_author, n_authors, affiliation), body) if p)


_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
INTRO_COUNTS = tuple(range(1, 9))
OUTRO_TEXT = "That's all for today. Tap a paper to read it."


def spoken_date(day: date) -> str:
    """'Friday, October 30' (English, independent of the machine's locale)."""
    return f"{_WEEKDAYS[day.weekday()]}, {_MONTHS[day.month - 1]} {day.day}"


def intro_text(count: int, day: date) -> str:
    """'Here are your six briefs for Friday, October 30.' (one: 'Here is your brief for ...')."""
    if count == 1:
        return f"Here is your brief for {spoken_date(day)}."
    return f"Here are your {_NUMBER_WORDS[count]} briefs for {spoken_date(day)}."


# ------------------------------------------------------------------ sentences

_ABBREV_NO_SPLIT = {"dr", "mr", "mrs", "ms", "prof", "st", "no", "vs", "inc", "ltd", "co", "jr", "sr", "fig", "eq", "sec", "approx", "al", "etc", "cf",
                    "e.g", "i.e", "u.s", "u.k", "ph.d", "m.d", "d.c", "a.i", "n.y"}
_SENT_END = re.compile(r"""([.!?…]+["')\]]*)(\s+)(?=["'(\[]?[A-Z0-9])""")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Character spans [start, end) of the sentences of a script, in order, covering every word (whitespace between them is excluded).
    A full stop after an initial ('J. Smith'), a known abbreviation ('Dr.') or inside a number never ends a sentence."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(text):
        before = text[start:m.start(1)]
        last = re.split(r"\s+", before.strip())[-1] if before.strip() else ""
        bare = last.rstrip(".").lower()
        if m.group(1).startswith(".") and (re.fullmatch(r"[A-Z]", last) or re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", last) or bare in _ABBREV_NO_SPLIT):
            continue
        spans.append((start, m.end(1)))
        start = m.end(2)
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans
