import re
import unicodedata

def normalize_persian_text(text: str) -> str:
    """
    Standardizes Persian text by unifying characters, removing diacritics, 
    punctuations, ZWNJ (\u200c), and normalizing whitespaces.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    translation_table = str.maketrans({
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "آ": "ا",
        "أ": "ا",
        "إ": "ا",
        "ۀ": "ه",
        "ة": "ه",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    })
    text = text.translate(translation_table)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    text = re.sub(r'[!?,.:;؛؟"\'()\-\[\]{}<>/\\]', " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()


def find_polarity_cues(text: str, cues: list[str]) -> list[str]:
    """Find whole-word cue matches within normalized text.

    Args:
        text: Target text segment (e.g., extracted signal evidence).
        cues: List of lexical cue phrases defined for a resilience node.

    Returns:
        list[str]: Cues from the input list that appear as isolated words or phrases
            in the normalized text.
    """
    normalized_text = normalize_persian_text(text)
    hits: list[str] = []

    for cue in cues:
        normalized_cue = normalize_persian_text(cue)
        if not normalized_cue:
            continue

        pattern = rf"(?<!\w){re.escape(normalized_cue)}(?!\w)"
        if re.search(pattern, normalized_text):
            hits.append(cue)

    return hits


# Backward compatibility alias
_find_polarity_cues = find_polarity_cues
