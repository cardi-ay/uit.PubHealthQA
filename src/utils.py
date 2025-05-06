import unicodedata
import re

def normalize_text(text: str) -> str:
    """
    Normalize a text string:
    - Convert to lowercase
    - Replace "Đ" and "đ" with "d"
    - Remove accents (but keep "d" normalized)
    - Replace non-alphanumeric characters with hyphens
    - Remove leading/trailing hyphens

    Example:
        " Cà phê Sữa đá! " → "ca-phe-sua-da"
    """
    if not text:
        return ""

    text = text.lower().strip()
    text = text.replace('đ', 'd').replace('Đ', 'd')

    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])

    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')

    return text
print(normalize_text("  Đắk Lắk Cà Phê! "))  # Output: "dak-lak-ca-phe"
print(normalize_text("Sầu riêng Ri6"))        # Output: "sau-rieng-ri6"
print(normalize_text("Đá đen"))               # Output: "da-den"