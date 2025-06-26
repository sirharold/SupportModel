import re

def extract_urls_from_answer(answer: str) -> list[str]:
    try:
        return re.findall(r'https?://[^\s\]\)"]+', answer)
    except Exception as e:
        print("Error extracting URLs:", e)
        return []