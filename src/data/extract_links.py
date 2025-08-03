import re
from urllib.parse import urlparse, urlunparse

def normalize_url(url: str) -> str:
    """
    Normalizes a URL by removing query parameters and fragments (anchors).
    
    Examples:
        https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-overview?view=azure-cli-latest#overview
        -> https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-overview
        
        https://learn.microsoft.com/azure/virtual-machines/windows/quick-create-portal?tabs=windows10#create-vm
        -> https://learn.microsoft.com/azure/virtual-machines/windows/quick-create-portal
    
    Args:
        url: The URL to normalize
        
    Returns:
        The normalized URL without query parameters and fragments
    """
    if not url or not url.strip():
        return ""
    
    try:
        # Parse the URL
        parsed = urlparse(url.strip())
        
        # Reconstruct without query parameters and fragments
        normalized = urlunparse((
            parsed.scheme,    # https
            parsed.netloc,    # learn.microsoft.com
            parsed.path,      # /en-us/azure/storage/blobs/storage-blob-overview
            '',               # params (empty)
            '',               # query (empty) - removes ?view=azure-cli-latest
            ''                # fragment (empty) - removes #overview
        ))
        
        return normalized
    except Exception as e:
        # If parsing fails, return the original URL stripped
        return url.strip()

def extract_urls_from_answer(answer: str) -> list[str]:
    try:
        raw_urls = re.findall(r'https?://[^\s\]\)"]+', answer)
        # Normalize all extracted URLs
        return [normalize_url(url) for url in raw_urls]
    except Exception as e:
        print("Error extracting URLs:", e)
        return []