# utils/auth.py
import os
from huggingface_hub import login, whoami
from huggingface_hub.utils import HfHubHTTPError

def ensure_huggingface_login(token: str | None = None):
    """
    Ensures the user is logged into Hugging Face. If not, it uses the provided token
    or a token from the environment to log in.
    """
    try:
        # Check if we are already logged in.
        whoami()
        print("[DEBUG] Already logged in to Hugging Face.")
        return
    except HfHubHTTPError:
        print("[DEBUG] Not logged in to Hugging Face. Attempting login.")
        
        # Use the provided token first, otherwise fall back to environment variables.
        if not token:
            token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

        if token:
            print("[DEBUG] Found token. Logging in.")
            login(token=token, add_to_git_credential=False)
            
            # Verify login was successful
            try:
                whoami()
                print("[DEBUG] Hugging Face login successful.")
            except HfHubHTTPError:
                raise RuntimeError("Hugging Face login failed. The provided token may be invalid.")
        else:
            raise RuntimeError("Hugging Face token not found.")