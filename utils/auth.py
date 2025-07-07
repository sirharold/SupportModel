# utils/auth.py
import os
from huggingface_hub import login, whoami
from huggingface_hub.utils import HfHubHTTPError

import torch
def ensure_huggingface_login():
    """
    Ensures the user is logged into Hugging Face.
    If not logged in, it attempts to log in using the HUGGINGFACE_TOKEN environment variable.
    """
    print("TORCH VERSION:  ", torch.__version__)

    try:
        # Check if we are already logged in
        print("[DEBUG AUTH] Checking Hugging Face login status...Before whoami")
        token = os.getenv("HUGGINGFACE_API_KEY") # Corrected to match your .env variable
        os.environ["HF_TOKEN"] = token
        whoami()
        print("[DEBUG AUTH] Already logged in to Hugging Face.")
    except HfHubHTTPError:
        # If not logged in, try to log in with the token from the environment
        print("[DEBUG AUTH] Not logged in to Hugging Face. Attempting login with token.")
        token = os.getenv("HUGGINGFACE_API_KEY") # Corrected to match your .env variable
        os.environ["HF_TOKEN"] = token
        if not token:
            raise RuntimeError(
                "Hugging Face token not found in environment variable HUGGINGFACE_API_KEY. "
                "Please ensure it is set in your .env file."
            )
        print("[DEBUG AUTH] before login")
        login(token)
        print("[DEBUG AUTH] Logged in to Hugging Face with token.")
        print("[DEBUG AUTH] Hugging Face login successful.")
    except Exception as e:
        print("[DEBUG AUTH] Error during Hugging Face login:", e)