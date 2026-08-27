import os
import requests
from dotenv import load_dotenv

load_dotenv()


def check_google_models():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    print("=" * 60)
    print("GOOGLE GEMINI API MODELS")
    print("=" * 60)
    if not api_key:
        print("[!] No GOOGLE_API_KEY found in .env\n")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200:
            print(f"[!] Error {res.status_code}: {res.json().get('error', {}).get('message', res.text)}\n")
            return

        models = res.json().get("models", [])
        print(f"[OK] Total Models Available: {len(models)}\n")
        print(f"{'Model Name':<38} | {'Input Limit':<12} | {'Supported Methods'}")
        print("-" * 80)
        for m in models:
            name = m.get("name", "").replace("models/", "")
            input_limit = m.get("inputTokenLimit", "N/A")
            methods = ", ".join(m.get("supportedGenerationMethods", []))
            # Filter to relevant chat and embedding models
            if any(k in name for k in ["gemini", "text-embedding", "embedding"]):
                print(f"{name:<38} | {str(input_limit):<12} | {methods}")
        print()
    except Exception as e:
        print(f"[!] Failed to fetch Google models: {e}\n")


def check_groq_models():
    api_key = os.getenv("GROQ_API_KEY")
    print("=" * 60)
    print("GROQ API MODELS & LIMITS")
    print("=" * 60)
    if not api_key:
        print("[!] No GROQ_API_KEY found in .env\n")
        return

    headers = {"Authorization": f"Bearer {api_key}"}
    url = "https://api.groq.com/openai/v1/models"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"[!] Error {res.status_code}: {res.json().get('error', {}).get('message', res.text)}\n")
            return

        models = res.json().get("data", [])
        print(f"[OK] Total Models Available: {len(models)}\n")
        print(f"{'Model ID':<35} | {'Context Window':<15} | {'Owned By'}")
        print("-" * 75)
        for m in models:
            m_id = m.get("id", "")
            ctx = m.get("context_window", "N/A")
            owned = m.get("owned_by", "")
            print(f"{m_id:<35} | {str(ctx):<15} | {owned}")
        print()
    except Exception as e:
        print(f"[!] Failed to fetch Groq models: {e}\n")


def check_openrouter_models():
    api_key = os.getenv("OPENROUTER_API_KEY")
    print("=" * 60)
    print("OPENROUTER API KEY STATUS & LIMITS")
    print("=" * 60)
    if not api_key:
        print("[!] No OPENROUTER_API_KEY found in .env\n")
        return

    headers = {"Authorization": f"Bearer {api_key}"}
    # Check key limits and credit balance
    try:
        key_res = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=15)
        if key_res.status_code == 200:
            data = key_res.json().get("data", {})
            label = data.get("label", "Key")
            usage = data.get("usage", 0)
            limit = data.get("limit", "Unlimited")
            rate_limit = data.get("rate_limit", {})
            print(f"[OK] Key Label: {label}")
            print(f"     Usage: ${usage:.4f} / Limit: {limit}")
            if rate_limit:
                print(f"     Rate Limit: {rate_limit.get('requests', 'N/A')} reqs / {rate_limit.get('interval', 'N/A')}")
        else:
            print(f"[!] Auth check error: {key_res.text}")
        print()
    except Exception as e:
        print(f"[!] Failed to fetch OpenRouter status: {e}\n")


if __name__ == "__main__":
    check_google_models()
    check_groq_models()
    check_openrouter_models()
