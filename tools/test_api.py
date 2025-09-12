import sys
import requests

API_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860/parse"
FILE_PATH = sys.argv[2] if len(sys.argv) > 2 else None
URL = sys.argv[3] if len(sys.argv) > 3 else None

if not FILE_PATH and not URL:
    print("Usage: python tools/test_api.py <api_url> <file_path> [remote_url]\n       OR python tools/test_api.py <api_url> '' <remote_url>")
    sys.exit(1)

files = None
data = {
    "prompt_mode": "prompt_layout_all_en",
    "fitz_preprocess": "true",
    "output": "md",
    "inline_md": "true",
}

if FILE_PATH:
    files = {"file": open(FILE_PATH, "rb")}
else:
    data["url"] = URL

resp = requests.post(API_URL, files=files, data=data)
print("Status:", resp.status_code)
content_type = resp.headers.get("Content-Type", "")
print("Content-Type:", content_type)

if "application/json" in content_type:
    try:
        js = resp.json()
        if "md_content" in js:
            print("MD length:", len(js["md_content"]))
            print(js["md_content"][:1000])
        else:
            print(js)
    except Exception:
        print(resp.text[:1000])
elif "text/markdown" in content_type:
    print(resp.text[:1000])
else:
    # Save attachments like zip/md
    disp = resp.headers.get("Content-Disposition", "")
    if "filename=" in disp:
        fname = disp.split("filename=")[-1].strip('"')
    else:
        fname = "response.bin"
    with open(fname, "wb") as f:
        f.write(resp.content)
    print(f"Saved attachment to {fname}")
