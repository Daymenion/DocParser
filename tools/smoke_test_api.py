import requests

url = "http://localhost:7860/api/parse"
files = {"file": ("demo_pdf1.pdf", open("demo/demo_pdf1.pdf", "rb"), "application/pdf")}
data = {"prompt_mode": "prompt_layout_all_en"}

resp = requests.post(url, files=files, data=data, timeout=120)
print(resp.status_code)
print(resp.json())
