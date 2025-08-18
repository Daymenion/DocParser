### REST API

Run via Docker Compose (see README.md). Then POST a file to `http://localhost:7860/api/parse` with multipart form-data (field: `file`). The response includes `cells_data` and optional `markdown`.

<h1 align="center">
dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model
</h1>

[![Blog](https://img.shields.io/badge/Blog-View_on_GitHub-333.svg?logo=github)](https://github.com/rednote-hilab/dots.ocr/blob/master/assets/blog.md)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/rednote-hilab/dots.ocr)

## Introduction

**dots.ocr** is a powerful, multilingual document parser that unifies layout detection and content recognition within a single vision-language model while maintaining good reading order. Despite its compact 1.7B-parameter LLM foundation, it achieves state-of-the-art(SOTA) performance.

1. **Powerful Performance:** **dots.ocr** achieves SOTA performance for text, tables, and reading order on [OmniDocBench](https://github.com/opendatalab/OmniDocBench), while delivering formula recognition results comparable to much larger models like Doubao-1.5 and gemini2.5-pro.
2. **Multilingual Support:** **dots.ocr** demonstrates robust parsing capabilities for low-resource languages, achieving decisive advantages across both layout detection and content recognition on our in-house multilingual documents benchmark.
3. **Unified and Simple Architecture:** By leveraging a single vision-language model, **dots.ocr** offers a significantly more streamlined architecture than conventional methods that rely on complex, multi-model pipelines. Switching between tasks is accomplished simply by altering the input prompt, proving that a VLM can achieve competitive detection results compared to traditional detection models like DocLayout-YOLO.
4. **Efficient and Fast Performance:** Built upon a compact 1.7B LLM, **dots.ocr** provides faster inference speeds than many other high-performing models based on larger foundations.

### Performance Comparison: dots.ocr vs. Competing Models

For detailed metrics, figures, datasets, and evaluation methodology, see:

- [Blog.md](assets/blog.md)

# Quick Start

## 1. Installation

### Install dots.ocr

```shell
conda create -n dots_ocr python=3.12
conda activate dots_ocr

git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr

# Install pytorch, see https://pytorch.org/get-started/previous-versions/ for your cuda version
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

If you have trouble with the installation, try our [Docker Image](https://hub.docker.com/r/rednotehilab/dots.ocr) for an easier setup, and follow these steps:

```shell
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
pip install -e .
```

### Download Model Weights

> 💡**Note:** Please use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) for the model save path. This is a temporary workaround pending our integration with Transformers.

```shell
python3 tools/download_model.py

# with modelscope
python3 tools/download_model.py --type modelscope
```

## 2. Deployment

### vLLM inference

We highly recommend using vllm for deployment and inference. All of our evaluations results are based on vllm version 0.9.1.
The [Docker Image](https://hub.docker.com/r/rednotehilab/dots.ocr) is based on the official vllm image. You can also follow [Dockerfile](https://github.com/rednote-hilab/dots.ocr/blob/master/docker/Dockerfile) to build the deployment environment by yourself.

```shell
# You need to register model to vllm at first
python3 tools/download_model.py
export hf_model_path=./weights/DotsOCR  # Path to your downloaded model weights, Please use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) for the model save path. This is a temporary workaround pending our integration with Transformers.
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' `which vllm`  # If you downloaded model weights by yourself, please replace `DotsOCR` by your model saved directory name, and remember to use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) 

# launch vllm server
CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.95  --chat-template-content-format string --served-model-name model --trust-remote-code

# If you get a ModuleNotFoundError: No module named 'DotsOCR', please check the note above on the saved model directory name.

# vllm api demo
python3 ./demo/demo_vllm.py --prompt_mode prompt_layout_all_en
```

### Hugginface inference

```shell
python3 demo/demo_hf.py
```

<details>
<summary><b>Hugginface inference details</b></summary>

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

model_path = "./weights/DotsOCR"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image_path = "demo/demo_image1.jpg"
prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

# Preparation for inference
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=24000)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

```

</details>

### Hugginface inference with CPU

Please refer to [CPU inference](https://github.com/rednote-hilab/dots.ocr/issues/1#issuecomment-3148962536)

## 3. Document Parse

**Based on vLLM server**, you can parse an image or a pdf file using the following commands:

```bash

# Parse all layout info, both detection and recognition
# Parse a single image
python3 dots_ocr/parser.py demo/demo_image1.jpg
# Parse a single PDF
python3 dots_ocr/parser.py demo/demo_pdf1.pdf  --num_thread 64  # try bigger num_threads for pdf with a large number of pages

# Layout detection only
python3 dots_ocr/parser.py demo/demo_image1.jpg --prompt prompt_layout_only_en

# Parse text only, except Page-header and Page-footer
python3 dots_ocr/parser.py demo/demo_image1.jpg --prompt prompt_ocr

# Parse layout info by bbox
python3 dots_ocr/parser.py demo/demo_image1.jpg --prompt prompt_grounding_ocr --bbox 163 241 1536 705

```

**Based on Transformers**, you can parse an image or a pdf file using the same commands above, just add `--use_hf true`.

> Notice: transformers is slower than vllm, if you want to use demo/* with transformers，just add `use_hf=True` in `DotsOCRParser(..,use_hf=True)`

<details>
<summary><b>Output Results</b></summary>

1. **Structured Layout Data** (`demo_image1.json`): A JSON file containing the detected layout elements, including their bounding boxes, categories, and extracted text.
2. **Processed Markdown File** (`demo_image1.md`): A Markdown file generated from the concatenated text of all detected cells.
   * An additional version, `demo_image1_nohf.md`, is also provided, which excludes page headers and footers for compatibility with benchmarks like Omnidocbench and olmOCR-bench.
3. **Layout Visualization** (`demo_image1.jpg`): The original image with the detected layout bounding boxes drawn on it.

</details>

## 4. Demo

### Web Interface
You can run the web demo with the following command, or try directly at [live demo](https://dotsocr.xiaohongshu.com/)

```bash
python demo/demo_gradio.py
```

We also provide a demo for grounding ocr:

```bash
python demo/demo_gradio_annotion.py
```

### REST API Server
For programmatic access and system integration, we provide a FastAPI-based REST API server:

```bash
# Install API dependencies
pip install fastapi uvicorn[standard] python-multipart

# Start the API server
python api/api_server.py

# Or with custom port
python api/api_server.py 7865
```

**API Features:**
- ✅ **PDF Upload & Parsing**: Send PDF files via HTTP POST and receive structured JSON
- ✅ **Image Upload & Parsing**: Support for PNG, JPG, JPEG images  
- ✅ **Multiple Prompt Modes**: Different parsing strategies (layout detection, OCR, etc.)
- ✅ **Structured Output**: JSON responses with layout data and markdown content
- ✅ **File Downloads**: Download complete results as ZIP files
- ✅ **Runtime Configuration**: Override server settings per request

**API Documentation**: `http://localhost:7860/docs`

**Example Usage:**
```python
import requests

# Parse a PDF
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:7860/api/parse/pdf",
        files={"file": f},
        data={"prompt_mode": "prompt_layout_all_en"}
    )

result = response.json()
print(f"Found {result['total_elements']} layout elements across {result['total_pages']} pages")
```

See `api/README.md` for complete API documentation and examples.

## Limitation & Future Work

- **Complex Document Elements:**

  - **Table&Formula**: dots.ocr is not yet perfect for high-complexity tables and formula extraction.
  - **Picture**: Pictures in documents are currently not parsed.
- **Parsing Failures:** The model may fail to parse under certain conditions:

  - When the character-to-pixel ratio is excessively high. Try enlarging the image or increasing the PDF parsing DPI (a setting of 200 is recommended). However, please note that the model performs optimally on images with a resolution under 11289600 pixels.
  - Continuous special characters, such as ellipses (`...`) and underscores (`_`), may cause the prediction output to repeat endlessly. In such scenarios, consider using alternative prompts like `prompt_layout_only_en`, `prompt_ocr`, or `prompt_grounding_ocr` ([details here](https://github.com/rednote-hilab/dots.ocr/blob/master/dots_ocr/utils/prompts.py)).
- **Performance Bottleneck:** Despite its 1.7B parameter LLM foundation, **dots.ocr** is not yet optimized for high-throughput processing of large PDF volumes.

