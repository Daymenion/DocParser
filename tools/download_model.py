from argparse import ArgumentParser
import os
import importlib
import subprocess
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', '-t', type=str, default="huggingface")
    parser.add_argument('--name', '-n', type=str, default="rednote-hilab/dots.ocr")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Attention: The model save dir dots.ocr should be replace by a name without `.` like DotsOCR, util we merge our code to transformers.")
    model_dir = os.path.join(script_dir, "weights/DotsOCR")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.type == "huggingface":
        # Ensure huggingface_hub is available by installing it via pip if missing
        try:
            importlib.import_module("huggingface_hub")
        except ImportError:
            print("Installing huggingface_hub via pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub"])
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install huggingface_hub: {e}") from e
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
    elif args.type == "modelscope":
        from modelscope import snapshot_download
        snapshot_download(repo_id=args.name, local_dir=model_dir)
    else:
        raise ValueError(f"Invalid type: {args.type}")
    
    print(f"model downloaded to {model_dir}")
