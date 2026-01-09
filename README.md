## Image-Informed Post-OCR Correction with Vision-Language Models
This repository contains the code for the paper "Image-Informed Post-OCR Correction with Vision-Language Models", where VLMs are adopted for post-OCR correction. We focus on the post-OCR correction of historical English, using BLN600, a parallel corpus of 19th century newspaper machine/human transcription.

### Usage

`data.py` - Create a dataset of paired OCR text, ground truth and image patch for model development

`finetune.py` & `eval.py` - Fine-tune and evaluate VLMs (Llama, Qwen) for post-OCR correction, using a YAML configuration file

```(bash)
pip install -r requirements.txt

python data.py

python finetune.py --config CONFIG

python eval.py --config CONFIG
```

### [PaddleOCRVL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) install (GPU)

```(bash)
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

python -m pip install -U "paddleocr[doc-parser]"

python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```
