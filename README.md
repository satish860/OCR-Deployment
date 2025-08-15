# DotsOCR Modal Deployment

A production-ready deployment of [DotsOCR](https://github.com/ucaslcl/DotsOCR) on Modal with GPU acceleration using vLLM.

## 🚀 Features

- **High-Performance OCR**: Advanced text extraction with table structure recognition
- **GPU Acceleration**: A100 GPU support for fast inference  
- **Large Context Support**: 131K token context length for processing large images
- **Multimodal Processing**: Handles complex documents with text, tables, and formulas
- **Production Ready**: FastAPI endpoints with proper error handling
- **Easy Deployment**: One-command deployment to Modal cloud

## 📁 Project Structure

```
OCR-Deployment/
├── modal_deploy.py          # Main Modal deployment file
├── test_modal_client.py     # Test client for the deployment
├── ocr_result.txt          # Example OCR output
└── dots.ocr/               # Complete DotsOCR model
    ├── weights/DotsOCR/    # Model weights and configuration
    ├── dots_ocr/           # Source code and utilities
    └── tools/              # Model download tools
```

## 🔧 Setup & Deployment

### Prerequisites

- [Modal account](https://modal.com/) with API token configured
- Python 3.11+

### Deploy to Modal

1. Clone the repository:
```bash
git clone https://github.com/satish860/OCR-Deployment.git
cd OCR-Deployment
```

2. Install Modal CLI:
```bash
pip install modal
```

3. Deploy to Modal:
```bash
modal deploy modal_deploy.py
```

4. The deployment will provide you with endpoint URLs like:
   - OCR Endpoint: `https://your-app--dotsocr-v2.modal.run/`
   - Health Check: `https://your-app--health-v2.modal.run/`

## 🧪 Testing

Test the deployment with the included test client:

```bash
python test_modal_client.py --modal_url "https://your-app--dotsocr-v2.modal.run/" --image_path "dots.ocr/demo/demo_image1.jpg"
```

### API Usage

Send POST requests to the OCR endpoint:

```python
import requests
import base64

# Load and encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Make OCR request
response = requests.post("https://your-app--dotsocr-v2.modal.run/", json={
    "image": image_b64,
    "prompt_mode": "prompt_layout_all_en",
    "temperature": 0.1,
    "top_p": 0.9
})

result = response.json()
print(result["result"])
```

## 📊 Example Output

The OCR system extracts structured text with bounding boxes and categories:

```json
{
  "success": true,
  "result": "[{\"bbox\": [628, 172, 1077, 194], \"category\": \"Page-header\", \"text\": \"EXPOSURE TO MEAT AND RISK OF LYMPHOMA\"}, ...]"
}
```

## 🏗️ Architecture

- **Base Image**: NVIDIA CUDA 12.8.1 with Python 3.12
- **Model**: DotsOCR with vLLM integration
- **GPU**: A100-40GB with 95% memory utilization
- **Context Length**: 131,072 tokens for large document processing
- **API Framework**: FastAPI with Modal's serverless infrastructure

## 🔍 Key Features

- **Advanced OCR**: Handles complex layouts, tables, formulas, and multiple languages
- **Structured Output**: Returns text with bounding boxes and element categories
- **Scalable**: Auto-scaling serverless deployment
- **Robust**: Proper error handling and health checks
- **Fast**: GPU-accelerated inference with optimized vLLM backend

## 📝 License

This project uses the DotsOCR model which has its own licensing terms. Please check the `dots.ocr/dots.ocr LICENSE AGREEMENT` file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

Built with ❤️ using [Modal](https://modal.com/) and [DotsOCR](https://github.com/ucaslcl/DotsOCR)