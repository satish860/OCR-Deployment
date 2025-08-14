import modal
from typing import Dict, Any

app = modal.App("dots-ocr")

image = (
    modal.Image.from_registry("rednotehilab/dots.ocr:vllm-openai-v0.9.1")
    .apt_install("curl")
)

@app.function(
    image=image,
    gpu="A100-40GB",
    scaledown_window=300,
    timeout=3600,
    max_containers=10,
)
@modal.fastapi_endpoint(method="POST", label="dots-ocr-api")
def process_ocr(request_data: Dict[str, Any]):
    """
    Process OCR requests using Dots.OCR service
    Compatible with OpenAI chat completions format
    """
    import requests
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=request_data,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return {
            "error": f"OCR processing failed: {str(e)}",
            "status": "error"
        }

@app.function(
    image=image,
    gpu="A100-40GB",
)
@modal.fastapi_endpoint(method="GET", label="health")
def health_check():
    """Health check endpoint"""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        return {"status": "healthy", "service": "dots-ocr"}
    except:
        return {"status": "unhealthy", "service": "dots-ocr"}

@app.function(
    image=image,
    gpu="A100-40GB",
    scaledown_window=300,
)
def extract_text_from_image(image_data, prompt: str = "Extract all text") -> str:
    """
    Direct function to extract text from image bytes
    """
    import requests
    import base64
    
    image_b64 = base64.b64encode(image_data).decode('utf-8')
    
    request_data = {
        "model": "dots.ocr",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }]
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=request_data,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "No text extracted"
            
    except Exception as e:
        return f"Error processing image: {str(e)}"