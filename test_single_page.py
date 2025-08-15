import time
import requests
import base64
import json
from PIL import Image
import io
import fitz  # PyMuPDF

def pdf_page_to_base64(pdf_path, page_num=0, dpi=150):
    """Convert a PDF page to base64 encoded image"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Render page to image
    mat = fitz.Matrix(dpi/72, dpi/72)  # 200 DPI
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    # Convert to PIL Image and then to base64
    pil_image = Image.open(io.BytesIO(img_data))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    doc.close()
    return img_base64, pil_image.size

def test_single_page():
    # Configuration
    pdf_path = "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"
    modal_endpoint = "https://marker--dotsocr-v2.modal.run"
    
    print(f"Loading PDF: {pdf_path}")
    
    # Convert first page to base64
    print("Converting page 0 to image...")
    start_convert = time.time()
    image_b64, image_size = pdf_page_to_base64(pdf_path, page_num=0)
    convert_time = time.time() - start_convert
    print(f"[OK] Page converted in {convert_time:.2f}s (size: {image_size})")
    
    # Prepare request data
    request_data = {
        "image": image_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    print(f"Sending request to Modal endpoint...")
    print(f"Image data size: {len(image_b64):,} characters")
    
    # Send request and measure time
    start_request = time.time()
    try:
        response = requests.post(modal_endpoint, json=request_data, timeout=300)  # 5 minute timeout
        end_request = time.time()
        
        request_time = end_request - start_request
        total_time = convert_time + request_time
        
        print(f"[OK] Request completed in {request_time:.2f}s")
        print(f"[DONE] Total time: {total_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                ocr_result = result.get("result", "")
                print(f"[INFO] OCR result length: {len(ocr_result):,} characters")
                
                # Save result to file
                with open("page_0_result.md", "w", encoding="utf-8") as f:
                    f.write(ocr_result)
                print(f"[SAVED] Result saved to page_0_result.md")
                
                # Print first 500 characters
                print(f"[PREVIEW] First 500 characters:")
                print("-" * 50)
                print(ocr_result[:500])
                if len(ocr_result) > 500:
                    print("...")
                print("-" * 50)
                
            else:
                print(f"[ERROR] OCR failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"[ERROR] HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out after 5 minutes")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    print("Testing single page OCR processing time")
    print("=" * 60)
    test_single_page()