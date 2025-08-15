import time
import requests
import base64
import json
from PIL import Image
import io
import fitz  # PyMuPDF

def pdf_to_base64_pages(pdf_path, dpi=150, max_pages=None):
    """Convert all PDF pages to base64 encoded images"""
    doc = fitz.open(pdf_path)
    pages = []
    
    total_pages = len(doc)
    if max_pages:
        total_pages = min(total_pages, max_pages)
    
    print(f"Converting {total_pages} pages from PDF...")
    
    for page_num in range(total_pages):
        page = doc[page_num]
        
        # Render page to image
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image and then to base64
        pil_image = Image.open(io.BytesIO(img_data))
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        pages.append({
            "page_number": page_num,
            "base64": img_base64,
            "size": pil_image.size
        })
        print(f"  Page {page_num}: {pil_image.size}")
    
    doc.close()
    return pages

def test_multi_page_batch():
    # Configuration
    pdf_path = "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"
    modal_endpoint = "https://marker--dotsocr-batch-v2.modal.run"
    max_pages = 3  # Limit for testing
    
    print(f"Testing multi-page batch OCR with {pdf_path}")
    print("=" * 60)
    
    # Convert all pages to base64
    print("Step 1: Converting PDF pages to images...")
    start_convert = time.time()
    pages = pdf_to_base64_pages(pdf_path, max_pages=max_pages)
    convert_time = time.time() - start_convert
    print(f"[OK] {len(pages)} pages converted in {convert_time:.2f}s")
    
    # Prepare batch request
    images_b64 = [page["base64"] for page in pages]
    total_size = sum(len(img) for img in images_b64)
    
    request_data = {
        "images": images_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    print(f"\nStep 2: Sending batch request...")
    print(f"Total images: {len(images_b64)}")
    print(f"Total data size: {total_size:,} characters")
    
    # Send batch request
    start_request = time.time()
    try:
        response = requests.post(modal_endpoint, json=request_data, timeout=600)  # 10 minute timeout
        end_request = time.time()
        
        request_time = end_request - start_request
        total_time = convert_time + request_time
        avg_time_per_page = request_time / len(pages)
        
        print(f"[OK] Batch request completed in {request_time:.2f}s")
        print(f"[INFO] Average time per page: {avg_time_per_page:.2f}s")
        print(f"[DONE] Total time: {total_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                batch_results = result.get("results", [])
                total_pages = result.get("total_pages", 0)
                
                print(f"\n[SUCCESS] Processed {total_pages} pages")
                
                # Save results
                output_data = {
                    "pdf_path": pdf_path,
                    "total_pages": total_pages,
                    "processing_time": request_time,
                    "avg_time_per_page": avg_time_per_page,
                    "results": batch_results
                }
                
                with open("multi_page_results.json", "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                print(f"[SAVED] Results saved to multi_page_results.json")
                
                # Print summary for each page
                print(f"\n[SUMMARY] Per-page results:")
                print("-" * 50)
                for i, page_result in enumerate(batch_results):
                    if page_result.get("success"):
                        ocr_text = page_result.get("result", "")
                        try:
                            # Try to parse as JSON to count elements
                            json_data = json.loads(ocr_text)
                            element_count = len(json_data) if isinstance(json_data, list) else 1
                        except:
                            element_count = "N/A"
                        
                        print(f"Page {i}: {len(ocr_text):,} chars, {element_count} elements")
                        
                        # Show first element for verification
                        if ocr_text and len(ocr_text) > 50:
                            preview = ocr_text[:100].replace('\n', ' ')
                            print(f"  Preview: {preview}...")
                    else:
                        error = page_result.get("error", "Unknown error")
                        print(f"Page {i}: ERROR - {error}")
                print("-" * 50)
                
            else:
                print(f"[ERROR] Batch OCR failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"[ERROR] HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out after 10 minutes")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def test_single_vs_batch_comparison():
    """Compare single page processing vs batch processing"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: Single vs Batch")
    print("=" * 60)
    
    pdf_path = "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"
    single_endpoint = "https://marker--dotsocr-v2.modal.run"
    batch_endpoint = "https://marker--dotsocr-batch-v2.modal.run"
    
    # Convert first page only
    pages = pdf_to_base64_pages(pdf_path, max_pages=1)
    image_b64 = pages[0]["base64"]
    
    # Test single endpoint
    print("\nTesting single endpoint...")
    start_time = time.time()
    single_request = {
        "image": image_b64,
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(single_endpoint, json=single_request, timeout=300)
        single_time = time.time() - start_time
        print(f"Single endpoint: {single_time:.2f}s")
    except Exception as e:
        print(f"Single endpoint failed: {e}")
        single_time = None
    
    # Test batch endpoint
    print("Testing batch endpoint...")
    start_time = time.time()
    batch_request = {
        "images": [image_b64],
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(batch_endpoint, json=batch_request, timeout=300)
        batch_time = time.time() - start_time
        print(f"Batch endpoint: {batch_time:.2f}s")
    except Exception as e:
        print(f"Batch endpoint failed: {e}")
        batch_time = None
    
    if single_time and batch_time:
        overhead = batch_time - single_time
        print(f"\nBatch overhead: {overhead:.2f}s ({overhead/single_time*100:.1f}%)")

if __name__ == "__main__":
    print("Testing multi-page OCR batch processing")
    test_multi_page_batch()
    test_single_vs_batch_comparison()