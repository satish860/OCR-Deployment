import time
import requests
import base64
import json
from PIL import Image
import io
import fitz  # PyMuPDF
import concurrent.futures
import threading

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

def process_single_page(args):
    """Process a single page via HTTP request to Modal single endpoint"""
    page_number, image_b64, endpoint, request_params = args
    
    # Create request data
    request_data = {
        "image": image_b64,
        **request_params
    }
    
    print(f"[Thread {threading.current_thread().name}] Starting OCR for page {page_number}")
    start_time = time.time()
    
    try:
        response = requests.post(endpoint, json=request_data, timeout=300)  # 5 minute timeout
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"[Thread {threading.current_thread().name}] Page {page_number} completed in {elapsed:.2f}s")
            
            return {
                "page_number": page_number,
                "success": True,
                "result": result.get("result", ""),
                "processing_time": elapsed,
                "thread_name": threading.current_thread().name
            }
        else:
            print(f"[Thread {threading.current_thread().name}] Page {page_number} failed with status {response.status_code}")
            return {
                "page_number": page_number,
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "processing_time": elapsed,
                "thread_name": threading.current_thread().name
            }
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"[Thread {threading.current_thread().name}] Page {page_number} timed out after {elapsed:.2f}s")
        return {
            "page_number": page_number,
            "success": False,
            "error": "Request timeout",
            "processing_time": elapsed,
            "thread_name": threading.current_thread().name
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Thread {threading.current_thread().name}] Page {page_number} failed: {e}")
        return {
            "page_number": page_number,
            "success": False,
            "error": str(e),
            "processing_time": elapsed,
            "thread_name": threading.current_thread().name
        }

def test_client_parallel_processing():
    """Test client-side parallel processing using ThreadPoolExecutor"""
    print("=" * 70)
    print("CLIENT-SIDE PARALLEL OCR PROCESSING TEST")
    print("=" * 70)
    
    pdf_path = "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"
    single_endpoint = "https://marker--dotsocr-v2.modal.run"
    
    # Test with different numbers of pages
    test_cases = [
        {"pages": 3, "max_workers": 3},
        {"pages": 6, "max_workers": 6}, 
        {"pages": 9, "max_workers": 9}
    ]
    
    for test_case in test_cases:
        page_count = test_case["pages"]
        max_workers = test_case["max_workers"]
        
        print(f"\n🚀 Testing {page_count} pages with {max_workers} parallel workers")
        print("-" * 50)
        
        # Convert pages to base64
        start_convert = time.time()
        pages = pdf_to_base64_pages(pdf_path, max_pages=page_count)
        convert_time = time.time() - start_convert
        print(f"PDF conversion completed in {convert_time:.2f}s")
        
        # Prepare arguments for parallel processing
        request_params = {
            "prompt_mode": "prompt_layout_all_en",
            "max_tokens": 1500,
            "temperature": 0.0,
            "top_p": 0.9
        }
        
        args_list = [
            (page["page_number"], page["base64"], single_endpoint, request_params)
            for page in pages
        ]
        
        # Process pages in parallel using ThreadPoolExecutor
        print(f"Starting {page_count} parallel HTTP requests to Modal...")
        start_parallel = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_single_page, args) for args in args_list]
            
            # Collect results as they complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Future failed: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "page_number": -1,
                        "processing_time": 0
                    })
        
        parallel_time = time.time() - start_parallel
        total_time = convert_time + parallel_time
        
        # Sort results by page number
        results.sort(key=lambda x: x.get("page_number", -1))
        
        # Analysis
        successful_pages = sum(1 for r in results if r.get("success", False))
        failed_pages = len(results) - successful_pages
        
        if successful_pages > 0:
            processing_times = [r.get("processing_time", 0) for r in results if r.get("success", False)]
            max_processing_time = max(processing_times) if processing_times else 0
            min_processing_time = min(processing_times) if processing_times else 0
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        else:
            max_processing_time = min_processing_time = avg_processing_time = 0
        
        print(f"\n📊 RESULTS for {page_count} pages:")
        print(f"✅ Successful: {successful_pages}/{page_count}")
        print(f"❌ Failed: {failed_pages}")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"⏱️ Parallel processing time: {parallel_time:.2f}s")
        print(f"⏱️ Max individual page time: {max_processing_time:.2f}s")
        print(f"⏱️ Min individual page time: {min_processing_time:.2f}s")
        print(f"⏱️ Avg individual page time: {avg_processing_time:.2f}s")
        
        # Calculate improvement vs sequential
        sequential_estimate = page_count * 15  # Assume 15s per page sequentially
        improvement = ((sequential_estimate - parallel_time) / sequential_estimate) * 100
        speedup = sequential_estimate / parallel_time if parallel_time > 0 else 0
        
        print(f"🚀 Speedup vs sequential: {speedup:.1f}x")
        print(f"🚀 Improvement vs sequential: {improvement:.1f}%")
        
        # Show thread distribution
        thread_usage = {}
        for result in results:
            thread_name = result.get("thread_name", "Unknown")
            thread_usage[thread_name] = thread_usage.get(thread_name, 0) + 1
        
        print(f"\n🧵 Thread distribution:")
        for thread_name, count in thread_usage.items():
            print(f"  {thread_name}: {count} pages")
        
        # Show per-page results
        print(f"\n📄 Per-page results:")
        for result in results:
            page_num = result.get("page_number", -1)
            success = "✅" if result.get("success", False) else "❌"
            time_taken = result.get("processing_time", 0)
            thread_name = result.get("thread_name", "Unknown")
            
            if result.get("success", False):
                ocr_result = result.get("result", "")
                char_count = len(ocr_result)
                try:
                    json_data = json.loads(ocr_result)
                    element_count = len(json_data) if isinstance(json_data, list) else 1
                except:
                    element_count = "N/A"
                
                print(f"  Page {page_num}: {success} {time_taken:.2f}s [{thread_name}] - {char_count} chars, {element_count} elements")
            else:
                error = result.get("error", "Unknown error")
                print(f"  Page {page_num}: {success} {time_taken:.2f}s [{thread_name}] - {error}")
        
        # Save results for this test case
        output_data = {
            "test_type": "client_parallel",
            "page_count": page_count,
            "max_workers": max_workers,
            "total_time": total_time,
            "parallel_time": parallel_time,
            "convert_time": convert_time,
            "successful_pages": successful_pages,
            "speedup": speedup,
            "improvement_percent": improvement,
            "thread_usage": thread_usage,
            "results": results
        }
        
        filename = f"client_parallel_{page_count}pages_results.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to {filename}")

if __name__ == "__main__":
    test_client_parallel_processing()