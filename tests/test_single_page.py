import time
import requests
import base64
import json
from PIL import Image
import io
import fitz  # PyMuPDF
import os
from openai import OpenAI
from dotenv import load_dotenv
import modal
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def pdf_page_to_base64(pdf_path, page_num=0, dpi=100):
    """Convert a PDF page to base64 encoded image"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Render page to image
    mat = fitz.Matrix(dpi/72, dpi/72)  # Optimized DPI
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    # Convert to PIL Image and then to base64 (optimized JPEG)
    pil_image = Image.open(io.BytesIO(img_data))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    doc.close()
    return img_base64, pil_image.size

def load_openai_client():
    """Load OpenAI client for chart analysis"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] OPENAI_API_KEY not found - chart processing disabled")
        return None
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[WARNING] Failed to initialize OpenAI client: {e}")
        return None

def extract_image_region(full_image_base64, bbox):
    """Extract chart region from full page image"""
    try:
        image_bytes = base64.b64decode(full_image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        x1, y1, x2, y2 = bbox
        cropped_image = pil_image.crop((x1, y1, x2, y2))
        
        buffer = io.BytesIO()
        cropped_image.save(buffer, format="PNG")
        cropped_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{cropped_base64}"
    except Exception as e:
        print(f"[ERROR] Failed to extract image region: {e}")
        return None

def is_likely_chart(element):
    """Check if OCR element is likely a chart"""
    if element.get("category") != "Picture":
        return False
    
    bbox = element.get("bbox", [])
    if len(bbox) != 4:
        return False
        
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Charts are usually reasonably sized (at least 100px in each dimension)
    if width < 100 or height < 100:
        return False
    
    # Reasonable aspect ratio (not too thin/wide)
    aspect_ratio = max(width / height, height / width) if height > 0 else float('inf')
    if aspect_ratio > 5:
        return False
    
    return True

def extract_table_from_chart_analysis(chart_description):
    """Extract table data from chart analysis description (markdown table format)"""
    import re
    
    # Look for markdown table structures in the description
    lines = chart_description.split('\n')
    table_data = []
    in_table = False
    
    for line in lines:
        line = line.strip()
        
        # Look for markdown table rows (lines with | separators)
        if '|' in line and len(line.split('|')) >= 3:
            # Skip markdown table separator lines (contain dashes)
            if re.match(r'^[\|\s\-]+$', line):
                continue
                
            # Clean up the line and split by pipe
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if row:  # Only add non-empty rows
                table_data.append(row)
                in_table = True
        elif in_table and not line:
            # Empty line after table - stop collecting
            break
        elif not in_table:
            # Look for structured data patterns like "Category: Value1, Value2"
            if ':' in line and any(char.isdigit() for char in line):
                # Try to extract structured data
                if any(keyword in line for keyword in ['Revenue', 'Sales', '$', '%', 'Total', 'million', 'billion']):
                    # This looks like financial data
                    parts = line.split(':')
                    if len(parts) == 2:
                        category = parts[0].strip()
                        values = parts[1].strip()
                        # Split values by comma or other separators
                        value_list = [v.strip() for v in re.split(r'[,;]', values) if v.strip()]
                        if value_list:
                            table_data.append([category] + value_list)
    
    # Return table data if we found at least 2 rows (header + data)
    if len(table_data) >= 2:
        print(f"[CHART] Extracted table with {len(table_data)} rows, {len(table_data[0]) if table_data else 0} columns")
        return table_data
    else:
        return None

def create_table_element_from_chart_data(chart_bbox, table_data):
    """Create a Table element from extracted chart table data"""
    if not table_data or len(table_data) < 2:
        return None
    
    try:
        # Create HTML table from the data
        html_table = "<table>"
        
        # First row as header
        if table_data[0]:
            html_table += "<thead><tr>"
            for cell in table_data[0]:
                html_table += f"<th>{cell}</th>"
            html_table += "</tr></thead>"
        
        # Remaining rows as body
        if len(table_data) > 1:
            html_table += "<tbody>"
            for row in table_data[1:]:
                html_table += "<tr>"
                for cell in row:
                    html_table += f"<td>{cell}</td>"
                html_table += "</tr>"
            html_table += "</tbody>"
        
        html_table += "</table>"
        
        # Create a new bbox slightly offset from the chart
        x1, y1, x2, y2 = chart_bbox
        table_bbox = [x1, y2 + 10, x2, y2 + 50]  # Position below the chart
        
        return {
            "bbox": table_bbox,
            "category": "Table",
            "text": html_table,
            "source": "chart_extraction"
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to create table element: {e}")
        return None

def analyze_chart_with_llm(chart_base64, client):
    """Analyze chart using OpenAI Vision API with enhanced structured prompt"""
    prompt = """Extract ALL data from this chart/graph in a structured format. Focus on extracting specific values that can be used for data analysis:

**REQUIRED OUTPUT FORMAT:**
1. Chart Type: [bar chart/line graph/pie chart/etc.]
2. Title: [exact title text]
3. Data Structure:
   - Categories/Labels: [list all x-axis labels or categories]
   - Data Series: [list all series names with their values]
   - Exact Values: [extract ALL numerical values with their corresponding labels]
4. Axis Information:
   - X-axis: [label and all values/categories]
   - Y-axis: [label, range, units]

5. **DATA TABLE (CRITICAL)**: Present ALL chart data in this exact markdown table format:
Category | Series1 | Series2 | Series3 | Total
---------|---------|---------|---------|-------
Row1     | Value1  | Value2  | Value3  | Sum
Row2     | Value4  | Value5  | Value6  | Sum

**CRITICAL REQUIREMENTS:**
- MUST include a markdown table with ALL values from the chart
- Extract EVERY numerical value visible in the chart
- Include all text labels, legends, and annotations
- Preserve exact numbers (don't round unless necessary)
- Format data so it can be easily parsed for business analysis
- Include units of measurement where shown
- Note any totals, percentages, or calculated values
- The data table is essential for downstream processing"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": chart_base64}}
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return {
            "success": True,
            "description": response.choices[0].message.content.strip(),
            "tokens_used": response.usage.total_tokens if response.usage else None
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def process_charts_in_ocr_result(ocr_result, full_image_base64, openai_client):
    """Process OCR result to find and analyze charts"""
    if not openai_client:
        return ocr_result, 0
    
    try:
        # Parse OCR result as JSON - handle leading text before JSON array
        cleaned_ocr = ocr_result.strip()
        if not cleaned_ocr.startswith('['):
            # Find the start of the JSON array
            json_start = cleaned_ocr.find('[')
            if json_start != -1:
                cleaned_ocr = cleaned_ocr[json_start:]
            else:
                # Not a JSON array format, return as-is
                return ocr_result, 0
        
        ocr_data = json.loads(cleaned_ocr)
        charts_processed = 0
        
        if isinstance(ocr_data, list):
            for element in ocr_data:
                if is_likely_chart(element):
                    print(f"[CHART] Found potential chart: {element.get('bbox')}")
                    
                    # Extract chart region
                    chart_image = extract_image_region(full_image_base64, element.get("bbox", []))
                    
                    if chart_image:
                        print(f"[CHART] Analyzing with OpenAI Vision API...")
                        analysis = analyze_chart_with_llm(chart_image, openai_client)
                        
                        if analysis["success"]:
                            # Set the main text to the chart description
                            element["text"] = analysis["description"]
                            
                            # Extract table data if present and add as structured data
                            table_data = extract_table_from_chart_analysis(analysis["description"])
                            
                            element["chart_analysis"] = {
                                "processed": True,
                                "tokens_used": analysis.get("tokens_used"),
                                "table_data": table_data if table_data else None
                            }
                            
                            # If we found table data, also add it as a new Table element
                            if table_data:
                                table_element = create_table_element_from_chart_data(element["bbox"], table_data)
                                if table_element:
                                    # Insert the table element right after the chart
                                    current_index = ocr_data.index(element)
                                    ocr_data.insert(current_index + 1, table_element)
                            
                            charts_processed += 1
                            print(f"[CHART] Successfully analyzed chart ({analysis.get('tokens_used', 'unknown')} tokens)")
                        else:
                            print(f"[CHART] Analysis failed: {analysis['error']}")
                            element["chart_analysis"] = {"processed": False, "error": analysis["error"]}
        
        return json.dumps(ocr_data), charts_processed
        
    except json.JSONDecodeError:
        # Not JSON format, return as-is
        return ocr_result, 0
    except Exception as e:
        print(f"[ERROR] Chart processing failed: {e}")
        return ocr_result, 0

async def test_single_page_async():
    # Async version for direct Modal calls
    pass

def test_single_page():
    # Configuration
    pdf_path = "input/44abcd07-58ab-4957-a66b-c03e82e11e6f.pdf"
    use_gpu_direct = True  # Set to True to use GPU direct HTTP endpoint
    gpu_direct_endpoint = "https://marker--gpu-direct.modal.run"  # Direct GPU HTTP endpoint
    fastapi_endpoint = "https://marker--dotsocr-v2.modal.run"  # Original FastAPI wrapper
    process_charts = False  # Set to True to enable chart processing (adds ~2s)
    
    # Choose endpoint
    if use_gpu_direct:
        modal_endpoint = gpu_direct_endpoint
        print("[INFO] Using GPU direct HTTP endpoint (bypasses FastAPI wrapper)")
    else:
        modal_endpoint = fastapi_endpoint
        print("[INFO] Using FastAPI wrapper HTTP endpoint")
    
    print(f"Loading PDF: {pdf_path}")
    
    # Initialize OpenAI client for chart processing
    openai_client = None
    if process_charts:
        print("Initializing OpenAI client for chart processing...")
        openai_client = load_openai_client()
        if openai_client:
            print("[OK] OpenAI client ready for chart analysis")
        else:
            print("[WARNING] Chart processing disabled - no OpenAI client")
    
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
        "max_tokens": 800,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    print(f"Image data size: {len(image_b64):,} characters")
    
    # Send request and measure time
    start_request = time.time()
    try:
        print(f"Sending request to: {modal_endpoint}")
        response = requests.post(modal_endpoint, json=request_data, timeout=120)  # 2 minute timeout
        end_request = time.time()
        
        request_time = end_request - start_request
        endpoint_type = "GPU Direct" if use_gpu_direct else "FastAPI"
        
        print(f"[OK] {endpoint_type} OCR completed in {request_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                ocr_result = result.get("result", "")
                print(f"[INFO] OCR result length: {len(ocr_result):,} characters")
                
                # Check processing method if available
                if result.get("processing_method"):
                    print(f"[INFO] Processing method: {result.get('processing_method')}")
                    
                success = True
            else:
                print(f"[ERROR] OCR failed: {result.get('error', 'Unknown error')}")
                success = False
        else:
            print(f"[ERROR] HTTP Error {response.status_code}: {response.text}")
            success = False
        
        if success:
                
            # Process charts if enabled
            charts_processed = 0
            chart_time = 0
            
            if process_charts and openai_client:
                print("\n[CHART] Starting chart processing...")
                start_chart = time.time()
                
                # Create full image base64 with data URL prefix
                full_image_b64 = f"data:image/png;base64,{image_b64}"
                
                ocr_result, charts_processed = process_charts_in_ocr_result(
                    ocr_result, image_b64, openai_client
                )
                
                chart_time = time.time() - start_chart
                print(f"[CHART] Chart processing completed in {chart_time:.2f}s")
                print(f"[CHART] Charts processed: {charts_processed}")
            
            total_time = convert_time + request_time + chart_time
            total_time = convert_time + request_time + chart_time
            method = "GPU-Direct" if use_gpu_direct else "FastAPI"
            print(f"[TIMING] Convert: {convert_time:.2f}s | OCR ({method}): {request_time:.2f}s | Charts: {chart_time:.2f}s | Total: {total_time:.2f}s")
                
            # Save result to file
            method_suffix = "_gpu_direct" if use_gpu_direct else "_fastapi"
            output_filename = f"page_0_result{method_suffix}_with_charts.md" if charts_processed > 0 else f"page_0_result{method_suffix}.md"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(ocr_result)
            print(f"[SAVED] Result saved to {output_filename}")
            
            # Print first 500 characters
            print(f"[PREVIEW] First 500 characters:")
            print("-" * 50)
            print(ocr_result[:500])
            if len(ocr_result) > 500:
                print("...")
            print("-" * 50)
            
            # Summary
            if charts_processed > 0:
                print(f"\n[SUCCESS] Processed {charts_processed} charts with LLM analysis")
            else:
                print(f"\n[INFO] No charts found or processed")
        else:
            print(f"\n[ERROR] OCR processing failed")
            
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out after 2 minutes")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing single page OCR processing time")
    print("=" * 60)
    test_single_page()