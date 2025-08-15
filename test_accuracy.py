import json
import re
import sys
from pathlib import Path

# Force UTF-8 encoding for console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def test_page_0_accuracy():
    """Test accuracy of OCR results in page_0_result.md"""
    print("Testing OCR Accuracy for page_0_result.md")
    print("=" * 60)
    
    # Read the OCR result
    result_file = Path("page_0_result.md")
    if not result_file.exists():
        print(f"[ERROR] File not found: {result_file}")
        return False
    
    with open(result_file, "r", encoding="utf-8") as f:
        ocr_text = f.read().strip()
    
    print(f"[INFO] OCR result length: {len(ocr_text):,} characters")
    
    # Parse JSON structure
    try:
        ocr_data = json.loads(ocr_text)
        print(f"[OK] Valid JSON structure with {len(ocr_data)} elements")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}")
        return False
    
    # Define expected Hindi text elements that should be present
    expected_texts = [
        "प्रेषक",  # Sender
        "जिला प्रोबेशन अधिकारी",  # District Probation Officer
        "संत कबीर नगर",  # Sant Kabir Nagar
        "सेवा में",  # Service to
        "मा० प्रधान मजिस्ट्रेट",  # Honorable Chief Magistrate
        "कृपया अपने आदेश",  # Please refer to your order
        "भवदीय",  # Yours truly
        "पत्रांक"  # Letter number
    ]
    
    # Test 1: Check for required text elements
    print(f"\n[TEST 1] Checking for key Hindi text elements...")
    found_texts = []
    missing_texts = []
    
    # Extract all text from OCR results
    all_text = ""
    for element in ocr_data:
        if element.get("text"):
            all_text += element["text"] + " "
    
    for expected_text in expected_texts:
        if expected_text in all_text:
            found_texts.append(expected_text)
            print(f"  [OK] Found: '{expected_text}'")
        else:
            missing_texts.append(expected_text)
            print(f"  [MISS] Missing: '{expected_text}'")
    
    accuracy_score = len(found_texts) / len(expected_texts) * 100
    print(f"  Text accuracy: {accuracy_score:.1f}% ({len(found_texts)}/{len(expected_texts)})")
    
    # Test 2: Check JSON structure integrity
    print(f"\n[TEST 2] Checking JSON structure integrity...")
    structure_errors = []
    
    required_fields = ["bbox", "category"]
    valid_categories = ["Text", "Picture", "Title", "Section-header", "Page-header", "Page-footer", "Table", "List-item", "Formula", "Caption", "Footnote"]
    
    for i, element in enumerate(ocr_data):
        # Check required fields
        for field in required_fields:
            if field not in element:
                structure_errors.append(f"Element {i}: Missing '{field}' field")
        
        # Text field is required only for non-Picture elements
        category = element.get("category", "")
        if category != "Picture" and "text" not in element:
            structure_errors.append(f"Element {i}: Missing 'text' field for {category} element")
        
        # Check bbox format [x1, y1, x2, y2]
        bbox = element.get("bbox")
        if bbox:
            if not isinstance(bbox, list) or len(bbox) != 4:
                structure_errors.append(f"Element {i}: Invalid bbox format")
            elif not all(isinstance(coord, (int, float)) for coord in bbox):
                structure_errors.append(f"Element {i}: Non-numeric bbox coordinates")
        
        # Check category validity
        category = element.get("category")
        if category and category not in valid_categories:
            structure_errors.append(f"Element {i}: Unknown category '{category}'")
    
    if structure_errors:
        print(f"  [ERROR] Structure errors found:")
        for error in structure_errors[:5]:  # Show first 5 errors
            print(f"    - {error}")
        if len(structure_errors) > 5:
            print(f"    ... and {len(structure_errors) - 5} more errors")
    else:
        print(f"  [OK] All elements have valid structure")
    
    # Test 3: Check specific date format
    print(f"\n[TEST 3] Checking date format recognition...")
    date_pattern = r"१२/०२/१"  # Expected date in Devanagari numerals
    date_found = False
    
    for element in ocr_data:
        text = element.get("text", "")
        if re.search(date_pattern, text):
            date_found = True
            print(f"  [OK] Found date pattern: '{date_pattern}' in text: '{text}'")
            break
    
    if not date_found:
        print(f"  [MISS] Date pattern '{date_pattern}' not found")
    
    # Test 4: Check element count and categories
    print(f"\n[TEST 4] Analyzing element distribution...")
    category_counts = {}
    for element in ocr_data:
        category = element.get("category", "Unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"  Total elements: {len(ocr_data)}")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    
    # Expected: mostly "Text" elements with one "Picture" element
    text_elements = category_counts.get("Text", 0)
    picture_elements = category_counts.get("Picture", 0)
    
    if text_elements >= 8 and picture_elements == 1:
        print(f"  [OK] Element distribution looks correct")
    else:
        print(f"  [WARN] Unexpected element distribution")
    
    # Overall assessment
    print(f"\n[SUMMARY] Overall Assessment:")
    print(f"  Text Accuracy: {accuracy_score:.1f}%")
    print(f"  Structure Errors: {len(structure_errors)}")
    print(f"  Date Recognition: {'OK' if date_found else 'MISS'}")
    
    # Determine overall success
    overall_success = (
        accuracy_score >= 80 and  # At least 80% of key texts found
        len(structure_errors) == 0 and  # No structure errors
        date_found  # Date pattern recognized
    )
    
    if overall_success:
        print(f"  Overall Result: PASS")
    else:
        print(f"  Overall Result: FAIL")
    
    return overall_success

def test_multi_page_results():
    """Test accuracy of multi-page results if available"""
    print(f"\n" + "=" * 60)
    print("Testing Multi-Page Results (if available)")
    print("=" * 60)
    
    results_file = Path("multi_page_results.json")
    if not results_file.exists():
        print(f"[INFO] No multi-page results file found. Run test_multi_page.py first.")
        return
    
    with open(results_file, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
    
    total_pages = batch_data.get("total_pages", 0)
    results = batch_data.get("results", [])
    
    print(f"[INFO] Found results for {total_pages} pages")
    
    successful_pages = 0
    failed_pages = 0
    
    for i, page_result in enumerate(results):
        if page_result.get("success"):
            successful_pages += 1
            ocr_text = page_result.get("result", "")
            
            try:
                page_data = json.loads(ocr_text)
                element_count = len(page_data) if isinstance(page_data, list) else 1
                print(f"  Page {i}: [OK] {element_count} elements, {len(ocr_text):,} chars")
            except json.JSONDecodeError:
                print(f"  Page {i}: [WARN] Invalid JSON structure")
        else:
            failed_pages += 1
            error = page_result.get("error", "Unknown error")
            print(f"  Page {i}: [FAIL] Failed - {error}")
    
    success_rate = successful_pages / total_pages * 100 if total_pages > 0 else 0
    print(f"\n[SUMMARY] Multi-page success rate: {success_rate:.1f}% ({successful_pages}/{total_pages})")
    
    return success_rate >= 90  # 90% success rate threshold

if __name__ == "__main__":
    print("OCR Accuracy Testing Suite")
    print("=" * 60)
    
    # Test single page accuracy
    single_page_success = test_page_0_accuracy()
    
    # Test multi-page accuracy if available
    multi_page_success = test_multi_page_results()
    
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Single Page Test: {'PASS' if single_page_success else 'FAIL'}")
    if Path("multi_page_results.json").exists():
        print(f"Multi Page Test: {'PASS' if multi_page_success else 'FAIL'}")
    
    overall_success = single_page_success and (multi_page_success or not Path("multi_page_results.json").exists())
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")