#!/usr/bin/env python3
"""
Test script for processing image with bounding box annotation
Similar to demo_gradio_annotation.py style but using Modal endpoint
"""

import argparse
import base64
import io
import requests
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import tempfile
import os

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def draw_bounding_box_on_image(image, bbox, label="OCR Region", color=(255, 0, 0), thickness=3):
    """
    Draw a bounding box on the image similar to gradio annotation style
    
    Args:
        image: PIL Image
        bbox: [x1, y1, x2, y2] coordinates
        label: Label text for the box
        color: RGB color tuple
        thickness: Line thickness
    
    Returns:
        PIL Image with bounding box drawn
    """
    # Create a copy to avoid modifying original
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    x1, y1, x2, y2 = bbox
    
    # Draw the bounding box rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Draw label above the box
    if font:
        text_bbox = draw.textbbox((x1, y1-25), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw background for text
        draw.rectangle([x1, y1-25, x1+text_width+6, y1-25+text_height+4], fill=color)
        draw.text((x1+3, y1-25+2), label, fill=(255, 255, 255), font=font)
    else:
        # Fallback without font
        draw.text((x1+5, y1-20), label, fill=color)
    
    return annotated_image

def test_modal_ocr_with_bbox(modal_url, image_path, bbox, output_dir="./output"):
    """
    Test the Modal OCR endpoint with bounding box (grounding OCR)
    
    Args:
        modal_url: Modal endpoint URL
        image_path: Path to test image
        bbox: [x1, y1, x2, y2] bounding box coordinates
        output_dir: Directory to save results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare image
    image = Image.open(image_path)
    print(f"Original image dimensions: {image.width} x {image.height}")
    
    # Create annotated image with bounding box
    annotated_image = draw_bounding_box_on_image(image, bbox, "OCR Region", (255, 0, 0), 3)
    
    # Save annotated image - convert RGBA to RGB if needed for JPEG
    annotated_path = os.path.join(output_dir, "annotated_image.jpg")
    if annotated_image.mode == 'RGBA':
        # Convert to RGB with white background
        rgb_image = Image.new('RGB', annotated_image.size, (255, 255, 255))
        rgb_image.paste(annotated_image, mask=annotated_image.split()[-1])
        rgb_image.save(annotated_path)
    else:
        annotated_image.save(annotated_path)
    print(f"Annotated image saved to: {annotated_path}")
    
    # Prepare the request data for grounding OCR
    image_b64 = image_to_base64(image_path)
    
    request_data = {
        "image": image_b64,
        "prompt_mode": "prompt_grounding_ocr",
        "bbox": bbox,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 2048
    }
    
    # Send request to Modal endpoint
    try:
        print(f"\nSending grounding OCR request to: {modal_url}")
        print(f"Image: {image_path}")
        print(f"Bounding box: {bbox}")
        print(f"Processing mode: Region OCR")
        
        response = requests.post(
            modal_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                ocr_text = result["result"]
                
                # Save OCR result to file
                result_path = os.path.join(output_dir, "ocr_result.txt")
                with open(result_path, "w", encoding="utf-8") as f:
                    f.write(ocr_text)
                
                # Create a summary file similar to gradio demo
                summary_path = os.path.join(output_dir, "processing_summary.md")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(f"""# OCR Processing Summary

## Image Information
- **Original Dimensions:** {image.width} x {image.height}
- **Processing Mode:** Region OCR (Grounding OCR)
- **Bounding Box Coordinates:** {bbox}
- **Image Path:** {image_path}

## OCR Result
```
{ocr_text}
```

## Files Generated
- `annotated_image.jpg` - Image with bounding box annotation
- `ocr_result.txt` - Raw OCR text result
- `processing_summary.md` - This summary file

## Request Details
- **Prompt Mode:** prompt_grounding_ocr
- **Temperature:** 0.1
- **Top P:** 0.9
- **Max Tokens:** 2048
""")
                
                print(f"\n=== OCR Processing Complete ===")
                print(f"Output directory: {output_dir}")
                print(f"Annotated image: {annotated_path}")
                print(f"OCR result: {result_path}")
                print(f"Summary: {summary_path}")
                print(f"\n=== OCR Result ===")
                print(ocr_text)
                print("=" * 50)
                
                return ocr_text
                
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"OCR failed: {error_msg}")
                
                # Save error info
                error_path = os.path.join(output_dir, "error.txt")
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(f"OCR Error: {error_msg}\n")
                    f.write(f"Request data: {json.dumps(request_data, indent=2)}\n")
                
                return None
        else:
            error_msg = f"HTTP Error {response.status_code}: {response.text}"
            print(f"Error: {error_msg}")
            
            # Save error info
            error_path = os.path.join(output_dir, "error.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"{error_msg}\n")
            
            return None
            
    except requests.RequestException as e:
        error_msg = f"Request error: {e}"
        print(f"Error: {error_msg}")
        
        # Save error info
        error_path = os.path.join(output_dir, "error.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"{error_msg}\n")
        
        return None

def create_layout_annotated_image(image_path, layout_data, output_dir="./output"):
    """Create annotated image from layout analysis results"""
    
    def get_category_color(category):
        """Get consistent colors for different layout categories"""
        color_map = {
            'Text': (0, 150, 0),         # Dark Green
            'Table': (255, 0, 0),        # Red
            'Page-footer': (128, 128, 128), # Gray
            'Title': (0, 0, 255),        # Blue
            'Section-header': (255, 165, 0), # Orange
            'Picture': (255, 20, 147),   # Deep Pink
        }
        return color_map.get(category, (0, 0, 0))  # Default to black
    
    # Load image
    image = Image.open(image_path)
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            font = ImageFont.load_default()
            small_font = font
        except:
            font = None
            small_font = None
    
    # Draw bounding boxes for each element
    for i, element in enumerate(layout_data):
        bbox = element.get('bbox', [])
        category = element.get('category', 'Unknown')
        
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = bbox
        color = get_category_color(category)
        
        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw category label
        label = f"{category}"
        if font:
            # Calculate text size and position
            text_bbox = draw.textbbox((x1, y1-25), label, font=small_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw background for text
            bg_y = max(0, y1-25)
            draw.rectangle([x1, bg_y, x1+text_width+4, bg_y+text_height+2], fill=color)
            draw.text((x1+2, bg_y+1), label, fill=(255, 255, 255), font=small_font)
        else:
            # Fallback without font
            draw.text((x1+2, max(0, y1-15)), label, fill=color)
    
    # Save annotated image
    annotated_path = os.path.join(output_dir, "layout_annotated.jpg")
    if annotated_image.mode == 'RGBA':
        # Convert to RGB with white background
        rgb_image = Image.new('RGB', annotated_image.size, (255, 255, 255))
        rgb_image.paste(annotated_image, mask=annotated_image.split()[-1])
        rgb_image.save(annotated_path)
    else:
        annotated_image.save(annotated_path)
    
    print(f"Layout annotated image saved to: {annotated_path}")
    return annotated_path

def test_modal_ocr_full_layout(modal_url, image_path, output_dir="./output"):
    """
    Test the Modal OCR endpoint with full layout analysis
    
    Args:
        modal_url: Modal endpoint URL
        image_path: Path to test image
        output_dir: Directory to save results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare image
    image = Image.open(image_path)
    print(f"Original image dimensions: {image.width} x {image.height}")
    
    # Prepare the request data for full layout OCR
    image_b64 = image_to_base64(image_path)
    
    request_data = {
        "image": image_b64,
        "prompt_mode": "prompt_layout_all_en",
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 4096
    }
    
    # Send request to Modal endpoint
    try:
        print(f"\nSending full layout OCR request to: {modal_url}")
        print(f"Image: {image_path}")
        print(f"Processing mode: Full Layout Analysis")
        
        response = requests.post(
            modal_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                ocr_result = result["result"]
                
                # Save OCR result to file
                result_path = os.path.join(output_dir, "layout_result.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    # Try to parse as JSON and pretty print, fallback to raw text
                    try:
                        json_data = json.loads(ocr_result)
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        f.write(ocr_result)
                
                # Create a summary file
                summary_path = os.path.join(output_dir, "layout_summary.md")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(f"""# Layout Analysis Summary

## Image Information
- **Original Dimensions:** {image.width} x {image.height}
- **Processing Mode:** Full Layout Analysis
- **Image Path:** {image_path}

## Layout Result
The complete layout analysis has been saved to `layout_result.json`.

## Files Generated
- `layout_result.json` - Complete layout analysis with bounding boxes and text
- `layout_summary.md` - This summary file

## Request Details
- **Prompt Mode:** prompt_layout_all_en
- **Temperature:** 0.1
- **Top P:** 0.9
- **Max Tokens:** 4096
""")
                
                print(f"\n=== Layout Analysis Complete ===")
                print(f"Output directory: {output_dir}")
                print(f"Layout result: {result_path}")
                print(f"Summary: {summary_path}")
                
                # Create annotated image from layout results
                try:
                    # Parse the layout JSON to create annotations
                    start_idx = ocr_result.find('[')
                    if start_idx != -1:
                        clean_json = ocr_result[start_idx:]
                        layout_data = json.loads(clean_json)
                        annotated_path = create_layout_annotated_image(image_path, layout_data, output_dir)
                        print(f"Layout annotated image: {annotated_path}")
                    else:
                        print("Could not parse layout data for annotation")
                except Exception as e:
                    print(f"Could not create annotated image: {e}")
                
                print(f"\n=== Layout Result Preview ===")
                print(ocr_result[:1000] + "..." if len(ocr_result) > 1000 else ocr_result)
                print("=" * 50)
                
                return ocr_result
                
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"Layout analysis failed: {error_msg}")
                return None
        else:
            error_msg = f"HTTP Error {response.status_code}: {response.text}"
            print(f"Error: {error_msg}")
            return None
            
    except requests.RequestException as e:
        error_msg = f"Request error: {e}"
        print(f"Error: {error_msg}")
        return None

def interactive_bbox_selection(image_path):
    """
    Help user select bounding box coordinates interactively
    """
    image = Image.open(image_path)
    print(f"\nImage dimensions: {image.width} x {image.height}")
    print("\nTo select a bounding box, provide coordinates in format: x1,y1,x2,y2")
    print("Where (x1,y1) is top-left corner and (x2,y2) is bottom-right corner")
    print("\nExample bounding boxes for common use cases:")
    print(f"- Top half: 0,0,{image.width},{image.height//2}")
    print(f"- Bottom half: 0,{image.height//2},{image.width},{image.height}")
    print(f"- Left half: 0,0,{image.width//2},{image.height}")
    print(f"- Right half: {image.width//2},0,{image.width},{image.height}")
    print(f"- Center region: {image.width//4},{image.height//4},{3*image.width//4},{3*image.height//4}")
    
    while True:
        try:
            bbox_input = input("\nEnter bounding box (x1,y1,x2,y2) or 'q' to quit: ").strip()
            if bbox_input.lower() == 'q':
                return None
            
            coords = [int(x.strip()) for x in bbox_input.split(',')]
            if len(coords) != 4:
                raise ValueError("Must provide exactly 4 coordinates")
            
            x1, y1, x2, y2 = coords
            
            # Validate coordinates
            if x1 >= x2 or y1 >= y2:
                raise ValueError("Invalid box: x2 must be > x1 and y2 must be > y1")
            if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height:
                raise ValueError(f"Coordinates must be within image bounds (0,0,{image.width},{image.height})")
            
            return coords
            
        except ValueError as e:
            print(f"Invalid input: {e}")
            print("Please try again.")

def main():
    parser = argparse.ArgumentParser(description="Test Modal DotsOCR with bounding box annotation")
    parser.add_argument("--modal_url", type=str, 
                       default="https://marker--dotsocr-v2.modal.run",
                       help="Modal endpoint URL")
    parser.add_argument("--image_path", type=str, 
                       default="tests/image.png",
                       help="Path to test image")
    parser.add_argument("--bbox", type=str, 
                       help="Bounding box coordinates as 'x1,y1,x2,y2'")
    parser.add_argument("--output_dir", type=str, 
                       default="./output",
                       help="Output directory for results")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive bounding box selection")
    parser.add_argument("--mode", type=str, choices=["bbox", "layout", "both"],
                       default="bbox", help="Processing mode: bbox only, layout only, or both")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    print(f"Testing Modal DotsOCR with Annotation")
    print(f"Modal URL: {args.modal_url}")
    print(f"Image: {args.image_path}")
    print("=" * 60)
    
    if args.mode in ["layout", "both"]:
        print("\n=== Running Full Layout Analysis ===")
        layout_result = test_modal_ocr_full_layout(args.modal_url, args.image_path, args.output_dir)
    
    if args.mode in ["bbox", "both"]:
        # Get bounding box coordinates
        bbox = None
        if args.bbox:
            try:
                bbox = [int(x.strip()) for x in args.bbox.split(',')]
                if len(bbox) != 4:
                    raise ValueError("Must provide exactly 4 coordinates")
            except ValueError as e:
                print(f"Invalid bbox format: {e}")
                return
        elif args.interactive:
            bbox = interactive_bbox_selection(args.image_path)
            if bbox is None:
                print("Goodbye!")
                return
        else:
            # Default bbox for lease agreement - targeting the main content area
            print("Using default bounding box for document content...")
            image = Image.open(args.image_path)
            # Use a reasonable default that covers most of the document
            bbox = [50, 100, image.width-50, image.height-100]
        
        print(f"Using bounding box: {bbox}")
        
        print("\n=== Running Bounding Box OCR ===")
        # Run OCR with bounding box
        bbox_result = test_modal_ocr_with_bbox(args.modal_url, args.image_path, bbox, args.output_dir)
    
    print(f"\nProcessing completed!")
    print(f"Check {args.output_dir} for detailed results")

if __name__ == "__main__":
    main()