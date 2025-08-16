"""
Generic LLM-based extraction utilities to convert OCR output to structured JSON.
Uses OpenAI API to extract data according to any provided JSON schema.
"""
import json
import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv


def load_openai_client() -> Optional[OpenAI]:
    """
    Load OpenAI client using API key from .env file or environment variable.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file or environment variables")
        print("Please add OPENAI_API_KEY=your_api_key to your .env file")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None


def extract_structured_data(ocr_text: str, json_schema: str, client: OpenAI, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Use OpenAI to extract structured data from OCR text according to JSON schema.
    
    Args:
        ocr_text: Raw OCR output from Modal
        json_schema: JSON schema defining the expected structure
        client: OpenAI client instance
        model: OpenAI model to use (default: gpt-4o)
        
    Returns:
        Dictionary with extraction results
    """
    
    # Create generic extraction prompt
    prompt = f"""You are a data extraction expert. Extract structured data from the provided OCR text according to the JSON schema.

OCR Text:
{ocr_text}

Required JSON Schema:
{json_schema}

Instructions:
1. Extract data from the OCR text that matches the provided JSON schema exactly
2. Return ONLY valid JSON that conforms to the schema
3. For numerical fields, ensure values are the correct data type (numbers, not strings)
4. For text fields, clean and normalize the text appropriately
5. If data is unclear or missing, use your best interpretation based on context
6. Ensure the output structure matches the schema precisely
7. Return only the JSON object with no additional text

JSON Output:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise data extraction expert. Return only valid JSON that exactly matches the provided schema."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        extracted_text = response.choices[0].message.content.strip()
        
        # Clean potential markdown formatting
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]
        extracted_text = extracted_text.strip()
        
        # Try to parse as JSON
        try:
            extracted_data = json.loads(extracted_text)
            return {
                "success": True,
                "data": extracted_data,
                "raw_response": extracted_text,
                "model_used": model
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON returned: {e}",
                "raw_response": extracted_text,
                "model_used": model
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"OpenAI API error: {e}",
            "raw_response": None,
            "model_used": model
        }


def compare_json_structures(ground_truth: Dict[str, Any], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic comparison function for any JSON structures.
    Compares field by field and provides detailed accuracy metrics.
    """
    def normalize_value(value):
        """Normalize values for comparison (handle type differences)."""
        if isinstance(value, str):
            return value.strip().lower()
        return value
    
    def compare_objects(gt_obj, ext_obj, path=""):
        """Recursively compare two objects."""
        results = {
            "matches": {},
            "mismatches": {},
            "missing_fields": {},
            "extra_fields": {}
        }
        
        if isinstance(gt_obj, dict) and isinstance(ext_obj, dict):
            # Compare dictionary objects
            gt_keys = set(gt_obj.keys())
            ext_keys = set(ext_obj.keys())
            
            # Check missing and extra fields
            missing = gt_keys - ext_keys
            extra = ext_keys - gt_keys
            common = gt_keys & ext_keys
            
            for key in missing:
                results["missing_fields"][f"{path}.{key}" if path else key] = gt_obj[key]
            
            for key in extra:
                results["extra_fields"][f"{path}.{key}" if path else key] = ext_obj[key]
            
            # Compare common fields
            for key in common:
                field_path = f"{path}.{key}" if path else key
                field_result = compare_objects(gt_obj[key], ext_obj[key], field_path)
                
                # Merge results
                results["matches"].update(field_result["matches"])
                results["mismatches"].update(field_result["mismatches"])
                results["missing_fields"].update(field_result["missing_fields"])
                results["extra_fields"].update(field_result["extra_fields"])
        
        elif isinstance(gt_obj, list) and isinstance(ext_obj, list):
            # Compare arrays - this is complex and domain-specific
            # For now, compare length and sample elements
            if len(gt_obj) == len(ext_obj):
                results["matches"][f"{path}_length"] = len(gt_obj)
                
                # Compare each element if they're objects
                for i, (gt_item, ext_item) in enumerate(zip(gt_obj, ext_obj)):
                    item_result = compare_objects(gt_item, ext_item, f"{path}[{i}]")
                    results["matches"].update(item_result["matches"])
                    results["mismatches"].update(item_result["mismatches"])
                    results["missing_fields"].update(item_result["missing_fields"])
                    results["extra_fields"].update(item_result["extra_fields"])
            else:
                results["mismatches"][f"{path}_length"] = {
                    "expected": len(gt_obj),
                    "got": len(ext_obj)
                }
        
        else:
            # Compare primitive values
            gt_norm = normalize_value(gt_obj)
            ext_norm = normalize_value(ext_obj)
            
            if gt_norm == ext_norm:
                results["matches"][path or "value"] = gt_obj
            else:
                results["mismatches"][path or "value"] = {
                    "expected": gt_obj,
                    "got": ext_obj
                }
        
        return results
    
    # Perform comparison
    comparison = compare_objects(ground_truth, extracted_data)
    
    # Calculate metrics
    total_matches = len(comparison["matches"])
    total_mismatches = len(comparison["mismatches"])
    total_missing = len(comparison["missing_fields"])
    total_extra = len(comparison["extra_fields"])
    
    total_fields = total_matches + total_mismatches + total_missing
    accuracy = (total_matches / total_fields * 100) if total_fields > 0 else 0
    
    return {
        "accuracy_percentage": accuracy,
        "total_fields": total_fields,
        "matches": total_matches,
        "mismatches": total_mismatches,
        "missing_fields": total_missing,
        "extra_fields": total_extra,
        "detailed_comparison": comparison,
        "summary": {
            "extraction_quality": "perfect" if accuracy == 100 else "good" if accuracy >= 80 else "needs_improvement",
            "major_issues": total_missing + total_mismatches,
            "completeness": ((total_fields - total_missing) / total_fields * 100) if total_fields > 0 else 0
        }
    }


def generate_comparison_report(comparison_results: Dict[str, Any], save_path: str = None) -> str:
    """
    Generate a detailed comparison report.
    
    Args:
        comparison_results: Results from compare_json_structures
        save_path: Optional path to save the report
        
    Returns:
        String containing the formatted report
    """
    report_lines = []
    report_lines.append("EXTRACTION COMPARISON REPORT")
    report_lines.append("=" * 50)
    
    # Overall metrics
    accuracy = comparison_results["accuracy_percentage"]
    report_lines.append(f"Overall Accuracy: {accuracy:.1f}%")
    report_lines.append(f"Total Fields: {comparison_results['total_fields']}")
    report_lines.append(f"Correct Fields: {comparison_results['matches']}")
    report_lines.append(f"Incorrect Fields: {comparison_results['mismatches']}")
    report_lines.append(f"Missing Fields: {comparison_results['missing_fields']}")
    report_lines.append(f"Extra Fields: {comparison_results['extra_fields']}")
    
    # Quality assessment
    quality = comparison_results["summary"]["extraction_quality"]
    completeness = comparison_results["summary"]["completeness"]
    report_lines.append(f"Extraction Quality: {quality}")
    report_lines.append(f"Data Completeness: {completeness:.1f}%")
    
    # Detailed issues
    detailed = comparison_results["detailed_comparison"]
    
    if detailed["mismatches"]:
        report_lines.append("\nMISMATCHED FIELDS:")
        for field, values in detailed["mismatches"].items():
            report_lines.append(f"  {field}: expected '{values['expected']}', got '{values['got']}'")
    
    if detailed["missing_fields"]:
        report_lines.append("\nMISSING FIELDS:")
        for field, value in detailed["missing_fields"].items():
            report_lines.append(f"  {field}: {value}")
    
    if detailed["extra_fields"]:
        report_lines.append("\nEXTRA FIELDS:")
        for field, value in detailed["extra_fields"].items():
            report_lines.append(f"  {field}: {value}")
    
    # Summary
    report_lines.append("\nSUMMARY:")
    if accuracy == 100:
        report_lines.append("Perfect extraction - all fields match exactly!")
    elif accuracy >= 90:
        report_lines.append("Excellent extraction with minor discrepancies")
    elif accuracy >= 70:
        report_lines.append("Good extraction with some issues to address")
    else:
        report_lines.append("Extraction needs significant improvement")
    
    report = "\n".join(report_lines)
    
    # Save if path provided
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved to: {save_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
    
    return report


def extract_and_compare(ocr_text: str, ground_truth: Dict[str, Any], json_schema: str, 
                       openai_client: OpenAI, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Complete workflow: extract structured data and compare with ground truth.
    
    Args:
        ocr_text: Raw OCR output
        ground_truth: Expected structured data
        json_schema: JSON schema for extraction
        openai_client: OpenAI client instance
        model: OpenAI model to use
        
    Returns:
        Complete results including extraction and comparison
    """
    # Step 1: Extract structured data
    extraction_result = extract_structured_data(ocr_text, json_schema, openai_client, model)
    
    if not extraction_result["success"]:
        return {
            "extraction_successful": False,
            "extraction_error": extraction_result["error"],
            "comparison_results": None
        }
    
    # Step 2: Compare with ground truth
    comparison_results = compare_json_structures(ground_truth, extraction_result["data"])
    
    return {
        "extraction_successful": True,
        "extracted_data": extraction_result["data"],
        "extraction_metadata": {
            "model_used": extraction_result["model_used"],
            "raw_response": extraction_result["raw_response"]
        },
        "comparison_results": comparison_results,
        "overall_success": comparison_results["accuracy_percentage"] == 100
    }


if __name__ == "__main__":
    print("Generic LLM-based OCR Extraction Utilities")
    print("This module provides functions to:")
    print("1. Extract structured data from OCR using OpenAI (any schema)")
    print("2. Compare extracted data with ground truth (any JSON structure)")
    print("3. Generate detailed accuracy reports")
    print("Required: OPENAI_API_KEY environment variable")