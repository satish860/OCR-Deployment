import time
import requests
import base64
import json
from PIL import Image
import io
import fitz  # PyMuPDF
from typing import Optional, List, Dict, Any, Union


class OCRClient:
    """
    Clean OCR client for direct GPU endpoint communication.
    No Modal SDK required - uses direct HTTP calls.
    """
    
    def __init__(self, process_url: str, health_url: str = None, timeout: int = 120):
        """
        Initialize OCR client.
        
        Args:
            process_url: Direct GPU process endpoint URL
            health_url: Health endpoint URL (optional, will derive from process_url if not provided)
            timeout: Request timeout in seconds
        """
        self.process_url = process_url
        self.health_url = health_url or process_url.replace("process", "health")
        self.timeout = timeout
    
    def check_health(self) -> Dict[str, Any]:
        """Check if the OCR service is healthy."""
        try:
            response = requests.get(self.health_url, timeout=10)
            if response.status_code == 200:
                return {"healthy": True, "details": response.json()}
            else:
                return {"healthy": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def pdf_page_to_base64(self, pdf_path: str, page_num: int = 0, dpi: int = 100) -> tuple[str, tuple[int, int]]:
        """
        Convert a PDF page to base64 encoded image.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            dpi: DPI for image conversion
            
        Returns:
            Tuple of (base64_string, image_size)
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render page to image
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image and then to base64 (optimized JPEG)
        pil_image = Image.open(io.BytesIO(img_data))
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        doc.close()
        return img_base64, pil_image.size
    
    def image_to_base64(self, image_path: str) -> tuple[str, tuple[int, int]]:
        """
        Convert image file to base64.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (base64_string, image_size)
        """
        pil_image = Image.open(image_path)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64, pil_image.size
    
    def process_image(
        self,
        image_b64: str,
        prompt_mode: str = "prompt_layout_all_en",
        max_tokens: int = 1500,
        temperature: float = 0.0,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Process a single image.
        
        Args:
            image_b64: Base64 encoded image
            prompt_mode: OCR prompt mode
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with OCR results
        """
        request_data = {
            "image": image_b64,
            "prompt_mode": prompt_mode,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            start_time = time.time()
            response = requests.post(self.process_url, json=request_data, timeout=self.timeout)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                result["processing_time"] = end_time - start_time
                return result
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "processing_time": end_time - start_time
                }
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Request timed out after {self.timeout} seconds"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}"}
    
    def process_batch(
        self,
        images_b64: List[str],
        prompt_mode: str = "prompt_layout_all_en",
        max_tokens: int = 1500,
        temperature: float = 0.0,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Process multiple images in a batch.
        
        Args:
            images_b64: List of base64 encoded images
            prompt_mode: OCR prompt mode
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with batch OCR results
        """
        request_data = {
            "images": images_b64,
            "prompt_mode": prompt_mode,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            start_time = time.time()
            response = requests.post(self.process_url, json=request_data, timeout=self.timeout)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                result["processing_time"] = end_time - start_time
                return result
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "processing_time": end_time - start_time
                }
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Request timed out after {self.timeout} seconds"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}"}
    
    def process_pdf_page(
        self,
        pdf_path: str,
        page_num: int = 0,
        prompt_mode: str = "prompt_layout_all_en",
        dpi: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            prompt_mode: OCR prompt mode
            dpi: DPI for PDF to image conversion
            **kwargs: Additional arguments for process_image
            
        Returns:
            Dictionary with OCR results
        """
        # Convert PDF page to base64
        image_b64, image_size = self.pdf_page_to_base64(pdf_path, page_num, dpi)
        
        # Process the image
        result = self.process_image(image_b64, prompt_mode, **kwargs)
        
        # Add metadata
        if result.get("success"):
            result["pdf_path"] = pdf_path
            result["page_number"] = page_num
            result["image_size"] = image_size
            result["dpi"] = dpi
        
        return result
    
    def process_image_file(
        self,
        image_path: str,
        prompt_mode: str = "prompt_layout_all_en",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image file.
        
        Args:
            image_path: Path to image file
            prompt_mode: OCR prompt mode
            **kwargs: Additional arguments for process_image
            
        Returns:
            Dictionary with OCR results
        """
        # Convert image to base64
        image_b64, image_size = self.image_to_base64(image_path)
        
        # Process the image
        result = self.process_image(image_b64, prompt_mode, **kwargs)
        
        # Add metadata
        if result.get("success"):
            result["image_path"] = image_path
            result["image_size"] = image_size
        
        return result
    
    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt modes."""
        return [
            "prompt_layout_all_en",
            "prompt_layout_only_en", 
            "prompt_ocr",
            "prompt_grounding_ocr"
        ]
    
    def save_result(self, result: Dict[str, Any], output_file: str) -> bool:
        """
        Save OCR result to file.
        
        Args:
            result: OCR result dictionary
            output_file: Output file path
            
        Returns:
            True if saved successfully
        """
        try:
            if result.get("success"):
                content = result.get("result", "")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(content)
                return True
            else:
                # Save error info
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Error: {result.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"Failed to save result: {e}")
            return False


# Convenience functions for quick usage
def quick_process_pdf(
    process_url: str,
    pdf_path: str,
    page_num: int = 0,
    prompt_mode: str = "prompt_layout_all_en"
) -> Dict[str, Any]:
    """
    Quick function to process a PDF page.
    
    Args:
        process_url: OCR process endpoint URL
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        prompt_mode: OCR prompt mode
        
    Returns:
        OCR result dictionary
    """
    client = OCRClient(process_url)
    return client.process_pdf_page(pdf_path, page_num, prompt_mode)


def quick_process_image(
    process_url: str,
    image_path: str,
    prompt_mode: str = "prompt_layout_all_en"
) -> Dict[str, Any]:
    """
    Quick function to process an image file.
    
    Args:
        process_url: OCR process endpoint URL
        image_path: Path to image file
        prompt_mode: OCR prompt mode
        
    Returns:
        OCR result dictionary
    """
    client = OCRClient(process_url)
    return client.process_image_file(image_path, prompt_mode)