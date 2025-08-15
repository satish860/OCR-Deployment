import modal
import os
from pathlib import Path

# Create the Modal app
app = modal.App("dotsocr-modal-pattern-v2")

# Model configuration
MODEL_DIR = "/workspace/weights/DotsOCR"
MODEL_NAME = "DotsOCR"

# Create volume for model caching
vllm_volume = modal.Volume.from_name("vllm-vol", create_if_missing=True)

# Create image following Modal's vLLM pattern
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .run_commands([
        "python3 -m pip install --upgrade pip",
        "python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "python3 -m pip install vllm==0.9.1",
        "python3 -m pip install flash-attn==2.8.0.post2 --no-build-isolation",
        "python3 -m pip install transformers==4.51.3 fastapi[standard] requests Pillow qwen_vl_utils huggingface_hub modelscope PyMuPDF"
    ])
    .add_local_dir("dots.ocr/dots_ocr", remote_path="/workspace/dots_ocr", copy=True)
    .add_local_dir("dots.ocr/weights/DotsOCR", remote_path="/workspace/weights/DotsOCR", copy=True)
    .add_local_file("dots.ocr/tools/download_model.py", remote_path="/workspace/tools/download_model.py", copy=True)
    .run_commands([
        # Download the actual model weights from Hugging Face after local files are copied
        "cd /workspace && python3 tools/download_model.py --type huggingface --name rednote-hilab/dots.ocr",
        "ls -la /workspace/weights/DotsOCR/ | head -20"  # Debug: show first 20 files
    ])
)

# GPU configuration
GPU_CONFIG = "A100-40GB"


@app.cls(
    gpu=GPU_CONFIG,
    image=vllm_image,
    volumes={"/cache": vllm_volume},
    timeout=60 * 20,  # 20 minutes
    scaledown_window=60 * 5,  # 5 minutes
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=10)
class vLLMModel:
    @modal.enter()
    def load_model(self):
        import time
        start = time.time()
        
        print("🔧 DEBUG: Starting @modal.enter() method")
        
        # Set up environment paths
        import sys
        import os
        os.environ["PYTHONPATH"] = "/workspace:/workspace/weights/DotsOCR:" + os.environ.get("PYTHONPATH", "")
        sys.path.insert(0, "/workspace")
        sys.path.insert(0, "/workspace/weights/DotsOCR")
        
        # Performance optimization: enable fast image processor
        os.environ["TRANSFORMERS_USE_FAST_PROCESSOR"] = "1"
        
        print("🔧 DEBUG: Environment paths set")
        print(f"🔧 DEBUG: MODEL_DIR = {MODEL_DIR}")
        
        print("🔧 DEBUG: Model should now have weights downloaded during image build")
        
        print("🥶 Cold boot: downloading model weights...")
        
        # Register DotsOCR model with vLLM
        try:
            import importlib.util
            from vllm import ModelRegistry
            
            print("🔧 DEBUG: About to register DotsOCR model")
            
            # Change working directory to the model directory so relative imports work
            import os
            original_cwd = os.getcwd()
            os.chdir("/workspace/weights/DotsOCR")
            
            # Add the DotsOCR directory to Python path as a package
            import sys
            if "/workspace/weights" not in sys.path:
                sys.path.insert(0, "/workspace/weights")
            
            # Now import as a proper module
            from DotsOCR import modeling_dots_ocr_vllm
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            ModelRegistry.register_model("DotsOCRForCausalLM", modeling_dots_ocr_vllm.DotsOCRForCausalLM)
            print("✅ DotsOCR model registered with vLLM")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not register DotsOCR model: {e}")
            import traceback
            traceback.print_exc()
        
        # Initialize vLLM engine
        try:
            print("🔧 DEBUG: About to import vLLM")
            from vllm import LLM, SamplingParams
            
            print("🔧 DEBUG: About to initialize LLM")
            self.llm = LLM(
                model=MODEL_DIR,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.95,  # Match original script
                disable_custom_all_reduce=True,
                # Performance optimization: use fast image processor
                tokenizer_mode="auto",
                # Let vLLM auto-detect max_model_len from config (131072)
                # Note: vLLM should auto-detect chat template from model's chat_template.json
            )
            print("🔧 DEBUG: LLM initialized successfully")
            
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize LLM: {e}")
            import traceback
            traceback.print_exc()
            # Set a flag so we know it failed and store the error
            self.llm = None
            self.llm_error = str(e)
            return
        
        # Load DotsOCR utilities
        try:
            print("🔧 DEBUG: Loading DotsOCR utilities")
            from dots_ocr.utils import dict_promptmode_to_prompt
            from dots_ocr.utils.image_utils import PILimage_to_base64
            
            self.prompts = dict_promptmode_to_prompt
            self.image_to_base64 = PILimage_to_base64
            print("🔧 DEBUG: Utilities loaded successfully")
            
        except Exception as e:
            print(f"❌ ERROR: Failed to load utilities: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"🏁 Model loaded in {time.time()-start:.1f}s")
        print(f"🔧 DEBUG: self.llm is: {type(self.llm) if hasattr(self, 'llm') else 'NOT SET'}")
        
        # No return needed for @modal.enter() method

    @modal.method()
    def generate(
        self,
        image_b64: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        print("🔧 DEBUG: Starting generate method")
        print(f"🔧 DEBUG: self has llm attribute: {hasattr(self, 'llm')}")
        if hasattr(self, 'llm'):
            print(f"🔧 DEBUG: self.llm type: {type(self.llm)}")
            print(f"🔧 DEBUG: self.llm is None: {self.llm is None}")
        
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        try:
            # Debug info for troubleshooting
            debug_info = {
                "has_llm_attr": hasattr(self, 'llm'),
                "llm_is_none": self.llm is None if hasattr(self, 'llm') else "no attr",
                "llm_type": str(type(self.llm)) if hasattr(self, 'llm') and self.llm is not None else "N/A",
                "llm_error": getattr(self, 'llm_error', 'No error stored'),
                "all_attrs": [attr for attr in dir(self) if not attr.startswith('_')]
            }
            
            if not hasattr(self, 'llm') or self.llm is None:
                return f"Error: Model not properly initialized. Debug: {debug_info}"
            
            # Format the input as DotsOCR expects
            prompt_text = f"<|img|><|imgpad|><|endofimg|>{prompt}"
            print(f"🔧 DEBUG: Formatted prompt: {prompt_text[:100]}...")
            
            # Convert base64 back to PIL image for vLLM generate API
            import base64
            import io
            from PIL import Image
            
            # Extract base64 data from data URL
            if image_b64.startswith("data:image/"):
                base64_data = image_b64.split(",")[1]
            else:
                base64_data = image_b64
                
            image_bytes = base64.b64decode(base64_data)
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            print(f"🔧 DEBUG: Using direct generate() with PIL image")
            
            # Generate response
            print("🔧 DEBUG: About to call self.llm.generate")
            outputs = self.llm.generate(
                prompts=[{
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": pil_image}
                }],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            print("🔧 DEBUG: Chat completed")
            
            if outputs and len(outputs) > 0 and outputs[0].outputs:
                result = outputs[0].outputs[0].text
                print(f"🔧 DEBUG: Generated result length: {len(result)}")
                return result
            else:
                return "No output generated"
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    @modal.method()
    def health_check(self):
        """Basic health check"""
        try:
            # Simple test to ensure model is loaded
            return {"status": "healthy", "model": MODEL_NAME}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Create a persistent model instance
model = vLLMModel()




def _process_ocr_request(image_data, prompt_mode="prompt_layout_all_en", max_tokens=1500, temperature=0.1, top_p=0.9, from_base64=True, bbox=None):
    """Synchronous OCR processing logic"""
    try:
        import base64
        import io
        from PIL import Image
        import sys
        
        sys.path.insert(0, "/workspace")
        from dots_ocr.utils import dict_promptmode_to_prompt
        from dots_ocr.utils.image_utils import PILimage_to_base64
        
        if not image_data:
            raise ValueError("No image data provided")
        
        # Validate prompt mode
        valid_prompt_modes = list(dict_promptmode_to_prompt.keys())
        if prompt_mode not in valid_prompt_modes:
            raise ValueError(f"Invalid prompt_mode '{prompt_mode}'. Valid options: {valid_prompt_modes}")
        
        # Validate grounding OCR requirements
        if prompt_mode == "prompt_grounding_ocr" and not bbox:
            raise ValueError("Grounding OCR requires 'bbox' parameter with format [x1, y1, x2, y2]")
        
        if bbox and prompt_mode != "prompt_grounding_ocr":
            raise ValueError("'bbox' parameter is only valid with 'prompt_grounding_ocr' mode")
        
        # Process image based on input format
        if from_base64:
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {e}")
        else:
            # Direct bytes input
            try:
                pil_image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                raise ValueError(f"Invalid image bytes: {e}")
        
        # Get the appropriate prompt
        prompt = dict_promptmode_to_prompt[prompt_mode]
        
        # Handle grounding OCR with bounding box
        if prompt_mode == "prompt_grounding_ocr" and bbox:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("bbox must be a list of 4 numbers [x1, y1, x2, y2]")
            prompt += str(bbox)
        
        # Convert image to the format expected by the model
        image_b64 = PILimage_to_base64(pil_image)
        
        # Call model directly
        result = model.generate.remote(
            image_b64=image_b64,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        return {
            "success": True,
            "result": result,
            "prompt_mode": prompt_mode,
            "bbox": bbox if bbox else None
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.function(image=vllm_image)
@modal.fastapi_endpoint(method="POST", label="dotsocr-v2")
def generate(request_data: dict):
    """
    Generate OCR results from base64 image data.
    Supports both single and batch processing with multiple prompt modes.
    
    Single image format:
    {
        "image": "base64_encoded_image",
        "prompt_mode": "prompt_layout_all_en",
        "bbox": [x1, y1, x2, y2],  // Optional, only for grounding OCR
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    Batch processing format:
    {
        "images": ["base64_encoded_image1", "base64_encoded_image2", ...],
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    Available prompt_mode options:
    - "prompt_layout_all_en": Full layout detection + text extraction (JSON format)
    - "prompt_layout_only_en": Layout detection only, no text content
    - "prompt_ocr": Simple text extraction without layout information
    - "prompt_grounding_ocr": Extract text from specific bounding box (requires bbox parameter)
    """
    # Check if this is a batch request
    if "images" in request_data:
        images = request_data.get("images", [])
        if not images:
            return {"success": False, "error": "No images provided"}
        
        prompt_mode = request_data.get("prompt_mode", "prompt_layout_all_en")
        max_tokens = request_data.get("max_tokens", 1500)
        temperature = request_data.get("temperature", 0.1)
        top_p = request_data.get("top_p", 0.9)
        bbox = request_data.get("bbox")  # For grounding OCR in batch
        
        results = []
        for i, image_data in enumerate(images):
            result = _process_ocr_request(
                image_data=image_data,
                prompt_mode=prompt_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                from_base64=True,
                bbox=bbox
            )
            # Add page number to result
            if result.get("success"):
                result["page_number"] = i
            results.append(result)
        
        return {
            "success": True,
            "total_pages": len(images),
            "results": results
        }
    
    # Single image processing
    else:
        return _process_ocr_request(
            image_data=request_data.get("image"),
            prompt_mode=request_data.get("prompt_mode", "prompt_layout_all_en"),
            max_tokens=request_data.get("max_tokens", 1500),
            temperature=request_data.get("temperature", 0.1),
            top_p=request_data.get("top_p", 0.9),
            from_base64=True,
            bbox=request_data.get("bbox")
        )






@app.function(image=vllm_image)
@modal.fastapi_endpoint(method="GET", label="health-v2")
def health():
    """Health check endpoint"""
    return model.health_check.remote()


# Local testing
if __name__ == "__main__":
    print("🚀 Deploying DotsOCR Modal app...")
    # You can add local testing code here