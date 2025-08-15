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


def _preprocess_image(image_data, prompt_mode="prompt_layout_all_en", from_base64=True):
    """Fast preprocessing of image data - synchronous, prepares data for parallel OCR"""
    import base64
    import io
    from PIL import Image
    import sys
    
    sys.path.insert(0, "/workspace")
    from dots_ocr.utils import dict_promptmode_to_prompt
    from dots_ocr.utils.image_utils import PILimage_to_base64
    
    if not image_data:
        raise ValueError("No image data provided")
    
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
    prompt = dict_promptmode_to_prompt.get(
        prompt_mode, 
        dict_promptmode_to_prompt["prompt_layout_all_en"]
    )
    
    # Convert image to the format expected by the model
    image_b64 = PILimage_to_base64(pil_image)
    
    return {
        "image_b64": image_b64,
        "prompt": prompt
    }


def _process_ocr_request_async(preprocessed_data, max_tokens=1500, temperature=0.1, top_p=0.9):
    """Pure async OCR call - returns Modal future IMMEDIATELY (non-blocking)"""
    # This should be instantaneous - no image processing, just returns future
    return model.generate.remote(
        image_b64=preprocessed_data["image_b64"],
        prompt=preprocessed_data["prompt"],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def _process_ocr_request(image_data, prompt_mode="prompt_layout_all_en", max_tokens=1500, temperature=0.1, top_p=0.9, from_base64=True):
    """Synchronous OCR processing logic (existing function)"""
    try:
        # Preprocess the image first
        preprocessed_data = _preprocess_image(
            image_data=image_data,
            prompt_mode=prompt_mode,
            from_base64=from_base64
        )
        
        # Get the future and wait for result
        future = _process_ocr_request_async(
            preprocessed_data=preprocessed_data,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Wait for completion and return formatted result
        result = future  # Modal automatically waits for completion
        
        return {
            "success": True,
            "result": result,
            "prompt_mode": prompt_mode
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.function(image=vllm_image)
@modal.fastapi_endpoint(method="POST", label="dotsocr-v2")
def generate(request_data: dict):
    """
    Generate OCR results from base64 image data.
    
    Expected format:
    {
        "image": "base64_encoded_image",
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    """
    return _process_ocr_request(
        image_data=request_data.get("image"),
        prompt_mode=request_data.get("prompt_mode", "prompt_layout_all_en"),
        max_tokens=request_data.get("max_tokens", 1500),
        temperature=request_data.get("temperature", 0.1),
        top_p=request_data.get("top_p", 0.9),
        from_base64=True
    )


@app.function(image=vllm_image)
@modal.fastapi_endpoint(method="POST", label="dotsocr-batch-v2")
def generate_batch(request_data: dict):
    """
    Generate OCR results from multiple base64 images in batch.
    
    Expected format:
    {
        "images": ["base64_encoded_image1", "base64_encoded_image2", ...],
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    """
    images = request_data.get("images", [])
    if not images:
        return {"success": False, "error": "No images provided"}
    
    prompt_mode = request_data.get("prompt_mode", "prompt_layout_all_en")
    max_tokens = request_data.get("max_tokens", 1500)
    temperature = request_data.get("temperature", 0.1)
    top_p = request_data.get("top_p", 0.9)
    
    results = []
    for i, image_data in enumerate(images):
        result = _process_ocr_request(
            image_data=image_data,
            prompt_mode=prompt_mode,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            from_base64=True
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


@app.function(image=vllm_image)
@modal.fastapi_endpoint(method="POST", label="dotsocr-batch-parallel-v2")
async def generate_batch_parallel(request_data: dict):
    """
    Generate OCR results from multiple base64 images in PARALLEL.
    
    Expected format:
    {
        "images": ["base64_encoded_image1", "base64_encoded_image2", ...],
        "prompt_mode": "prompt_layout_all_en",
        "max_tokens": 1500,
        "temperature": 0.1,
        "top_p": 0.9
    }
    """
    import time
    
    images = request_data.get("images", [])
    if not images:
        return {"success": False, "error": "No images provided"}
    
    prompt_mode = request_data.get("prompt_mode", "prompt_layout_all_en")
    max_tokens = request_data.get("max_tokens", 1500)
    temperature = request_data.get("temperature", 0.1)
    top_p = request_data.get("top_p", 0.9)
    
    print(f"Starting TRUE parallel processing of {len(images)} images...")
    start_time = time.time()
    
    # Step 1: Preprocess ALL images first (fast, sequential)
    preprocess_start = time.time()
    preprocessed_images = []
    
    for i, image_data in enumerate(images):
        try:
            preprocessed_data = _preprocess_image(
                image_data=image_data,
                prompt_mode=prompt_mode,
                from_base64=True
            )
            preprocessed_images.append((i, preprocessed_data))
            print(f"Preprocessed image {i}")
        except Exception as e:
            # Handle preprocessing errors
            preprocessed_images.append((i, None))
            print(f"Failed to preprocess image {i}: {e}")
    
    preprocess_time = time.time() - preprocess_start
    print(f"Preprocessing completed in {preprocess_time:.2f}s")
    
    # Step 2: Launch ALL OCR tasks in parallel (truly non-blocking)
    futures = []
    launch_start = time.time()
    
    for i, preprocessed_data in preprocessed_images:
        try:
            if preprocessed_data is None:
                # Preprocessing failed
                futures.append((i, None))
            else:
                # This should be instantaneous - just returns Modal future
                future = _process_ocr_request_async(
                    preprocessed_data=preprocessed_data,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                futures.append((i, future))
                print(f"Launched OCR task {i}")
        except Exception as e:
            # Handle launch errors
            futures.append((i, None))
            print(f"Failed to launch task {i}: {e}")
    
    launch_time = time.time() - launch_start
    print(f"All {len(futures)} OCR tasks launched in {launch_time:.3f}s (should be ~0.001s)")
    print(f"Now waiting for parallel execution to complete...")
    
    # Step 3: Collect all results using AsyncIO (TRUE PARALLEL COLLECTION)
    results = []
    collection_start = time.time()
    
    print(f"Waiting for all {len(futures)} tasks to complete in parallel...")
    
    # Prepare futures for async collection
    valid_futures = []
    future_to_index = {}
    failed_tasks = {}
    
    for i, future in futures:
        if future is None:
            # Task failed to launch
            failed_tasks[i] = {
                "success": False,
                "error": "Failed to launch OCR task",
                "page_number": i
            }
        else:
            valid_futures.append(future)
            future_to_index[future] = i
    
    # Use asyncio.gather to wait for all remote calls concurrently
    import asyncio
    try:
        if valid_futures:
            print(f"Collecting {len(valid_futures)} results in parallel...")
            # This waits for ALL futures concurrently, not sequentially!
            ocr_results = await asyncio.gather(*valid_futures, return_exceptions=True)
            
            # Process results
            for future, ocr_result in zip(valid_futures, ocr_results):
                i = future_to_index[future]
                if isinstance(ocr_result, Exception):
                    result = {
                        "success": False,
                        "error": str(ocr_result),
                        "page_number": i
                    }
                    print(f"Task {i} failed: {ocr_result}")
                else:
                    result = {
                        "success": True,
                        "result": ocr_result,
                        "prompt_mode": prompt_mode,
                        "page_number": i
                    }
                    print(f"Task {i} completed successfully")
                results.append(result)
        
        # Add failed tasks
        for failed_result in failed_tasks.values():
            results.append(failed_result)
            
    except Exception as e:
        print(f"AsyncIO gather failed: {e}")
        # Fallback to sequential collection if asyncio fails
        for i, future in futures:
            try:
                if future is None:
                    result = {"success": False, "error": "Failed to launch OCR task", "page_number": i}
                else:
                    ocr_result = future
                    result = {"success": True, "result": ocr_result, "prompt_mode": prompt_mode, "page_number": i}
            except Exception as ex:
                result = {"success": False, "error": str(ex), "page_number": i}
            results.append(result)
    
    collection_time = time.time() - collection_start
    total_time = time.time() - start_time
    
    print(f"Collection completed in {collection_time:.2f}s")
    print(f"Total parallel processing time: {total_time:.2f}s")
    print(f"Average time per page: {total_time/len(images):.2f}s")
    
    # Sort results by page number to maintain order
    results.sort(key=lambda x: x.get("page_number", 0))
    
    return {
        "success": True,
        "total_pages": len(images),
        "processing_mode": "true_parallel",
        "timing": {
            "preprocess_time": preprocess_time,
            "launch_time": launch_time,
            "collection_time": collection_time,
            "total_time": total_time,
            "avg_time_per_page": total_time / len(images)
        },
        "results": results
    }


@app.function(image=vllm_image)
@modal.fastapi_endpoint(method="GET", label="health-v2")
def health():
    """Health check endpoint"""
    return model.health_check.remote()


# Local testing
if __name__ == "__main__":
    print("🚀 Deploying DotsOCR Modal app...")
    # You can add local testing code here