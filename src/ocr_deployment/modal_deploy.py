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
    .run_commands(
        [
            "python3 -m pip install --upgrade pip",
            "python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "python3 -m pip install vllm==0.9.1",
            "python3 -m pip install flash-attn==2.8.0.post2 --no-build-isolation",
            "python3 -m pip install transformers==4.51.3 fastapi[standard] requests Pillow qwen_vl_utils huggingface_hub modelscope PyMuPDF",
        ]
    )
    .add_local_dir("dots.ocr/dots_ocr", remote_path="/workspace/dots_ocr", copy=True)
    .add_local_dir(
        "dots.ocr/weights/DotsOCR", remote_path="/workspace/weights/DotsOCR", copy=True
    )
    .add_local_file(
        "dots.ocr/tools/download_model.py",
        remote_path="/workspace/tools/download_model.py",
        copy=True,
    )
    .run_commands(
        [
            # Download the actual model weights from Hugging Face after local files are copied
            "cd /workspace && python3 tools/download_model.py --type huggingface --name rednote-hilab/dots.ocr",
            "ls -la /workspace/weights/DotsOCR/ | head -20",  # Debug: show first 20 files
        ]
    )
)

# GPU configuration
GPU_CONFIG = "H100"


@app.cls(
    gpu=GPU_CONFIG,
    image=vllm_image,
    volumes={"/cache": vllm_volume},
    timeout=60 * 60,  # 60 minutes - handle concurrent load
    scaledown_window=60 * 30,  # 30 minutes - keep containers warm longer
    min_containers=1,  # Keep at least 1 container always running
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class vLLMModel:
    @modal.enter()
    def load_model(self):
        import time

        start = time.time()

        print("🔧 DEBUG: Starting @modal.enter() method")

        # Set up environment paths
        import sys
        import os

        os.environ["PYTHONPATH"] = (
            "/workspace:/workspace/weights/DotsOCR:" + os.environ.get("PYTHONPATH", "")
        )
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

            ModelRegistry.register_model(
                "DotsOCRForCausalLM", modeling_dots_ocr_vllm.DotsOCRForCausalLM
            )
            print("✅ DotsOCR model registered with vLLM")

        except Exception as e:
            print(f"⚠️ Warning: Could not register DotsOCR model: {e}")
            import traceback

            traceback.print_exc()

        # Initialize vLLM engine
        try:
            print("🔧 DEBUG: About to import vLLM")
            from vllm import LLM

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

        print(f"🏁 Model loaded in {time.time() - start:.1f}s")
        print(
            f"🔧 DEBUG: self.llm is: {type(self.llm) if hasattr(self, 'llm') else 'NOT SET'}"
        )

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
        if hasattr(self, "llm"):
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
                "has_llm_attr": hasattr(self, "llm"),
                "llm_is_none": self.llm is None if hasattr(self, "llm") else "no attr",
                "llm_type": str(type(self.llm))
                if hasattr(self, "llm") and self.llm is not None
                else "N/A",
                "llm_error": getattr(self, "llm_error", "No error stored"),
                "all_attrs": [attr for attr in dir(self) if not attr.startswith("_")],
            }

            if not hasattr(self, "llm") or self.llm is None:
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

            print("🔧 DEBUG: Using direct generate() with PIL image")

            # Generate response
            print("🔧 DEBUG: About to call self.llm.generate")
            outputs = self.llm.generate(
                prompts=[
                    {"prompt": prompt_text, "multi_modal_data": {"image": pil_image}}
                ],
                sampling_params=sampling_params,
                use_tqdm=False,
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
    def generate_batch(
        self,
        images_b64: list,
        prompts: list,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        """Generate OCR results for multiple images in a single vLLM batch call"""
        print(f"DEBUG: Starting batch generate with {len(images_b64)} images")

        if len(images_b64) != len(prompts):
            return f"Error: Number of images ({len(images_b64)}) must match number of prompts ({len(prompts)})"

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        try:
            if not hasattr(self, "llm") or self.llm is None:
                return f"Error: Model not properly initialized"

            # Prepare all prompts and images for batch processing
            batch_prompts = []
            pil_images = []

            for i, (image_b64, prompt_mode) in enumerate(zip(images_b64, prompts)):
                # Convert prompt_mode to actual prompt text using the loaded prompts
                if hasattr(self, "prompts") and prompt_mode in self.prompts:
                    actual_prompt = self.prompts[prompt_mode]
                else:
                    actual_prompt = prompt_mode  # Fallback to using as-is

                # Format the input as DotsOCR expects
                prompt_text = f"<|img|><|imgpad|><|endofimg|>{actual_prompt}"

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

                # Add to batch
                batch_prompts.append(
                    {"prompt": prompt_text, "multi_modal_data": {"image": pil_image}}
                )
                pil_images.append(pil_image)
                print(f"DEBUG: Prepared image {i} for batch processing")

            print(
                f"DEBUG: About to call batch vLLM generate with {len(batch_prompts)} prompts"
            )

            # Single batch call to vLLM - this is where the magic happens!
            outputs = self.llm.generate(
                prompts=batch_prompts, sampling_params=sampling_params, use_tqdm=False
            )

            print(f"DEBUG: Batch generation completed, got {len(outputs)} outputs")

            # Extract results
            results = []
            for i, output in enumerate(outputs):
                if output.outputs and len(output.outputs) > 0:
                    result = output.outputs[0].text
                    results.append(result)
                    print(f"DEBUG: Extracted result {i}, length: {len(result)}")
                else:
                    results.append("")
                    print(f"DEBUG: No output for image {i}")

            return results

        except Exception as e:
            print(f"ERROR: Batch generation failed: {e}")
            import traceback

            traceback.print_exc()
            # Return a list of error messages, one per image
            return [f"Error: {str(e)}"] * len(images_b64)

    @modal.method()
    def health_check(self):
        """Basic health check"""
        try:
            # Simple test to ensure model is loaded
            return {"status": "healthy", "model": MODEL_NAME}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    @modal.fastapi_endpoint(method="POST", label="gpu-direct")
    def gpu_direct_endpoint(self, request_data: dict):
        """Direct HTTP endpoint to GPU container (bypasses FastAPI wrapper)"""
        try:
            # Extract parameters
            image_b64 = request_data.get("image")
            prompt_mode = request_data.get("prompt_mode", "prompt_layout_all_en")
            max_tokens = request_data.get("max_tokens", 1500)
            temperature = request_data.get("temperature", 0.1)
            top_p = request_data.get("top_p", 0.9)
            
            if not image_b64:
                return {"success": False, "error": "No image provided"}
            
            # Convert prompt_mode to actual prompt text
            PROMPT_DICT = {
                "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.""",
                "prompt_layout_only_en": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",
                "prompt_ocr": """Extract the text content from this image.""",
                "prompt_grounding_ocr": """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",
            }
            
            prompt_text = PROMPT_DICT.get(prompt_mode, prompt_mode)
            
            # Call the generate method directly on this instance
            result = self.generate(
                image_b64,
                prompt_text,
                max_tokens,
                temperature,
                top_p
            )
            
            return {
                "success": True,
                "result": result,
                "prompt_mode": prompt_mode,
                "processing_method": "gpu_direct"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Create a persistent model instance
model = vLLMModel()


def _process_ocr_batch(
    images_data,
    prompt_mode="prompt_layout_all_en",
    max_tokens=1500,
    temperature=0.1,
    top_p=0.9,
    from_base64=True,
    bbox=None,
):
    """True batch OCR processing using vLLM's batch capabilities"""
    try:
        import base64
        import io
        from PIL import Image
        import sys

        sys.path.insert(0, "/workspace")
        from dots_ocr.utils import dict_promptmode_to_prompt
        from dots_ocr.utils.image_utils import PILimage_to_base64

        if not images_data or len(images_data) == 0:
            raise ValueError("No images data provided")

        # Validate prompt mode
        valid_prompt_modes = list(dict_promptmode_to_prompt.keys())
        if prompt_mode not in valid_prompt_modes:
            raise ValueError(
                f"Invalid prompt_mode '{prompt_mode}'. Valid options: {valid_prompt_modes}"
            )

        # Validate grounding OCR requirements
        if prompt_mode == "prompt_grounding_ocr" and not bbox:
            raise ValueError(
                "Grounding OCR requires 'bbox' parameter with format [x1, y1, x2, y2]"
            )

        if bbox and prompt_mode != "prompt_grounding_ocr":
            raise ValueError(
                "'bbox' parameter is only valid with 'prompt_grounding_ocr' mode"
            )

        # Get the appropriate prompt
        base_prompt = dict_promptmode_to_prompt[prompt_mode]

        # Handle grounding OCR with bounding box
        if prompt_mode == "prompt_grounding_ocr" and bbox:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("bbox must be a list of 4 numbers [x1, y1, x2, y2]")
            base_prompt += str(bbox)

        # Process all images and prepare for batch call
        images_b64 = []
        prompts = []

        for i, image_data in enumerate(images_data):
            if not image_data:
                raise ValueError(f"No image data provided for image {i}")

            # Process image based on input format
            if from_base64:
                # Decode base64 image
                try:
                    image_bytes = base64.b64decode(image_data)
                    pil_image = Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    raise ValueError(f"Invalid base64 image data for image {i}: {e}")
            else:
                # Direct bytes input
                try:
                    pil_image = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    raise ValueError(f"Invalid image bytes for image {i}: {e}")

            # Convert image to the format expected by the model
            image_b64 = PILimage_to_base64(pil_image)
            images_b64.append(image_b64)
            prompts.append(base_prompt)

        print(f"DEBUG: About to call batch processing with {len(images_b64)} images")

        # Call the new batch processing method - THIS IS THE KEY OPTIMIZATION!
        batch_results = model.generate_batch.remote(
            images_b64=images_b64,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Format results
        results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, str) and result.startswith("Error:"):
                results.append(
                    {
                        "success": False,
                        "error": result,
                        "page_number": i,
                        "prompt_mode": prompt_mode,
                        "bbox": bbox if bbox else None,
                    }
                )
            else:
                results.append(
                    {
                        "success": True,
                        "result": result,
                        "page_number": i,
                        "prompt_mode": prompt_mode,
                        "bbox": bbox if bbox else None,
                    }
                )

        return results

    except Exception as e:
        # Return error for all images
        error_results = []
        num_images = len(images_data) if images_data else 1
        for i in range(num_images):
            error_results.append(
                {
                    "success": False,
                    "error": str(e),
                    "page_number": i,
                    "prompt_mode": prompt_mode,
                    "bbox": bbox if bbox else None,
                }
            )
        return error_results


def _process_ocr_request(
    image_data,
    prompt_mode="prompt_layout_all_en",
    max_tokens=1500,
    temperature=0.1,
    top_p=0.9,
    from_base64=True,
    bbox=None,
):
    """Single image OCR processing (legacy method for compatibility)"""
    # Use batch processing with a single image for consistency
    batch_results = _process_ocr_batch(
        images_data=[image_data],
        prompt_mode=prompt_mode,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        from_base64=from_base64,
        bbox=bbox,
    )

    # Return the single result
    if batch_results and len(batch_results) > 0:
        result = batch_results[0]
        # Remove page_number for single image compatibility
        if "page_number" in result:
            del result["page_number"]
        return result
    else:
        return {"success": False, "error": "No result returned from batch processing"}


@app.function(
    image=modal.Image.debian_slim().pip_install("fastapi[standard]", "Pillow")
)
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

    # Local prompt dictionary to avoid module dependencies
    PROMPT_DICT = {
        "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.""",
        "prompt_layout_only_en": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",
        "prompt_ocr": """Extract the text content from this image.""",
        "prompt_grounding_ocr": """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",
    }
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

        # Convert prompt_mode to actual prompt text
        actual_prompt = PROMPT_DICT.get(prompt_mode, prompt_mode)

        # Handle grounding OCR with bounding box
        if prompt_mode == "prompt_grounding_ocr" and bbox:
            if isinstance(bbox, list) and len(bbox) == 4:
                actual_prompt += str(bbox)

        print(
            f"DEBUG: Processing batch of {len(images)} images with TRUE batch processing"
        )

        # Call GPU container directly for batch processing
        batch_results = model.generate_batch.remote(
            images_b64=images,
            prompts=[actual_prompt] * len(images),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Format results
        results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, str) and result.startswith("Error:"):
                results.append(
                    {
                        "success": False,
                        "error": result,
                        "page_number": i,
                        "prompt_mode": prompt_mode,
                        "bbox": bbox if bbox else None,
                    }
                )
            else:
                results.append(
                    {
                        "success": True,
                        "result": result,
                        "page_number": i,
                        "prompt_mode": prompt_mode,
                        "bbox": bbox if bbox else None,
                    }
                )

        return {
            "success": True,
            "total_pages": len(images),
            "processing_mode": "true_vllm_batch",
            "results": results,
        }

    # Single image processing
    else:
        image = request_data.get("image")
        prompt_mode = request_data.get("prompt_mode", "prompt_layout_all_en")
        max_tokens = request_data.get("max_tokens", 1500)
        temperature = request_data.get("temperature", 0.1)
        top_p = request_data.get("top_p", 0.9)
        bbox = request_data.get("bbox")

        # Convert prompt_mode to actual prompt text
        actual_prompt = PROMPT_DICT.get(prompt_mode, prompt_mode)

        # Handle grounding OCR with bounding box
        if prompt_mode == "prompt_grounding_ocr" and bbox:
            if isinstance(bbox, list) and len(bbox) == 4:
                actual_prompt += str(bbox)

        # Call GPU container directly for single image
        batch_results = model.generate_batch.remote(
            images_b64=[image],
            prompts=[actual_prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Return single result
        if batch_results and len(batch_results) > 0:
            result = batch_results[0]
            if isinstance(result, str) and result.startswith("Error:"):
                return {
                    "success": False,
                    "error": result,
                    "prompt_mode": prompt_mode,
                    "bbox": bbox,
                }
            else:
                return {
                    "success": True,
                    "result": result,
                    "prompt_mode": prompt_mode,
                    "bbox": bbox,
                }
        else:
            return {
                "success": False,
                "error": "No result returned",
                "prompt_mode": prompt_mode,
                "bbox": bbox,
            }


@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint(method="GET", label="health-v2")
def health():
    """Health check endpoint"""
    return model.health_check.remote()


# Local testing
if __name__ == "__main__":
    print("🚀 Deploying DotsOCR Modal app...")
    # You can add local testing code here
