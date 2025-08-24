import modal
import os
import json
from pathlib import Path
from fastapi import Query

# Create the Modal app
app = modal.App("dotsocr-gpu-v3")

# Model configuration
MODEL_DIR = "/workspace/weights/DotsOCR"
MODEL_NAME = "DotsOCR"

# Create volume for model caching
vllm_volume = modal.Volume.from_name("vllm-vol", create_if_missing=True)

# Create image following Modal's vLLM pattern - PRESERVING ALL OPTIMIZATIONS
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

# GPU configuration - PRESERVING ALL OPTIMIZATIONS
GPU_CONFIG = "H100"

# The GPU endpoint will be instantiated by Modal automatically

# Prompt dictionary
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


@app.cls(
    gpu=GPU_CONFIG,
    image=vllm_image,
    volumes={"/cache": vllm_volume},
    timeout=60 * 60,
    scaledown_window=60 * 30,
    min_containers=1,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=5)  # Increased concurrency for better single image performance
class GPUEndpoint:
    @modal.enter()
    def load_model(self):
        import time
        start = time.time()
        print("DEBUG: Starting modal.enter() method")

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

        print("DEBUG: Environment paths set")
        print(f"DEBUG: MODEL_DIR = {MODEL_DIR}")

        # Register DotsOCR model with vLLM
        try:
            from vllm import ModelRegistry
            print("DEBUG: About to register DotsOCR model")

            # Change working directory to the model directory so relative imports work
            original_cwd = os.getcwd()
            os.chdir("/workspace/weights/DotsOCR")

            # Add the DotsOCR directory to Python path as a package
            if "/workspace/weights" not in sys.path:
                sys.path.insert(0, "/workspace/weights")

            # Import as a proper module
            from DotsOCR import modeling_dots_ocr_vllm
            os.chdir(original_cwd)

            ModelRegistry.register_model(
                "DotsOCRForCausalLM", modeling_dots_ocr_vllm.DotsOCRForCausalLM
            )
            print("SUCCESS: DotsOCR model registered with vLLM")

        except Exception as e:
            print(f"WARNING: Could not register DotsOCR model: {e}")
            import traceback
            traceback.print_exc()

        # Initialize vLLM engine - PRESERVING ALL OPTIMIZATIONS
        try:
            print("DEBUG: About to import vLLM")
            from vllm import LLM

            print("DEBUG: About to initialize LLM")
            self.llm = LLM(
                model=MODEL_DIR,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.95,  # Match original script
                disable_custom_all_reduce=True,
                tokenizer_mode="auto",
            )
            print("DEBUG: LLM initialized successfully")

        except Exception as e:
            print(f"ERROR: Failed to initialize LLM: {e}")
            import traceback
            traceback.print_exc()
            self.llm = None
            self.llm_error = str(e)
            return

        # Load DotsOCR utilities
        try:
            print("DEBUG: Loading DotsOCR utilities")
            from dots_ocr.utils import dict_promptmode_to_prompt
            from dots_ocr.utils.image_utils import PILimage_to_base64

            self.prompts = dict_promptmode_to_prompt
            self.image_to_base64 = PILimage_to_base64
            print("DEBUG: Utilities loaded successfully")

        except Exception as e:
            print(f"ERROR: Failed to load utilities: {e}")
            import traceback
            traceback.print_exc()

        print(f"Model loaded in {time.time() - start:.1f}s")


    def generate_batch(
        self,
        images_b64: list,
        prompts: list,
        max_tokens: int = 1500,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        """Generate OCR results for multiple images in a single vLLM batch call"""
        print(f"DEBUG: Starting batch generate with {len(images_b64)} images")

        if len(images_b64) != len(prompts):
            return [f"Error: Number of images ({len(images_b64)}) must match number of prompts ({len(prompts)})"] * len(images_b64)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        try:
            if not hasattr(self, "llm") or self.llm is None:
                return ["Error: Model not properly initialized"] * len(images_b64)

            # Prepare all prompts and images for batch processing
            import base64
            import io
            from PIL import Image
            
            batch_prompts = []

            for i, (image_b64, prompt_text) in enumerate(zip(images_b64, prompts)):
                # Format the input as DotsOCR expects
                formatted_prompt = f"<|img|><|imgpad|><|endofimg|>{prompt_text}"

                # Optimized base64 decode - extract data once
                if image_b64.startswith("data:image/"):
                    base64_data = image_b64.split(",", 1)[1]  # More efficient split
                else:
                    base64_data = image_b64

                # Fast base64 decode and PIL creation
                image_bytes = base64.b64decode(base64_data)
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Add to batch
                batch_prompts.append(
                    {"prompt": formatted_prompt, "multi_modal_data": {"image": pil_image}}
                )
                print(f"DEBUG: Prepared image {i} ({pil_image.size}) for batch processing")

            print(f"DEBUG: About to call batch vLLM generate with {len(batch_prompts)} prompts")

            # Single batch call to vLLM - PRESERVING OPTIMIZATION
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

    @modal.fastapi_endpoint(method="POST", label="process")
    def process_ocr(self, request_data: dict):
        """
        Direct GPU processing endpoint - no hopping, all optimizations preserved
        
        Single image format:
        {
            "image": "base64_encoded_image",
            "prompt_mode": "prompt_layout_all_en",
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
        """
        
        try:
            # Check if this is a batch request
            if "images" in request_data:
                # Batch processing
                images = request_data.get("images", [])
                if not images:
                    return {"success": False, "error": "No images provided"}

                prompt_mode = request_data.get("prompt_mode", "prompt_layout_all_en")
                max_tokens = request_data.get("max_tokens", 1500)
                temperature = request_data.get("temperature", 0.1)
                top_p = request_data.get("top_p", 0.9)

                # Convert prompt_mode to actual prompt text
                actual_prompt = PROMPT_DICT.get(prompt_mode, prompt_mode)

                print(f"DEBUG: Processing batch of {len(images)} images")

                # Call batch processing directly on GPU (no hopping)
                batch_results = self.generate_batch(
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
                        results.append({
                            "success": False,
                            "error": result,
                            "page_number": i,
                            "prompt_mode": prompt_mode,
                        })
                    else:
                        results.append({
                            "success": True,
                            "result": result,
                            "page_number": i,
                            "prompt_mode": prompt_mode,
                        })

                return {
                    "success": True,
                    "total_pages": len(images),
                    "processing_mode": "gpu_direct_batch",
                    "results": results,
                }

            else:
                # Single image processing
                image = request_data.get("image")
                if not image:
                    return {"success": False, "error": "No image provided"}
                    
                prompt_mode = request_data.get("prompt_mode", "prompt_layout_all_en")
                max_tokens = request_data.get("max_tokens", 1500)
                temperature = request_data.get("temperature", 0.1)
                top_p = request_data.get("top_p", 0.9)

                # Convert prompt_mode to actual prompt text
                actual_prompt = PROMPT_DICT.get(prompt_mode, prompt_mode)

                print(f"DEBUG: Processing single image")

                # Call batch processing directly on GPU for single image (no hopping)
                batch_results = self.generate_batch(
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
                            "processing_method": "gpu_direct"
                        }
                    else:
                        return {
                            "success": True,
                            "result": result,
                            "prompt_mode": prompt_mode,
                            "processing_method": "gpu_direct"
                        }
                else:
                    return {
                        "success": False,
                        "error": "No result returned",
                        "prompt_mode": prompt_mode,
                    }

        except Exception as e:
            return {"success": False, "error": f"Processing failed: {str(e)}"}



    @modal.fastapi_endpoint(method="GET", label="health") 
    def health(self):
        """Lightweight health check endpoint on GPU container"""
        try:
            return {
                "status": "healthy",
                "model": MODEL_NAME,
                "gpu_config": GPU_CONFIG,
                "deployment": "gpu_direct_v3",
                "model_loaded": hasattr(self, "llm") and self.llm is not None
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Local testing
if __name__ == "__main__":
    print("Deploying DotsOCR GPU-direct Modal app...")