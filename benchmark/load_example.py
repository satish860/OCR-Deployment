"""
Load a single example from the OCR benchmark dataset and save it locally.
"""
import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image


def load_single_example(index=0, save_dir="benchmark/data"):
    """
    Load a single example from the OCR benchmark dataset.
    
    Args:
        index (int): Index of the example to load (default: 0 for first example)
        save_dir (str): Directory to save the example files
    """
    print(f"Loading OCR benchmark dataset...")
    
    # Load the dataset with streaming to avoid downloading everything
    try:
        dataset = load_dataset("getomni-ai/ocr-benchmark", split="test", streaming=True)
        print(f"Dataset loaded successfully in streaming mode")
        
        # Convert to list to get the first few examples
        examples = list(dataset.take(index + 1))
        if len(examples) <= index:
            print(f"Not enough examples available. Got {len(examples)}, need {index + 1}")
            return None
        example = examples[index]
        print(f"Successfully loaded example {index}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Example is already loaded from the streaming dataset
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save the image
    image = example["image"]
    image_path = save_path / f"example_{index:03d}.png"
    image.save(image_path)
    print(f"Image saved to: {image_path}")
    
    # Save the ground truth JSON
    json_truth_path = save_path / f"example_{index:03d}_truth.json"
    with open(json_truth_path, "w", encoding="utf-8") as f:
        json.dump(example["true_json_output"], f, indent=2, ensure_ascii=False)
    print(f"Ground truth JSON saved to: {json_truth_path}")
    
    # Save the ground truth markdown
    md_truth_path = save_path / f"example_{index:03d}_truth.md"
    with open(md_truth_path, "w", encoding="utf-8") as f:
        f.write(example["true_markdown_output"])
    print(f"Ground truth Markdown saved to: {md_truth_path}")
    
    # Save the metadata and schema
    metadata_path = save_path / f"example_{index:03d}_metadata.json"
    metadata = {
        "id": example["id"],
        "metadata": example["metadata"],
        "json_schema": example["json_schema"]
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {metadata_path}")
    
    # Print summary
    print("\n=== Example Summary ===")
    print(f"ID: {example['id']}")
    print(f"Image size: {image.size}")
    print(f"JSON schema keys: {list(example['json_schema'].keys()) if example['json_schema'] else 'None'}")
    print(f"Metadata keys: {list(example['metadata'].keys()) if example['metadata'] else 'None'}")
    
    return {
        "image_path": str(image_path),
        "json_truth_path": str(json_truth_path),
        "md_truth_path": str(md_truth_path),
        "metadata_path": str(metadata_path),
        "example_data": example
    }


def main():
    """Main function to load and save a single example."""
    print("OCR Benchmark - Single Example Loader")
    print("=" * 40)
    
    result = load_single_example()
    
    if result:
        print("\nExample loaded successfully!")
        print(f"Files saved in: {Path('benchmark/data').absolute()}")
    else:
        print("Failed to load example.")


if __name__ == "__main__":
    main()