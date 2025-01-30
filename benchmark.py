"""
Code Review Assistant Benchmark System
Hardware Target: NVIDIA 4070TI (12GB VRAM) + Ryzen 7 7700X
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from huggingface_hub import snapshot_download
import csv
import json
from time import perf_counter
from tqdm import tqdm
import os
import gc
from datasets import load_dataset

with open("model_configs.json") as f:
    MODEL_CONFIGS = json.load(f)

def generate_responses(models_to_test, output_file="raw_responses.json"):
    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.3")
    test_samples = dataset.select(range(1))
    
    all_responses = []
    
    for model_name in models_to_test:
        print(f"\n🔧 Initializing {model_name}...")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            config = MODEL_CONFIGS[model_name]

            quant_config_dict = config.get("quant_config", {})
            quant_config = BitsAndBytesConfig(**quant_config_dict) if quant_config_dict else None
            
            tokenizer = AutoTokenizer.from_pretrained(
                config["local_dir"],
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load model with safety parameters
            model = AutoModelForCausalLM.from_pretrained(
                config["local_dir"],
                quantization_config=quant_config,
                trust_remote_code=True,
                local_files_only=True,
                device_map="auto",
                offload_folder="offload",
                torch_dtype=torch.float16
            )
            
            print(f"✅ {model_name} loaded successfully")
            
            # Test only 1 sample
            for sample in test_samples:
                print("\n📝 Testing sample:", sample["task_id"])
                
                # Test both prompt types
                for prompt_type in ["complete_prompt", "instruct_prompt"]:
                    try:
                        prompt = sample[prompt_type]
                        print(f"\n⚡ Generating {prompt_type} response...")
                        
                        # Time the generation
                        start_time = perf_counter()
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=config["max_new_tokens"]
                        )
                        latency = perf_counter() - start_time
                        
                        # Decode and print immediately
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        print(f"\n💬 Response generated in {latency:.2f}s:")
                        print("="*50)
                        print(response[:500] + "...")  # Show first 500 chars
                        print("="*50)
                        
                        # Store results
                        all_responses.append({
                            "model": model_name,
                            "task_id": sample["task_id"],
                            "prompt_type": prompt_type,
                            "prompt": prompt,
                            "latency": latency,
                            "response": response
                        })
                        
                    except Exception as e:
                        print(f"❌ Generation failed: {str(e)}")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            continue
    
    # Save results
    with open(output_file, "a") as f:
        json.dump(all_responses, f, indent=2)
    print(f"\n💾 Saved results to {output_file}")

if __name__ == "__main__":
    selected_models = ["DeepSeek-Coder-V2-Lite"]
    generate_responses(selected_models)