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
from datasets import load_dataset

# =====================
# MODEL CONFIGURATIONS
# =====================

def load_quant_config(quant_dict):
    """Convert JSON quant config to BitsAndBytesConfig"""
    if not quant_dict:
        return None
    return BitsAndBytesConfig(**quant_dict)

with open("model_configs.json") as f:
    MODEL_CONFIGS = json.load(f)

# =====================
# INFERENCE PIPELINE
# =====================

class CodeAssistant:
    def __init__(self, model_name):
        self.config = MODEL_CONFIGS[model_name]
        
        print(f"\n🔧 Loading {model_name} from {self.config.get('local_dir', 'default')}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("local_dir", self.config["model_id"]),
            trust_remote_code=True,
            revision="main",
            local_files_only=True
        )
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.get("local_dir", self.config["model_id"]),
                trust_remote_code=True,
                quantization_config=load_quant_config(self.config.get("quant_config")),
                device_map=self.config.get("device_map", "auto"),
                torch_dtype=torch.float16,
                local_files_only=True,
                max_memory=self.config.get("max_memory"),
                offload_folder=self.config.get("offload_folder"),
                low_cpu_mem_usage=self.config.get("low_cpu_mem_usage", False)
            )
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            raise

# ==================
# EVALUATION SYSTEM
# ==================

def run_benchmark(models_to_test):
    # Load dataset with correct split version
    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.3")
    test_samples = dataset.select(range(10))  # First 10 samples
    results = []
    
    print("\n🚀 Starting benchmark process...")
    
    for model_name in models_to_test:
        print(f"\n🔧 Initializing {model_name}...")
        try:
            assistant = CodeAssistant(model_name)
            print(f"✅ {model_name} loaded | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        except Exception as e:
            print(f"❌ Load failed: {str(e)}")
            continue

        model_results = {
            "model": model_name,
            "dataset_evaluations": [],
            "average_latency": 0,
            "max_vram": torch.cuda.max_memory_allocated() / (1024 ** 3)
        }
        
        total_latency = 0
        evaluator = DeepSeekEvaluator()  # Initialize quality evaluator
        
        # Evaluate on dataset prompts
        for sample in tqdm(test_samples, desc=f"Testing {model_name}"):
            for prompt_type in ["complete_prompt", "instruct_prompt"]:
                try:
                    # Get actual dataset prompt
                    test_prompt = sample[prompt_type]
                    
                    # Generate response
                    start_time = perf_counter()
                    response, latency = assistant.generate(test_prompt)
                    total_latency += latency
                    
                    # Evaluate response quality
                    evaluation = evaluator.evaluate_response(response)
                    
                    model_results["dataset_evaluations"].append({
                        "task_id": sample["task_id"],
                        "prompt_type": prompt_type,
                        "response": response,
                        "evaluation": evaluation,
                        "latency": latency,
                        "tokens_generated": len(assistant.tokenizer.encode(response))
                    })
                    
                except Exception as e:
                    print(f"❌ Error in {prompt_type}: {str(e)}")
                    model_results["dataset_evaluations"].append({
                        "task_id": sample["task_id"],
                        "prompt_type": prompt_type,
                        "error": str(e)
                    })

        model_results["average_latency"] = total_latency / (len(test_samples)*2)  # 2 prompts per sample
        results.append(model_results)
        
        # Cleanup
        del assistant
        torch.cuda.empty_cache()
    
    return results

# ========================
# DEEPSEEK EVALUATOR
# ========================

class DeepSeekEvaluator:
    def __init__(self):
        self.evaluator = pipeline(
            task="text-generation",
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            local_dir= "models/DeepSeek-R1-Qwen-14B",
            device_map="auto",
            torch_dtype=torch.float16,
            model_kwargs={
                "load_in_4bit": True,
                "device_map": "sequential",
                "offload_folder": "offload",
                "max_memory": {"0": "10GiB", "cpu": "30GiB"}
            }
        )
    
    def evaluate_response(self, response):
        evaluation_prompt = f"""Analyze this code submission:
        
        {response}
        
        Provide scores (1-10) for:
        - Correctness
        - Efficiency
        - Readability
        - Security
        - Documentation
        
        Include brief rationale for each score."""
        
        return self.evaluator(evaluation_prompt)[0]["generated_text"]

# =============
# MAIN EXECUTION
# =============

if __name__ == "__main__":
    selected_models = ["DeepSeek-Coder-V2-Lite"]
    print("🔨 Starting benchmark setup for: ", selected_models)
    
    # Select models to benchmark
    print(f"🔍 Selected models: {', '.join(selected_models)}")
    
    # Run benchmark
    print("\n⏱️  Starting benchmark...")
    start_time = perf_counter()
    
    benchmark_results = run_benchmark(selected_models)
    
    total_time = perf_counter() - start_time
    print(f"\n⏳ Total benchmark time: {total_time/60:.2f} minutes")
    
    # Save results
    with open("final_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print("💾 Saved results to final_results.json")

    print("\n✅ Benchmark completed successfully!")