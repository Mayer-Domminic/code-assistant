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

# =====================
# MODEL CONFIGURATIONS
# =====================

MODEL_CONFIGS = {
    "DeepSeek-R1-32B": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "quant_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ),
        "batch_size": 1,
        "context_window": 4096,
        "max_new_tokens": 1024,
        "gpu_priority": True,
        "flash_attention": False
    },
    "StarCoder2-15B": {
        "model_id": "bigcode/starcoder2-15b",
        "quant_config": None,  # Use FP16
        "batch_size": 2,
        "context_window": 16384,
        "max_new_tokens": 2048,
        "gpu_priority": True,
        "flash_attention": True
    },
    "Phi-4": {
        "model_id": "microsoft/phi-4",
        "quant_config": None,
        "batch_size": 4,
        "context_window": 2048,
        "max_new_tokens": 1024,
        "gpu_priority": True,
        "flash_attention": False
    },
    "DeepSeek-R1-8B": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "quant_config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        ),
        "batch_size": 4,
        "context_window": 4096,
        "max_new_tokens": 1024,
        "gpu_priority": True,
        "flash_attention": False
    },
    "DeepSeek-Coder-V2-Lite": {
        "model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "quant_config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        ),
        "batch_size": 4,
        "context_window": 16384,
        "max_new_tokens": 2048,
        "gpu_priority": True,
        "flash_attention": False
    },
    "StarCoder2-7B": {
        "model_id": "bigcode/starcoder2-7b",
        "quant_config": None,  # FP16
        "batch_size": 4,
        "context_window": 16384,
        "max_new_tokens": 2048,
        "gpu_priority": True,
        "flash_attention": False
    },
    "Qwen2.5-32B-Instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "quant_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ),
        "batch_size": 1,
        "context_window": 32768,
        "max_new_tokens": 2048,
        "gpu_priority": True,
        "flash_attention": False
    },
    "Qwen2.5-14B-Instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "quant_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ),
        "batch_size": 2,
        "context_window": 32768,
        "max_new_tokens": 2048,
        "gpu_priority": True,
        "flash_attention": False
    }
}

# =================
# FULL PROMPT SET
# =================

PROMPTS = {
    "feature_creation": """[INST]
    As a senior full-stack developer, create a Python implementation for a {feature} with:
    1. Complete type annotations
    2. Detailed docstrings
    3. Comprehensive error handling
    4. Unit test template
    5. Performance optimizations
    
    Requirements: {requirements}
    
    Example input/output: {examples}
    [/INST]""",

    "bug_fix": """[INST]
    Analyze and fix this Python code bug:
    
    {code}
    
    Error message: {error}
    
    Required:
    1. Explain the root cause
    2. Provide fixed code
    3. Suggest prevention strategies
    [/INST]""",

    "feature_plan": """[INST]
    Create technical plan for {feature} including:
    
    1. Architecture diagram (Mermaid syntax)
    2. REST API specification (OpenAPI format)
    3. Database schema (SQL tables with relationships)
    4. Security considerations
    5. Deployment strategy
    
    Technical constraints: {constraints}
    [/INST]"""
}

# ======================
# MODEL DOWNLOADER
# ======================

def download_models():
    """Download all models with smart caching"""
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n🔽 Downloading {model_name}...")
        snapshot_download(
            repo_id=config["model_id"],
            local_dir=f"models/{model_name}",
            max_workers=4,
            ignore_patterns=["*.bin", "*.h5", "*.ot"],
            token=True  # Add your HF token here
        )

# =====================
# INFERENCE PIPELINE
# =====================

class CodeAssistant:
    def __init__(self, model_name):
        self.config = MODEL_CONFIGS[model_name]
        self.device = "cuda" if self.config["gpu_priority"] else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_id"],
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"],
            trust_remote_code=True,
            quantization_config=self.config["quant_config"],
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation=(
                "flash_attention_2" 
                if self.config["flash_attention"] 
                else None
            )
        )
        
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            batch_size=self.config["batch_size"],
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )

    def generate(self, prompt):
        """Run generation with performance monitoring"""
        start_time = perf_counter()
        
        output = self.pipe(
            prompt,
            max_new_tokens=self.config["max_new_tokens"],
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        
        latency = perf_counter() - start_time
        return output[0]["generated_text"], latency

# ==================
# EVALUATION SYSTEM
# ==================

def run_benchmark(models_to_test):
    results = []
    
    print("\n🚀 Starting benchmark process...")
    
    for model_name in models_to_test:
        print(f"\n🔧 Initializing {model_name}...")
        try:
            assistant = CodeAssistant(model_name)
            print(f"✅ {model_name} loaded successfully | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)}")
            continue

        model_results = {
            "model": model_name,
            "evaluations": [],
            "average_latency": 0,
            "total_vram": torch.cuda.max_memory_allocated() / (1024 ** 3)
        }
        
        total_latency = 0
        
        # Test all prompt types with individual tracking
        for prompt_name, prompt_template in PROMPTS.items():
            print(f"\n📝 Generating {prompt_name} prompt for {model_name}...")
            
            try:
                test_prompt = prompt_template.format(
                    feature="user authentication system",
                    requirements="JWT-based, rate limiting, OAuth2 support",
                    examples="Input: valid credentials\nOutput: access token",
                    code="def divide(a, b): return a / b",
                    error="ZeroDivisionError",
                    constraints="Must use PostgreSQL and Redis"
                )
                
                print("⚡ Generating response...")
                response, latency = assistant.generate(test_prompt)
                total_latency += latency
                
                tokens = len(assistant.tokenizer.encode(response))
                print(f"✅ Generated {tokens} tokens in {latency:.2f}s")
                
                model_results["evaluations"].append({
                    "prompt_type": prompt_name,
                    "response": response,
                    "latency": latency,
                    "tokens_generated": tokens
                })
                
            except Exception as e:
                print(f"❌ Error processing {prompt_name}: {str(e)}")
                model_results["evaluations"].append({
                    "prompt_type": prompt_name,
                    "error": str(e)
                })
        
        model_results["average_latency"] = total_latency / len(PROMPTS) if len(PROMPTS) > 0 else 0
        results.append(model_results)
        print(f"\n🏁 Completed {model_name} | Avg latency: {model_results['average_latency']:.2f}s")
        
        # Cleanup memory
        del assistant
        torch.cuda.empty_cache()
        print(f"🧹 Cleaned up {model_name} resources")
    
    return results

# ========================
# DEEPSEEK EVALUATOR
# ========================

class DeepSeekEvaluator:
    def __init__(self):
        self.evaluator = pipeline(
            task="text-generation",
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            device_map="auto",
            torch_dtype=torch.float16,
            model_kwargs={"load_in_4bit": True}
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
    print("🔨 Starting benchmark setup...")
    
    # Select models to benchmark
    selected_models = ["Phi-4", "DeepSeek-Coder-V2-Lite", "StarCoder2-15B"]
    print(f"🔍 Selected models: {', '.join(selected_models)}")
    
    # Run benchmark
    print("\n⏱️ Starting benchmark...")
    start_time = perf_counter()
    
    benchmark_results = run_benchmark(selected_models)
    
    total_time = perf_counter() - start_time
    print(f"\n⏳ Total benchmark time: {total_time/60:.2f} minutes")
    
    # Save results
    with open("final_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print("💾 Saved results to final_results.json")

    print("\n✅ Benchmark completed successfully!")