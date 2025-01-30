from datasets import load_dataset
from vllm import LLM, SamplingParams
import gc
import torch


dataset = load_dataset("bigcode/bigcodebench", split="v0.1.3")
prompts  = dataset.select(range(1))

sampling_params = SamplingParams(temperature=0.6)

gc.collect()
torch.cuda.empty_cache()

llm = LLM(
    model="models/DeepSeek-Coder-V2-Lite",
    trust_remote_code=True,
    quantization="awq",  # Use AWQ quantization
    dtype="half",  # Use float16 instead of bfloat16
    max_model_len=8192,  # Reduce context length
    enforce_eager=True  # Disable graph optimization
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")