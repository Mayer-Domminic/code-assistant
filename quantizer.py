from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from awq.models import AwqConfig

# Load model and tokenizer
model = AutoAWQForCausalLM.from_pretrained("models/DeepSeek-Coder-V2-Lite")
tokenizer = AutoTokenizer.from_pretrained("models/DeepSeek-Coder-V2-Lite")

# Define quantization config
quant_config = AwqConfig(
    bits=4,
    group_size=128,
    version="exllama"
)

model.model.quant_config = quant_config
model.model.config.use_cache = True

# Quantize with proper parameters
quant_path = "models/DeepSeek-Coder-V2-Lite-awq"
model.quantize(
    tokenizer=tokenizer,
    quant_config=quant_config,
    work_dir=quant_path,
    safetensors=True
)