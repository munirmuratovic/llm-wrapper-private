from fastapi import FastAPI

app = FastAPI()

from torch import cuda, bfloat16
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-chat-hf"

device = "cpu"
#f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"


quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

# begin initializing HF items, need auth token for these
hf_auth = "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"
model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_auth)

# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     config=model_config,
#     quantization_config=bnb_config,
#     device_map='auto',
#     use_auth_token=hf_auth
# )

device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}

model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map=device_map,
    quantization_config=quantization_config,
)

model_8bit.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

generate_text = transformers.pipeline(
    model=model_8bit,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task="text-generation",
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
)

res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])


# Define a POST route to handle chat requests
@app.post("/chat")
async def chat(request: str):
    # Process the chat request using the model

    # Return the model's response
    return {"response": res}
