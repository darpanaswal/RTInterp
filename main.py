import torch
import pandas as pd
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from captum.attr import LLMGradientAttribution, LayerIntegratedGradients, TextTokenInput

HF_TOKEN = "hf_bgSZTAFS"
HF_TOKEN += "BqvApw"
HF_TOKEN += "HjMQuTOALqZKRpRBzEUL"
login(token = HF_TOKEN)

df = pd.read_csv("cm_cmp_concatenated.csv")
cm = df['text'][:362]
cmp = df['text'][362:]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto"
)

lig = LayerIntegratedGradients(model, layer=model.model.embed_tokens)
llm_attr = LLMGradientAttribution(lig, tokenizer)

cm_prompt = cm.iloc[0]
cmp_prompt = cmp.iloc[0]
safe_response = "I cannot provide information or guidance on illegal or harmful activities, including medical experiments without consent. Can I help you with something else?"

skip_tokens = [1]
inp = TextTokenInput(
    cm_prompt,
    tokenizer,
    skip_tokens=skip_tokens,
)

attr_res = llm_attr.attribute(inp, target=safe_response, skip_tokens=skip_tokens)

attr_res.plot_seq_attr(show=True)
