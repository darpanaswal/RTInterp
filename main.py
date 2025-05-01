import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from captum.attr import LLMGradientAttribution, LayerIntegratedGradients, TextTokenInput

HF_TOKEN = "hf_bgSZTAFS"
HF_TOKEN += "BqvApw"
HF_TOKEN += "HjMQuTOALqZKRpRBzEUL"
login(token = HF_TOKEN)

df = pd.read_csv("interpolated_data.csv")
cm = df['cm']
safe_responses = df['cm_response']
cmp = df['cmp']

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
    device_map="auto",  # Let Accelerate decide optimal placement
    offload_folder="offload",  # Folder where offloaded weights will be stored
    offload_state_dict=True  # Ensures offload actually triggers
)
analysis_layer = model.model.layers[15]
layer_name = "layer_15"  # You can make this dynamic if needed from the object

lig = LayerIntegratedGradients(model, layer=analysis_layer)
llm_attr = LLMGradientAttribution(lig, tokenizer)

# Tokens to skip during attribution
skip_tokens = [1]

def process_sequence_attributions(cm_prompts, cmp_prompts, prefix):
    dir_name = f"{prefix}_sequence_attributions"
    os.makedirs(dir_name, exist_ok=True)

    max_len = max(len(cm_prompts), len(cmp_prompts))

    for i in range(max_len - 1, -1, -1):
        for label, prompt_list in [('cm', cm_prompts), ('cmp', cmp_prompts)]:
            if i < len(prompt_list):
                print(f"Processing {label} sequence attribution for prompt {i+1}/{len(prompt_list)}")
                try:
                    inp = TextTokenInput(prompt_list[i], tokenizer, skip_tokens=skip_tokens)
                    attr_res = llm_attr.attribute(inp, target=safe_responses[i], skip_tokens=skip_tokens)

                    fig_seq, _ = attr_res.plot_seq_attr(show=False)
                    seq_filename = f"{dir_name}/{label}_seq_attribution_plot_{i:03d}.png"
                    fig_seq.savefig(seq_filename, dpi=300, bbox_inches='tight')
                    plt.close(fig_seq)
                except Exception as e:
                    print(f"Failed {label} sequence attribution for prompt {i}: {e}")

def process_token_attributions(cm_prompts, cmp_prompts, prefix):
    dir_name = f"{prefix}_token_attributions"
    os.makedirs(dir_name, exist_ok=True)

    max_len = max(len(cm_prompts), len(cmp_prompts))

    for i in range(15 - 1, -1, -1):
        for label, prompt_list in [('cm', cm_prompts), ('cmp', cmp_prompts)]:
            if i < len(prompt_list):
                print(f"Processing {label} token attribution for prompt {i+1}/{len(prompt_list)}")
                try:
                    inp = TextTokenInput(prompt_list[i], tokenizer, skip_tokens=skip_tokens)
                    attr_res = llm_attr.attribute(inp, target=safe_responses[i], skip_tokens=skip_tokens)

                    fig_tok, _ = attr_res.plot_token_attr(show=False)
                    tok_filename = f"{dir_name}/{label}_token_attribution_plot_{i:03d}.png"
                    fig_tok.savefig(tok_filename, dpi=300, bbox_inches='tight')
                    plt.close(fig_tok)
                except Exception as e:
                    print(f"Failed {label} token attribution for prompt {i}: {e}")


# Run both functions
process_sequence_attributions(cm, cmp, prefix=layer_name)
print("All attribution plots saved.")
