import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from captum.attr import LLMGradientAttribution, LayerIntegratedGradients, TextTokenInput

HF_TOKEN = ""
login(token = HF_TOKEN)

df = pd.read_csv("interpretability_data.csv")
cm = df['cm']
safe_responses = df['cm_response']
cmp = df['cmp']
eng = df['eng']

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
analysis_layer = model.model.layers[0]
layer_name = "layer1"  # You can make this dynamic if needed from the object

lig = LayerIntegratedGradients(model, layer=analysis_layer)
llm_attr = LLMGradientAttribution(lig, tokenizer)

# Tokens to skip during attribution
skip_tokens = ['<|begin_of_text|>']

def process_sequence_attributions(cm_prompts, cmp_prompts, eng_prompts, prefix):
    base_dir = f"{prefix}_sequence_attributions"
    os.makedirs(base_dir, exist_ok=True)

    # Create subdirectories for cm and cmp
    cm_dir = os.path.join(base_dir, "cm")
    cmp_dir = os.path.join(base_dir, "cmp")
    eng_dir = os.path.join(base_dir, "eng")
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(cmp_dir, exist_ok=True)
    os.makedirs(eng_dir, exist_ok=True)

    max_len = max(len(cm_prompts), len(cmp_prompts))

    for i in range(max_len):
        for label, prompt_list, sub_dir in [('cm', cm_prompts, cm_dir), ('cmp', cmp_prompts, cmp_dir), ('eng', eng_prompts, eng_dir)]:
            if i < len(prompt_list):
                print(f"Processing {label} sequence attribution for prompt {i+1}/{len(prompt_list)}")
                try:
                    inp = TextTokenInput(prompt_list[i], tokenizer, skip_tokens=skip_tokens)
                    attr_res = llm_attr.attribute(inp, target=safe_responses[i], skip_tokens=skip_tokens)

                    fig_seq, _ = attr_res.plot_seq_attr(show=False)
                    seq_filename = os.path.join(sub_dir, f"{label}_seq_attribution_plot_{i:03d}.png")
                    fig_seq.savefig(seq_filename, dpi=300, bbox_inches='tight')
                    plt.close(fig_seq)
                except Exception as e:
                    print(f"Failed {label} sequence attribution for prompt {i}: {e}")

def save_combined_sequence_attributions(cm_prompts, cmp_prompts, eng_prompts, safe_responses, prefix):
    records = []

    num_samples = min(len(cm_prompts), len(cmp_prompts), len(eng_prompts), len(safe_responses))

    for i in range(num_samples):
        print(f"Processing sequence attribution for sample {i+1}/{num_samples}")
        try:
            # English attribution
            inp_eng = TextTokenInput(eng_prompts[i], tokenizer, skip_tokens=skip_tokens)
            attr_eng = llm_attr.attribute(inp_eng, target=safe_responses[i], skip_tokens=skip_tokens)
            tokens_eng = attr_eng.input_tokens
            scores_eng = attr_eng.seq_attr.detach().cpu().tolist()
            # # CM attribution
            # inp_cm = TextTokenInput(cm_prompts[i], tokenizer, skip_tokens=skip_tokens)
            # attr_cm = llm_attr.attribute(inp_cm, target=safe_responses[i], skip_tokens=skip_tokens)
            # tokens_cm = attr_cm.input_tokens
            # scores_cm = attr_cm.seq_attr.detach().cpu().tolist()

            # # CMP attribution
            # inp_cmp = TextTokenInput(cmp_prompts[i], tokenizer, skip_tokens=skip_tokens)
            # attr_cmp = llm_attr.attribute(inp_cmp, target=safe_responses[i], skip_tokens=skip_tokens)
            # tokens_cmp = attr_cmp.input_tokens
            # scores_cmp = attr_cmp.seq_attr.detach().cpu().tolist()

            # One row for this prompt pair
            records.append({
                'tokens_eng': tokens_eng,
                'eng_attribution_score': scores_eng,
            })

        except Exception as e:
            print(f"Failed attribution at index {i}: {e}")

    df = pd.DataFrame(records)
    output_path = f"{prefix}_combined_sequence_attributions.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved combined attribution DataFrame to: {output_path}")

start, end = 10, 15

# Slice the relevant prompt and response lists
cm_subset = cm[start:end].tolist()
cmp_subset = cmp[start:end].tolist()
eng_subset = eng[start:end].tolist()
safe_responses_subset = safe_responses[start:end].tolist()

# Call the function with the subset
save_combined_sequence_attributions(cm_subset, cmp_subset, eng_subset, safe_responses_subset, prefix=f"{layer_name}")
