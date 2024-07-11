# MalayaLLM: Gemma [മലയാളം/Malayalam]

<img src="https://cdn-uploads.huggingface.co/production/uploads/64e65800e44b2668a56f9731/bipVMulaNJ9um46ecYpR4.png" alt="Baby MalayaLLM" width="300" height="auto">

# Introducing the Developer:
Discover the mind behind this model and stay updated on their contributions to the field
https://www.linkedin.com/in/vishnu-prasad-j/

# Model description
The MalayaLLM models have been improved and customized expanding upon the groundwork laid by the original Gemma model.

- **Model type:** A 7B Gemma finetuned model on Malayalam tokens.
- **Language(s):** Malayalam and English
- **Datasets:**
  * [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset)
  * [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
- **Source Model:** [MalayaLLM_Gemma_7B_Base_V1](https://huggingface.co/VishnuPJ/MalayaLLM_Gemma_7B_Base_V1)
- **Instruct Model:** [MalayaLLM_Gemma_7B_Instruct_V1](https://huggingface.co/VishnuPJ/MalayaLLM_Gemma_7B_Instruct_V1)
- **GGUF Model:** [MalayaLLM_Gemma_7B_Instruct_V1_GGUF](https://huggingface.co/VishnuPJ/MalayaLLM_Gemma_7B_Instruct_V1_GGUF)
- **Training Precision:** `float16`

# Model Update
Latest Gemma2-9B trained model is here :[MalayaLLM:Gemma-2-9B](https://huggingface.co/collections/VishnuPJ/malayallm-malayalam-gemma-2-9b-6689843413da7de7c57b5b8c)

## A simple example code

```python
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
#!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

import sentencepiece as spm
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="VishnuPJ/MalayaLLM_Gemma_7B_Instruct_V1",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
#### Giving Instruction with Input
'''
alpaca_prompt_1 = """ഒരു  ചുമതല  വിവരിക്കുന്ന  ഒരു  നിർദ്ദേശം  ചുവടെയുണ്ട്.
 അഭ്യർത്ഥന  ശരിയായി  പൂർത്തിയാക്കുന്ന  ഒരു  പ്രതികരണം  എഴുതുക.".
### നിർദ്ദേശം:
{}
### ഇൻപുട്ട്:
{}
### പ്രതികരണം:
{}"""
inputs = tokenizer([
    alpaca_prompt_1.format(
        # "Continue the fibonnaci sequence.", # instruction
        """താഴെ ഉള്ള വാക്യത്തിൽ "അത്" എന്ന് പറയുന്നത് എന്തിനെ ആണ് ?""", # instruction
""" ഒരു വാഹനം കയറ്റം കയറുക ആയിരുന്നു .അതിൽ 4 ആൾക്കാർ ഉണ്ടായിരുന്നു. """, # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
# Printing the result
print(tokenizer.batch_decode(outputs)[0].split("പ്രതികരണം:\n")[-1])
'''
## Giving Instruction only.
alpaca_prompt_2 = """ഒരു  ചുമതല  വിവരിക്കുന്ന  ഒരു  നിർദ്ദേശം  ചുവടെയുണ്ട്.
 അഭ്യർത്ഥന  ശരിയായി  പൂർത്തിയാക്കുന്ന  ഒരു  പ്രതികരണം  എഴുതുക.".
### നിർദ്ദേശം:
{}
### പ്രതികരണം:
{}"""
while True:
    # Taking user input for the instruction
    instruction = input("Enter the instruction (or type 'exit' to quit): ")
    if instruction.lower() == 'exit':
        break
    # Preparing the input for the model
    inputs = tokenizer([
        alpaca_prompt_2.format(
            instruction,
            "",  # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")
    # Generating the output
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    # Printing the result
    print(tokenizer.batch_decode(outputs)[0].split("പ്രതികരണം:\n")[-1])
print("Program terminated.")
''''
```
## Example Output
```
Enter instruction (or 'exit' to end): ഒരു സമചതുരത്തിന്റെ ഒരു വശം 4 cm ആണെങ്കിൽ , അതിന്റെ area കണ്ടുപിടിക്കുക..
സമചതുരത്തിന്റെ area 16 cm2 ആണ്.<eos>.
Enter instruction (or 'exit' to end): ഇന്ത്യയുടെ അടുത്ത് സ്ഥിതി ചെയുന്ന നാല് രാജ്യങ്ങളുടെ പേര് പറയുക.
"ഇന്ത്യയ്ക്ക് സമീപമുള്ള നാല് രാജ്യങ്ങൾ ഇവയാണ്:
- നേപ്പാൾ
- ഭൂട്ടാൻ
- ടിബറ്റ് (ചൈന)
- പാകിസ്ഥാൻ"<eos>
Enter instruction (or 'exit' to end):exit
```
## Made Using UNSLOTH

Thanks to [Unsloth](https://github.com/unslothai/unsloth), the process of fine-tuning large language models (LLMs) has become much easier and more efficient.
<img src="https://cdn-uploads.huggingface.co/production/uploads/64e65800e44b2668a56f9731/WPt_FKUWDdc6--l_Qnb-G.png" alt="Unsloth" width="300" height="auto">

# 🌟Happy coding💻🌟
