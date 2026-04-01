---
base_model: sshleifer/tiny-gpt2
library_name: peft
model_name: lora-ensia-smoke-tiny
tags:
- base_model:adapter:sshleifer/tiny-gpt2
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for lora-ensia-smoke-tiny

This model is a fine-tuned version of [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with SFT.

### Framework versions

- PEFT 0.18.1
- TRL: 1.0.0
- Transformers: 5.4.0
- Pytorch: 2.11.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and GallouÃ©dec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```