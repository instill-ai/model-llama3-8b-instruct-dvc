---
Task: TextGenerationChat
Tags:
  - TextGenerationChat
  - Llama-3-8B-Instruct
---

# Model-Llama-3-8B-Instruct-dvc

🔥🔥🔥 Deploy [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model and on [Instill-Core](https://github.com/instill-ai/instill-core).

This repository contains the Llama-3-8B Instruct Model, managed using [DVC](https://dvc.org/).

```
{
    "task_inputs": [
        {
            "text_generation_chat": {
                "prompt": "Who are you?",
                "system_message": "You are a lovely cat, named Penguin.",
                "max_new_tokens": 512,
                "top_k": 10,
                "temperature": 0.7
            }
        }
    ]
}
```
