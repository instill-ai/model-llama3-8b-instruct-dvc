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
                "conversation": "[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},{\"role\": \"user\", \"content\": \"Who won the world series in 2020?\"},{\"role\": \"assistant\", \"content\": \"The Los Angeles Dodgers won the World Series in 2020.\"},{\"role\": \"user\", \"content\": \"Where was it played?\"}]",
                "max_new_tokens": "100",
                "temperature": "0.8",
                "top_k": "20",
                "random_seed": "0",
                "extra_params": "{\"top_p\": 0.8, \"frequency_penalty\": 1.2}"
            }
        }
    ]
}
```
