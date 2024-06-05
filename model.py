# pylint: skip-file
import random
import torch
import transformers

import numpy as np

from instill.helpers.const import DataType, TextGenerationChatInput
from instill.helpers.ray_io import (
    serialize_byte_tensor,
    StandardTaskIO,
)

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    construct_text_generation_chat_metadata_response,
    construct_text_generation_chat_infer_response,
)


@instill_deployment
class Llama3Instruct:
    def __init__(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )

    def ModelMetadata(self, req):
        return construct_text_generation_chat_metadata_response(req)

    async def __call__(self, req):
        task_text_generation_chat_input: TextGenerationChatInput = (
            StandardTaskIO.parse_task_text_generation_chat_input(request=req)
        )
        if task_text_generation_chat_input.temperature <= 0.0:
            task_text_generation_chat_input.temperature = 0.8

        if task_text_generation_chat_input.random_seed > 0:
            random.seed(task_text_generation_chat_input.random_seed)
            np.random.seed(task_text_generation_chat_input.random_seed)

        CHECK_FIRST_ROLE_IS_USER = True
        COMBINED_CONSEQUENCE_PROMPTS = True
        prompt_roles = ["user", "assistant", "system"]

        prompt_conversation = []
        default_system_message = task_text_generation_chat_input.system_message
        if default_system_message is None:
            default_system_message = (
                "You are a helpful, respectful and honest assistant. "
                "Always answer as helpfully as possible, while being safe.  "
                "Your answers should not include any harmful, unethical, racist, "
                "sexist, toxic, dangerous, or illegal content. Please ensure that "
                "your responses are socially unbiased and positive in nature. "
                "If a question does not make any sense, or is not factually coherent, "
                "explain why instead of answering something not correct. If you don't "
                "know the answer to a question, please don't share false information."
            )

        prompt_conversation.append(
            {"role": "system", "content": default_system_message}
        )

        if (
            task_text_generation_chat_input.chat_history is not None
            and len(task_text_generation_chat_input.chat_history) > 0
        ):
            for chat_entity in task_text_generation_chat_input.chat_history:
                chat_message = None
                if len(chat_entity["content"]) > 1:
                    raise ValueError(
                        "Multiple text message detected"
                        " in a single chat history entity"
                    )

                if chat_entity["content"][0]["type"] == "text":
                    if "Content" in chat_entity["content"][0]:
                        chat_message = chat_entity["content"][0]["Content"]["Text"]
                    elif "Text" in chat_entity["content"][0]:
                        chat_message = chat_entity["content"][0]["Text"]
                    elif "text" in chat_entity["content"][0]:
                        chat_message = chat_entity["content"][0]["text"]
                    else:
                        raise ValueError(
                            f"Unknown structure of chat_hisoty: {task_text_generation_chat_input.chat_history}"
                        )
                else:
                    raise ValueError(
                        "Unsupported chat_hisotry message type"
                        ", expected 'text'"
                        f" but get {chat_entity['content'][0]['type']}"
                    )

                if chat_entity["role"] not in prompt_roles:
                    raise ValueError(
                        f"Role `{chat_entity['role']}` is not in supported"
                        f"role list ({','.join(prompt_roles)})"
                    )
                elif (
                    chat_entity["role"] == prompt_roles[-1]
                    and default_system_message is not None
                    and len(default_system_message) > 0
                ):
                    continue
                elif chat_message is None:
                    raise ValueError(
                        f"No message found in chat_history. {chat_message}"
                    )

                if CHECK_FIRST_ROLE_IS_USER:
                    if (
                        len(prompt_conversation) == 1
                        and chat_entity["role"] != prompt_roles[0]
                    ):
                        prompt_conversation.append({"role": "user", "content": " "})
                if COMBINED_CONSEQUENCE_PROMPTS:
                    if (
                        len(prompt_conversation) > 0
                        and prompt_conversation[-1]["role"] == chat_entity["role"]
                    ):
                        last_conversation = prompt_conversation.pop()
                        chat_message = (
                            f"{last_conversation['content']}\n\n{chat_message}"
                        )
                prompt_conversation.append(
                    {"role": chat_entity["role"], "content": chat_message}
                )

        prompt = task_text_generation_chat_input.prompt
        if COMBINED_CONSEQUENCE_PROMPTS:
            if (
                len(prompt_conversation) > 0
                and prompt_conversation[-1]["role"] == prompt_roles[0]
            ):
                last_conversation = prompt_conversation.pop()
                prompt = f"{last_conversation['content']}\n\n{prompt}"
        prompt_conversation.append({"role": "user", "content": prompt})

        conv = self.pipeline.tokenizer.apply_chat_template(
            prompt_conversation, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        sequences = self.pipeline(
            conv,
            do_sample=True,
            top_k=task_text_generation_chat_input.top_k,
            temperature=task_text_generation_chat_input.temperature,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=terminators,
            max_new_tokens=task_text_generation_chat_input.max_new_tokens,
            **task_text_generation_chat_input.extra_params,
        )

        max_output_len = 0

        text_outputs = []
        for seq in sequences:
            generated_text = seq["generated_text"][len(conv) :].strip().encode("utf-8")
            text_outputs.append(generated_text)
            max_output_len = max(max_output_len, len(generated_text))
        text_outputs_len = len(text_outputs)
        task_output = serialize_byte_tensor(np.asarray(text_outputs))

        return construct_text_generation_chat_infer_response(
            req=req,
            shape=[text_outputs_len, max_output_len],
            raw_outputs=[task_output],
        )


entrypoint = InstillDeployable(Llama3Instruct).get_deployment_handle()
