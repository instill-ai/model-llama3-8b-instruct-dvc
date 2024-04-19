# pylint: skip-file
import os

TORCH_GPU_DEVICE_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{TORCH_GPU_DEVICE_ID}"

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
    construct_infer_response,
    construct_metadata_response,
    Metadata,
)


@instill_deployment
class Llama3Instruct:
    def __init__(self, model_path: str):
        self.application_name = "_".join(model_path.split("/")[3:5])
        self.deployement_name = model_path.split("/")[4]
        print(f"application_name: {self.application_name}")
        print(f"deployement_name: {self.deployement_name}")
        print(f"torch version: {torch.__version__}")

        print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count() : {torch.cuda.device_count()}")
        print(f"torch.cuda.current_device() : {torch.cuda.current_device()}")
        print(f"torch.cuda.device(0) : {torch.cuda.device(0)}")
        print(f"torch.cuda.get_device_name(0) : {torch.cuda.get_device_name(0)}")

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )

    def ModelMetadata(self, req):
        resp = construct_metadata_response(
            req=req,
            inputs=[
                Metadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="prompt_images",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="chat_history",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="system_message",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="max_new_tokens",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="temperature",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                Metadata(
                    name="top_k",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="random_seed",
                    datatype=str(DataType.TYPE_UINT64.name),
                    shape=[1],
                ),
                Metadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                Metadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1, -1],
                ),
            ],
        )
        return resp

    # async def ModelInfer(self, request: ModelInferRequest) -> ModelInferResponse:
    async def __call__(self, req):
        task_text_generation_chat_input: TextGenerationChatInput = (
            StandardTaskIO.parse_task_text_generation_chat_input(request=req)
        )
        print("----------------________")
        print(task_text_generation_chat_input)
        print("----------------________")

        print("print(task_text_generation_chat.prompt")
        print(task_text_generation_chat_input.prompt)
        print("-------\n")

        print("print(task_text_generation_chat.prompt_images")
        print(task_text_generation_chat_input.prompt_images)
        print("-------\n")

        print("print(task_text_generation_chat.chat_history")
        print(task_text_generation_chat_input.chat_history)
        print("-------\n")

        print("print(task_text_generation_chat.system_message")
        print(task_text_generation_chat_input.system_message)
        if len(task_text_generation_chat_input.system_message) is not None:
            if len(task_text_generation_chat_input.system_message) == 0:
                task_text_generation_chat_input.system_message = None
        print("-------\n")

        print("print(task_text_generation_chat.max_new_tokens")
        print(task_text_generation_chat_input.max_new_tokens)
        print("-------\n")

        print("print(task_text_generation_chat.temperature")
        print(task_text_generation_chat_input.temperature)
        print("-------\n")

        print("print(task_text_generation_chat.top_k")
        print(task_text_generation_chat_input.top_k)
        print("-------\n")

        print("print(task_text_generation_chat.random_seed")
        print(task_text_generation_chat_input.random_seed)
        print("-------\n")

        print("print(task_text_generation_chat.stop_words")
        print(task_text_generation_chat_input.stop_words)
        print("-------\n")

        print("print(task_text_generation_chat.extra_params")
        print(task_text_generation_chat_input.extra_params)
        print("-------\n")

        if task_text_generation_chat_input.temperature <= 0.0:
            task_text_generation_chat_input.temperature = 0.8

        if task_text_generation_chat_input.random_seed > 0:
            random.seed(task_text_generation_chat_input.random_seed)
            np.random.seed(task_text_generation_chat_input.random_seed)

        # Process chat_history
        # Preprocessing

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
                    # This structure comes from google protobuf `One of` Syntax, where an additional layer in Content
                    # [{'role': 'system', 'content': [{'type': 'text', 'Content': {'Text': "What's in this image?"}}]}]
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

                # TODO: support image message in chat history
                # self.messages.append([role, message])
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
                    raise ValueError(
                        "it's ambiguious to set `system_message` and "
                        f"using role `{prompt_roles[-1]}` simultaneously"
                    )
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
            print("Output No Clean ----")
            print(seq["generated_text"])
            print("Output Clean ----")
            print(seq["generated_text"][len(conv) :])
            print("---")
            generated_text = seq["generated_text"][len(conv) :].strip().encode("utf-8")
            text_outputs.append(generated_text)
            max_output_len = max(max_output_len, len(generated_text))
        text_outputs_len = len(text_outputs)
        task_output = serialize_byte_tensor(np.asarray(text_outputs))
        # task_output = StandardTaskIO.parse_task_text_generation_output(sequences)

        print("Output:")
        print(task_output)
        print(type(task_output))

        return construct_infer_response(
            req=req,
            outputs=[
                Metadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[text_outputs_len, max_output_len],
                )
            ],
            raw_outputs=[task_output],
        )


deployable = InstillDeployable(
    Llama3Instruct,
    model_weight_or_folder_name="Meta-Llama-3-8B-Instruct/",
    use_gpu=True,
)

deployable.update_min_replicas(1)
deployable.update_max_replicas(1)
