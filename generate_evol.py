import os
import json
import dataclasses
import random
import string
import time
from typing import Optional, Sequence, Union

import openai
from tqdm import tqdm
from dotenv import load_dotenv
from async_api import OpenAIMultiClient



@dataclasses.dataclass
class OpenAIDecodingArguments:
    """These are the values used in the WizardCoder paper"""
    max_tokens: int = 2048
    temperature: float = 1
    top_p: float = 0.9
    n: int = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False
    stop: Optional[Sequence[str]] = None


def convert_alpaca_to_evol(file_path: str, lines: bool = False):
    """Convert the Instruction/Input/Output format of Alpaca Instruct datasets
    to the Evol-Instruct format of Instruction/Output. Inputs are appended to the
    instructions.
    
    Args:
        file_path: the file path to a single JSON file in alpaca format
        lines: Set to True if the input is a JSONL file, the default is False
        
    Returns: a list of the instruction-output pairs generated from the alpaca set"""
    result = []
    if lines:
        with open(file_path, "r") as json_file:
            loaded_json = [json.loads(line) for line in json_file]
        for record in loaded_json:
            if record["instances"][0]["input"]:
                record["instruction"] += '\n' + record["instances"][0]["input"]
            result.append({
                "instruction": record["instruction"],
                "output": record["instances"][0]["output"]
            })
    else:
        with open(file_path, "r") as json_file:
            loaded_json = json.load(json_file)
        for record in loaded_json:
            if record["input"]:
                record["instruction"] += '\n' + record["input"]
            result.append({
                "instruction": record["instruction"],
                "output": record["output"]
            })
    with open("converted_alpaca.json", "w") as fp:
        json.dump(result, fp)
    return result


def load_instructions(file_path: str):
    """Load in JSON file in Evol Format"""
    with open(file_path, "r") as json_file:
        loaded_json = json.load(json_file)
    return loaded_json


def evolve_instructions(instructions, api) -> None:
    methods = [
    'Add new constraints and requirements to the original problem, adding approximately 10 additional words.',
    'Replace a commonly used requirement in the programming task with a less common and more specific one.',
    'If the original problem can be solved with only a few logical steps, please add more reasoning steps.',
    'Provide a piece of erroneous code as a reference to increase misdirection.',
    'Propose higher time or space complexity requirements, but please refrain from doing so frequently.'
    ]
    for task in instructions:
        chosen_method = random.choice(methods)
        prompt = f"Please increase the difficulty of the given programming test question a bit.\n\nYou can increase the difficulty using, but not limited to, the following methods:\n{chosen_method}\n\n{task['instruction']}"
        while True:
            try:
                api.request(data={
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }]
                }, metadata={'original_prompt': task['instruction'], 'method': chosen_method})
                break
            except openai.error.OpenAIError as e:
                print(e)
                time.sleep(2)


def generate_responses(instructions, api) -> None:
    for task in instructions:
        while True:
            try:
                api.request(data={
                    "messages": [{
                        "role": "user",
                        "content": task["instruction"]
                    }]
                }, metadata={'prompt': task['instruction']})
                break
            except openai.error.OpenAIError as e:
                print(e)
                time.sleep(2)


def check_instruction(instruction: str) -> bool:
    """Check the generated instruction with several checks. If it returns True,
    then the instruction should be discarded"""
    #TODO Checking for the "sorry" case of bad responses
    #TODO Check when it simply copies from the previous instruction
    #TODO Check when the generated text only contains stop words
    if not instruction:
        return True
    if len(instruction.split()) <= 3:
        return True
    if instruction[0] in string.punctuation:
        return True
    if not instruction[0].isascii():
        return True
    return False


def generate_evol_instruct_set(
    output_dir="./generation/",
    seed_tasks_path="./seed_evol.json",
    evolutions=3,
    temperature=1,
    max_tokens=2048,
    frequency_penalty=0,
    top_p=0.9,
    model_name="gpt-3.5-turbo"
):
    load_dotenv(override=True)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    decoding_args = OpenAIDecodingArguments(
        temperature=temperature,
        max_tokens=max_tokens,  # hard-code to maximize the length. the requests will be automatically adjusted
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        stop=["\n20", "20.", "20."],
    )
    api = OpenAIMultiClient(endpoint="chats", data_template={"model": model_name, **decoding_args.__dict__})
    
    prev_tasks = load_instructions(seed_tasks_path)
    for evolution in range(1, evolutions+1):
        start_time = time.time()
        print(f'Evolution {evolution}:')

        # 1. Evolving Instructions
        print("Generating New Instructions")
        new_tasks = []
        api = OpenAIMultiClient(
            endpoint="chats",
            concurrency=50,
            max_retries=10,
            wait_interval=2,
            retry_multiplier=1,
            retry_max=30,
            data_template={"model": "gpt-3.5-turbo", **decoding_args.__dict__},
        )
        api.run_request_function(evolve_instructions, prev_tasks, api)
        for _, evolved_response in tqdm(enumerate(api), total=len(prev_tasks)):
            if check_instruction(
                evolved_response.response["choices"][0]["message"]["content"]
            ):
                continue
            new_tasks.append({"instruction": evolved_response.response["choices"][0]["message"]["content"]})
        # print("Before we close the API")
        # api.close()
        # print("After we close the API")

        # 2. Generating Responses to the New Instructions
        print("Generating New Responses")
        new_dataset = []
        api = OpenAIMultiClient(
            endpoint="chats",
            concurrency=50,
            max_retries=10,
            wait_interval=2,
            retry_multiplier=1,
            retry_max=30,
            data_template={"model": "gpt-3.5-turbo", **decoding_args.__dict__},
        )
        api.run_request_function(generate_responses, new_tasks, api)
        for _, new_response in tqdm(enumerate(api), total=len(new_tasks)):
            if check_instruction(
                new_response.response["choices"][0]["message"]["content"]
            ):
                continue
            new_dataset.append({
                "instruction": new_response.metadata["prompt"],
                "output": new_response.response["choices"][0]["message"]["content"]
            })
        # api.close()

        # 3. Output Evolution to a JSON file
        output_file = output_dir + "evol-instruct-" + str(evolution) + '.json'
        with open(output_file, "w") as json_file:
            json.dump(new_dataset, json_file)
        prev_tasks = new_dataset
        evolution_time = time.time() - start_time
        print(f'Evolution {evolution} complete, took {evolution_time:.2f}s')


if __name__ == "__main__":
    generate_evol_instruct_set()