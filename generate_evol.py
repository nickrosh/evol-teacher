import os
import json
import glob
import logging
import dataclasses
import random
import time
from typing import Optional, Sequence

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


def convert_alpaca_to_evol(
    file_path: str, 
    lines: bool = False,
    output_file: str = "converted_alpaca.json"
):
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
    with open(output_file, "w") as fp:
        json.dump(result, fp)
    return result


def merge_evolutions(output_dir: str, output_file: str = "merged_datasets.json"):
    """merge all jsons currently in the output_dir folder. This should be the
    evolved datasets and the original dataset. Will deposit the merged dataset 
    in the same folder"""
    merged_json = []
    for json_file in glob.glob(os.path.join(output_dir, "*.json")):
        with open(json_file, "r") as file:
            merged_json.extend(json.load(file))
    with open(os.path.join(output_dir, output_file), "w") as output_file:
              json.dump(merged_json, output_file)


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
        prompt = f"Please increase the difficulty of the given programming test question a bit.\n\nYou can increase the difficulty using, but not limited to, the following methods:\n{chosen_method}\n\n#Given Test#\n{task['instruction']}\n\n#Rewritten Test#\n"
        api.request(data={
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }, metadata={'original_prompt': task['instruction'], 'method': chosen_method})


def generate_responses(instructions, api) -> None:
    for task in instructions:
        api.request(data={
            "messages": [{
                "role": "user",
                "content": task["instruction"]
            }]
        }, metadata={'prompt': task['instruction']})


def check_instruction(instruction) -> bool:
    """Check the generated instruction with several checks. If it returns True,
    then the instruction should be discarded"""
    # I'm certain there is something here that they didn't mention in the paper.
    # I decided not to do the LLM original vs New Equality Check. Might try it later
    # Not happy with the post-processing, the paper throws away 10% of instructions in the
    # first evol, I'm only throwing away 4%, but they look fine on visual inspection
    # I probably also need to make separate functions for instructions and responses
    
    #TODO
    # The paper describes 2 situations as instruction failure:
    # 1. evolved instruction does not provide info gain vs original, check using LLM (not implemented)
    # 2. The instruction obviously copies from the generation prompt, e.g. containing #Rewritten Prompt#
    content = instruction.response["choices"][0]["message"]["content"]
    if not content:
        return True
    if len(content.split()) <= 3:
        return True
    if not content[0].isascii():
        return True
    if instruction.response["usage"]["completion_tokens"] >= 1000:
        return True
    # HTML and other code starts with punctuation
    # if instruction[0] in string.punctuation:
    #     return True
    return False


def check_response(response) -> bool:
    """Check the generated instruction with several checks. If it returns True,
    then the instruction should be discarded""" 
    #TODO
    # The paper describes 2 situations as response failure:
    # 1. Instruction makes it difficult to generate a response. Response contains "sorry" and is short
    # 2. Response only contains punctuation and stop words
    content = response.response["choices"][0]["message"]["content"]
    if not content:
        return True
    if len(content.split()) <= 3:
        return True
    if not content[0].isascii():
        return True
    if response.response["usage"]["total_tokens"] >= 2000:
        return True
    # HTML and other code starts with punctuation
    # if instruction[0] in string.punctuation:
    #     return True
    return False


def generate_evol_instruct_set(
    output_dir="./generation/",
    seed_tasks_path="./generation/converted_alpaca_20k",
    evolutions=3,
    temperature=1,
    max_tokens=2048,
    frequency_penalty=0,
    top_p=0.9,
    model_name="gpt-3.5-turbo",
    api_concurrency=10,
    max_retries=100,
    wait_interval=10,
    retry_multiplier=2,
    retry_max=30
):
    """Take in seed dataset and Evolve Dataset by creating new instructions for each,
    and then generate responses for each new instruction. Repeat the process for multiple
    evolutions, each time evolving the previously evolved set."""
    load_dotenv(override=True)
    logging.basicConfig(filename="app.log", filemode="w", format='%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # If you are using Azure OAI Service, your rate limits will be much higher
    if os.getenv("API_TYPE") == "azure":
        openai.api_type = os.getenv("API_TYPE")
        openai.api_base = os.getenv("AZURE_API_BASE")
        openai.api_version = os.getenv("AZURE_API_VERSION")
        model_name = "gpt-35-turbo" if model_name == "gpt-3.5-turbo" else model_name
    decoding_args = OpenAIDecodingArguments(
        temperature=temperature,
        max_tokens=max_tokens,  # hard-code to maximize the length. the requests will be automatically adjusted
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        stop=["\n20", "20.", "20."],
    )

    prev_tasks = load_instructions(seed_tasks_path)
    start_time = time.time()
    for evolution in range(1, evolutions+1):
        evolution_start_time = time.time()
        print(f'Evolution {evolution}:')

        # 1. Evolving Instructions
        print("Generating New Instructions")
        new_tasks = []
        api = OpenAIMultiClient(
            endpoint="chats",
            concurrency=api_concurrency,
            max_retries=max_retries,
            wait_interval=wait_interval,
            retry_multiplier=retry_multiplier,
            retry_max=retry_max,
            data_template={"model": model_name, **decoding_args.__dict__},
        )
        api.run_request_function(evolve_instructions, prev_tasks, api)
        for _, evolved_response in tqdm(enumerate(api), total=len(prev_tasks)):
            if check_instruction(
                evolved_response
            ):
                continue
            new_tasks.append({"instruction": evolved_response.response["choices"][0]["message"]["content"]})

        # 2. Generating Responses to the New Instructions
        print("Generating New Responses")
        new_dataset = []
        api = OpenAIMultiClient(
            endpoint="chats",
            concurrency=api_concurrency,
            max_retries=max_retries,
            wait_interval=wait_interval,
            retry_multiplier=retry_multiplier,
            retry_max=retry_max,
            data_template={"model": model_name, **decoding_args.__dict__},
        )
        api.run_request_function(generate_responses, new_tasks, api)
        for _, new_response in tqdm(enumerate(api), total=len(new_tasks)):
            if check_response(
                new_response
            ):
                continue
            new_dataset.append({
                "instruction": new_response.metadata["prompt"],
                "output": new_response.response["choices"][0]["message"]["content"]
            })

        # 3. Output Evolution to a JSON file
        output_file = output_dir + "evol-instruct-" + str(evolution) + '.json'
        with open(output_file, "w") as json_file:
            json.dump(new_dataset, json_file)
        prev_tasks = new_dataset
        evolution_time = time.time() - evolution_start_time
        print(f'Evolution {evolution} complete, took {evolution_time:.2f}s')
    final_time = time.time() - start_time
    print(f'All Computation complete, total run took {final_time:.2f}s')

if __name__ == "__main__":
    # convert_alpaca_to_evol(file_path="./data/code_alpaca_20k.json")
    generate_evol_instruct_set()
    merge_evolutions(output_dir="./generation/")