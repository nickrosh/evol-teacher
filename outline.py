
prompt_template = """Please increase the difficulty of the given programming test question a bit.

You can increase the difficulty using, but not limited to, the following methods:
{method}

{question}"""


prompt = f"Please increase the difficulty of the given programming test question a bit.\n\nYou can increase the difficulty using, but not limited to, the following methods:\n{chosen_method}\n\n{task['instruction']}"
prompt2 = f"Please increase the difficulty of the given programming test question a bit.\n\nYou can increase the difficulty using, but not limited to, the following methods:\n{chosen_method}\n\n#Given Test#\n{task['instruction']}\n\n#Rewritten Test#\n"


methods = [
    'Add new constraints and requirements to the original problem, adding approximately 10 additional words.',
    'Replace a commonly used requirement in the programming task with a less common and more specific one.',
    'If the original problem can be solved with only a few logical steps, please add more reasoning steps.',
    'Provide a piece of erroneous code as a reference to increase misdirection.',
    'Propose higher time or space complexity requirements, but please refrain from doing so frequently.'
]

fine_tuning_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def load_code_alpaca():
    """get the 20k examples and load them to our dataset. We need to convert
    the alpaca Instruction/Input/Output style to Evol-Instruct style. So we
    just concatenate the Instruction and Input."""


def evolve_instructions(initial_data, methods):
    """for M evolutions, take each instruction and a randomly selected method, 
    and generate a new instruction"""


def generate_responses(new_instructions):
    """generate responses for the generated instructions"""


def eliminate_instructions(instructions):
    """Elminate Evolutions for the following reasons:
    
    1. The evolved instruction does not provide any information gain compared to the
    original one. You use an LLM to decide this.

    2. The Evolved instruction makes it hard for the LLM to generate a response. They found
    that when the response contained 'sorry' and is relatively short (i.e. less than 80 words)
    it often indicates that the LLM struggles to respons to the evolved instruction.
    
    3. The response/instruction only contains punctuations or stop words.

    4. The evolved instruction obviously copies some words from the evolving prompt
    such as 'given prompt', 'rewritten prompt', '#Rewritten prompt#', etc.
    """

def save_dataset(final_dataset):
    """Save the data to a jsonl or something to then be used for fine tuning"""

