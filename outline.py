
prompt_template = """Please increase the difficulty of the given programming test question a bit.

You can increase the difficulty using, but not limited to, the following methods:
{method}

{question}"""

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
    """get the 20k examples and load them to our dataset"""


def evolve_instructions(initial_data, methods):
    """for M evolutions, take each instruction and a randomly selected method, and generate a new instruction using the API"""


def prune_instructions(instructions):
    """remove bad results like if the LLM refused the answer the question"""


def generate_responses(instructions):
    """Take the instructions we made and generate responses to them using the same API"""


def save_dataset(final_dataset):
    """Save the data to a jsonl or something to then be used for fine tuning"""

