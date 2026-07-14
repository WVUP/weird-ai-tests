from weird_ai.instruction_dataset import InstructionDataset
from weird_ai.tokenizer import SimpleCharacterTokenizer
from weird_ai.instruction_data import format_full_example


def make_data():
    return [
        {"instruction": "Write a silly parody chorus.", "input": "", "output": "My toaster sings the blues."},
        {"instruction": "Rewrite as a pirate song.", "input": "I remember your smile.", "output": "Arrr, I remember yer grin."},
    ]


def make_tokenizer(data):
    text = "\n".join(format_full_example(entry) for entry in data)
    return SimpleCharacterTokenizer(text)


def test_instruction_dataset_length():
    data = make_data()
    dataset = InstructionDataset(data, make_tokenizer(data))
    assert len(dataset) == 2


def test_instruction_dataset_returns_list_of_token_ids():
    data = make_data()
    dataset = InstructionDataset(data, make_tokenizer(data))
    item = dataset[0]
    assert isinstance(item, list)
    assert len(item) > 0
    assert all(isinstance(token_id, int) for token_id in item)
