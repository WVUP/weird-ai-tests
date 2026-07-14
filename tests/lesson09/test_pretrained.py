import torch

from weird_ai import pretrained
from weird_ai.pretrained import (
    decode_generated_tokens,
    generate_text,
    generate_with_reasoning,
    get_device,
    load_model,
    load_tokenizer,
    tokenize_prompt,
)


class FakeTokenizer:
    def __init__(self):
        self.eos_token = "<eos>"
        self.eos_token_id = 99
        self.pad_token = None
        self.last_prompt = None

    def __call__(self, prompt, return_tensors=None):
        self.last_prompt = prompt
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "decoded generated text"


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model_name):
        tokenizer = FakeTokenizer()
        tokenizer.loaded_model_name = model_name
        return tokenizer


class FakeModel:
    def __init__(self):
        self.loaded_model_name = None
        self.device = None
        self.eval_was_called = False
        self.generate_kwargs = None

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        self.eval_was_called = True
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4, 5]])


class FakeAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(model_name):
        model = FakeModel()
        model.loaded_model_name = model_name
        return model


def test_get_device_uses_preferred_device():
    device = get_device("cpu")

    assert device.type == "cpu"


def test_load_tokenizer_uses_auto_tokenizer_and_sets_pad_token(monkeypatch):
    monkeypatch.setattr(pretrained, "AutoTokenizer", FakeAutoTokenizer)

    tokenizer = load_tokenizer("fake-model")

    assert tokenizer.loaded_model_name == "fake-model"
    assert tokenizer.pad_token == tokenizer.eos_token


def test_load_model_uses_auto_model_moves_to_device_and_sets_eval(monkeypatch):
    monkeypatch.setattr(pretrained, "AutoModelForCausalLM", FakeAutoModelForCausalLM)

    model = load_model("fake-model", device="cpu")

    assert model.loaded_model_name == "fake-model"
    assert model.device.type == "cpu"
    assert model.eval_was_called is True


def test_tokenize_prompt_returns_tensors_on_selected_device():
    tokenizer = FakeTokenizer()

    inputs = tokenize_prompt(tokenizer, "Write a parody about Python.", device="cpu")

    assert tokenizer.last_prompt == "Write a parody about Python."
    assert inputs["input_ids"].device.type == "cpu"
    assert inputs["attention_mask"].device.type == "cpu"


def test_decode_generated_tokens_decodes_first_row():
    tokenizer = FakeTokenizer()
    token_ids = torch.tensor([[1, 2, 3, 4]])

    text = decode_generated_tokens(tokenizer, token_ids)

    assert text == "decoded generated text"


def test_generate_text_calls_model_generate_with_sampling_options():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    text = generate_text(
        "Write a parody about debugging.",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=12,
        temperature=0.7,
        top_k=25,
        do_sample=True,
        device="cpu",
    )

    assert text == "decoded generated text"
    assert model.generate_kwargs["max_new_tokens"] == 12
    assert model.generate_kwargs["temperature"] == 0.7
    assert model.generate_kwargs["top_k"] == 25
    assert model.generate_kwargs["do_sample"] is True
    assert model.generate_kwargs["pad_token_id"] == tokenizer.eos_token_id


def test_generate_with_reasoning_wraps_prompt_before_generating(monkeypatch):
    captured = {}

    def fake_generate_text(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "reasoning output"

    monkeypatch.setattr(pretrained, "generate_text", fake_generate_text)

    result = generate_with_reasoning(
        "Explain why attention helps lyric generation.",
        model="fake-model-object",
        tokenizer="fake-tokenizer-object",
        max_new_tokens=40,
        temperature=0.5,
        top_k=10,
        device="cpu",
    )

    assert result == "reasoning output"
    assert "Explain why attention helps lyric generation." in captured["prompt"]
    assert "Think step-by-step" in captured["prompt"]
    assert captured["kwargs"]["model"] == "fake-model-object"
    assert captured["kwargs"]["tokenizer"] == "fake-tokenizer-object"
    assert captured["kwargs"]["max_new_tokens"] == 40
