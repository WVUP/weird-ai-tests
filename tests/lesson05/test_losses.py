import math

import torch
import torch.nn as nn

from weird_ai.losses import (
    calc_loss_batch,
    calc_loss_loader,
    calculate_perplexity,
)


class DummyLanguageModel(nn.Module):
    """
    Small fake language model for testing loss functions.

    It returns logits shaped like a real language model:
    (batch_size, num_tokens, vocab_size)
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_batch):
        batch_size, num_tokens = input_batch.shape

        return torch.randn(
            batch_size,
            num_tokens,
            self.vocab_size,
            requires_grad=True
        )


def test_calc_loss_batch_returns_tensor():
    model = DummyLanguageModel(vocab_size=10)
    device = torch.device("cpu")

    input_batch = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])

    target_batch = torch.tensor([
        [2, 3, 4],
        [5, 6, 7],
    ])

    loss = calc_loss_batch(
        input_batch=input_batch,
        target_batch=target_batch,
        model=model,
        device=device
    )

    assert torch.is_tensor(loss)
    assert loss.dim() == 0


def test_calc_loss_batch_is_positive():
    model = DummyLanguageModel(vocab_size=10)
    device = torch.device("cpu")

    input_batch = torch.tensor([[1, 2, 3]])
    target_batch = torch.tensor([[2, 3, 4]])

    loss = calc_loss_batch(input_batch, target_batch, model, device)

    assert loss.item() > 0


def test_calc_loss_loader_returns_float():
    model = DummyLanguageModel(vocab_size=10)
    device = torch.device("cpu")

    batches = [
        (
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[2, 3, 4]])
        ),
        (
            torch.tensor([[4, 5, 6]]),
            torch.tensor([[5, 6, 7]])
        ),
    ]

    loss = calc_loss_loader(
        data_loader=batches,
        model=model,
        device=device
    )

    assert isinstance(loss, float)
    assert loss > 0


def test_calc_loss_loader_honors_num_batches():
    model = DummyLanguageModel(vocab_size=10)
    device = torch.device("cpu")

    batches = [
        (
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[2, 3, 4]])
        ),
        (
            torch.tensor([[4, 5, 6]]),
            torch.tensor([[5, 6, 7]])
        ),
        (
            torch.tensor([[7, 8, 9]]),
            torch.tensor([[8, 9, 0]])
        ),
    ]

    loss = calc_loss_loader(
        data_loader=batches,
        model=model,
        device=device,
        num_batches=1
    )

    assert isinstance(loss, float)
    assert loss > 0


def test_calc_loss_loader_empty_loader_returns_nan():
    model = DummyLanguageModel(vocab_size=10)
    device = torch.device("cpu")

    loss = calc_loss_loader(
        data_loader=[],
        model=model,
        device=device
    )

    assert math.isnan(loss)


def test_calculate_perplexity_from_tensor():
    loss = torch.tensor(2.0)

    perplexity = calculate_perplexity(loss)

    assert torch.is_tensor(perplexity)
    assert torch.allclose(perplexity, torch.exp(loss))


def test_calculate_perplexity_from_float():
    loss = 2.0

    perplexity = calculate_perplexity(loss)

    assert torch.is_tensor(perplexity)
    assert torch.allclose(perplexity, torch.exp(torch.tensor(loss)))