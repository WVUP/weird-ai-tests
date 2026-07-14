from pathlib import Path

import torch
import torch.nn as nn

from weird_ai.trainer import (
    evaluate_model,
    save_checkpoint,
    load_checkpoint,
)


class DummyLanguageModel(nn.Module):
    """
    Small trainable language model for testing trainer utilities.
    """

    def __init__(self, vocab_size=10, emb_dim=8):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.output = nn.Linear(emb_dim, vocab_size)

    def forward(self, input_batch):
        x = self.embedding(input_batch)
        logits = self.output(x)

        return logits


def make_fake_batches():
    return [
        (
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[2, 3, 4], [5, 6, 7]])
        ),
        (
            torch.tensor([[2, 3, 4], [5, 6, 7]]),
            torch.tensor([[3, 4, 5], [6, 7, 8]])
        ),
    ]


def test_evaluate_model_returns_two_losses():
    model = DummyLanguageModel()
    device = torch.device("cpu")

    train_loader = make_fake_batches()
    val_loader = make_fake_batches()

    train_loss, val_loss = evaluate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        eval_iter=1
    )

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert train_loss > 0
    assert val_loss > 0


def test_save_checkpoint_creates_file(tmp_path):
    model = DummyLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        train_losses=[1.0, 0.9],
        val_losses=[1.1, 1.0],
        track_tokens_seen=[100, 200],
        checkpoint_path=checkpoint_path
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()


def test_load_checkpoint_restores_metadata(tmp_path):
    model = DummyLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=3,
        train_losses=[1.0, 0.9],
        val_losses=[1.1, 1.0],
        track_tokens_seen=[100, 200],
        checkpoint_path=checkpoint_path
    )

    new_model = DummyLanguageModel()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.001)

    metadata = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu")
    )

    assert metadata["epoch"] == 3
    assert metadata["train_losses"] == [1.0, 0.9]
    assert metadata["val_losses"] == [1.1, 1.0]
    assert metadata["track_tokens_seen"] == [100, 200]


def test_load_checkpoint_restores_model_weights(tmp_path):
    model = DummyLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        train_losses=[],
        val_losses=[],
        track_tokens_seen=[],
        checkpoint_path=checkpoint_path
    )

    new_model = DummyLanguageModel()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.001)

    load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu")
    )

    for original_param, loaded_param in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(original_param, loaded_param)