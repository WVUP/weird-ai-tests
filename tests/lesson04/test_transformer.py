import torch

from weird_ai.layer_norm import LayerNorm
from weird_ai.feed_forward import GELU, FeedForward
from weird_ai.transformer import TransformerBlock


def test_layer_norm_output_shape():
    x = torch.rand(2, 3, 768)

    layer_norm = LayerNorm(emb_dim=768)

    output = layer_norm(x)

    assert output.shape == x.shape


def test_layer_norm_mean_is_close_to_zero():
    x = torch.rand(2, 3, 768)

    layer_norm = LayerNorm(emb_dim=768)

    output = layer_norm(x)

    mean = output.mean(dim=-1)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


def test_layer_norm_variance_is_close_to_one():
    x = torch.rand(2, 3, 768)

    layer_norm = LayerNorm(emb_dim=768)

    output = layer_norm(x)

    variance = output.var(dim=-1, unbiased=False)

    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-4)


def test_gelu_output_shape():
    x = torch.rand(2, 3, 768)

    gelu = GELU()

    output = gelu(x)

    assert output.shape == x.shape


def test_feed_forward_output_shape():
    x = torch.rand(2, 3, 768)

    feed_forward = FeedForward(emb_dim=768)

    output = feed_forward(x)

    assert output.shape == x.shape


def test_feed_forward_expands_and_contracts():
    feed_forward = FeedForward(emb_dim=768)

    first_linear = feed_forward.layers[0]
    second_linear = feed_forward.layers[2]

    assert first_linear.in_features == 768
    assert first_linear.out_features == 3072

    assert second_linear.in_features == 3072
    assert second_linear.out_features == 768


def test_transformer_block_output_shape():
    x = torch.rand(2, 8, 768)

    block = TransformerBlock(
        emb_dim=768,
        context_length=8,
        num_heads=12,
        dropout=0.1,
        qkv_bias=False
    )

    output = block(x)

    assert output.shape == x.shape


def test_transformer_block_preserves_batch_and_token_dimensions():
    x = torch.rand(4, 16, 768)

    block = TransformerBlock(
        emb_dim=768,
        context_length=16,
        num_heads=12,
        dropout=0.1,
        qkv_bias=False
    )

    output = block(x)

    assert output.shape[0] == 4
    assert output.shape[1] == 16
    assert output.shape[2] == 768