import torch

from weird_ai.attention import SimpleSelfAttention, SelfAttention, CausalAttention


def test_simple_self_attention_shape():
    x = torch.rand(4, 3)

    attention = SimpleSelfAttention()
    context_vectors, attention_weights = attention(x)

    assert context_vectors.shape == (4, 3)
    assert attention_weights.shape == (4, 4)


def test_self_attention_shape():
    x = torch.rand(4, 3)

    attention = SelfAttention(
        embedding_dim=3,
        output_dim=2
    )

    context_vectors, attention_weights = attention(x)

    assert context_vectors.shape == (4, 2)
    assert attention_weights.shape == (4, 4)


def test_causal_attention_shape():
    x = torch.rand(2, 4, 3)

    attention = CausalAttention(
        embedding_dim=3,
        output_dim=2,
        context_length=4
    )

    context_vectors = attention(x)

    assert context_vectors.shape == (2, 4, 2)

def test_causal_mask_prevents_future_attention():
    attention = CausalAttention(
        embedding_dim=3,
        output_dim=2,
        context_length=4
    )

    mask = attention.mask.bool()

    assert mask[0, 1]
    assert mask[0, 2]
    assert mask[0, 3]
    assert not mask[1, 0]