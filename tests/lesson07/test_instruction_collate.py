import torch

from weird_ai.instruction_collate import custom_collate_fn


def test_custom_collate_returns_two_tensors():
    inputs, targets = custom_collate_fn([[1, 2, 3], [4, 5]], pad_token_id=0)
    assert torch.is_tensor(inputs)
    assert torch.is_tensor(targets)


def test_custom_collate_shapes_match():
    inputs, targets = custom_collate_fn([[1, 2, 3], [4, 5]], pad_token_id=0)
    assert inputs.shape == targets.shape
    assert inputs.shape[0] == 2


def test_custom_collate_pads_to_longest_sequence():
    inputs, targets = custom_collate_fn([[1, 2, 3], [4, 5]], pad_token_id=0)
    assert inputs.shape == (2, 3)
    assert targets.shape == (2, 3)


def test_custom_collate_creates_shifted_targets():
    inputs, targets = custom_collate_fn([[1, 2, 3]], pad_token_id=0)
    assert inputs.tolist()[0] == [1, 2, 3]
    assert targets.tolist()[0] == [2, 3, 0]


def test_custom_collate_masks_extra_padding_targets():
    inputs, targets = custom_collate_fn([[1, 2, 3], [4]], pad_token_id=0, ignore_index=-100)
    assert targets.tolist()[1] == [0, -100, -100]


def test_custom_collate_respects_allowed_max_length():
    inputs, targets = custom_collate_fn([[1, 2, 3, 4, 5], [6, 7]], pad_token_id=0, allowed_max_length=3)
    assert inputs.shape[1] == 3
    assert targets.shape[1] == 3
