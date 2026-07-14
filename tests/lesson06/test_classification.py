import pandas as pd
import torch
from torch.utils.data import DataLoader

from weird_ai.classification import (
    TinyLyricsClassifier,
    calculate_accuracy,
    classify_text,
    label_to_name,
)

from weird_ai.classification_dataset import LyricsClassificationDataset
from weird_ai.tokenizer import SimpleCharacterTokenizer


def make_test_dataframe():
    return pd.DataFrame(
        [
            {"text": "I walked alone in the rain", "label": 0},
            {"text": "The night was quiet and cold", "label": 0},
            {"text": "My sandwich joined a polka band", "label": 1},
            {"text": "The toaster wore neon shoes", "label": 1},
        ]
    )


def make_tokenizer(df):
    all_text = "\n".join(df["text"].tolist())

    return SimpleCharacterTokenizer(all_text)


def test_classification_dataset_length():
    df = make_test_dataframe()
    tokenizer = make_tokenizer(df)

    dataset = LyricsClassificationDataset(df, tokenizer)

    assert len(dataset) == 4


def test_classification_dataset_returns_tensors():
    df = make_test_dataframe()
    tokenizer = make_tokenizer(df)

    dataset = LyricsClassificationDataset(df, tokenizer)

    input_ids, label = dataset[0]

    assert torch.is_tensor(input_ids)
    assert torch.is_tensor(label)


def test_classification_dataset_uses_fixed_length():
    df = make_test_dataframe()
    tokenizer = make_tokenizer(df)

    dataset = LyricsClassificationDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_length=12,
    )

    input_ids, label = dataset[0]

    assert input_ids.shape == (12,)


def test_classification_dataset_truncates_long_text():
    df = make_test_dataframe()
    tokenizer = make_tokenizer(df)

    dataset = LyricsClassificationDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_length=5,
    )

    input_ids, label = dataset[0]

    assert len(input_ids) == 5


def test_classification_dataset_pads_short_text():
    df = make_test_dataframe()
    tokenizer = make_tokenizer(df)

    dataset = LyricsClassificationDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_length=50,
        pad_token_id=0,
    )

    input_ids, label = dataset[0]

    assert len(input_ids) == 50


def test_tiny_lyrics_classifier_output_shape():
    df = make_test_dataframe()
    tokenizer = make_tokenizer(df)

    dataset = LyricsClassificationDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_length=20,
    )

    loader = DataLoader(dataset, batch_size=2)

    input_batch, label_batch = next(iter(loader))

    model = TinyLyricsClassifier(
        vocab_size=len(tokenizer.chars),
        emb_dim=16,
        num_classes=2,
    )

    logits = model(input_batch)

    assert logits.shape == (2, 2)


def test_calculate_accuracy_returns_float():
    df = make_test_dataframe()
    tokenizer = make_tokenizer(df)

    dataset = LyricsClassificationDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_length=20,
    )

    loader = DataLoader(dataset, batch_size=2)

    model = TinyLyricsClassifier(
        vocab_size=len(tokenizer.chars),
        emb_dim=16,
        num_classes=2,
    )

    accuracy = calculate_accuracy(
        data_loader=loader,
        model=model,
        device=torch.device("cpu"),
    )

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_classify_text_returns_integer_label():
    df = make_test_dataframe()
    tokenizer = make_tokenizer(df)

    model = TinyLyricsClassifier(
        vocab_size=len(tokenizer.chars),
        emb_dim=16,
        num_classes=2,
    )

    label = classify_text(
        text="My microwave started singing opera",
        model=model,
        tokenizer=tokenizer,
        max_length=20,
        device=torch.device("cpu"),
    )

    assert isinstance(label, int)
    assert label in [0, 1]


def test_label_to_name_known_labels():
    assert label_to_name(0) == "serious"
    assert label_to_name(1) == "silly/comedic"


def test_label_to_name_unknown_label():
    assert label_to_name(99) == "unknown"