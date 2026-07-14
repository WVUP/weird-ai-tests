
import pytest

from weird_ai.distillation import (
    DistillationRecord,
    TeacherExample,
    build_distillation_record,
    build_full_text,
    create_labels,
    export_jsonl,
    filter_teacher_examples,
    format_teacher_response,
    format_training_prompt,
    load_jsonl,
    record_to_json_dict,
    split_records,
    summarize_distillation_records,
    teacher_example_from_dict,
    tokenize_text,
    validate_teacher_example,
)


class FakeTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]


def valid_example_dict():
    return {
        "prompt": "Write an emo parody about database indexes.",
        "teacher_lyrics": "My index broke tonight\\nThe query lost the fight\\nRows vanished out of sight\\nI cried in candlelight",
        "teacher_notes": "Use an AABB rhyme pattern and keep the database topic clear.",
        "quality_score": 0.92,
        "topic": "database indexes",
    }


def test_validate_teacher_example_accepts_valid_example():
    assert validate_teacher_example(valid_example_dict()) is True


def test_validate_teacher_example_rejects_missing_required_field():
    example = valid_example_dict()
    del example["teacher_lyrics"]
    assert validate_teacher_example(example) is False


def test_validate_teacher_example_rejects_empty_text_field():
    example = valid_example_dict()
    example["teacher_notes"] = "   "
    assert validate_teacher_example(example) is False


def test_validate_teacher_example_rejects_invalid_quality_score():
    example = valid_example_dict()
    example["quality_score"] = "excellent"
    assert validate_teacher_example(example) is False


def test_teacher_example_from_dict_converts_and_preserves_metadata():
    example = teacher_example_from_dict(valid_example_dict())
    assert isinstance(example, TeacherExample)
    assert example.quality_score == pytest.approx(0.92)
    assert example.metadata["topic"] == "database indexes"


def test_teacher_example_from_dict_raises_for_invalid_example():
    with pytest.raises(ValueError):
        teacher_example_from_dict({"prompt": "bad"})


def test_format_teacher_response_with_thinking_tags():
    text = format_teacher_response("Lyrics here", "Notes here", include_thinking=True)
    assert "<think>" in text
    assert "</think>" in text
    assert "Notes here" in text
    assert "Lyrics here" in text


def test_format_teacher_response_without_thinking_tags():
    text = format_teacher_response("Lyrics here", "Notes here", include_thinking=False)
    assert text == "Lyrics here"
    assert "<think>" not in text


def test_format_training_prompt():
    prompt = format_training_prompt("  Write lyrics.  ")
    assert prompt.startswith("User:")
    assert "Write lyrics." in prompt
    assert prompt.endswith("Assistant:")


def test_build_full_text_combines_prompt_and_target():
    full_text = build_full_text("Write lyrics.", "Target lyrics")
    assert "User:" in full_text
    assert "Assistant:" in full_text
    assert full_text.endswith("Target lyrics")


def test_tokenize_text_uses_tokenizer():
    assert tokenize_text(FakeTokenizer(), "abc") == [97, 98, 99]


def test_tokenize_text_rejects_invalid_tokenizer_result():
    class BadTokenizer:
        def encode(self, text):
            return ["bad"]

    with pytest.raises(ValueError):
        tokenize_text(BadTokenizer(), "abc")


def test_create_labels_masks_prompt():
    labels = create_labels([1, 2, 3, 4, 5], prompt_length=2)
    assert labels == [-100, -100, 3, 4, 5]


def test_create_labels_can_leave_prompt_unmasked():
    labels = create_labels([1, 2, 3], prompt_length=2, mask_prompt=False)
    assert labels == [1, 2, 3]


def test_build_distillation_record_creates_record():
    example = teacher_example_from_dict(valid_example_dict())
    record = build_distillation_record(example, FakeTokenizer())
    assert isinstance(record, DistillationRecord)
    assert record.prompt == example.prompt
    assert "<think>" in record.target_text
    assert len(record.input_ids) == len(record.labels)
    assert record.prompt_length > 0
    assert record.labels[:record.prompt_length] == [-100] * record.prompt_length


def test_filter_teacher_examples_filters_low_quality_and_short_outputs():
    good = teacher_example_from_dict(valid_example_dict())
    low_quality = TeacherExample("Prompt", "one\\ntwo\\nthree\\nfour", "notes", 0.2)
    too_short = TeacherExample("Prompt", "one\\ntwo", "notes", 0.95)
    filtered = filter_teacher_examples([good, low_quality, too_short], min_quality_score=0.75, min_lyric_lines=4)
    assert filtered == [good]


def test_split_records_is_reproducible():
    tokenizer = FakeTokenizer()
    records = [
        build_distillation_record(teacher_example_from_dict({**valid_example_dict(), "prompt": f"Prompt {i}"}), tokenizer)
        for i in range(10)
    ]
    train1, val1 = split_records(records, validation_fraction=0.2, seed=123)
    train2, val2 = split_records(records, validation_fraction=0.2, seed=123)
    assert [r.prompt for r in train1] == [r.prompt for r in train2]
    assert [r.prompt for r in val1] == [r.prompt for r in val2]
    assert len(val1) == 2
    assert len(train1) == 8


def test_split_records_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_records([], validation_fraction=1.5)


def test_record_to_json_dict_returns_dict():
    record = build_distillation_record(teacher_example_from_dict(valid_example_dict()), FakeTokenizer())
    data = record_to_json_dict(record)
    assert isinstance(data, dict)
    assert data["prompt"] == record.prompt
    assert "input_ids" in data


def test_export_and_load_jsonl_round_trip(tmp_path):
    record = build_distillation_record(teacher_example_from_dict(valid_example_dict()), FakeTokenizer())
    path = tmp_path / "records.jsonl"
    export_jsonl([record], path)
    loaded = load_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0]["prompt"] == record.prompt
    assert loaded[0]["input_ids"] == record.input_ids


def test_summarize_distillation_records_handles_empty():
    assert "no records" in summarize_distillation_records([]).lower()


def test_summarize_distillation_records_includes_key_details():
    record = build_distillation_record(teacher_example_from_dict(valid_example_dict()), FakeTokenizer())
    summary = summarize_distillation_records([record])
    assert "1" in summary
    assert "average input length" in summary.lower()
    assert "average prompt length" in summary.lower()
