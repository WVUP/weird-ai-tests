from weird_ai.instruction_data import (
    format_input,
    format_response,
    format_full_example,
    validate_instruction_entry,
)


def make_entry():
    return {
        "instruction": "Write a silly parody chorus.",
        "input": "The original song is about heartbreak.",
        "output": "My pizza left me crying in the rain.",
    }


def test_format_input_includes_instruction():
    formatted = format_input(make_entry())
    assert "### Instruction:" in formatted
    assert "Write a silly parody chorus." in formatted


def test_format_input_includes_nonempty_input():
    formatted = format_input(make_entry())
    assert "### Input:" in formatted
    assert "heartbreak" in formatted


def test_format_input_skips_empty_input_section():
    entry = make_entry()
    entry["input"] = ""
    formatted = format_input(entry)
    assert "### Input:" not in formatted


def test_format_response_includes_response_section():
    response = format_response(make_entry())
    assert "### Response:" in response
    assert "pizza" in response


def test_format_full_example_contains_all_parts():
    full_text = format_full_example(make_entry())
    assert "### Instruction:" in full_text
    assert "### Input:" in full_text
    assert "### Response:" in full_text


def test_validate_instruction_entry_valid():
    assert validate_instruction_entry(make_entry())


def test_validate_instruction_entry_missing_key():
    entry = make_entry()
    del entry["output"]
    assert not validate_instruction_entry(entry)


def test_validate_instruction_entry_empty_instruction():
    entry = make_entry()
    entry["instruction"] = ""
    assert not validate_instruction_entry(entry)
