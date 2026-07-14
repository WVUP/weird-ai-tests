from weird_ai.evaluate import (
    calculate_rhyme_score,
    calculate_structure_score,
    evaluate_parody,
    extract_final_parody,
    normalize_lyrics,
)


SAMPLE_RESPONSE = """
Planning:
- Topic: debugging
- Funny exaggeration: the bug keeps returning

Final Parody:
I chased a bug into a log
It hid inside my startup fog
I fixed one line and pet the dog
Then found it back inside the log
"""


def test_extract_final_parody_uses_marker():
    result = extract_final_parody(SAMPLE_RESPONSE)

    assert "Planning:" not in result
    assert result.startswith("I chased a bug")
    assert "Final Parody:" not in result


def test_extract_final_parody_falls_back_to_full_text():
    text = "  I saw a cat\n  It wore a hat  "

    result = extract_final_parody(text)

    assert result == "I saw a cat\n  It wore a hat"


def test_normalize_lyrics_removes_blank_lines_and_strips_whitespace():
    text = "\n  I saw a cat  \n\n It wore a hat\n"

    lines = normalize_lyrics(text)

    assert lines == ["I saw a cat", "It wore a hat"]


def test_calculate_rhyme_score_detects_adjacent_rhyme():
    text = """
    I saw a cat
    It wore a hat
    I found a dog
    It sat on a log
    """

    score = calculate_rhyme_score(text)

    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_calculate_structure_score_rewards_target_line_count():
    text = """
    I saw a cat
    It wore a hat
    I found a dog
    It sat on a log
    """

    score = calculate_structure_score(text, target_line_count=4)

    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_evaluate_parody_returns_expected_keys():
    result = evaluate_parody(SAMPLE_RESPONSE, target_line_count=4)

    assert result["line_count"] == 4
    assert result["lines"][0] == "I chased a bug into a log"
    assert isinstance(result["rhyme_scheme"], list)
    assert isinstance(result["syllable_counts"], list)
    assert 0.0 <= result["rhyme_score"] <= 1.0
    assert 0.0 <= result["structure_score"] <= 1.0
    assert 0.0 <= result["overall_score"] <= 1.0
    assert isinstance(result["passed"], bool)
