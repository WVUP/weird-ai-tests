"""
Tests for the Lesson 11 Weird AI generator assignment.

These tests use fake generation and evaluation functions so students can
validate the inference-time scaling logic without loading a large model.
"""

import pytest

from weird_ai.generator import (
    GenerationCandidate,
    build_generation_prompt,
    format_candidate_report,
    generate_best,
    generate_multiple,
    generate_once,
    rank_candidates,
    score_candidate,
    select_best_candidate,
)


def fake_generation_func(prompt, temperature=1.0, top_p=None, max_new_tokens=120, **kwargs):
    return (
        f"Prompt: {prompt}\n"
        f"temperature={temperature}\n"
        f"top_p={top_p}\n"
        f"max_new_tokens={max_new_tokens}"
    )


def fake_evaluation_func(text):
    # Higher score for texts containing more exclamation marks.
    score = min(1.0, text.count("!") / 3)
    return {
        "overall_score": score,
        "rhyme_score": score,
        "syllable_consistency_score": 0.75,
        "structure_score": 0.8,
    }


def test_build_generation_prompt_contains_assignment_details():
    prompt = build_generation_prompt("database indexes", style="emo pop punk", line_count=6)

    assert "database indexes" in prompt.lower()
    assert "emo pop punk" in prompt.lower()
    assert "6" in prompt
    assert "lyric" in prompt.lower()


def test_generate_once_forwards_sampling_arguments():
    text = generate_once(
        "Write a parody.",
        fake_generation_func,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=42,
    )

    assert "Write a parody." in text
    assert "temperature=0.7" in text
    assert "top_p=0.9" in text
    assert "max_new_tokens=42" in text


def test_generate_multiple_returns_requested_number_of_candidates():
    candidates = generate_multiple(
        "Write a parody.",
        fake_generation_func,
        num_candidates=4,
        temperature=1.2,
    )

    assert len(candidates) == 4
    assert all("temperature=1.2" in candidate for candidate in candidates)


def test_generate_multiple_rejects_invalid_candidate_count():
    with pytest.raises(ValueError):
        generate_multiple("Prompt", fake_generation_func, num_candidates=0)


def test_score_candidate_uses_overall_score():
    candidate = score_candidate("line one\nline two!!!", fake_evaluation_func)

    assert isinstance(candidate, GenerationCandidate)
    assert candidate.score == 1.0
    assert candidate.evaluation["overall_score"] == 1.0
    assert candidate.text == "line one\nline two!!!"


def test_rank_candidates_sorts_from_best_to_worst():
    texts = [
        "no excitement",
        "one!",
        "three!!!",
        "two!!",
    ]

    ranked = rank_candidates(texts, fake_evaluation_func)

    assert [candidate.score for candidate in ranked] == [1.0, 2 / 3, 1 / 3, 0.0]
    assert ranked[0].text == "three!!!"
    assert ranked[0].index == 3


def test_select_best_candidate_returns_highest_score():
    candidates = [
        GenerationCandidate(text="low", score=0.1, evaluation={}, index=1),
        GenerationCandidate(text="high", score=0.9, evaluation={}, index=2),
        GenerationCandidate(text="middle", score=0.5, evaluation={}, index=3),
    ]

    best = select_best_candidate(candidates)

    assert best.text == "high"
    assert best.index == 2


def test_select_best_candidate_rejects_empty_list():
    with pytest.raises(ValueError):
        select_best_candidate([])


def test_generate_best_returns_best_text_and_ranked_candidates():
    outputs = iter(["bad", "better!!", "best!!!"])

    def sequential_generator(prompt, **kwargs):
        return next(outputs)

    result = generate_best(
        "Write a parody.",
        sequential_generator,
        fake_evaluation_func,
        num_candidates=3,
    )

    assert result["best_text"] == "best!!!"
    assert result["best_score"] == 1.0
    assert len(result["candidates"]) == 3
    assert result["candidates"][0].text == "best!!!"


def test_format_candidate_report_includes_scores():
    candidates = [
        GenerationCandidate(
            text="example",
            score=0.88,
            evaluation={
                "overall_score": 0.88,
                "rhyme_score": 0.9,
                "syllable_consistency_score": 0.75,
                "structure_score": 1.0,
            },
            index=1,
        )
    ]

    report = format_candidate_report(candidates)

    assert "Candidate 1" in report
    assert "0.88" in report
    assert "rhyme" in report.lower()
    assert "syllable" in report.lower()
