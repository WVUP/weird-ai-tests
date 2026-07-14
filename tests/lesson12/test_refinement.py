"""
Tests for the Lesson 12 self-refinement assignment.

These tests use fake generation and evaluation functions so the refinement loop can
be tested without loading a model.
"""

import pytest

from weird_ai.refinement import (
    RefinementStep,
    build_refinement_prompt,
    choose_better_text,
    get_score,
    iterative_refinement,
    refine_once,
    should_continue,
    summarize_refinement,
)


def fake_evaluation(text):
    """Score text by counting exclamation marks."""
    score = min(1.0, text.count("!") / 4)
    return {
        "overall_score": score,
        "rhyme_score": score,
        "syllable_consistency_score": 0.75,
        "structure_score": 0.80,
        "line_count": len([line for line in text.splitlines() if line.strip()]),
    }


def test_get_score_returns_float_for_valid_score():
    assert get_score({"overall_score": "0.75"}) == 0.75


def test_get_score_returns_zero_for_missing_score():
    assert get_score({"rhyme_score": 0.5}) == 0.0


def test_get_score_returns_zero_for_invalid_score():
    assert get_score({"overall_score": "not a number"}) == 0.0


def test_build_refinement_prompt_contains_context_and_instructions():
    prompt = build_refinement_prompt(
        original_prompt="Write a parody about SQL joins.",
        current_text="My joins are gone",
        evaluation={
            "overall_score": 0.4,
            "rhyme_score": 0.2,
            "syllable_consistency_score": 0.5,
            "structure_score": 0.6,
        },
        focus=["stronger rhymes", "more consistent line lengths"],
        preserve=["topic", "emotional tone"],
    )

    prompt_lower = prompt.lower()

    assert "sql joins" in prompt_lower
    assert "my joins are gone" in prompt_lower
    assert "0.4" in prompt
    assert "stronger rhymes" in prompt_lower
    assert "emotional tone" in prompt_lower
    assert "only" in prompt_lower
    assert "lyrics" in prompt_lower


def test_choose_better_text_accepts_higher_score():
    selected_text, selected_eval, accepted = choose_better_text(
        "old",
        "new!",
        fake_evaluation("old"),
        fake_evaluation("new!"),
    )

    assert selected_text == "new!"
    assert selected_eval["overall_score"] == 0.25
    assert accepted is True


def test_choose_better_text_rejects_lower_score():
    selected_text, selected_eval, accepted = choose_better_text(
        "old!!!",
        "new!",
        fake_evaluation("old!!!"),
        fake_evaluation("new!"),
    )

    assert selected_text == "old!!!"
    assert selected_eval["overall_score"] == 0.75
    assert accepted is False


def test_choose_better_text_rejects_equal_score_when_improvement_required():
    selected_text, selected_eval, accepted = choose_better_text(
        "old!",
        "new!",
        fake_evaluation("old!"),
        fake_evaluation("new!"),
        require_improvement=True,
    )

    assert selected_text == "old!"
    assert accepted is False


def test_choose_better_text_accepts_equal_score_when_improvement_not_required():
    selected_text, selected_eval, accepted = choose_better_text(
        "old!",
        "new!",
        fake_evaluation("old!"),
        fake_evaluation("new!"),
        require_improvement=False,
    )

    assert selected_text == "new!"
    assert accepted is True


def test_should_continue_stops_at_max_iterations():
    assert should_continue(iteration=3, max_iterations=3, current_score=0.5) is False


def test_should_continue_stops_at_target_score():
    assert should_continue(iteration=1, max_iterations=3, current_score=0.9, target_score=0.8) is False


def test_should_continue_stops_on_no_improvement():
    assert should_continue(
        iteration=1,
        max_iterations=3,
        current_score=0.5,
        last_improved=False,
        stop_on_no_improvement=True,
    ) is False


def test_should_continue_allows_more_iterations():
    assert should_continue(
        iteration=1,
        max_iterations=3,
        current_score=0.5,
        target_score=0.9,
        last_improved=True,
    ) is True


def test_refine_once_returns_refinement_step():
    def fake_generator(prompt, **kwargs):
        return "better lyrics!!"

    step = refine_once("Write a parody.", "rough lyrics", fake_generator, fake_evaluation)

    assert isinstance(step, RefinementStep)
    assert step.previous_text == "rough lyrics"
    assert step.revised_text == "better lyrics!!"
    assert step.previous_score == 0.0
    assert step.revised_score == 0.5
    assert step.accepted is True
    assert "Write a parody." in step.prompt


def test_iterative_refinement_improves_until_target_score():
    outputs = iter(["revision one!", "revision two!!", "revision three!!!!"])

    def fake_generator(prompt, **kwargs):
        return next(outputs)

    result = iterative_refinement(
        "Write a parody.",
        "initial",
        fake_generator,
        fake_evaluation,
        max_iterations=5,
        target_score=1.0,
    )

    assert result["initial_score"] == 0.0
    assert result["final_score"] == 1.0
    assert result["final_text"] == "revision three!!!!"
    assert len(result["iterations"]) == 3
    assert result["stopped_reason"] == "target_score"


def test_iterative_refinement_stops_on_no_improvement():
    outputs = iter(["better!!", "worse!"])

    def fake_generator(prompt, **kwargs):
        return next(outputs)

    result = iterative_refinement(
        "Write a parody.",
        "initial",
        fake_generator,
        fake_evaluation,
        max_iterations=5,
        stop_on_no_improvement=True,
    )

    assert result["final_text"] == "better!!"
    assert result["final_score"] == 0.5
    assert len(result["iterations"]) == 2
    assert result["iterations"][-1].accepted is False
    assert result["stopped_reason"] == "no_improvement"


def test_iterative_refinement_rejects_invalid_max_iterations():
    with pytest.raises(ValueError):
        iterative_refinement(
            "Write a parody.",
            "initial",
            lambda prompt, **kwargs: "revision",
            fake_evaluation,
            max_iterations=0,
        )


def test_summarize_refinement_includes_key_details():
    result = {
        "initial_score": 0.0,
        "final_score": 0.75,
        "stopped_reason": "max_iterations",
        "iterations": [
            RefinementStep(
                iteration=1,
                prompt="prompt",
                previous_text="old",
                revised_text="new!!!",
                previous_score=0.0,
                revised_score=0.75,
                accepted=True,
                evaluation=fake_evaluation("new!!!"),
            )
        ],
    }

    summary = summarize_refinement(result)

    assert "Initial score" in summary
    assert "Final score" in summary
    assert "0.75" in summary
    assert "accepted" in summary.lower()
    assert "max_iterations" in summary
