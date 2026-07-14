"""
Tests for Lesson 13 reward utilities.

These tests use a fake evaluator so students can validate reward and advantage
logic without generating text or loading a model.
"""

import pytest

from weird_ai.rewards import (
    RolloutReward,
    build_rollout_reward,
    clamp,
    compute_group_advantages,
    extract_score,
    get_best_rollout,
    get_worst_rollout,
    normalize_reward,
    prepare_reward_batch,
    rank_by_reward,
    reward_from_evaluation,
    summarize_reward_batch,
)


def fake_evaluator(text):
    # Score is based on exclamation marks for predictable tests.
    score = min(1.0, text.count("!") / 4)
    return {
        "overall_score": score,
        "rhyme_score": score,
        "structure_score": 0.75,
        "syllable_consistency_score": 0.80,
    }


def test_clamp_limits_values():
    assert clamp(-0.5) == 0.0
    assert clamp(0.5) == 0.5
    assert clamp(1.5) == 1.0


def test_clamp_uses_custom_range():
    assert clamp(15, minimum=0, maximum=10) == 10
    assert clamp(-3, minimum=0, maximum=10) == 0
    assert clamp(7, minimum=0, maximum=10) == 7


def test_extract_score_reads_float():
    assert extract_score({"overall_score": "0.75"}) == 0.75


def test_extract_score_missing_returns_zero():
    assert extract_score({"rhyme_score": 0.4}) == 0.0


def test_extract_score_invalid_returns_zero():
    assert extract_score({"overall_score": "bad"}) == 0.0


def test_normalize_reward_default_range():
    assert normalize_reward(0.75) == 0.75


def test_normalize_reward_custom_range():
    assert normalize_reward(75, minimum=0, maximum=100) == 0.75


def test_normalize_reward_clamps_out_of_range():
    assert normalize_reward(150, minimum=0, maximum=100) == 1.0
    assert normalize_reward(-50, minimum=0, maximum=100) == 0.0


def test_normalize_reward_handles_zero_width_range():
    assert normalize_reward(5, minimum=1, maximum=1) == 0.0


def test_reward_from_evaluation_extracts_and_normalizes():
    evaluation = {"overall_score": 80}
    assert reward_from_evaluation(evaluation, minimum=0, maximum=100) == 0.8


def test_compute_group_advantages_subtracts_mean():
    rewards = [0.0, 0.5, 1.0]
    advantages = compute_group_advantages(rewards)

    assert advantages == pytest.approx([-0.5, 0.0, 0.5])


def test_compute_group_advantages_empty_input():
    assert compute_group_advantages([]) == []


def test_compute_group_advantages_normalized():
    rewards = [0.0, 0.5, 1.0]
    advantages = compute_group_advantages(rewards, normalize=True)

    assert sum(advantages) == pytest.approx(0.0)
    assert advantages[0] < 0
    assert advantages[2] > 0


def test_compute_group_advantages_all_same_rewards():
    rewards = [0.5, 0.5, 0.5]
    advantages = compute_group_advantages(rewards, normalize=True)

    assert advantages == pytest.approx([0.0, 0.0, 0.0])


def test_build_rollout_reward_creates_dataclass():
    record = build_rollout_reward(
        prompt="Write lyrics.",
        text="lyrics!",
        evaluation={"overall_score": 0.25},
        reward=0.25,
        advantage=-0.1,
        index=1,
    )

    assert isinstance(record, RolloutReward)
    assert record.prompt == "Write lyrics."
    assert record.text == "lyrics!"
    assert record.reward == 0.25
    assert record.advantage == -0.1
    assert record.index == 1


def test_prepare_reward_batch_builds_records_and_advantages():
    rollouts = ["bad", "ok!!", "great!!!!"]
    batch = prepare_reward_batch("Write lyrics.", rollouts, fake_evaluator)

    assert len(batch) == 3
    assert [item.index for item in batch] == [1, 2, 3]
    assert [item.reward for item in batch] == pytest.approx([0.0, 0.5, 1.0])
    assert [item.advantage for item in batch] == pytest.approx([-0.5, 0.0, 0.5])
    assert all(item.prompt == "Write lyrics." for item in batch)


def test_prepare_reward_batch_empty_rollouts():
    assert prepare_reward_batch("Prompt", [], fake_evaluator) == []


def test_rank_by_reward_highest_first():
    batch = prepare_reward_batch("Prompt", ["bad", "great!!!!", "ok!!"], fake_evaluator)
    ranked = rank_by_reward(batch)

    assert [item.reward for item in ranked] == pytest.approx([1.0, 0.5, 0.0])
    assert [item.index for item in batch] == [1, 2, 3]  # original not mutated


def test_rank_by_reward_lowest_first():
    batch = prepare_reward_batch("Prompt", ["bad", "great!!!!", "ok!!"], fake_evaluator)
    ranked = rank_by_reward(batch, descending=False)

    assert [item.reward for item in ranked] == pytest.approx([0.0, 0.5, 1.0])


def test_get_best_and_worst_rollouts():
    batch = prepare_reward_batch("Prompt", ["bad", "great!!!!", "ok!!"], fake_evaluator)

    assert get_best_rollout(batch).text == "great!!!!"
    assert get_worst_rollout(batch).text == "bad"


def test_get_best_and_worst_empty_batch():
    assert get_best_rollout([]) is None
    assert get_worst_rollout([]) is None


def test_summarize_reward_batch_contains_key_information():
    batch = prepare_reward_batch("Prompt", ["bad", "great!!!!", "ok!!"], fake_evaluator)
    summary = summarize_reward_batch(batch)

    assert "3" in summary
    assert "average" in summary.lower()
    assert "best" in summary.lower()
    assert "worst" in summary.lower()
    assert "advantage" in summary.lower()


def test_summarize_reward_batch_handles_empty_batch():
    summary = summarize_reward_batch([])

    assert "no rollouts" in summary.lower()
