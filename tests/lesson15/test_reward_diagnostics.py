"""
Tests for Lesson 14 reward diagnostics.
"""

import pytest

from weird_ai.reward_diagnostics import (
    DiagnosticReport,
    DiagnosticWarning,
    advantage_statistics,
    build_diagnostic_report,
    combine_rewards,
    detect_repetition_hacking,
    detect_reward_collapse,
    detect_short_high_reward,
    format_reward,
    get_nonempty_lines,
    moving_average,
    repeated_line_ratio,
    repeated_word_ratio,
    summarize_diagnostic_report,
    summarize_values,
)


def test_moving_average_basic():
    assert moving_average([1, 2, 3, 4], window_size=2) == pytest.approx([1.0, 1.5, 2.5, 3.5])


def test_moving_average_window_larger_than_values():
    assert moving_average([2, 4], window_size=10) == pytest.approx([2.0, 3.0])


def test_moving_average_rejects_invalid_window():
    with pytest.raises(ValueError):
        moving_average([1, 2, 3], window_size=0)


def test_summarize_values_empty():
    stats = summarize_values([])
    assert stats == {"count": 0, "average": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}


def test_summarize_values_nonempty():
    stats = summarize_values([0.0, 0.5, 1.0])
    assert stats["count"] == 3
    assert stats["average"] == pytest.approx(0.5)
    assert stats["min"] == 0.0
    assert stats["max"] == 1.0
    assert stats["std"] > 0


def test_advantage_statistics():
    stats = advantage_statistics([-0.5, 0.0, 0.5])
    assert stats["average"] == pytest.approx(0.0)
    assert stats["std"] > 0


def test_detect_reward_collapse_returns_warning():
    warning = detect_reward_collapse([0.5, 0.5, 0.5], min_std=0.01)
    assert isinstance(warning, DiagnosticWarning)
    assert "collapse" in warning.code


def test_detect_reward_collapse_returns_none_for_varied_rewards():
    assert detect_reward_collapse([0.0, 0.5, 1.0], min_std=0.01) is None


def test_get_nonempty_lines():
    assert get_nonempty_lines("\nline one\n\n line two \n") == ["line one", "line two"]


def test_repeated_line_ratio():
    assert repeated_line_ratio("same\nsame\nnew\nsame") == pytest.approx(0.5)


def test_repeated_line_ratio_empty():
    assert repeated_line_ratio("") == 0.0


def test_repeated_word_ratio():
    assert repeated_word_ratio("night night fight light night") == pytest.approx(2 / 5)


def test_repeated_word_ratio_empty():
    assert repeated_word_ratio("") == 0.0


def test_detect_short_high_reward_flags_suspicious_output():
    warning = detect_short_high_reward("night\nfight", reward=0.95, min_lines=4)
    assert isinstance(warning, DiagnosticWarning)
    assert "short" in warning.code


def test_detect_short_high_reward_ignores_low_reward_short_output():
    assert detect_short_high_reward("night\nfight", reward=0.3, min_lines=4) is None


def test_detect_repetition_hacking_flags_repeated_lines_and_words():
    text = "night night\nnight night\nnight night\nnew line"
    warnings = detect_repetition_hacking(text, max_repeated_line_ratio=0.2, max_repeated_word_ratio=0.2)
    codes = {warning.code for warning in warnings}
    assert "repeated_lines" in codes
    assert "repeated_words" in codes


def test_format_reward_good_output():
    text = "\n".join([
        "My query broke at dawn",
        "The missing rows were gone",
        "I joined my heart to you",
        "But nulls came crashing through",
        "The index lost its way",
        "The server cried all day",
        "I wrote one final view",
        "And queried dreams of you",
    ])
    assert format_reward(text, target_line_count=8) == pytest.approx(1.0)


def test_format_reward_penalizes_explanation_text():
    text = "Here are the lyrics:\nnight\nfight\nlight\nright"
    assert format_reward(text, target_line_count=4) < 1.0


def test_format_reward_penalizes_wrong_line_count():
    text = "night\nfight"
    assert format_reward(text, target_line_count=8, line_tolerance=1) < 1.0


def test_combine_rewards_weighted_average():
    assert combine_rewards(0.8, 0.4, creative_weight=0.75, format_weight=0.25) == pytest.approx(0.7)


def test_combine_rewards_clamps_result():
    assert combine_rewards(2.0, 2.0) == 1.0
    assert combine_rewards(-1.0, -1.0) == 0.0


def test_combine_rewards_rejects_negative_weights():
    with pytest.raises(ValueError):
        combine_rewards(0.8, 0.4, creative_weight=-1.0)


def test_build_diagnostic_report_includes_stats_and_warnings():
    texts = ["night\nfight", "same\nsame\nsame\nsame", "good one\ngood two\ngood three\ngood four"]
    rewards = [0.95, 0.9, 0.2]
    advantages = [0.3, 0.25, -0.55]
    report = build_diagnostic_report(rewards, advantages, texts)
    assert isinstance(report, DiagnosticReport)
    assert report.reward_count == 3
    assert report.reward_average == pytest.approx(sum(rewards) / 3)
    assert len(report.warnings) >= 1


def test_build_diagnostic_report_handles_no_texts():
    report = build_diagnostic_report([0.0, 0.5, 1.0], [-0.5, 0.0, 0.5])
    assert report.reward_count == 3
    assert isinstance(report.warnings, list)


def test_summarize_diagnostic_report_contains_key_details():
    report = build_diagnostic_report([0.0, 0.5, 1.0], [-0.5, 0.0, 0.5])
    summary = summarize_diagnostic_report(report)
    assert "reward" in summary.lower()
    assert "advantage" in summary.lower()
    assert "3" in summary


def test_summarize_diagnostic_report_mentions_no_warnings():
    report = build_diagnostic_report([0.0, 0.5, 1.0], [-0.5, 0.0, 0.5])
    summary = summarize_diagnostic_report(report)
    assert "no warnings" in summary.lower()
