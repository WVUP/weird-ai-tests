from weird_ai.reasoning import (
    build_reasoning_prompt,
    build_parody_reasoning_prompt,
    build_closed_world_prompt,
    build_open_world_prompt,
)


def test_build_reasoning_prompt_includes_original_prompt():
    prompt = build_reasoning_prompt("Explain why transformers use attention.")

    assert "Explain why transformers use attention." in prompt


def test_build_reasoning_prompt_requests_step_by_step_reasoning():
    prompt = build_reasoning_prompt("What is 17 times 23?")

    assert "Think step-by-step" in prompt
    assert "Final Answer" in prompt


def test_build_parody_reasoning_prompt_includes_topic():
    prompt = build_parody_reasoning_prompt("computer networking")

    assert "computer networking" in prompt


def test_build_parody_reasoning_prompt_includes_creative_planning_steps():
    prompt = build_parody_reasoning_prompt("debugging Python")

    assert "Think step-by-step" in prompt
    assert "Identify the topic" in prompt
    assert "funny" in prompt.lower()
    assert "rhymes" in prompt.lower()
    assert "Final Parody" in prompt


def test_build_closed_world_prompt_includes_all_premises():
    premises = [
        "All birds can fly.",
        "A penguin is a bird.",
    ]

    prompt = build_closed_world_prompt(premises, "Can a penguin fly?")

    assert "All birds can fly." in prompt
    assert "A penguin is a bird." in prompt
    assert "Can a penguin fly?" in prompt


def test_build_closed_world_prompt_restricts_to_premises():
    prompt = build_closed_world_prompt(
        ["All birds can fly.", "A penguin is a bird."],
        "Can a penguin fly?"
    )

    assert "only" in prompt.lower()
    assert "provided premises" in prompt.lower()
    assert "Reasoning" in prompt
    assert "Answer" in prompt


def test_build_open_world_prompt_allows_background_knowledge():
    prompt = build_open_world_prompt(
        ["All birds can fly.", "A penguin is a bird."],
        "Can a penguin fly?"
    )

    assert "background knowledge" in prompt.lower()
    assert "conflict" in prompt.lower() or "contradiction" in prompt.lower()
    assert "Reasoning" in prompt
    assert "Answer" in prompt


def test_closed_world_and_open_world_prompts_are_different():
    premises = ["All birds can fly.", "A penguin is a bird."]
    question = "Can a penguin fly?"

    closed_prompt = build_closed_world_prompt(premises, question)
    open_prompt = build_open_world_prompt(premises, question)

    assert closed_prompt != open_prompt
