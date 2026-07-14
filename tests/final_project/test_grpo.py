"""Tests for Lesson 16 Mini-GRPO utilities."""
import pytest
import torch
from weird_ai.grpo import *

def fake_evaluation_func(text):
    return {"overall_score": min(1.0, text.count("!") / 4), "length": len(text)}

def test_extract_reward_valid(): assert extract_reward({"overall_score":"0.75"}) == pytest.approx(0.75)
def test_extract_reward_missing_or_invalid():
    assert extract_reward({}) == 0.0
    assert extract_reward({"overall_score":"bad"}) == 0.0

def test_compute_group_advantages_basic(): assert compute_group_advantages([0.0,0.5,1.0]) == pytest.approx([-0.5,0.0,0.5])
def test_compute_group_advantages_empty(): assert compute_group_advantages([]) == []
def test_compute_group_advantages_normalized():
    a=compute_group_advantages([0.0,0.5,1.0], normalize=True)
    assert sum(a)==pytest.approx(0.0); assert a[0]<0 and a[-1]>0

def test_generate_rollouts_count():
    calls={"count":0}
    def generator(prompt, **kwargs):
        calls["count"] += 1
        return f"{prompt} rollout {calls['count']}!"
    outputs=generate_rollouts("Prompt", generator, num_rollouts=3, temperature=0.7)
    assert len(outputs)==3 and outputs[0].endswith("1!") and calls["count"]==3

def test_generate_rollouts_rejects_invalid_count():
    with pytest.raises(ValueError): generate_rollouts("Prompt", lambda p, **k: p, num_rollouts=0)

def test_evaluate_rollouts_returns_evaluations_and_rewards():
    ev, rewards = evaluate_rollouts(["a","b!!","c!!!!"], fake_evaluation_func)
    assert len(ev)==3 and rewards == pytest.approx([0.0,0.5,1.0])

def test_sequence_logprob_sum_and_mean():
    t=torch.tensor([-1.0,-2.0,-3.0])
    assert sequence_logprob_from_token_logprobs(t,reduction="sum").item()==pytest.approx(-6.0)
    assert sequence_logprob_from_token_logprobs(t,reduction="mean").item()==pytest.approx(-2.0)

def test_sequence_logprob_rejects_bad_reduction():
    with pytest.raises(ValueError): sequence_logprob_from_token_logprobs(torch.tensor([-1.0]), reduction="max")

def test_selected_token_logprobs():
    logits=torch.tensor([[2.0,1.0,0.0],[0.0,3.0,1.0]])
    targets=torch.tensor([0,1])
    selected=selected_token_logprobs(logits,targets)
    expected=torch.log_softmax(logits,dim=-1)[torch.arange(2),targets]
    assert torch.allclose(selected, expected)

def test_selected_token_logprobs_rejects_bad_shapes():
    with pytest.raises(ValueError): selected_token_logprobs(torch.tensor([1.0,2.0]), torch.tensor([0]))
    with pytest.raises(ValueError): selected_token_logprobs(torch.zeros(2,3), torch.tensor([0,1,2]))

def test_compute_policy_gradient_loss():
    logprobs=torch.tensor([-1.0,-2.0,-3.0]); advantages=torch.tensor([1.0,0.0,-1.0])
    loss=compute_policy_gradient_loss(logprobs,advantages)
    assert loss.item()==pytest.approx(-((1*-1)+(0*-2)+(-1*-3))/3)

def test_compute_policy_gradient_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError): compute_policy_gradient_loss(torch.zeros(2), torch.zeros(3))

def test_compute_entropy_from_logits():
    entropy=compute_entropy_from_logits(torch.zeros(2,4))
    assert entropy.item()==pytest.approx(torch.log(torch.tensor(4.0)).item())

def test_build_rollout_records():
    records=build_rollout_records("Prompt", ["a","b"], [{"overall_score":0.1},{"overall_score":0.9}], [0.1,0.9], [-0.4,0.4], [-2.0,-1.0])
    assert len(records)==2 and isinstance(records[0], Rollout) and records[0].index==1 and records[1].reward==pytest.approx(0.9)

def test_build_rollout_records_rejects_length_mismatch():
    with pytest.raises(ValueError): build_rollout_records("Prompt", ["a"], [], [], [], [])

def test_mini_grpo_update_changes_parameter():
    torch.manual_seed(0)
    model=torch.nn.Linear(1,1,bias=False); optimizer=torch.optim.SGD(model.parameters(), lr=0.1)
    outputs=iter(["bad","better!!","best!!!!"])
    def generator(prompt, **kwargs): return next(outputs)
    def logprob_func(prompt, text):
        feature=torch.tensor([[len(text)/10.0]], dtype=torch.float32)
        return model(feature).squeeze()
    before=model.weight.detach().clone()
    result=mini_grpo_update(prompt="Prompt", model=model, optimizer=optimizer, generation_func=generator, evaluation_func=fake_evaluation_func, logprob_func=logprob_func, num_rollouts=3)
    after=model.weight.detach().clone()
    assert isinstance(result, MiniGRPOResult) and result.optimizer_step_completed and len(result.rollouts)==3 and not torch.allclose(before, after)

def test_summarize_mini_grpo_result():
    result=MiniGRPOResult("Prompt", [Rollout("Prompt","text",{"overall_score":1.0},1.0,0.5,-1.2,1)], 0.123, 1.0, 0.5, True)
    summary=summarize_mini_grpo_result(result)
    assert "Prompt" in summary and "loss" in summary.lower() and "reward" in summary.lower() and "optimizer" in summary.lower()
