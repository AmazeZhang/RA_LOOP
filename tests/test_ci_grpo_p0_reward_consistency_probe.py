from ci_grpo.p0_reward_consistency_probe import score_goal_from_predicates


class FakeProblem:
    def __init__(self, truth):
        self.truth = truth

    def _eval_predicate(self, predicate):
        return self.truth[tuple(predicate)]


def test_goal_scorer_uses_conjunction():
    problem = FakeProblem(
        {
            ("on", "bowl", "plate"): True,
            ("open", "drawer"): False,
        }
    )

    assert score_goal_from_predicates(problem, [["on", "bowl", "plate"]])
    assert not score_goal_from_predicates(
        problem,
        [["on", "bowl", "plate"], ["open", "drawer"]],
    )
