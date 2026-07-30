from ci_grpo.p0_state_goal_probe import canonical_json, scene_signature


def test_scene_signature_excludes_instruction_goal_and_interest():
    common = {
        "problem_name": "scene",
        "fixtures": {"fixture": ["table"]},
        "regions": {"left": {"target": "table"}},
        "objects": {"object": ["bowl"]},
        "scene_properties": {},
        "initial_state": [["on", "bowl", "left"]],
    }
    first = {
        **common,
        "language_instruction": ["put", "left"],
        "goal_state": [["on", "bowl", "left"]],
        "obj_of_interest": ["left"],
    }
    second = {
        **common,
        "language_instruction": ["put", "right"],
        "goal_state": [["on", "bowl", "right"]],
        "obj_of_interest": ["right"],
    }

    assert canonical_json(scene_signature(first)) == canonical_json(scene_signature(second))


def test_scene_signature_detects_initial_state_change():
    base = {
        "problem_name": "scene",
        "fixtures": {},
        "regions": {},
        "objects": {"object": ["bowl"]},
        "scene_properties": {},
        "initial_state": [["on", "bowl", "left"]],
    }
    changed = {**base, "initial_state": [["on", "bowl", "right"]]}

    assert canonical_json(scene_signature(base)) != canonical_json(scene_signature(changed))
