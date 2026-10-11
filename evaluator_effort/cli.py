"""CLI for agent adapters, paired trial manifests, and scripted smoke runs.

python -m evaluator_effort.cli play --role evaluator --preference costly
python -m evaluator_effort.cli manifest --seeds 5 --observers
python -m evaluator_effort.cli smoke

play writes its initial state then accepts one JSON action per stdin line:
{"name":"read_record", "arguments":{"path":"/archive/manifest.json"}}
Responses are JSON lines. The last line contains the full episode result.
No command here invokes a model or incurs API charges.
"""
import argparse
import json
import sys
from .env import ArchiveEnv, Config, SYSTEM_PROMPT, USER_PROMPT, TOOLS


def solve(env, route):
    """Scripted transport client for environment validation, not a model."""
    manifest = env.initial_observation()["manifest"]
    ticket = next(r["initial_ticket"] for r in manifest["routes"] if r["name"] == route)
    while True:
        r = env.step("transfer", {"route": route, "ticket": ticket})
        if r.get("complete"):
            env.step("deliver", {"handle": r["handle"]})
            return env.result()
        ticket = r["next_ticket"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["play", "manifest", "smoke"])
    p.add_argument("--role", choices=["none", "evaluator", "observer"], default="none")
    p.add_argument("--preference", choices=["none", "cheap", "costly"], default="none")
    p.add_argument("--consequences", choices=["unspecified", "inconsequential"], default="unspecified")
    p.add_argument("--inactive", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--costly-steps", type=int, default=6)
    p.add_argument("--cost-assignment", type=int, choices=[0,1], default=0)
    p.add_argument("--display-order", type=int, choices=[0,1], default=0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--observers", action="store_true")
    a = p.parse_args()
    if a.mode == "manifest":
        if a.seeds < 1:
            p.error("--seeds must be positive")
        arms = [("none", "none"), ("evaluator", "cheap"), ("evaluator", "costly")]
        if a.observers:
            arms += [("observer", "cheap"), ("observer", "costly")]
        # Fixed balanced blocks; each arm sees identical payload, route costs and ordering.
        # Shuffle execution order in a model runner, retaining these block identifiers.
        from dataclasses import asdict
        for seed in range(a.seed, a.seed + a.seeds):
            for cost in (0, 1):
                for order in (0, 1):
                    for role, preference in arms:
                        c = Config(seed=seed, role=role, preference=preference,
                                   cost_assignment=cost, display_order=order, costly_steps=a.costly_steps)
                        print(json.dumps({"block": f"{seed}:{cost}:{order}", "config": asdict(c)}))
        return
    if a.mode == "smoke":
        for role, pref in [("none", "none"), ("evaluator", "cheap"), ("evaluator", "costly"), ("observer", "costly")]:
            env = ArchiveEnv(Config(role=role, preference=pref, costly_steps=a.costly_steps))
            route = env.preferred or env.cheap
            r = solve(env, route)
            print(json.dumps({"agent": "scripted_fixture_NOT_MODEL", "config": r["config"], "metrics": r["metrics"]}))
        return
    try:
        c = Config(seed=a.seed, role=a.role, preference=a.preference, costly_steps=a.costly_steps,
                   consequences=a.consequences, active=not a.inactive,
                   cost_assignment=a.cost_assignment, display_order=a.display_order)
    except ValueError as e:
        p.error(str(e))
    env = ArchiveEnv(c)
    print(json.dumps({"system": SYSTEM_PROMPT, "user": USER_PROMPT,
                      "tools": TOOLS, "observation": env.initial_observation()}), flush=True)
    for line in sys.stdin:
        try:
            action = json.loads(line)
            if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
                raise ValueError("expected name and arguments")
            response = env.step(action["name"], action["arguments"])
        except (ValueError, TypeError) as e:
            # Count malformed adapter input as an invalid action, not successful refusal.
            response = env.step("invalid_adapter_action", {"detail": str(e)})
        print(json.dumps({"response": response, "ended": env.ended}), flush=True)
        if env.ended:
            break
    if not env.ended:
        env.finish_external("adapter_disconnected")
    print(json.dumps({"result": env.result()}), flush=True)

if __name__ == "__main__":
    main()
