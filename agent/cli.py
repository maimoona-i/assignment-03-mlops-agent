"""Command-line chat:  python -m agent.cli "Which dataset version was used for experiment 25?"  [--confirm]"""
from __future__ import annotations

import argparse
import json

from agent.logging_utils import configure_logging
from agent.main import get_agent


def main() -> None:
    p = argparse.ArgumentParser(description="Talk to the MLOps agent")
    p.add_argument("message")
    p.add_argument("--confirm", action="store_true", help="explicitly confirm state-changing operations")
    p.add_argument("--degradation-report", default=None)
    p.add_argument("--json", action="store_true", help="print the full structured response")
    a = p.parse_args()
    configure_logging()
    resp = get_agent().handle(a.message, confirm=a.confirm, degradation_report=a.degradation_report)
    if a.json:
        print(json.dumps(resp.model_dump(), indent=2))
    else:
        print(f"[request {resp.request_id} | mode={resp.mode} | tools={[c.tool + ':' + c.status for c in resp.tool_calls]}]\n")
        print(resp.answer)


if __name__ == "__main__":
    main()
