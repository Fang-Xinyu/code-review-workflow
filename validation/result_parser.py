from schemas import TestResult, Phase, Failure


def classify_failure(failure):
    source = "/input/solution.py" in [*failure.source_frames, failure.location]
    if failure.exception_type in ("ModuleNotFoundError", "ImportError"):
        return "test_input"
    if failure.exception_type in ("SyntaxError", "IndentationError") and source:
        return "source_syntax"
    if failure.phase == "call" and (failure.exception_type == "AssertionError" or (
            failure.exception_type == "Failed" and "/input/test_solution.py" in failure.source_frames
            and any(p.endswith("/site-packages/_pytest/outcomes.py") for p in failure.source_frames))):
        return "assertion"
    return "source_exception" if source else "unknown"


def parse_report(payload, *, run_id, source_hash, tests_hash, environment_id,
                 report_path="", stdout="", stderr="", truncated=False):
    base = dict(run_id=run_id, source_hash=source_hash, tests_hash=tests_hash,
                environment_id=environment_id, report_path=report_path, stdout=stdout, stderr=stderr,
                truncated=truncated, status="error", failure_kind="environment")
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return TestResult(**base)
    try:
        results = {k: [Phase.model_validate(x) for x in v] for k, v in payload["results"].items()}
        failures = [Failure.model_validate(x) for x in payload["failures"]]
        collected = payload["collected"]
        if not isinstance(collected, list) or len(collected) != len(set(collected)):
            raise ValueError("测试身份异常")
        passed, failed, skipped, xfailed = [], [], [], []
        complete = bool(payload["complete"]) and not payload["truncated"]
        for nodeid in collected:
            phases = results.get(nodeid, [])
            if any(p.wasxfail for p in phases):
                xfailed.append(nodeid)
            if any(p.outcome == "failed" for p in phases):
                failed.append(nodeid)
            elif any(p.outcome == "skipped" for p in phases):
                skipped.append(nodeid)
            elif ([p.phase for p in phases] == ["setup", "call", "teardown"]
                  and all(p.outcome == "passed" for p in phases)):
                passed.append(nodeid)
            else:
                complete = False
        if set(results) - set(collected):
            complete = False
        exit_code = int(payload["exit_code"])
        trunc = truncated or bool(payload["truncated"])
        status, kind = "inconclusive", "unknown"
        if not complete or trunc or skipped or xfailed:
            pass
        elif failures:
            kinds = [classify_failure(f) for f in failures]
            if all(k in ("assertion", "source_exception", "source_syntax") for k in kinds):
                status, kind = "failed", ("source_syntax" if "source_syntax" in kinds else kinds[0])
            elif "test_input" in kinds:
                status, kind = "inconclusive", "test_input"
        elif exit_code == 0 and collected and len(passed) == len(collected):
            status, kind = "passed", "none"
        elif exit_code not in (0, 1, 2, 5):
            status, kind = "error", "environment"
        if kind == "source_syntax" and not collected:
            complete = False
        return TestResult(**{**base, "status": status, "failure_kind": kind, "complete": complete,
            "truncated": trunc, "collected": collected, "passed": passed, "failed": failed,
            "skipped": skipped, "xfailed": xfailed, "results": results, "failures": failures,
            "exit_code": exit_code, "duration_ms": payload["duration_ms"]})
    except (ValueError, KeyError, TypeError):
        return TestResult(**base)
