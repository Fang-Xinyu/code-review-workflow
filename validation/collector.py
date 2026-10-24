"""Fixed in-image pytest plugin. Never accepts uploaded configuration/plugins."""
import json
import os
import platform
import time
from pathlib import Path

import pytest


class Collector:
    def __init__(self):
        self.started = time.monotonic()
        self.items = []
        self.phases = {}
        self.failures = []
        self.truncated = False

    def pytest_collection_finish(self, session):
        self.items = [item.nodeid for item in session.items]

    def failure(self, report, phase, call=None):
        text = str(report.longrepr)
        if len(text) > 32000:
            self.truncated = True
        frames = []
        exc_type = ""
        if call is not None and call.excinfo is not None:
            exc_type = call.excinfo.typename
            frames = [str(e.path) for e in call.excinfo.traceback]
        if phase == "collect":
            entries = getattr(getattr(report.longrepr, "reprtraceback", None), "reprentries", [])
            frames.extend(str(getattr(getattr(e, "reprfileloc", None), "path", "")) for e in entries)
        if not exc_type:
            crash = getattr(report.longrepr, "reprcrash", None)
            message = getattr(crash, "message", "")
            words = message.split(":")[0].split()
            exc_type = words[-1] if words else ""
        if phase == "collect" and 'File "/input/solution.py"' in text and "SyntaxError:" in text:
            exc_type = "SyntaxError"
            frames.append("/input/solution.py")
        if phase == "collect" and 'File "/input/solution.py"' in text and "IndentationError:" in text:
            exc_type = "IndentationError"
            frames.append("/input/solution.py")
        self.failures.append({"nodeid": report.nodeid or "test_solution.py", "phase": phase,
                              "exception_type": exc_type, "traceback": text[:32000],
                              "location": str(getattr(getattr(report.longrepr, "reprcrash", None), "path", "")),
                              "source_frames": frames})

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        self.phases.setdefault(report.nodeid, []).append({"phase": report.when, "outcome": report.outcome,
                     "duration_ms": int(report.duration * 1000), "wasxfail": hasattr(report, "wasxfail")})
        if report.failed:
            self.failure(report, report.when, call)

    def pytest_collectreport(self, report):
        if report.failed:
            self.failure(report, "collect")
        if report.skipped:
            self.truncated = True

    def pytest_sessionfinish(self, session, exitstatus):
        payload = {"version": 1, "python": platform.python_version(), "pytest": pytest.__version__,
                   "collected": self.items, "results": self.phases, "failures": self.failures,
                   "exit_code": int(exitstatus), "complete": True, "truncated": self.truncated,
                   "duration_ms": int((time.monotonic() - self.started) * 1000)}
        data = json.dumps(payload, ensure_ascii=False).encode()
        if len(data) > 2 * 1024 * 1024:
            data = json.dumps({"complete": False, "truncated": True}).encode()
        Path("/tmp/report.partial").write_bytes(data)
        os.replace("/tmp/report.partial", "/tmp/report.json")
