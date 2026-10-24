import json
import tempfile
import time
import uuid
from pathlib import Path

from schemas import TestResult, digest
from .result_parser import parse_report
from .sandbox import Sandbox, SandboxError, REPORT_LIMIT

PROBE_SOURCE = "VALUE = 7\n"
PROBE_TESTS = '''import os
import socket
import fcntl
import struct
from pathlib import Path
from solution import VALUE

def test_environment():
    assert VALUE == 7
    assert os.getuid() == 65532
    assert not any("KEY" in k or "TOKEN" in k or "SECRET" in k for k in os.environ)
    # Linux can expose dormant tunnel devices even with --network=none.
    # They must all be down, with no default route and no network administration capability.
    with socket.socket() as interface_socket:
        for _, name in socket.if_nameindex():
            if name == "lo":
                continue
            data = fcntl.ioctl(interface_socket.fileno(), 0x8913, struct.pack("256s", name.encode()))
            assert not (struct.unpack("H", data[16:18])[0] & 1), name
    routes = Path("/proc/net/route").read_text().splitlines()[1:]
    assert not any(row.split()[1] == "00000000" for row in routes if row.split())
    for path in ("/root-write-probe", "/input/solution.py", "/input/test_solution.py"):
        try:
            with open(path, "a") as handle:
                handle.write("x")
        except OSError:
            pass
        else:
            assert False, "read-only boundary failed"
    cg = Path("/sys/fs/cgroup")
    assert (cg / "memory.max").read_text().strip() == "536870912"
    assert (cg / "pids.max").read_text().strip() == "64"
    quota, period = map(int, (cg / "cpu.max").read_text().split())
    assert quota == period
    stat = os.statvfs("/tmp")
    assert stat.f_blocks * stat.f_frsize <= 64 * 1024 * 1024
    status = Path("/proc/self/status").read_text()
    assert "NoNewPrivs:\\t1" in status
    assert "CapEff:\\t0000000000000000" in status
    sock = socket.socket()
    sock.settimeout(.2)
    try:
        sock.connect(("1.1.1.1", 80))
    except OSError:
        pass
    else:
        assert False, "network boundary failed"
    finally:
        sock.close()
'''


class TestRunner:
    __test__ = False

    def __init__(self, sandbox=None, timeout=15):
        self.sandbox = sandbox or Sandbox()
        self.timeout = timeout
        self.ready = False
        self.environment = {}

    def preflight(self):
        self.ready = False
        self.sandbox.inspect_image()
        with tempfile.TemporaryDirectory(prefix="codeflow-probe-") as tmp:
            result = self._execute(PROBE_SOURCE, PROBE_TESTS, Path(tmp) / "probe")
        if result.status != "passed":
            raise SandboxError("隔离预检未完整通过：" + result.failure_kind)
        self.ready = True
        self.environment.update({"python": "3.12.8", "pytest": "8.3.5", "probe": "passed"})
        return dict(self.environment)

    def run_tests(self, snapshot, immutable_test_suite, directory):
        if not self.ready:
            raise SandboxError("真实执行前必须完成隔离预检")
        return self._execute(snapshot, immutable_test_suite, Path(directory))

    def _execute(self, source, tests, directory):
        directory.mkdir(parents=True, exist_ok=False)
        inputs = directory / "input"
        inputs.mkdir(mode=0o755)
        for name, content in (("solution.py", source), ("test_solution.py", tests)):
            path = inputs / name
            path.write_text(content, encoding="utf-8")
            path.chmod(0o444)
        name = "codeflow-" + uuid.uuid4().hex
        base = dict(run_id=name, source_hash=digest(source), tests_hash=digest(tests),
                    environment_id=self.sandbox.image_id, report_path=str(directory / "report.json"))
        started = time.monotonic()
        created = False
        result = None
        try:
            rc, _, _, trunc, _ = self.sandbox.command(*self.sandbox.create_command(name, inputs))
            if rc or trunc:
                raise SandboxError("创建受限容器失败")
            created = True
            self.environment = self.sandbox.check_container(name)
            rc, *_ = self.sandbox.command("start", name)
            if rc:
                raise SandboxError("启动受限容器失败")
            payload = None
            report_text = ""
            while time.monotonic() - started < self.timeout:
                rc, text, _, report_truncated, _ = self.sandbox.command(
                    "exec", name, "/usr/local/bin/python", "-I", "-c",
                    "import pathlib,sys; p=pathlib.Path('/tmp/report.json'); sys.stdout.buffer.write(p.read_bytes()) if p.is_file() else sys.exit(3)",
                    timeout=max(.1, self.timeout - (time.monotonic() - started)), limit=REPORT_LIMIT)
                if rc == 0:
                    if report_truncated:
                        raise SandboxError("机器报告超过大小限制")
                    report_text = text
                    payload = json.loads(text)
                    break
                rc2, running, *_ = self.sandbox.command("inspect", "--format={{.State.Running}}", name,
                    timeout=max(.1, self.timeout - (time.monotonic() - started)))
                if rc2 or running.strip() != "true":
                    raise SandboxError("测试容器或采集器异常退出")
                time.sleep(.1)
            # Stop the whole execution boundary before retrieving logs; descendants cannot outlive the deadline.
            rc, *_ = self.sandbox.command("kill", name, timeout=3)
            if rc:
                raise SandboxError("执行容器终止失败")
            _, stdout, stderr, trunc, _ = self.sandbox.command("logs", name)
            (directory / "stdout.txt").write_text(stdout, encoding="utf-8")
            (directory / "stderr.txt").write_text(stderr, encoding="utf-8")
            if payload is None:
                result = TestResult(**base, status="error", failure_kind="timeout", stdout=stdout,
                                    stderr=stderr, truncated=trunc)
            else:
                (directory / "report.json").write_text(report_text, encoding="utf-8")
                if payload.get("python") != "3.12.8" or payload.get("pytest") != "8.3.5":
                    raise SandboxError("固定 Python/pytest 版本不符")
                result = parse_report(payload, **base, stdout=stdout, stderr=stderr, truncated=trunc)
        except (SandboxError, OSError, ValueError):
            kind = "timeout" if time.monotonic() - started >= self.timeout else "environment"
            result = TestResult(**base, status="error", failure_kind=kind)
        finally:
            if created:
                try:
                    self.sandbox.cleanup(name)
                except (SandboxError, OSError):
                    result = TestResult(**base, status="error", failure_kind="environment",
                                        stderr="容器清理失败，自动执行已停止")
                    self.ready = False
        result.duration_ms = int((time.monotonic() - started) * 1000)
        return result
