import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

IMAGE = "codeflow-test:1.0"
LOG_LIMIT = 65536
REPORT_LIMIT = 2 * 1024 * 1024


def find_docker():
    executable = shutil.which("docker")
    if executable:
        return executable
    # Desktop apps can retain the PATH from before Docker was installed.
    candidates = [Path.home() / ".docker" / "bin" / "docker",
                  Path("/Applications/Docker.app/Contents/Resources/bin/docker")]
    return next((str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


class SandboxError(RuntimeError):
    pass


def bounded_process(command, timeout=15, limit=LOG_LIMIT):
    """Drain both pipes while retaining only bounded bytes (no unbounded communicate)."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    buffers = [bytearray(), bytearray()]
    truncated = [False, False]

    def read(pipe, index):
        while chunk := pipe.read(8192):
            left = limit - len(buffers[index])
            buffers[index].extend(chunk[:max(0, left)])
            if len(chunk) > left:
                truncated[index] = True
        pipe.close()

    threads = [threading.Thread(target=read, args=(pipe, i), daemon=True)
               for i, pipe in enumerate((process.stdout, process.stderr))]
    for t in threads:
        t.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
    for t in threads:
        t.join(timeout=2)
    return process.returncode, *(bytes(b).decode("utf-8", "replace") for b in buffers), any(truncated), timed_out


class Sandbox:
    def __init__(self):
        self.docker = find_docker()
        self.image_id = ""

    def command(self, *args, timeout=10, limit=LOG_LIMIT):
        if not self.docker:
            raise SandboxError("未找到 Docker；不会在宿主机执行输入")
        result = bounded_process([self.docker, *args], timeout, limit)
        if result[4]:
            raise SandboxError("Docker 控制命令超时")
        return result

    def inspect_image(self):
        rc, out, _, trunc, _ = self.command("image", "inspect", IMAGE)
        if rc or trunc:
            raise SandboxError("固定测试镜像不存在；请显式构建镜像")
        image = json.loads(out)[0]
        if image["Config"].get("Labels", {}).get("codeflow.collector") != "1":
            raise SandboxError("测试镜像采集器标识不符")
        self.image_id = image["Id"]
        return self.image_id

    def create_command(self, name, directory):
        return ["create", "--pull=never", "--name", name, "--network=none", "--read-only",
                "--user=65532:65532", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                "--cpus=1", "--memory=512m", "--memory-swap=512m", "--pids-limit=64",
                "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777", "--ulimit", "nofile=256:256",
                "--log-driver=local", "--log-opt=max-size=1m", "--log-opt=max-file=1", "--log-opt=compress=false",
                "--mount", f"type=bind,source={Path(directory).resolve()},target=/input,readonly",
                "--workdir=/input", self.image_id]

    def check_container(self, name):
        rc, out, _, trunc, _ = self.command("inspect", name)
        if rc or trunc:
            raise SandboxError("无法核对容器限制")
        data = json.loads(out)[0]
        h, c = data["HostConfig"], data["Config"]
        mounts = data["Mounts"]
        allowed = {"PATH", "LANG", "GPG_KEY", "PYTHON_VERSION", "PYTHON_SHA256",
                   "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "HOME"}
        if (not h["ReadonlyRootfs"] or h["NetworkMode"] != "none" or h["Privileged"]
                or h["Memory"] != 512 * 1024 * 1024 or h["MemorySwap"] != h["Memory"]
                or h["NanoCpus"] != 10**9 or h["PidsLimit"] != 64
                or not all(x in h.get("Tmpfs", {}).get("/tmp", "") for x in ("noexec", "nosuid", "nodev", "size=64m"))
                or c["User"] != "65532:65532" or "ALL" not in h["CapDrop"]
                or not any(x in ("no-new-privileges:true", "no-new-privileges") for x in h["SecurityOpt"])
                or len([m for m in mounts if m["Type"] == "bind"]) != 1
                or any(m["RW"] or m["Destination"] != "/input" for m in mounts if m["Type"] == "bind")
                or not all(e.split("=", 1)[0] in allowed for e in c["Env"])):
            raise SandboxError("容器安全边界未生效或镜像环境包含未允许变量")
        return {"image_id": self.image_id, "user": c["User"], "network": h["NetworkMode"],
                "memory": h["Memory"], "nano_cpus": h["NanoCpus"], "pids_limit": h["PidsLimit"],
                "read_only": h["ReadonlyRootfs"], "tmpfs": h["Tmpfs"]}

    def cleanup(self, name):
        rc, *_ = self.command("rm", "-f", name)
        if rc:
            raise SandboxError("执行容器清理失败，需要检查 Docker")
