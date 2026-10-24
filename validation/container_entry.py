import os
import subprocess
import sys
import time

# The parent remains alive so the host can retrieve tmpfs report before cleanup.
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        import pytest
        from collector import Collector
        code = pytest.main(["-q", "-c", "/opt/empty.ini", "--rootdir=/input", "--confcutdir=/input",
                            "-p", "no:cacheprovider", "--assert=plain", "--tb=long", "/input/test_solution.py"],
                           plugins=[Collector()])
        sys.exit(code)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": "/input:/opt",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "HOME": "/tmp"}
    subprocess.run([sys.executable, "/opt/container_entry.py", "child"], env=env, check=False)
    # Missing report is a system error, signalled independently of pytest output.
    if not os.path.isfile("/tmp/report.json"):
        sys.exit(70)
    while True:
        time.sleep(1)
