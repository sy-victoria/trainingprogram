import shutil
import tempfile
from pathlib import Path
from .common import ROOT, read_json, write_json
from .sandbox import get_sandbox


def verify_agent(directory):
    directory = Path(directory)
    config = read_json(directory / "config.json")
    with tempfile.TemporaryDirectory(prefix="tp-acceptance-") as tmp:
        work = Path(tmp) / "workspace"
        shutil.copytree(directory / "workspace", work)
        # Restore pristine public tests and add tests never shown to the agent.
        shutil.rmtree(work / "tests", ignore_errors=True)
        shutil.copytree(ROOT / "tasks" / config["task"] / "workspace/tests", work / "tests")
        shutil.copy2(ROOT / "tasks" / config["task"] / "acceptance.py", work / "tests/test_acceptance.py")
        result = get_sandbox().run(work, ["python", "-m", "unittest", "discover", "-s", "tests", "-v"], timeout=60)
    result["passed"] = result["returncode"] == 0 and not result["timeout"]
    write_json(directory / "acceptance.json", result)
    return result
