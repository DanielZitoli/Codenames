"""The website installs codemaster without the `pipeline` extra (root CLAUDE.md boundary
#2), so the online modules must never pull in torch, transformers, or `codemaster.pipeline`
itself. This runs as a subprocess so prior test imports in the same process can't taint it.
"""

import subprocess
import sys

FORBIDDEN = ("torch", "transformers", "codemaster.pipeline")

_CHECK_SCRIPT = """
import sys
import codemaster.engine
import codemaster.agents
import codemaster.eval

forbidden = {forbidden!r}
leaked = [m for m in sys.modules if m in forbidden or any(
    m == f or m.startswith(f + ".") for f in forbidden
)]
if leaked:
    print("LEAKED:" + ",".join(sorted(leaked)))
    sys.exit(1)
print("OK")
""".format(forbidden=FORBIDDEN)


def test_online_modules_do_not_import_pipeline_or_torch():
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"online modules pulled in a forbidden dependency: {result.stdout} {result.stderr}"
    )
    assert result.stdout.strip() == "OK"
