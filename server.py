# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Serve from this source tree, including compatibility in spawned schedulers."""
import importlib.util
import os
from pathlib import Path
import sys


# Multiprocessing imports the original entry as __mp_main__ in each scheduler.
# Tokenization and the public API are unchanged; there is no ID-only entry shim.
if __name__ in ("__main__", "__mp_main__") and os.environ.get("SGLANG_TACTIC_ROWS", "0") != "0":
    _spec = importlib.util.spec_from_file_location(
        "_flash_next_tr_compat", Path(__file__).resolve().with_name("tr_compat.py"))
    _compat = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_compat)
    _compat.install()

if __name__ == "__main__":
    from sglang.cli.main import main
    sys.argv[0] = "sglang"
    sys.exit(main())
