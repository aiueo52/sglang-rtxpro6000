# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

import asyncio
import pickle
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CREDENTIALS = ("admin_api_key", "api_key", "ssl_keyfile_password")


class TestDumpRedaction(unittest.TestCase):
    def _check_dumps(self, credentials):
        context = get_context()
        argv = [
            "sglang.launch_server",
            "--model-path",
            "dummy",
            "--admin-api-key",
            "dump-test-secret-argv-admin",
            "--api-key=dump-test-secret-argv-api",
            "--port",
            "30000",
        ]
        original_argv = argv.copy()
        secrets = [value for value in credentials.values() if value is not None]
        secrets.extend((argv[4], argv[5].split("=", 1)[1]))
        with (
            tempfile.TemporaryDirectory() as folder,
            context.override_server_args(device="cpu", **credentials) as server_args,
            patch.object(sys, "argv", argv),
            envs.SGLANG_PYSPY_DUMP_BEFORE_CRASH.override(False),
            envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH.override(False),
        ):
            original_args = pickle.dumps(server_args)

            manager = object.__new__(TokenizerManager)
            manager.server_args = server_args
            manager.model_path = "dummy"
            manager.served_model_name = "dummy"
            manager.dump_requests_folder = str(Path(folder) / "requests")
            manager.dump_requests_threshold = 1
            manager.dump_requests_exclude_meta_keys = []
            manager.dump_request_list = []
            manager.crash_dump_folder = str(Path(folder) / "crashes")
            manager.crash_dump_request_list = deque()
            manager.crash_dump_performed = False
            manager.rid_to_state = {}
            state = SimpleNamespace(
                obj={"text": "nonsecret request"},
                time_stats=SimpleNamespace(created_time=1.0, finished_time=2.0),
            )
            output = {"text": "nonsecret response"}

            async def write_requests():
                manager.dump_requests(state, output)
                # This isolated loop contains only the real background writer.
                await asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}))

            asyncio.run(write_requests())
            manager.record_request_for_crash_dump(state, output)
            manager.dump_requests_before_crash(hostname="cpu-test")

            request_files = list((Path(folder) / "requests").glob("*.pkl"))
            crash_files = list((Path(folder) / "crashes" / "cpu-test").glob("*.pkl"))
            self.assertEqual(len(request_files), 1)
            self.assertEqual(len(crash_files), 1)
            for filename in request_files + crash_files:
                with self.subTest(dump=filename.parent.name):
                    raw = filename.read_bytes()
                    for secret in secrets:
                        self.assertNotIn(secret.encode(), raw)
                    dumped = pickle.loads(raw)
                    self.assertEqual(len(dumped["requests"]), 1)
                    self.assertEqual(dumped["requests"][0][:2], (state.obj, output))
                    for key in ("server_args", "resolved_config"):
                        snapshot = dumped[key]
                        self.assertIsNotNone(snapshot)
                        for name, value in credentials.items():
                            actual = (
                                snapshot[name]
                                if isinstance(snapshot, dict)
                                else getattr(snapshot, name)
                            )
                            self.assertEqual(
                                actual, None if value is None else "<redacted>"
                            )
                    if filename in crash_files:
                        command = dumped["launch_command"]
                        self.assertIn("--admin-api-key", command)
                        self.assertIn("--api-key=", command)
                        self.assertIn("--model-path dummy", command)
                        self.assertIn("--port 30000", command)

            self.assertEqual(pickle.dumps(server_args), original_args)
            self.assertEqual(sys.argv, original_argv)
            for name, value in credentials.items():
                self.assertEqual(getattr(server_args, name), value)
                self.assertEqual(getattr(context.config_bag("serving"), name), value)

    def test_set_credentials(self):
        self._check_dumps({name: f"dump-test-secret-{name}" for name in CREDENTIALS})

    def test_unset_credentials(self):
        self._check_dumps({name: None for name in CREDENTIALS})


if __name__ == "__main__":
    unittest.main()
