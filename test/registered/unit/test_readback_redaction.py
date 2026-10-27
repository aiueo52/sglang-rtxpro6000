# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.entrypoints import engine, http_server
from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CREDENTIALS = ("admin_api_key", "api_key", "ssl_keyfile_password")


class TestReadbackRedaction(unittest.TestCase):
    def _check_readbacks(self, credentials):
        context = get_context()
        with context.override_server_args(device="cpu", **credentials) as server_args:
            self.assertIsInstance(server_args, ServerArgs)
            expected = {
                name: None if value is None else "<redacted>"
                for name, value in credentials.items()
            }

            def check(readback):
                for name, value in expected.items():
                    self.assertEqual(readback[name], value)
                encoded = json.dumps(readback, default=str)
                for value in credentials.values():
                    if value:
                        self.assertNotIn(value, encoded)
                self.assertEqual(readback["model_path"], "dummy")

            check(server_args.resolved_dict())
            internal_state = context.resolved_server_args_dict()
            check(internal_state)

            manager = object.__new__(TokenizerManager)
            manager.server_args = server_args
            manager.startup_time = {}
            manager.model_path = "dummy"
            manager.served_model_name = "dummy"
            manager.get_internal_state = AsyncMock(return_value=[internal_state])
            state = SimpleNamespace(tokenizer_manager=manager, scheduler_info={})
            with patch.object(http_server, "_global_state", state):
                for route in (http_server.server_info, http_server.get_server_info):
                    with self.subTest(route=route.__name__):
                        response = asyncio.run(route())
                        check(response)
                        check(response["internal_states"][0])

            bridge = object.__new__(RuntimeHandle)
            bridge.tokenizer_manager = manager
            bridge.scheduler_info = {}
            check(json.loads(bridge.get_server_info()))

            loop = asyncio.new_event_loop()
            try:
                offline_engine = SimpleNamespace(
                    loop=loop,
                    tokenizer_manager=manager,
                    _scheduler_init_result=SimpleNamespace(scheduler_infos=[{}]),
                )
                check(engine.Engine.get_server_info(offline_engine))
            finally:
                loop.close()

            check(manager._dump_config_snapshot())

            # This real constructor logs before rejecting Rust-server mode,
            # before atexit registration, port allocation or process startup.
            with (
                patch.object(engine, "load_plugins"),
                engine.envs.SGLANG_RUST_SERVER.override(True),
                self.assertLogs(engine.logger, level="INFO") as captured,
            ):
                with self.assertRaisesRegex(ValueError, "SGLANG_RUST_SERVER"):
                    engine.Engine(server_args=server_args)
            messages = [record.getMessage() for record in captured.records]
            log_lines = [line for line in messages if line.startswith("server_args=")]
            self.assertEqual(len(log_lines), 1)
            for name, value in expected.items():
                self.assertIn(f"'{name}': {value!r}", log_lines[0])
            for value in credentials.values():
                if value:
                    self.assertNotIn(value, log_lines[0])

            # Readbacks must not replace credentials used by auth or TLS.
            for name, value in credentials.items():
                self.assertEqual(getattr(server_args, name), value)
                self.assertEqual(getattr(context.config_bag("serving"), name), value)

    def test_set_credentials(self):
        self._check_readbacks(
            {name: f"release-test-secret-{name}" for name in CREDENTIALS}
        )

    def test_unset_credentials(self):
        self._check_readbacks({name: None for name in CREDENTIALS})

    def test_empty_credentials_are_not_none(self):
        self._check_readbacks({name: "" for name in CREDENTIALS})


if __name__ == "__main__":
    unittest.main()
