# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""CPU auth checks using server functions and decorators without importing workers."""

import __future__
import ast
import importlib.util
import itertools
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import Depends, FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"
SERVER = SRT / "entrypoints/http_server.py"
TREE = ast.parse(SERVER.read_text())
FUNCTIONS = {
    node.name: node
    for node in TREE.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
ERROR = "authentication with multiple tokenizer workers is not supported in this release"
ADMIN_ROUTES = (
    "load_lora_adapter_from_tensors",
    "load_lora_adapter",
    "unload_lora_adapter",
    "update_weights_from_tensor",
    "update_weights_from_disk",
    "update_weights_from_distributed",
    "update_weights_from_ipc",
    "update_weight_version",
)
# Inference routes keep upstream authentication: open unless --api-key is set.
INFERENCE_ROUTES = (
    "generate_request",
    "encode_request",
    "classify_request",
    "openai_v1_completions",
    "openai_v1_chat_completions",
    "openai_v1_embeddings",
    "anthropic_v1_messages",
    "sagemaker_chat_completions",
    "vertex_generate",
)

# Load the lightweight auth module without importing the runtime package.
spec = importlib.util.spec_from_file_location("_http_server_auth_under_test", SRT / "utils/auth.py")
auth = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = auth
spec.loader.exec_module(auth)


def server_function(name, namespace=None):
    namespace = {} if namespace is None else namespace
    module = ast.Module(body=[FUNCTIONS[name]], type_ignores=[])
    exec(
        compile(module, str(SERVER), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace[name]


def route_app(api_key, admin_api_key, names=ADMIN_ROUTES):
    app = FastAPI()
    calls = []
    namespace = {
        "app": app,
        "auth_level": auth.auth_level,
        "AuthLevel": auth.AuthLevel,
        "Depends": Depends,
        "validate_json_request": lambda: None,
        "ORJSONResponse": ORJSONResponse,
        "SGLangORJSONResponse": ORJSONResponse,
        "os": os,
    }
    routes = []
    for name in names:

        async def endpoint():
            calls.append(True)
            return {"ok": True}

        endpoint.__name__ = name
        for decorator in reversed(FUNCTIONS[name].decorator_list):
            apply = eval(
                compile(ast.Expression(decorator), str(SERVER), "eval"), namespace
            )
            endpoint = apply(endpoint)
        route = app.routes[-1]
        routes.extend((method, route.path) for method in sorted(route.methods))
    auth.add_api_key_middleware(app, api_key=api_key, admin_api_key=admin_api_key)
    return app, calls, routes


class TestTokenizerWorkerAuth(unittest.TestCase):
    def test_set_and_unset_keys(self):
        validate = server_function("_validate_tokenizer_worker_auth")
        for workers, api_key, admin_api_key in itertools.product(
            (1, 2, 4), (None, "", "user"), (None, "", "admin")
        ):
            with self.subTest(
                workers=workers, api_key=api_key, admin_api_key=admin_api_key
            ):
                args = SimpleNamespace(
                    tokenizer_worker_num=workers,
                    api_key=api_key,
                    admin_api_key=admin_api_key,
                )
                if workers > 1 and (api_key is not None or admin_api_key is not None):
                    with self.assertRaises(ValueError) as error:
                        validate(args)
                    self.assertEqual(str(error.exception), ERROR)
                else:
                    self.assertIsNone(validate(args))

    def test_launch_rejects_before_starting_workers(self):
        launch_subprocesses = Mock(side_effect=AssertionError("workers must not start"))
        namespace = {
            "Engine": SimpleNamespace(_launch_subprocesses=launch_subprocesses),
            "_validate_tokenizer_worker_auth": server_function(
                "_validate_tokenizer_worker_auth"
            ),
            "init_tokenizer_manager": None,
            "run_scheduler_process": None,
            "run_detokenizer_process": None,
            "_execute_server_warmup": None,
        }
        launch = server_function("launch_server", namespace)
        for api_key, admin_api_key in (
            ("user", None),
            (None, "admin"),
            ("user", "admin"),
            ("", None),
            (None, ""),
        ):
            with self.subTest(api_key=api_key, admin_api_key=admin_api_key):
                args = SimpleNamespace(
                    tokenizer_worker_num=2, api_key=api_key, admin_api_key=admin_api_key
                )
                with self.assertRaises(ValueError) as error:
                    launch(args)
                self.assertEqual(str(error.exception), ERROR)
                launch_subprocesses.assert_not_called()


class TestRouteAuth(unittest.TestCase):
    def test_admin_key_required_when_set(self):
        for api_key in (None, "user"):
            app, calls, routes = route_app(api_key, "admin")
            with TestClient(app) as client:
                for method, path in routes:
                    with self.subTest(api_key=api_key, method=method, path=path):
                        calls.clear()
                        for token in (None, "user", "wrong"):
                            headers = (
                                {} if token is None else {"Authorization": f"Bearer {token}"}
                            )
                            self.assertEqual(
                                client.request(method, path, headers=headers).status_code,
                                401,
                            )
                        self.assertEqual(calls, [])
                        response = client.request(
                            method, path, headers={"Authorization": "Bearer admin"}
                        )
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(calls, [True])

    def test_inference_routes_keep_upstream_auth(self):
        for api_key in (None, "user"):
            app, calls, routes = route_app(api_key, "admin", INFERENCE_ROUTES)
            self.assertEqual(len({path for _, path in routes}), len(INFERENCE_ROUTES))
            with TestClient(app) as client:
                for method, path in routes:
                    with self.subTest(api_key=api_key, method=method, path=path):
                        calls.clear()
                        self.assertEqual(
                            client.request(method, path).status_code,
                            401 if api_key else 200,
                        )
                        if api_key:
                            response = client.request(
                                method, path, headers={"Authorization": "Bearer user"}
                            )
                            self.assertEqual(response.status_code, 200)
                        self.assertEqual(calls, [True])

    def test_no_admin_key_preserves_legacy_access(self):
        for api_key in (None, "user"):
            app, _, routes = route_app(api_key, None)
            with TestClient(app) as client:
                for method, path in routes:
                    with self.subTest(api_key=api_key, method=method, path=path):
                        self.assertEqual(
                            client.request(method, path).status_code,
                            401 if api_key else 200,
                        )
                        headers = (
                            {} if api_key is None else {"Authorization": "Bearer user"}
                        )
                        self.assertEqual(
                            client.request(method, path, headers=headers).status_code,
                            200,
                        )


if __name__ == "__main__":
    unittest.main()
