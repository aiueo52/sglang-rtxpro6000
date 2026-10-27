# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.srt.arg_groups.overrides import (
    _NONSECRET_TOKEN_OPTIONS,
    _sensitive_argv_options,
    redacted_argv,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


SUFFIXES = (
    "key", "token", "secret", "password", "passwd", "auth", "cred", "creds",
    "credential", "credentials",
)
LAUNCHER = Path(__file__).resolve().parents[3] / "serve-local.sh"


class TestArgvRedaction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        cls.parser = parser
        options = {
            option.lstrip("-").lower().replace("_", "-")
            for action in parser._actions
            for option in action.option_strings
            if option.startswith("--")
        }
        cls.sensitive = {
            name
            for name in options - _NONSECRET_TOKEN_OPTIONS
            if name.endswith(SUFFIXES)
        }
        cls.nonsensitive_prefixes = {
            name
            for name in options - cls.sensitive
            if any(option.startswith(name) for option in cls.sensitive)
        }

    def check_both(self, args, expected, admin_api_key="test-admin-secret"):
        original = list(args)
        self.assertEqual(redacted_argv(args), expected)
        self.assertEqual(args, original)
        with tempfile.TemporaryDirectory() as folder:
            env = {
                "PATH": os.defpath,
                "HOME": folder,
                "XDG_CACHE_HOME": str(Path(folder) / "cache"),
                "TARGET_MODEL": folder,
                "PYTHON": sys.executable,
                "FLASH_NEXT_ADMIN_API_KEY": admin_api_key,
                "SGLANG_TEST_TOKEN": "test-env-secret",
            }
            result = subprocess.run(
                ["bash", str(LAUNCHER), "--print", *args],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        printed = json.loads(result.stdout)
        self.assertEqual(printed["argv"][-len(args) - 1 : -1], expected)
        self.assertEqual(printed["argv"][-1:], ["--admin-api-key=<redacted>"])
        self.assertEqual(printed["env"]["SGLANG_TEST_TOKEN"], "<redacted>")
        self.assertNotIn(admin_api_key, result.stdout + result.stderr)
        self.assertNotIn("test-env-secret", result.stdout + result.stderr)
        for arg, redacted in zip(args, expected):
            if arg != redacted:
                secret = arg if redacted == "<redacted>" else arg.partition("=")[2]
                if secret == "--" or secret in printed["argv"][:-len(args) - 1]:
                    # Structural tokens can occur in the launcher's own arguments.
                    self.assertNotIn(secret, printed["argv"][-len(args) - 1 : -1])
                    self.assertNotIn(secret, result.stderr)
                else:
                    self.assertNotIn(secret, result.stdout + result.stderr)

    def check_hidden_flags(self, flags):
        args, expected = [], []
        for index, flag in enumerate(flags):
            args.extend((flag, f"hidden-split-{index}", f"{flag}=hidden-joined-{index}"))
            expected.extend((flag, "<redacted>", f"{flag}=<redacted>"))
        self.check_both(args, expected)

    def test_embedded_lists_match_real_parser(self):
        self.assertEqual(
            _NONSECRET_TOKEN_OPTIONS,
            frozenset({
                "bucket-time-to-first-token", "kt-max-deferred-experts-per-token",
            }),
        )
        self.assertEqual(
            _sensitive_argv_options(), (self.sensitive, self.nonsensitive_prefixes)
        )
        # Parse only the isolated --print Python, without importing it.
        code = LAUNCHER.read_text().split("-I -B -c '\n", 1)[1]
        code = code.rsplit("\n' \"${CMD[@]}\"", 1)[0]
        embedded = {
            node.targets[0].id: (
                frozenset(ast.literal_eval(node.value.args[0]))
                if isinstance(node.value, ast.Call)
                else ast.literal_eval(node.value)
            )
            for node in ast.parse(code).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in (
                "_NONSECRET_TOKEN_OPTIONS", "sensitive_options", "nonsensitive_prefixes",
            )
        }
        self.assertEqual(embedded["_NONSECRET_TOKEN_OPTIONS"], _NONSECRET_TOKEN_OPTIONS)
        self.assertEqual(embedded["sensitive_options"], self.sensitive)
        self.assertEqual(embedded["nonsensitive_prefixes"], self.nonsensitive_prefixes)

    def test_required_abbreviations(self):
        self.check_hidden_flags(
            ("--api-k", "--ssl-keyfile-pass", "--admin-api", "--hf-tok", "--apikey")
        )

    def test_sensitive_option_boundaries(self):
        cases = (
            (["--api-k"], ["--api-k"]),
            (
                ["--api-k", "--admin-api-key", "direct-admin-secret"],
                ["--api-k", "--admin-api-key", "<redacted>"],
            ),
            (["--api-k", "-12345"], ["--api-k", "<redacted>"]),
            (
                ["--admin-api-key", "-FAKE"],
                ["--admin-api-key", "<redacted>"],
            ),
            (["--api-key", "-a b"], ["--api-key", "<redacted>"]),
            (
                ["--api-k", "--", "KEY"],
                ["--api-k", "<redacted>", "<redacted>"],
            ),
            (["--admin-api-key=-FAKE"], ["--admin-api-key=<redacted>"]),
            (["--api-key", "-dashkey"], ["--api-key", "<redacted>"]),
            (
                ["--api-k", "--service-key", "chained-secret"],
                ["--api-k", "<redacted>", "<redacted>"],
            ),
            (
                ["--api-k", "--admin-api-key", "KEY"],
                ["--api-k", "--admin-api-key", "<redacted>"],
            ),
            (
                ["--ssl-keyfile-pass", "--port", "8000"],
                ["--ssl-keyfile-pass", "<redacted>", "8000"],
            ),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                self.check_both(args, expected)

    def test_dash_prefixed_admin_env_key(self):
        secret = "-FAKE" + os.urandom(16).hex()
        self.check_both([], [], admin_api_key=secret)
        parsed = self.parser.parse_args(
            ["--model-path", "dummy", f"--admin-api-key={secret}"]
        )
        self.assertEqual(parsed.admin_api_key, secret)

    def test_all_parser_sensitive_prefixes(self):
        flags = sorted({
            "--" + name[:length]
            for name in self.sensitive
            for length in range(1, len(name) + 1)
            if name[:length] not in self.nonsensitive_prefixes
        })
        self.check_hidden_flags(flags)

    def test_suffix_and_prefix_normalization(self):
        flags = [
            prefix + suffix.upper()
            for prefix in ("--", "---SeRvIcE_", "-service-")
            for suffix in SUFFIXES
        ]
        flags.extend(("--API_K", "--SSL_KEYFILE_PASS", "--ADMIN_API", "--HF_TOK"))
        self.check_hidden_flags(flags)

    def test_nonsensitive_values_stay_visible(self):
        pairs = (
            ("--ssl-keyfile", "/tmp/ssl-key.pem"),
            ("--SSL_KEYFILE", "/tmp/other-key.pem"),
            ("--speculative-num-draft-tokens", "8"),
            ("--speculative-token-map", "/tmp/token-map.json"),
            ("--bucket-time-to-first-token", "0.5"),
            ("--kt-max-deferred-experts-per-token", "4"),
            ("--tokenizer-path", "/tmp/tokenizer"),
            ("--key-file", "/tmp/key-file"),
            ("--secret-config", "/tmp/secret-config"),
            ("--password-file", "/tmp/password-file"),
            ("--auth-backend", "backend"),
            ("--credentials-file", "/tmp/credentials-file"),
        )
        args = ["python3", "-P", "-B", "/tmp/server.py", "serve"]
        args.extend(arg for flag, value in pairs for arg in (flag, value, f"{flag}={value}"))
        self.check_both(args, args.copy())


if __name__ == "__main__":
    unittest.main()
