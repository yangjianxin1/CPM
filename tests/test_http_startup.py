"""Offline startup contract tests; no ML packages, model files, or sockets required."""
import ast
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get("CPM_SOURCE_ROOT", str(Path(__file__).resolve().parents[1])))


class HttpStartupTests(unittest.TestCase):
    def startup(self, argv=(), environ=None):
        app = types.SimpleNamespace(config={})
        app.route = lambda *a, **kw: lambda fn: fn
        runs = []
        app.run = lambda **kw: runs.append(kw)
        model = types.SimpleNamespace(eval=lambda: None)
        model.to = lambda device: model
        tokenizer = types.SimpleNamespace(
            convert_tokens_to_ids=lambda token: 0,
            sep_token_id=1, unk_token_id=2)
        cuda = types.SimpleNamespace(is_available=lambda: False)
        torch = types.ModuleType("torch")
        torch.cuda = cuda
        nn = types.ModuleType("torch.nn")
        functional = types.ModuleType("torch.nn.functional")
        torch.nn = nn
        nn.functional = functional
        transformers = types.ModuleType("transformers")
        transformers.GPT2LMHeadModel = types.SimpleNamespace(from_pretrained=lambda path: model)
        transformers.CpmTokenizer = lambda **kw: tokenizer
        utils = types.ModuleType("utils")
        utils.top_k_top_p_filtering = lambda *a, **kw: None
        utils.set_logger = lambda path: types.SimpleNamespace(info=lambda *a: None)
        flask = types.ModuleType("flask")
        flask.Flask = lambda name: app
        flask.redirect = flask.url_for = flask.request = None
        tqdm = types.ModuleType("tqdm")
        tqdm.trange = range
        modules = {"torch": torch, "torch.nn": nn, "torch.nn.functional": functional,
                   "transformers": transformers, "utils": utils, "flask": flask, "tqdm": tqdm}
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["http_service.py"] + list(argv)):
            with patch.dict(os.environ, environ or {}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                state = runpy.run_path(str(ROOT / "http_service.py"), run_name="__main__")
        self.assertEqual(len(runs), 1)
        return runs[0], state

    def test_default_is_loopback(self):
        run, _ = self.startup()
        self.assertEqual(run["host"], "127.0.0.1")
        self.assertEqual(run["port"], 8085)

    def test_debugger_and_reloader_disabled(self):
        run, _ = self.startup()
        self.assertIs(run.get("debug"), False)
        self.assertIs(run.get("use_debugger"), False)
        self.assertIs(run.get("use_reloader"), False)

    def test_explicit_host_and_existing_arguments(self):
        run, state = self.startup(["--host", "0.0.0.0", "--port", "9090", "--context_len", "123", "--no_cuda",
                                   "--device", "0", "--temperature", "0.7", "--topk", "4", "--topp", "0.9",
                                   "--model_path", "test-model", "--log_path", "test.log"])
        self.assertEqual((run["host"], run["port"]), ("0.0.0.0", 9090))
        args = state["args"]
        self.assertEqual((args.context_len, args.device, args.temperature, args.topk, args.topp), (123, "0", 0.7, 4, 0.9))
        self.assertEqual((args.model_path, args.log_path), ("test-model", "test.log"))
        self.assertFalse(args.cuda)

    def test_environment_cannot_enable_debug(self):
        run, _ = self.startup(environ={"FLASK_DEBUG": "1", "FLASK_ENV": "development"})
        self.assertIs(run.get("debug"), False)
        self.assertIs(run.get("use_debugger"), False)
        self.assertIs(run.get("use_reloader"), False)

    def test_original_port_argument_still_works(self):
        run, _ = self.startup(["--port", "8086"])
        self.assertEqual(run["port"], 8086)

    def test_help_exits_normally(self):
        with self.assertRaises(SystemExit) as result:
            self.startup(["--help"])
        self.assertEqual(result.exception.code, 0)

    def test_no_debug_opt_in_was_added(self):
        with self.assertRaises(SystemExit) as result:
            self.startup(["--debug"])
        self.assertEqual(result.exception.code, 2)

    def test_python36_syntax(self):
        for name in ("http_service.py", "tests/test_http_startup.py"):
            source = (ROOT / name) if (ROOT / name).exists() else Path(__file__)
            kwargs = {"feature_version": (3, 6)} if sys.version_info >= (3, 8) else {}
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source), **kwargs)


if __name__ == "__main__":
    unittest.main()
