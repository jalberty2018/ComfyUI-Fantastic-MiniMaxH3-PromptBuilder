import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "minimax_h3_security_tests"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, data, status=200, headers=None):
        self.data = data
        self.status = status
        self.headers = headers or {}
        self.method = "POST"
        self.path = "/test"


class FakeRoutes:
    def __init__(self):
        self.gets = {}
        self.posts = {}

    def _register(self, collection, path):
        def decorator(handler):
            collection[path] = handler
            return handler
        return decorator

    def get(self, path):
        return self._register(self.gets, path)

    def post(self, path):
        return self._register(self.posts, path)


class FakeRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.method = "POST"
        self.path = "/test"


package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
sys.modules[f"{PACKAGE}.media_io"] = types.ModuleType(f"{PACKAGE}.media_io")

routes = FakeRoutes()
server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(
    instance=types.SimpleNamespace(routes=routes))
sys.modules["server"] = server

aiohttp = types.ModuleType("aiohttp")
aiohttp.web = types.SimpleNamespace(
    json_response=lambda data, status=200, headers=None:
        FakeResponse(data, status, headers))
sys.modules["aiohttp"] = aiohttp

nodes = load_module(f"{PACKAGE}.nodes", ROOT / "nodes.py")
web_api = load_module(f"{PACKAGE}.web_api", ROOT / "web_api.py")


class PrefixContainmentTests(unittest.TestCase):
    def test_rejects_parent_traversal_in_any_prefix_part(self):
        with self.assertRaisesRegex(ValueError, "stay inside"):
            nodes.MiniMaxH3FilenamePrefix().build(
                "renders", "../outside", "off", "clip")

    def test_normalizes_absolute_and_windows_style_prefixes(self):
        self.assertEqual(nodes._contain_prefix("/renders/clip"), "renders/clip")
        self.assertEqual(nodes._contain_prefix(r"C:\renders\clip"), "renders/clip")


class RouteTokenTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_is_same_origin_only_and_not_cached(self):
        handler = routes.gets["/minimax_h3/token"]
        response = await handler(FakeRequest({"Sec-Fetch-Site": "same-origin"}))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.data["token"], web_api._TOKEN)
        self.assertEqual(response.headers["Cache-Control"], "no-store")

        for site in ("cross-site", "same-site"):
            with self.subTest(site=site):
                response = await handler(FakeRequest({"Sec-Fetch-Site": site}))
                self.assertEqual(response.status, 403)

    async def test_every_post_route_requires_the_session_token(self):
        self.assertGreater(len(routes.posts), 0)
        for path, handler in routes.posts.items():
            with self.subTest(path=path):
                response = await handler(FakeRequest())
                self.assertEqual(response.status, 403)
                self.assertTrue(response.data.get("token_required"))

    async def test_stale_token_is_rejected(self):
        handler = routes.posts["/minimax_h3/upload"]
        response = await handler(FakeRequest({web_api.TOKEN_HEADER: "stale"}))
        self.assertEqual(response.status, 403)
        self.assertTrue(response.data["token_required"])



class RefModDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_download_is_attachment_and_stays_in_library(self):
        import tempfile
        from unittest.mock import patch
        refmods = load_module(f"{PACKAGE}.refmods", ROOT / "refmods.py")
        handler = routes.gets["/minimax_h3/refmods/download"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "library"
            root.mkdir()
            (root / "hero.safetensors").write_bytes(b"refmod")
            (Path(directory) / "outside.safetensors").write_bytes(b"private")
            with patch.object(refmods, "search_dirs", return_value=[str(root)]), patch.object(
                web_api.web, "FileResponse", create=True,
                side_effect=lambda path, headers: FakeResponse(path, headers=headers)
            ):
                request = FakeRequest()
                request.query = {"name": "hero.safetensors"}
                response = await handler(request)
                self.assertEqual(response.status, 200)
                self.assertEqual(Path(response.data), root / "hero.safetensors")
                self.assertTrue(response.headers["Content-Disposition"].startswith("attachment;"))
                for name in ["../outside.safetensors", str(Path(directory) / "outside.safetensors"),
                             "missing.safetensors", "hero.json", "hero#0.safetensors"]:
                    request.query = {"name": name}
                    response = await handler(request)
                    self.assertIn(response.status, (400, 404), name)


class RefModUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_validation_collision_and_cleanup(self):
        import json
        import struct
        import tempfile
        from unittest.mock import patch
        try:
            import safetensors
        except ImportError:
            self.skipTest("safetensors is required for upload integration tests")
        refmods = load_module(f"{PACKAGE}.refmods", ROOT / "refmods.py")

        async def upload(filename, payload):
            class Field:
                name = "file"
                async def read_chunk(self):
                    chunk, self.payload = self.payload, b""
                    return chunk
            field = Field()
            field.filename, field.payload = filename, payload
            class Reader:
                async def next(self):
                    return field
            request = FakeRequest({web_api.TOKEN_HEADER: web_api._TOKEN})
            async def multipart():
                return Reader()
            request.multipart = multipart
            return await routes.posts["/minimax_h3/refmods/upload"](request)

        def fixture(metadata):
            header = json.dumps({
                "__metadata__": {"refmod_meta": json.dumps(metadata)},
                "latent": {"dtype": "F32", "shape": [1, 24, 1, 2, 2],
                           "data_offsets": [0, 384]},
            }).encode()
            header += b" " * (-len(header) % 8)
            return struct.pack("<Q", len(header)) + header + bytes(384)

        with tempfile.TemporaryDirectory() as directory:
            paths = types.SimpleNamespace(models_dir=directory)
            root = Path(directory) / "refmods"
            with patch.object(web_api, "folder_paths", paths), patch.object(
                refmods, "search_dirs", return_value=[str(root)]
            ):
                payload = fixture({"kind": "image", "latent_t": 1,
                                   "latent_h": 2, "latent_w": 2})
                first = await upload("../../hero.safetensors", payload)
                second = await upload("../../hero.safetensors", payload)
                self.assertEqual(first.status, 200, first.data)
                self.assertEqual(second.status, 200, second.data)
                self.assertNotEqual(first.data["item"]["name"], second.data["item"]["name"])
                self.assertEqual(first.data["item"]["visual"]["kind"], "image")
                self.assertEqual(len(list(root.rglob("*.safetensors"))), 2)
                for filename, data in [("bad.txt", payload),
                                       ("bad.safetensors", b"invalid"),
                                       ("model.safetensors", fixture({})),
                                       ("truncated.safetensors", payload[:-1])]:
                    response = await upload(filename, data)
                    self.assertEqual(response.status, 400, response.data)
                self.assertEqual(len(list(root.iterdir())), 2)
                self.assertFalse(list(root.rglob("*.part")))

if __name__ == "__main__":
    unittest.main()
