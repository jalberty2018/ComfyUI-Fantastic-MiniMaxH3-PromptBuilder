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



class FakeStreamResponse(FakeResponse):
    def __init__(self, headers):
        super().__init__(b"", headers=headers)
    async def prepare(self, request):
        pass
    async def write(self, data):
        self.data += data
    async def write_eof(self):
        pass


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
            (root / "hero.png").write_bytes(b"preview")
            (root / "hero.json").write_bytes(b"{}")
            (Path(directory) / "outside.safetensors").write_bytes(b"private")
            with patch.object(refmods, "search_dirs", return_value=[str(root)]), patch.object(
                web_api.web, "StreamResponse", create=True, side_effect=FakeStreamResponse
            ):
                request = FakeRequest()
                request.query = {"name": "hero.safetensors"}
                response = await handler(request)
                self.assertEqual(response.status, 200)
                import io, zipfile
                with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                    self.assertEqual(archive.read("hero.safetensors"), b"refmod")
                    self.assertEqual(archive.read("hero.png"), b"preview")
                    self.assertEqual(archive.read("hero.json"), b"{}")
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
                import io, zipfile
                def archive_bytes(entries):
                    output = io.BytesIO()
                    with zipfile.ZipFile(output, "w") as archive:
                        for name, value in entries.items():
                            archive.writestr(name, value)
                    return output.getvalue()
                zipped = archive_bytes({"hero.safetensors": payload})
                first = await upload("../../hero.zip", zipped)
                second = await upload("../../hero.zip", zipped)
                self.assertEqual(first.status, 200, first.data)
                self.assertEqual(second.status, 200, second.data)
                self.assertNotEqual(first.data["item"]["name"], second.data["item"]["name"])
                self.assertEqual(first.data["item"]["visual"]["kind"], "image")
                self.assertEqual(len(list(root.rglob("*.safetensors"))), 2)
                for filename, data in [("bad.txt", payload),
                                       ("hero.safetensors", payload),
                                       ("bad.zip", b"invalid"),
                                       ("bad.safetensors", b"invalid"),
                                       ("model.safetensors", fixture({})),
                                       ("truncated.safetensors", payload[:-1])]:
                    response = await upload(filename, data)
                    self.assertEqual(response.status, 400, response.data)
                self.assertEqual(len(list(root.iterdir())), 2)
                self.assertFalse(list(root.rglob("*.part")))
                response = await upload("hero.zip", archive_bytes({
                    "hero.safetensors": payload, "hero.png": b"preview", "hero.json": b"{}"
                }))
                self.assertEqual(response.status, 200, response.data)
                item = response.data["item"]
                self.assertIsNotNone(item["preview"])
                request = FakeRequest()
                request.query = {"name": item["visual"]["file"] + ".safetensors"}
                with patch.object(web_api.web, "StreamResponse", create=True, side_effect=FakeStreamResponse):
                    downloaded = await routes.gets["/minimax_h3/refmods/download"](request)
                restored = await upload("restored.zip", downloaded.data)
                self.assertEqual(restored.status, 200, restored.data)
                self.assertIsNotNone(restored.data["item"]["preview"])
                paired = await upload("pair.zip", archive_bytes({
                    "pair_visual.safetensors": payload,
                    "pair_audio.safetensors": fixture({"kind": "audio", "latent_t": 1}),
                    "pair.png": b"shared preview",
                }))
                self.assertEqual(paired.status, 200, paired.data)
                self.assertEqual(len(paired.data["items"]), 1)
                request.query = {"name": paired.data["item"]["visual"]["file"] + ".safetensors"}
                with patch.object(web_api.web, "StreamResponse", create=True, side_effect=FakeStreamResponse):
                    pair_download = await routes.gets["/minimax_h3/refmods/download"](request)
                with zipfile.ZipFile(io.BytesIO(pair_download.data)) as archive:
                    self.assertEqual(len(archive.namelist()), 3)
                    self.assertTrue(any(n.endswith("pair_audio.safetensors") for n in archive.namelist()))
                sidecar = await upload("sidecar.zip", archive_bytes({
                    "legacy.safetensors": fixture(None),
                    "legacy.json": json.dumps({"kind": "image", "latent_t": 1,
                                                "latent_h": 2, "latent_w": 2}).encode(),
                }))
                self.assertEqual(sidecar.status, 200, sidecar.data)
                count = len(list(root.iterdir()))
                for entries in [{"../escape.json": b"{}"}, {"/escape.json": b"{}"},
                                {"C:/escape.json": b"{}"}, {"run.py": b"pass"},
                                {"bad.safetensors": b"invalid"}]:
                    response = await upload("bad.zip", archive_bytes(entries))
                    self.assertEqual(response.status, 400, response.data)
                    self.assertEqual(len(list(root.iterdir())), count)


if __name__ == "__main__":
    unittest.main()
