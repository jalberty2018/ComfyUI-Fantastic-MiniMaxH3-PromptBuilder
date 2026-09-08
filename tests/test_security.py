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

        response = await handler(FakeRequest({"Sec-Fetch-Site": "cross-site"}))
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


if __name__ == "__main__":
    unittest.main()
