import tempfile
import unittest
from pathlib import Path

from och import tools


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _write(self, rel: str, content: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    # read_file ----------------------------------------------------------

    def test_read_file_numbers_lines(self):
        p = self._write("a.txt", "one\ntwo\nthree\n")
        out = tools.read_file(str(p))
        self.assertIn("     1\tone", out)
        self.assertIn("     3\tthree", out)

    def test_read_file_offset_limit(self):
        p = self._write("a.txt", "\n".join(str(i) for i in range(10)))
        out = tools.read_file(str(p), offset=5, limit=2)
        self.assertIn("     6\t5", out)
        self.assertIn("     7\t6", out)
        self.assertNotIn("\t7", out)
        self.assertIn("more lines", out)

    def test_read_file_missing(self):
        self.assertIn("Error", tools.read_file(str(self.root / "nope.txt")))

    # write_file / edit_file ----------------------------------------------

    def test_write_file_creates_parents(self):
        p = self.root / "deep" / "dir" / "f.txt"
        out = tools.write_file(str(p), "hello")
        self.assertIn("Wrote", out)
        self.assertEqual(p.read_text(), "hello")

    def test_edit_file_unique_replacement(self):
        p = self._write("a.py", "x = 1\ny = 2\n")
        out = tools.edit_file(str(p), "y = 2", "y = 3")
        self.assertIn("Edited", out)
        self.assertEqual(p.read_text(), "x = 1\ny = 3\n")

    def test_edit_file_rejects_missing_and_ambiguous(self):
        p = self._write("a.py", "a\na\n")
        self.assertIn("not found", tools.edit_file(str(p), "zzz", "q"))
        self.assertIn("2 times", tools.edit_file(str(p), "a", "q"))
        self.assertEqual(p.read_text(), "a\na\n")  # unchanged

    # listing / searching ---------------------------------------------------

    def test_list_directory(self):
        self._write("f.txt", "x")
        (self.root / "sub").mkdir()
        out = tools.list_directory(str(self.root))
        self.assertIn("sub/", out)
        self.assertIn("f.txt", out)

    def test_glob_files(self):
        self._write("a.py", "")
        self._write("sub/b.py", "")
        self._write("c.txt", "")
        out = tools.glob_files("*.py", str(self.root))
        self.assertIn("a.py", out)
        self.assertIn("b.py", out)
        self.assertNotIn("c.txt", out)

    def test_grep_finds_matches_and_skips_binary(self):
        self._write("a.py", "def foo():\n    pass\n")
        (self.root / "bin.dat").write_bytes(b"\x00\xfffoo")
        out = tools.grep(r"def \w+", str(self.root))
        self.assertIn("a.py:1", out)
        self.assertNotIn("bin.dat", out)

    def test_grep_invalid_regex(self):
        self.assertIn("invalid regex", tools.grep("[", str(self.root)))

    # bash ------------------------------------------------------------------

    def test_bash_captures_output_and_exit_code(self):
        self.assertEqual(tools.bash("echo hi"), "hi")
        out = tools.bash("echo oops >&2; exit 3")
        self.assertIn("oops", out)
        self.assertIn("[exit code: 3]", out)

    def test_bash_timeout(self):
        out = tools.bash("sleep 5", timeout=1)
        self.assertIn("timed out", out)

    # registry ----------------------------------------------------------------

    def test_schemas_shape(self):
        registry = tools.build_tools()
        schemas = tools.tool_schemas(registry)
        names = {s["function"]["name"] for s in schemas}
        self.assertEqual(
            names,
            {"read_file", "write_file", "edit_file", "list_directory", "glob", "grep", "bash"},
        )
        for s in schemas:
            params = s["function"]["parameters"]
            self.assertEqual(params["type"], "object")
            for prop in params["properties"].values():
                self.assertNotIn("_required", prop)
        edit = next(s for s in schemas if s["function"]["name"] == "edit_file")
        self.assertEqual(
            sorted(edit["function"]["parameters"]["required"]),
            ["new_string", "old_string", "path"],
        )

    def test_mutating_flags(self):
        registry = tools.build_tools()
        self.assertTrue(registry["bash"].mutating)
        self.assertTrue(registry["write_file"].mutating)
        self.assertFalse(registry["read_file"].mutating)


if __name__ == "__main__":
    unittest.main()
