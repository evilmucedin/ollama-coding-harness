import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from och import session


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(session, "SESSIONS_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_and_load_roundtrip(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        path = session.save(messages, model="m:7b")
        data = session.load(path)
        self.assertEqual(data["messages"], messages)
        self.assertEqual(data["model"], "m:7b")

    def test_latest_picks_newest(self):
        self.assertIsNone(session.latest())
        session.save([], "m", path=Path(self.tmp.name) / "session-20260101-000000.json")
        p2 = Path(self.tmp.name) / "session-20260102-000000.json"
        session.save([], "m", path=p2)
        self.assertEqual(session.latest(), p2)

    def test_load_rejects_garbage(self):
        p = Path(self.tmp.name) / "session-bad.json"
        p.write_text(json.dumps({"nope": True}))
        with self.assertRaises(ValueError):
            session.load(p)


if __name__ == "__main__":
    unittest.main()
