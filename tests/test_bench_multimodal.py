import json
import tempfile
import unittest
from pathlib import Path

from bench_multimodal import write_report


class WriteReportTest(unittest.TestCase):
    def test_creates_missing_parent_directories(self) -> None:
        report = {"summary": {"successful_requests": 100}}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "benchmark-results" / "native-vlm.json"
            write_report(output_path, report)

            self.assertEqual(json.loads(output_path.read_text()), report)


if __name__ == "__main__":
    unittest.main()
