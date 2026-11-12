from __future__ import annotations

import unittest

from mrz_reader.passport_evaluation_review import strict_auto_approval


class PassportEvaluationReviewTests(unittest.TestCase):
    def test_strict_auto_approval_requires_all_three_conditions(self) -> None:
        self.assertTrue(strict_auto_approval(True, True, True))
        self.assertFalse(strict_auto_approval(False, True, True))
        self.assertFalse(strict_auto_approval(True, False, True))
        self.assertFalse(strict_auto_approval(True, True, False))
        self.assertFalse(strict_auto_approval(True, True, None))


if __name__ == "__main__":
    unittest.main()
