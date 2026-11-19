import unittest

from tinker_lab.safety import LiveRunBlocked, require_live_run_authorization


class LiveRunAuthorizationTests(unittest.TestCase):
    def test_refuses_a_paid_call_without_explicit_execute_flag(self) -> None:
        with self.assertRaisesRegex(LiveRunBlocked, "--execute"):
            require_live_run_authorization(execute=False, api_key="present")

    def test_refuses_a_paid_call_without_a_tinker_key(self) -> None:
        with self.assertRaisesRegex(LiveRunBlocked, "TINKER_API_KEY"):
            require_live_run_authorization(execute=True, api_key=None)

    def test_allows_a_paid_call_only_when_both_checks_pass(self) -> None:
        require_live_run_authorization(execute=True, api_key="present")


if __name__ == "__main__":
    unittest.main()
