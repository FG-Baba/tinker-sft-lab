class LiveRunBlocked(RuntimeError):
    """Raised before a potentially billed Tinker operation."""


def require_live_run_authorization(*, execute: bool, api_key: str | None) -> None:
    if not execute:
        raise LiveRunBlocked("Refusing a billed Tinker call without the explicit --execute flag.")
    if not api_key:
        raise LiveRunBlocked("TINKER_API_KEY is required for a live Tinker run.")
