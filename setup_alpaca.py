"""One-step Alpaca paper setup: prompt, store, verify.

The keys cannot be generated for you — they live behind your Alpaca login and the
secret is shown once, in your browser. Everything after that point is automated
here so it is a single command.

The secret is read with getpass, so it is never echoed to the terminal and never
appears in shell history or a transcript. It is stored as a Windows *user*
environment variable, not written to any file in this repo.

    venv\\Scripts\\python setup_alpaca.py
"""
from __future__ import annotations

import getpass
import os
import subprocess
import sys


def store(name: str, value: str) -> bool:
    """Persist a user environment variable via setx, without echoing it."""
    try:
        p = subprocess.run(["setx", name, value], capture_output=True, text=True,
                           timeout=20)
        return p.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  setx failed: {exc}", file=sys.stderr)
        return False


def main() -> int:
    """Collect credentials, store them, and verify against the paper endpoint."""
    print("Alpaca PAPER setup")
    print("-" * 52)
    print("On alpaca.markets: switch the toggle to PAPER, then")
    print("Home -> API Keys -> Generate New Key. Copy both before closing.\n")

    key = input("API key ID (starts PK...): ").strip()
    if not key:
        print("no key entered — aborted")
        return 2
    if key.startswith("AK"):
        # Live keys start AK, paper keys PK. Catching this here is better than a
        # confusing 403 later against an endpoint that will not accept them.
        print("\nThat looks like a LIVE key (starts 'AK'). This project is paper-only.")
        print("Switch the Alpaca toggle to Paper and generate a paper key (PK...).")
        return 3

    secret = getpass.getpass("Secret key (hidden, not echoed): ").strip()
    if not secret:
        print("no secret entered — aborted")
        return 2

    print("\nstoring...")
    ok = store("ALPACA_API_KEY", key) and store("ALPACA_SECRET_KEY", secret)
    print("  stored as user environment variables" if ok else "  FAILED to store")
    if not ok:
        return 4

    # setx only affects NEW processes, so verify using values set in-process.
    os.environ["ALPACA_API_KEY"] = key
    os.environ["ALPACA_SECRET_KEY"] = secret

    print("\nverifying against the paper endpoint...")
    try:
        from paper_broker import AlpacaPaper
        api = AlpacaPaper()
        acct = api.account()
        clock = api.clock()
        pos = api.positions()
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
        print("\n  Keys were stored, but the connection failed. Common causes:")
        print("   - keys generated on the Live tab instead of Paper")
        print("   - secret truncated when copied")
        print("   - key regenerated after copying (invalidates the old one)")
        return 5

    print(f"  CONNECTED")
    print(f"  account status : {acct.get('status')}")
    print(f"  equity         : {float(acct.get('equity', 0)):,.2f} (paper)")
    print(f"  buying power   : {float(acct.get('buying_power', 0)):,.2f}")
    print(f"  open positions : {len(pos)}")
    print(f"  market         : {'OPEN' if clock.get('is_open') else 'closed'}")

    print("\nDone. Open a NEW terminal (setx does not affect this one), then:")
    print("  venv\\Scripts\\python paper_broker.py --check")
    print("  venv\\Scripts\\python paper_broker.py --dry-run")
    print("  venv\\Scripts\\python paper_broker.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
