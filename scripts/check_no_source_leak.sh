#!/usr/bin/env bash
# Fails if any tracked/staged file contains a word from .source-denylist.
# The denylist itself is gitignored (it would otherwise leak what it protects).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
DENY=.source-denylist
[ -f "$DENY" ] || { echo "no $DENY, skipping leak check"; exit 0; }
pattern=$(grep -v '^\s*\(#\|$\)' "$DENY" | paste -sd'|' -)
[ -n "$pattern" ] || exit 0
if git ls-files -co --exclude-standard -z | xargs -0 grep -IniE "$pattern" -- 2>/dev/null; then
  echo "ERROR: source identifiers found (see above). Remove them before committing." >&2
  exit 1
fi
