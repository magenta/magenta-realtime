#!/bin/bash
# Copyright 2026 Google LLC

# Launch Vite dev servers and their host apps for development.
#
# Usage: ./scripts/dev-all.sh [0-3]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SHOW_LOGS=false
MODE=""

SERVERS_ONLY=false
for arg in "$@"; do
  if [ "$arg" = "--log" ]; then
    SHOW_LOGS=true
  elif [ "$arg" = "--servers-only" ] || [ "$arg" = "--server-only" ]; then
    SERVERS_ONLY=true
  elif [[ "$arg" =~ ^[0-3]$ ]]; then
    MODE="$arg"
  fi
done

if [ -z "$MODE" ]; then
  echo "Which apps would you like to launch?"
  echo "  0) All apps"
  echo "  1) Plugin"
  echo "  2) Jam"
  echo "  3) Collider"
  echo ""
  read -p "Enter choice [0-3, default 0]: " MODE
  MODE="${MODE:-0}"
fi

if [[ ! "$MODE" =~ ^[0-3]$ ]]; then
  echo "Invalid option: $MODE"
  exit 1
fi

CLEANED_UP=false
cleanup() {
    if $CLEANED_UP; then return; fi
    CLEANED_UP=true
    echo ""
    echo "Shutting down dev servers and closing apps..."
    kill 0 2>/dev/null
    if ! $SERVERS_ONLY; then
        [[ "$MODE" == 0 || "$MODE" == 1 ]] && osascript -e 'quit app "MRT2"' 2>/dev/null || true
        [[ "$MODE" == 0 || "$MODE" == 2 ]] && osascript -e 'quit app "MRT2 - Jam"' 2>/dev/null || true
        [[ "$MODE" == 0 || "$MODE" == 3 ]] && osascript -e 'quit app "MRT2 - Collider"' 2>/dev/null || true
    fi
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

echo ""
echo "Starting Vite dev servers..."
echo ""

if [[ "$MODE" == 0 || "$MODE" == 1 ]]; then
  echo "  mrt2/react_ui   → http://localhost:62420"
  (cd "$EXAMPLES_DIR/mrt2/react_ui" && npm run dev) &
fi

if [[ "$MODE" == 0 || "$MODE" == 2 ]]; then
  echo "  jam             → http://localhost:62421"
  (cd "$EXAMPLES_DIR/jam/ui" && npm run dev) &
fi

if [[ "$MODE" == 0 || "$MODE" == 3 ]]; then
  echo "  collider → http://localhost:62419"
  (cd "$EXAMPLES_DIR/collider/ui" && npm run dev) &
fi

if ! $SERVERS_ONLY; then
  echo ""
  echo "Waiting 3s for servers to start..."
  sleep 3

  echo "Launching apps..."

  launch_app() {
    local app_name="$1"
    local bundle_name="$2"
    local bin_name="$3"

    if $SHOW_LOGS; then
      open "$HOME/Applications/${bundle_name}.app/Contents/MacOS/${bin_name}" 2>/dev/null && echo "  ✓ ${app_name} (with logs)" || echo "  ✗ ${app_name} (not found)"
    else
      open "$HOME/Applications/${bundle_name}.app" 2>/dev/null && echo "  ✓ ${app_name}" || echo "  ✗ ${app_name} (not found)"
    fi
  }

  [[ "$MODE" == 0 || "$MODE" == 1 ]] && launch_app "MRT2" "MRT2" "MRT2"
  [[ "$MODE" == 0 || "$MODE" == 2 ]] && launch_app "MRT2 - Jam" "MRT2 - Jam" "MRT2_Jam"
  [[ "$MODE" == 0 || "$MODE" == 3 ]] && launch_app "MRT2 - Collider" "MRT2 - Collider" "MRT2_Collider"
fi

echo ""
echo "Dev servers running. Press Ctrl+C to stop all."
wait
