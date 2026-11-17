#!/usr/bin/env bash
# Installs vi-mcp as a systemd user service. Idempotent. Creates an instance the first time.
#   deploy/install.sh [instance-name]     (default: $USER, with the reflex right)
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:-$USER}"
cfg="${VI_MCP_CONFIG:-$HOME/.config/vi-mcp/config.json}"
tokdatei="$HOME/.config/vi-mcp/token-$name"
# "instances" since 0.51.0, "instanzen" before — checking only one would renew an existing instance's token and lock its clients out
if ! python3 -c "import json,sys; c=json.load(open('$cfg')); sys.exit(0 if '$name' in (c.get('instances') or c.get('instanzen') or {}) else 1)" 2>/dev/null; then
  umask 077
  # the token file is written right after the instance is created — without its directory the token would be lost
  mkdir -p "$(dirname "$tokdatei")"
  python3 -m vi_mcp instance "$name" --reflexes | tail -1 > "$tokdatei"
  chmod 600 "$tokdatei"
  echo "instance $name created; token in $tokdatei (0600)"
fi
mkdir -p "$HOME/.config/systemd/user"
# the unit names %h/Dev/code/vi-mcp; point it to wherever this checkout lives
sed "s#%h/Dev/code/vi-mcp#$PWD#g" deploy/vi-mcp.service > "$HOME/.config/systemd/user/vi-mcp.service"
chmod 0644 "$HOME/.config/systemd/user/vi-mcp.service"
# optional, but without scikit-learn the router never trains (hints stay in heuristic order)
python3 -c "import sklearn" 2>/dev/null || echo "note: scikit-learn missing — the router cannot train (Debian: python3-sklearn, Arch: python-scikit-learn, Fedora: python3-scikit-learn)"
systemctl --user daemon-reload
systemctl --user enable --now vi-mcp.service
systemctl --user restart vi-mcp.service
sleep 1
systemctl --user --no-pager --lines=5 status vi-mcp.service
curl -fsS "http://127.0.0.1:$(python3 -m vi_mcp port)/health"
