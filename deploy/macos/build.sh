#!/usr/bin/env bash
# Builds the macOS release parts (0.68.1):
#   dist/vi-mcp-<version>.tar.gz   the source archive the Homebrew formula installs (git-tracked files, never docs/LOKAL*),
#                                  reproducible: sorted, fixed owner, the commit's time, gzip without a name/time stamp
#   dist/vi-mcp.rb                 the formula for the tap imoes/homebrew-vi-mcp, with this archive's checksum
# Prints both paths. The release uploads the archive (it is in SHA256SUMS) and pushes the formula to the tap.
set -euo pipefail
cd "$(dirname "$0")/../.."
version=$(python3 -c "import vi_mcp; print(vi_mcp.__version__)")
mkdir -p dist
bau=$(mktemp -d); trap 'rm -rf "$bau"' EXIT
src="$bau/vi-mcp-$version"
mkdir -p "$src"
git ls-files -z --cached | grep -zv '^docs/LOKAL' | xargs -0 -I{} cp --parents {} "$src/"
when=$(git log -1 --format=%ct)
tar -C "$bau" --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$when" -cf - "vi-mcp-$version" | gzip -n -9 > "dist/vi-mcp-$version.tar.gz"
sum=$(sha256sum "dist/vi-mcp-$version.tar.gz" | cut -d' ' -f1)
sed -e "s/@VERSION@/$version/g" -e "s/@SHA256@/$sum/" deploy/macos/vi-mcp.rb.in > dist/vi-mcp.rb
echo "dist/vi-mcp-$version.tar.gz"
echo "dist/vi-mcp.rb"
