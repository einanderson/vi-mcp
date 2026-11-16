#!/usr/bin/env bash
# Builds dist/vi-mcp-<version>-<rel>-any.pkg.tar.zst for Arch Linux and CachyOS from the git-tracked files of the working tree.
# Without makepkg the build runs in an archlinux container (as the RPM build runs in Fedora). docs/LOKAL* never enters the package.
set -euo pipefail
cd "$(dirname "$0")/../.."
version=$(python3 -c "import vi_mcp; print(vi_mcp.__version__)")
rel="${ARCH_REL:-1}"
mkdir -p dist
bau=$(mktemp -d); trap 'rm -rf "$bau"' EXIT
src="$bau/vi-mcp-$version"
mkdir -p "$src"
git ls-files -z --cached --others --exclude-standard | grep -zv '^docs/LOKAL' | xargs -0 -I{} cp --parents {} "$src/"
tar -C "$bau" -czf "$bau/vi-mcp-$version.tar.gz" "vi-mcp-$version"
sum=$(sha256sum "$bau/vi-mcp-$version.tar.gz" | cut -d' ' -f1)
pk="$bau/pkg"; mkdir -p "$pk"
sed -e "s/@VERSION@/$version/" -e "s/@REL@/$rel/" -e "s/@SHA256@/$sum/" deploy/arch/PKGBUILD.in > "$pk/PKGBUILD"
cp deploy/arch/vi-mcp.install "$bau/vi-mcp-$version.tar.gz" "$pk/"
if command -v makepkg >/dev/null && [ "$(id -u)" != 0 ]; then
    (cd "$pk" && PKGDEST="$PWD" makepkg -f --nodeps --noconfirm >/dev/null)
else
    chmod -R a+rwX "$pk"
    # makepkg refuses root: build as a throw-away user, then hand every file back to the caller so the cleanup can remove it
    docker run --rm --network host -e http_proxy -e https_proxy -e no_proxy -e HOST_ID="$(id -u):$(id -g)" -v "$pk:/pkg" archlinux:base-devel \
        bash -c 'useradd -m b && chown -R b /pkg && su b -c "cd /pkg && PKGDEST=/pkg makepkg -f --nodeps --noconfirm >/dev/null"; s=$?; chown -R "$HOST_ID" /pkg; exit $s'
fi
cp "$pk"/vi-mcp-*.pkg.tar.zst dist/
echo "dist/$(cd "$pk" && ls vi-mcp-*.pkg.tar.zst)"
