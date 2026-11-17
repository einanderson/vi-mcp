"""0.68.1: every platform is built and released — the Arch package (buildable since 0.67, never released) and macOS (Homebrew formula)."""
import re
import subprocess
from pathlib import Path

from vi_mcp import setup

ROOT = Path(__file__).resolve().parents[1]


def test_restart_on_macos_uses_brew_services(monkeypatch):
    calls = []
    monkeypatch.setattr(setup.sys, "platform", "darwin")
    monkeypatch.setattr(setup, "WINDOWS", False)
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None)
    monkeypatch.setattr(setup.subprocess, "run", lambda cmd, **k: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    assert setup._restart() is None and calls == [["brew", "services", "restart", "vi-mcp"]]


def test_restart_on_macos_without_brew_names_the_reason(monkeypatch):
    monkeypatch.setattr(setup.sys, "platform", "darwin")
    monkeypatch.setattr(setup, "WINDOWS", False)
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    assert "Homebrew" in setup._restart()


def test_the_release_builds_signs_and_uploads_every_platform():
    s = (ROOT / "deploy/release.sh").read_text()
    for build in ("deploy/debian/build.sh", "deploy/rpm/build.sh", "deploy/arch/build.sh", "deploy/windows/build.sh",
                  "deploy/macos/build.sh", "deploy/clingo-object/build_wheel.py"):
        assert build in s, build
    sums = next(line for line in s.splitlines() if "> SHA256SUMS" in line)
    create = next(line for line in s.splitlines() if "gh release create" in line)
    for var in ("deb", "rpm", "pkg", "msi", "whl", "src"):
        assert f'"$(basename "${var}")"' in sums, var
        assert f'"${var}"' in create, var


def test_the_formula_template_installs_the_hook_where_setup_looks_and_has_no_placeholders_left_but_two():
    t = (ROOT / "deploy/macos/vi-mcp.rb.in").read_text()
    assert set(re.findall(r"@[A-Z0-9]+@", t)) == {"@VERSION@", "@SHA256@"}
    assert '(libexec/"lib").install "vi_mcp", "clingo_object"' in t and '(libexec/"lib/hook").install "hook/vi_hook.py"' in t
    # setup.hook_path falls back to <package dir>/../hook/vi_hook.py — libexec/lib/vi_mcp → libexec/lib/hook
    assert 'parents[1] / "hook" / "vi_hook.py"' in (ROOT / "vi_mcp/setup.py").read_text()
    assert "clingo==5.8.3" in t and "clingo==5.8.3" in (ROOT / "deploy/windows/install.ps1").read_text()  # one pinned clingo
