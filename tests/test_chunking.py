"""Structure-aware chunking of long content (vi_mcp/chunking.py, ported from the IT-AIKB chunker v2): the cases of its test suite in
their Markdown form, plus the shape of a real imported page (path line, table, headings, fenced code). Invented host names only."""
from vi_mcp import chunking as ch


def _all_text(chunks) -> str:
    return "\n".join(c.body for c in chunks)


def test_heading_path_prefixes_every_chunk():
    md = ("Introduction to the server.\n\n## Configuration\n\nGeneral.\n\n### HAProxy\n\ntimeout client 6h\n\n"
          "## Operation\n\nRestart with systemctl.\n")
    chunks = ch.chunk_markdown("proxy07.example.org", md, min_chars=0)
    assert [c.header_path for c in chunks] == [
        "proxy07.example.org",
        "proxy07.example.org > Configuration",
        "proxy07.example.org > Configuration > HAProxy",
        "proxy07.example.org > Operation",  # an h2 closes the h3 below it
    ]
    assert chunks[2].text.startswith("proxy07.example.org > Configuration > HAProxy\n\n")
    assert "timeout client 6h" in chunks[2].body


def test_fenced_code_stays_verbatim_and_whole():
    code = '```bash\ncmd 2>&1 | tee /var/log/x.log\nif [ $a -gt 1 ]; then echo "->"; fi\n```'
    chunks = ch.chunk_markdown("Backup", f"## Script\n\n{code}\n")
    assert len(chunks) == 1 and code in chunks[0].body


def test_a_heading_inside_a_code_fence_is_code_not_a_section():
    md = "## Setup\n\n```sh\n# not a heading\necho hi\n```\n\nafter\n"
    chunks = ch.chunk_markdown("T", md, min_chars=0)
    assert all(c.header_path == "T > Setup" for c in chunks)
    assert "# not a heading" in _all_text(chunks)


def test_table_rows_stay_together_and_oversized_tables_repeat_the_header():
    rows = "\n".join(f"| host{i:02d} | 10.0.0.{i} |" for i in range(120))
    md = f"## Hosts\n\n| Name | IP |\n|---|---|\n{rows}\n"
    chunks = ch.chunk_markdown("Inventory", md, max_chars=600)
    assert len(chunks) > 1
    for c in chunks:
        assert c.body.startswith("| Name | IP |\n|---|---|\n")
        assert len(c.text) <= 600
    assert all(f"| host{i:02d} | 10.0.0.{i} |" in _all_text(chunks) for i in range(120))


def test_long_sections_are_grouped_without_losing_text():
    paragraphs = [f"Paragraph {i} describes step {i} of the setup in detail." for i in range(80)]
    chunks = ch.chunk_markdown("VPN", "## Guide\n\n" + "\n\n".join(paragraphs))
    small = ch.chunk_markdown("VPN", "## Guide\n\n" + "\n\n".join(paragraphs), max_chars=500)
    assert len(small) > len(chunks) >= 1
    assert all(len(c.text) <= 500 for c in small)
    assert all(p in _all_text(small) for p in paragraphs)


def test_a_short_tail_is_merged_into_its_predecessor_in_the_same_section():
    body = "x" * 900
    chunks = ch.chunk_markdown("T", f"## A\n\n{body}\n\n{'y' * 650}\n\nshort tail.", max_chars=1600)
    assert len(chunks) == 1 and chunks[0].body.endswith("short tail.")


def test_a_single_overlong_line_is_split_never_dropped():
    line = "z" * 5000
    chunks = ch.chunk_markdown("T", line, max_chars=800)
    assert "".join(c.body for c in chunks) == line
    assert all(len(c.text) <= 800 for c in chunks)


def test_manual_page_entries_carry_their_name_into_the_path():
    md = ("# Tools\n\n## NAME\n\nexport-ad - export the address book\n\n## Description\n\nWrites a CSV.\n\n"
          "## NAME\n\nimport-ad - import the address book\n\n## Description\n\nReads a CSV.\n")
    paths = [c.header_path for c in ch.chunk_markdown("Manual", md, min_chars=0)]
    assert "Manual > Tools > export-ad > Description" in paths
    assert "Manual > Tools > import-ad > Description" in paths


def test_the_shape_of_an_imported_page():
    md = ("Path: IT › Systems › Cluster › host7.example.org\n\n| host7.example.org | 192.0.2.23 |\n|---|---|\n\nHardware:\n\n"
          "Board X470\nCPU 16 cores\n\n## Network configuration:\n\n### Proxy:\n\n/etc/apt/apt.conf:\n\n"
          '```bash\nAcquire::http::Proxy "http://proxy.example.org:80";\n```\n\n## Disks\n\nTwo NVMe in a mirror.\n')
    chunks = ch.chunk_markdown("host7.example.org", md, min_chars=0)
    paths = [c.header_path for c in chunks]
    assert paths[0] == "host7.example.org" and "host7.example.org > Network configuration: > Proxy:" in paths
    assert "host7.example.org > Disks" in paths
    proxy = next(c for c in chunks if c.header_path.endswith("Proxy:"))
    assert 'Acquire::http::Proxy "http://proxy.example.org:80";' in proxy.body
    assert "| host7.example.org | 192.0.2.23 |" in chunks[0].body


def test_a_table_that_is_only_an_oversized_header_row_keeps_its_text():
    # Found on a production copy: the original chunker dropped such a table completely (no body rows to put the header before).
    row = "| " + " # ".join(f"cell {i} with some words" for i in range(150)) + " |"
    chunks = ch.chunk_markdown("T", f"## Rollout\n\n{row}\n|---|\n", max_chars=800)
    import re
    assert re.sub(r"\s+", "", row) in re.sub(r"\s+", "", "".join(c.body for c in chunks))


def test_a_paragraph_written_as_a_heading_stays_content_and_the_path_stays_short():
    long_heading = "# " + "word " * 400
    chunks = ch.chunk_markdown("T", f"{long_heading}\n\nbody text\n", min_chars=0)
    assert all(len(c.header_path) <= len("T > ") + ch.MAX_HEADING_CHARS + 2 for c in chunks)
    assert all(len(c.text) <= ch.DEFAULT_MAX_CHARS for c in chunks)
    import re
    assert "word" * 400 in re.sub(r"\s+", "", "".join(c.body for c in chunks))  # split mid-line without punctuation: compare without spaces


def test_plain_text_prefixes_the_source_header():
    text = "\n\n".join(f"Para {i} " + "w " * 120 for i in range(10))
    chunks = ch.chunk_plain_text("Page > manual.pdf", text, max_chars=700)
    assert len(chunks) > 1 and all(c.text.startswith("Page > manual.pdf\n\n") for c in chunks)


def test_empty_content_has_no_chunks_and_the_hash_carries_the_version():
    assert ch.chunk_markdown("T", "  \n") == [] and ch.chunk_plain_text("T", "") == []
    h = ch.versioned_hash("abc")
    assert h == f"c{ch.CHUNKER_VERSION}:abc" and ch.hash_digest(h) == "abc" and ch.hash_digest("abc") == "abc"
