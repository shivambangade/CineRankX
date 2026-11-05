"""Find markdown tables whose rows have a different number of cells than the header (unescaped pipes)."""

import glob
import re

SPLIT = re.compile(r"(?<!\\)\|")
for f in sorted(glob.glob("docs/_build/chapters/*.md")):
    lines = open(f, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append((i + 1, lines[i]))
                i += 1
            count = lambda line: len(SPLIT.split(line.strip().strip("|")))
            header = count(block[0][1])
            for number, line in block:
                if count(line) != header and not re.match(r"^\|\s*:?-", line):
                    print(f.split("/")[-1], number, "cols", count(line), "expected", header, line[:80])
        else:
            i += 1
