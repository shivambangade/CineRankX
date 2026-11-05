"""Extract a grounded inventory of the repository (modules, classes, functions, tests) via AST."""
import ast, json, subprocess
from pathlib import Path

ROOT = Path(".")
INCLUDE = ["app.py", "src", "scripts", "tests"]
out = {}
for entry in INCLUDE:
    paths = [ROOT / entry] if entry.endswith(".py") else sorted((ROOT / entry).rglob("*.py"))
    for p in paths:
        if "__pycache__" in str(p):
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        info = {"lines": src.count("\n") + 1, "doc": (ast.get_docstring(tree) or "").strip(),
                "classes": [], "functions": []}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [{"name": f.name, "args": [a.arg for a in f.args.args], "doc": (ast.get_docstring(f) or "").split("\n")[0]}
                           for f in node.body if isinstance(f, ast.FunctionDef)]
                info["classes"].append({"name": node.name, "doc": (ast.get_docstring(node) or "").split("\n")[0], "methods": methods})
            elif isinstance(node, ast.FunctionDef):
                info["functions"].append({"name": node.name, "args": [a.arg for a in node.args.args],
                                          "doc": (ast.get_docstring(node) or "").split("\n")[0]})
        out[str(p).replace("\\", "/")] = info
Path("docs/_build/inventory.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(len(out), "files;", sum(v["lines"] for v in out.values()), "lines")
tests = {k: [f["name"] for c in v["classes"] for f in c["methods"] if f["name"].startswith("test_")] + [f["name"] for f in v["functions"] if f["name"].startswith("test_")] for k, v in out.items() if k.startswith("tests/")}
print({k.split("/")[-1]: len(v) for k, v in tests.items()}, sum(len(v) for v in tests.values()))
