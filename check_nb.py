import json

with open("chess_pipeline_v2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")
print()

# Check for any issues
for i, cell in enumerate(cells):
    ctype = cell.get("cell_type", "MISSING")
    cid = cell.get("id", "MISSING_ID")
    
    # Check source exists
    src = cell.get("source", None)
    if src is None:
        print(f"WARNING: Cell {i} ({cid}) has no 'source' key!")
    elif not isinstance(src, list):
        print(f"WARNING: Cell {i} ({cid}) source is not a list! Type: {type(src)}")
    
    # Check for code cells with issues
    if ctype == "code":
        if "outputs" not in cell:
            print(f"WARNING: Code cell {i} ({cid}) missing 'outputs' key")
        if "execution_count" not in cell:
            print(f"WARNING: Code cell {i} ({cid}) missing 'execution_count' key")

# Show the scratch and bayesian cells
print("\nNew cells location:")
for i, cell in enumerate(cells):
    cid = cell.get("id", "")
    if "scratch" in cid or "bayesian" in cid:
        preview = "".join(cell.get("source", []))[:60].replace("\n", " ")
        print(f"  Cell {i}: {cell['cell_type']} id={cid} -> {preview}...")

print("\nNotebook JSON is valid.")
