import json

with open("chess_pipeline_v2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Find and remove duplicate cells by id
seen_ids = set()
unique_cells = []
removed = 0

for cell in cells:
    cid = cell.get("id", "")
    if cid in seen_ids:
        removed += 1
        print(f"Removing duplicate: {cid}")
    else:
        seen_ids.add(cid)
        unique_cells.append(cell)

nb["cells"] = unique_cells

with open("chess_pipeline_v2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nRemoved {removed} duplicate cells. Total cells now: {len(unique_cells)}")
