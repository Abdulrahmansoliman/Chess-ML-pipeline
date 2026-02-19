import json

with open("chess_pipeline_v2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# IDs of the from-scratch LogReg cells to move
scratch_ids = [
    "scratch-logreg-md",
    "scratch-logreg-code", 
    "scratch-training-md",
    "scratch-training-code",
    "scratch-comparison-md",
    "scratch-comparison-code",
    "scratch-eval-code",
]

# 1. Extract and remove scratch cells from current position
scratch_cells = []
for cell_id in scratch_ids:
    for i, cell in enumerate(cells):
        if cell.get("id") == cell_id:
            scratch_cells.append(cells.pop(i))
            break

print(f"Extracted {len(scratch_cells)} cells")

# 2. Update the markdown motivation (first cell)
if scratch_cells:
    scratch_cells[0]["source"] = [
        "## Section 8: Logistic Regression from Scratch\n",
        "\n",
        "### Why Build It Myself?\n",
        "\n",
        "I don't want to just call `model.fit()` and move on. I want to prove that I actually understand what's happening under the hood. Building logistic regression from scratch serves three purposes:\n",
        "\n",
        "1. **Proving understanding**: If I can implement the math myself and get matching results, it means I truly understand the algorithm — not just the API.\n",
        "2. **Validation**: Comparing my scratch model to sklearn's output acts as a cross-check. If both give similar coefficients and accuracy, I know the math is correct.\n",
        "3. **Full control**: Writing my own training loop lets me track the loss at every iteration, visualize convergence, and experiment with things sklearn doesn't expose (custom learning rates, different regularization, etc.).\n",
        "\n",
        "### The Math (Quick Recap)\n",
        "\n",
        "For a single sample $x_i$ with label $y_i \\in \\{0, 1\\}$:\n",
        "\n",
        "1. **Linear score**: $z_i = w^\\top x_i + b$\n",
        "2. **Sigmoid**: $\\hat{y}_i = \\sigma(z_i) = \\frac{1}{1 + e^{-z_i}}$\n",
        "3. **Binary cross-entropy loss**: $\\mathcal{L} = -\\frac{1}{n}\\sum_{i=1}^{n}[y_i \\log(\\hat{y}_i) + (1-y_i)\\log(1-\\hat{y}_i)]$\n",
        "4. **Gradients**:\n",
        "   - $\\frac{\\partial \\mathcal{L}}{\\partial w} = \\frac{1}{n} X^\\top (\\hat{y} - y)$\n",
        "   - $\\frac{\\partial \\mathcal{L}}{\\partial b} = \\frac{1}{n} \\sum (\\hat{y}_i - y_i)$\n",
        "5. **Update rule** (gradient descent): $w \\leftarrow w - \\alpha \\cdot \\nabla_w \\mathcal{L}$, same for $b$\n",
        "\n",
        "I also add **L2 regularization** (same as sklearn's default) which adds $\\frac{\\lambda}{2n}\\|w\\|^2$ to the loss and $\\frac{\\lambda}{n}w$ to the weight gradient. The goal is that my from-scratch implementation should converge to essentially the same solution as sklearn."
    ]

# 3. Find insertion point: after feature importance, before model summary
# Look for the "Model 1 Summary" or the cell with id containing "fff9cfc4" or similar
# Let's find the cell that starts the Model 1 Summary section
insert_idx = None
for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if "Model 1 Summary" in src and "Is This Actually Useful" in src:
        insert_idx = i
        print(f"Found 'Model 1 Summary' at cell index {i}")
        break

if insert_idx is None:
    # Fallback: look for the "Section 7" experiments header
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if "Section 7" in src and "Ablation" in src:
            insert_idx = i
            print(f"Found 'Section 7' at cell index {i}")
            break

if insert_idx is None:
    print("ERROR: Could not find insertion point")
    exit(1)

# 4. Insert scratch cells at the found position
for j, cell in enumerate(scratch_cells):
    cells.insert(insert_idx + j, cell)

print(f"Inserted {len(scratch_cells)} cells at index {insert_idx}")

# 5. Write back
with open("chess_pipeline_v2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! From-scratch LogReg section moved to before Model 1 Summary.")
