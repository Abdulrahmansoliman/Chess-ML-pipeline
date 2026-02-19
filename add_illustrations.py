import json

with open("chess_pipeline_v2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ============================================================
# ILLUSTRATION 1: ML Pipeline Flowchart
# Insert after intro section (find "section2-md")
# ============================================================
pipeline_md = {
    "cell_type": "markdown",
    "id": "pipeline-diagram-md",
    "metadata": {},
    "source": [
        "### My ML Pipeline at a Glance\n",
        "\n",
        "Before diving into the code, here's the full pipeline I'm building — from raw chess games to actionable predictions:"
    ]
}

pipeline_code = {
    "cell_type": "code",
    "execution_count": None,
    "id": "pipeline-diagram-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Pipeline Illustration\n",
        "fig, ax = plt.subplots(figsize=(14, 4))\n",
        "ax.set_xlim(0, 14)\n",
        "ax.set_ylim(0, 4)\n",
        "ax.axis('off')\n",
        "ax.set_title('Chess ML Pipeline Overview', fontsize=16, fontweight='bold', pad=20)\n",
        "\n",
        "# Define pipeline stages\n",
        "stages = [\n",
        "    {'name': 'Raw PGN\\nData', 'x': 0.5, 'color': '#E8D5B7', 'desc': '1,451 games'},\n",
        "    {'name': 'Data\\nCleaning', 'x': 2.7, 'color': '#B7D5E8', 'desc': '1,250 games'},\n",
        "    {'name': 'Feature\\nExtraction', 'x': 4.9, 'color': '#B7E8C4', 'desc': '11 features'},\n",
        "    {'name': 'Train/Test\\nSplit', 'x': 7.1, 'color': '#E8E8B7', 'desc': '80/20'},\n",
        "    {'name': 'Model\\nTraining', 'x': 9.3, 'color': '#D5B7E8', 'desc': 'LogReg + RF'},\n",
        "    {'name': 'Evaluation\\n& Insights', 'x': 11.5, 'color': '#E8B7B7', 'desc': 'Accuracy, PR, ROC'},\n",
        "]\n",
        "\n",
        "box_w, box_h = 1.8, 1.8\n",
        "\n",
        "for i, s in enumerate(stages):\n",
        "    # Draw box\n",
        "    rect = plt.Rectangle((s['x'], 1.1), box_w, box_h, \n",
        "                          linewidth=2, edgecolor='#333', facecolor=s['color'],\n",
        "                          alpha=0.85, zorder=2)\n",
        "    rect.set_clip_on(False)\n",
        "    ax.add_patch(rect)\n",
        "    # Stage name\n",
        "    ax.text(s['x'] + box_w/2, 2.15, s['name'], ha='center', va='center',\n",
        "            fontsize=10, fontweight='bold', zorder=3)\n",
        "    # Description\n",
        "    ax.text(s['x'] + box_w/2, 0.65, s['desc'], ha='center', va='center',\n",
        "            fontsize=8, fontstyle='italic', color='#555', zorder=3)\n",
        "    # Arrow (except last)\n",
        "    if i < len(stages) - 1:\n",
        "        ax.annotate('', xy=(stages[i+1]['x'] - 0.05, 2.0),\n",
        "                    xytext=(s['x'] + box_w + 0.05, 2.0),\n",
        "                    arrowprops=dict(arrowstyle='->', color='#333', lw=2),\n",
        "                    zorder=1)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# Find section2-md and insert before it
for i, cell in enumerate(cells):
    if cell.get("id") == "section2-md":
        cells.insert(i, pipeline_code)
        cells.insert(i, pipeline_md)
        print(f"Inserted pipeline diagram before cell {i} (section2-md)")
        break

# ============================================================
# ILLUSTRATION 2: Logistic Regression Architecture
# Insert right after the "scratch-logreg-md" cell
# ============================================================
arch_md = {
    "cell_type": "markdown",
    "id": "logreg-arch-md",
    "metadata": {},
    "source": [
        "### Logistic Regression as a \"Neural Network\"\n",
        "\n",
        "Logistic regression is actually the simplest possible neural network — a single neuron with a sigmoid activation. Here's what it looks like visually:"
    ]
}

arch_code = {
    "cell_type": "code",
    "execution_count": None,
    "id": "logreg-arch-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Logistic Regression Architecture Diagram\n",
        "fig, ax = plt.subplots(figsize=(12, 7))\n",
        "ax.set_xlim(-1, 11)\n",
        "ax.set_ylim(-1, 8)\n",
        "ax.axis('off')\n",
        "ax.set_title('Logistic Regression Model Architecture', fontsize=14, fontweight='bold', pad=15)\n",
        "\n",
        "# Input features (left side)\n",
        "features = ['rating_diff', 'base_sec', 'inc_sec', 'is_white',\n",
        "            'castled', 'opp_castled', 'queen_moved',\n",
        "            'pawn_moves', 'minor_moves', 'captures', 'checks']\n",
        "\n",
        "n_feat = len(features)\n",
        "y_positions = np.linspace(0.5, 7, n_feat)\n",
        "\n",
        "# Draw input nodes\n",
        "for j, (feat, yp) in enumerate(zip(features, y_positions)):\n",
        "    color = '#B7D5E8' if j < 4 else '#B7E8C4'  # blue=context, green=behavior\n",
        "    circle = plt.Circle((1.0, yp), 0.3, color=color, ec='#333', lw=1.5, zorder=3)\n",
        "    ax.add_patch(circle)\n",
        "    ax.text(-0.1, yp, feat, ha='right', va='center', fontsize=8, fontweight='bold')\n",
        "    # Connection line to sum node\n",
        "    ax.plot([1.3, 4.4], [yp, 3.75], color='#999', lw=0.8, alpha=0.6, zorder=1)\n",
        "\n",
        "# Sum node (weighted sum)\n",
        "sum_circle = plt.Circle((5.0, 3.75), 0.55, color='#E8E8B7', ec='#333', lw=2, zorder=3)\n",
        "ax.add_patch(sum_circle)\n",
        "ax.text(5.0, 3.75, '$\\\\Sigma$', ha='center', va='center', fontsize=20, fontweight='bold', zorder=4)\n",
        "ax.text(5.0, 2.8, '$z = w^T x + b$', ha='center', va='center', fontsize=9, fontstyle='italic')\n",
        "\n",
        "# Arrow from sum to sigmoid\n",
        "ax.annotate('', xy=(7.0, 3.75), xytext=(5.6, 3.75),\n",
        "            arrowprops=dict(arrowstyle='->', color='#333', lw=2))\n",
        "\n",
        "# Sigmoid node\n",
        "sig_circle = plt.Circle((7.5, 3.75), 0.55, color='#D5B7E8', ec='#333', lw=2, zorder=3)\n",
        "ax.add_patch(sig_circle)\n",
        "ax.text(7.5, 3.75, '$\\\\sigma$', ha='center', va='center', fontsize=20, fontweight='bold', zorder=4)\n",
        "ax.text(7.5, 2.8, '$\\\\sigma(z) = \\\\frac{1}{1+e^{-z}}$', ha='center', va='center', fontsize=9, fontstyle='italic')\n",
        "\n",
        "# Arrow from sigmoid to output\n",
        "ax.annotate('', xy=(9.2, 3.75), xytext=(8.1, 3.75),\n",
        "            arrowprops=dict(arrowstyle='->', color='#333', lw=2))\n",
        "\n",
        "# Output node\n",
        "out_circle = plt.Circle((9.7, 3.75), 0.45, color='#E8B7B7', ec='#333', lw=2, zorder=3)\n",
        "ax.add_patch(out_circle)\n",
        "ax.text(9.7, 3.75, 'P(win)', ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)\n",
        "\n",
        "# Legend\n",
        "legend_y = -0.5\n",
        "ax.add_patch(plt.Rectangle((0.5, legend_y), 0.4, 0.3, color='#B7D5E8', ec='#333'))\n",
        "ax.text(1.1, legend_y + 0.15, 'Context features (4)', fontsize=8, va='center')\n",
        "ax.add_patch(plt.Rectangle((4.0, legend_y), 0.4, 0.3, color='#B7E8C4', ec='#333'))\n",
        "ax.text(4.6, legend_y + 0.15, 'Behavior features (7)', fontsize=8, va='center')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# Find scratch-logreg-md and insert architecture diagram after it
for i, cell in enumerate(cells):
    if cell.get("id") == "scratch-logreg-md":
        cells.insert(i + 1, arch_code)
        cells.insert(i + 1, arch_md)
        print(f"Inserted architecture diagram after cell {i} (scratch-logreg-md)")
        break

# ============================================================
# ILLUSTRATION 3: Training Loop Visualization
# Insert right before "scratch-training-code"
# ============================================================
training_loop_md = {
    "cell_type": "markdown",
    "id": "training-loop-diagram-md",
    "metadata": {},
    "source": [
        "### How Training Works: The Gradient Descent Loop\n",
        "\n",
        "Here's what happens inside `model.fit()` — the iterative process of learning the weights:"
    ]
}

training_loop_code = {
    "cell_type": "code",
    "execution_count": None,
    "id": "training-loop-diagram-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Training Loop Illustration\n",
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        "ax.set_xlim(0, 10)\n",
        "ax.set_ylim(0, 8)\n",
        "ax.axis('off')\n",
        "ax.set_title('Gradient Descent Training Loop', fontsize=14, fontweight='bold', pad=15)\n",
        "\n",
        "# Step boxes arranged in a cycle\n",
        "steps = [\n",
        "    {'name': '1. Initialize\\nWeights', 'x': 3.5, 'y': 6.5, 'color': '#E8D5B7',\n",
        "     'detail': 'w ~ N(0, 0.01), b = 0'},\n",
        "    {'name': '2. Forward Pass', 'x': 7.0, 'y': 5.0, 'color': '#B7D5E8',\n",
        "     'detail': 'z = Xw + b,  y_hat = sigmoid(z)'},\n",
        "    {'name': '3. Compute Loss', 'x': 7.0, 'y': 2.5, 'color': '#E8B7B7',\n",
        "     'detail': 'L = -mean(y*log(y_hat) + ...)'},\n",
        "    {'name': '4. Compute\\nGradients', 'x': 3.5, 'y': 1.0, 'color': '#B7E8C4',\n",
        "     'detail': 'dw = X.T @ (y_hat - y) / n'},\n",
        "    {'name': '5. Update\\nWeights', 'x': 0.5, 'y': 2.5, 'color': '#D5B7E8',\n",
        "     'detail': 'w = w - lr * dw'},\n",
        "    {'name': '6. Converged?', 'x': 0.5, 'y': 5.0, 'color': '#E8E8B7',\n",
        "     'detail': 'Check iterations or loss'},\n",
        "]\n",
        "\n",
        "bw, bh = 2.2, 1.2\n",
        "\n",
        "for j, s in enumerate(steps):\n",
        "    rect = plt.Rectangle((s['x'], s['y'] - bh/2), bw, bh,\n",
        "                          linewidth=2, edgecolor='#333', facecolor=s['color'],\n",
        "                          alpha=0.85, zorder=2)\n",
        "    ax.add_patch(rect)\n",
        "    ax.text(s['x'] + bw/2, s['y'] + 0.1, s['name'], ha='center', va='center',\n",
        "            fontsize=10, fontweight='bold', zorder=3)\n",
        "    ax.text(s['x'] + bw/2, s['y'] - 0.4, s['detail'], ha='center', va='center',\n",
        "            fontsize=7, fontstyle='italic', color='#444', zorder=3)\n",
        "\n",
        "# Arrows connecting steps in a loop\n",
        "# 1 -> 2\n",
        "ax.annotate('', xy=(7.0, 5.6), xytext=(5.7, 6.3),\n",
        "            arrowprops=dict(arrowstyle='->', color='#333', lw=2, connectionstyle='arc3,rad=-0.2'))\n",
        "# 2 -> 3\n",
        "ax.annotate('', xy=(8.1, 3.1), xytext=(8.1, 4.4),\n",
        "            arrowprops=dict(arrowstyle='->', color='#333', lw=2))\n",
        "# 3 -> 4\n",
        "ax.annotate('', xy=(5.7, 1.2), xytext=(7.0, 1.9),\n",
        "            arrowprops=dict(arrowstyle='->', color='#333', lw=2, connectionstyle='arc3,rad=-0.2'))\n",
        "# 4 -> 5\n",
        "ax.annotate('', xy=(2.7, 2.0), xytext=(3.5, 1.2),\n",
        "            arrowprops=dict(arrowstyle='->', color='#333', lw=2, connectionstyle='arc3,rad=-0.2'))\n",
        "# 5 -> 6\n",
        "ax.annotate('', xy=(1.6, 4.4), xytext=(1.6, 3.1),\n",
        "            arrowprops=dict(arrowstyle='->', color='#333', lw=2))\n",
        "# 6 -> 2 (loop back) with 'No' label\n",
        "ax.annotate('', xy=(7.0, 5.3), xytext=(2.7, 5.3),\n",
        "            arrowprops=dict(arrowstyle='->', color='#C44', lw=2, linestyle='dashed'))\n",
        "ax.text(4.85, 5.55, 'No → repeat', fontsize=8, ha='center', color='#C44', fontweight='bold')\n",
        "# 6 -> Done\n",
        "ax.annotate('', xy=(1.6, 6.8), xytext=(1.6, 5.6),\n",
        "            arrowprops=dict(arrowstyle='->', color='#4A4', lw=2))\n",
        "ax.text(1.6, 7.2, 'Yes → Done!', fontsize=9, ha='center', color='#4A4', fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# Find scratch-training-code and insert before it
for i, cell in enumerate(cells):
    if cell.get("id") == "scratch-training-code":
        cells.insert(i, training_loop_code)
        cells.insert(i, training_loop_md)
        print(f"Inserted training loop diagram before cell {i} (scratch-training-code)")
        break

# Write back
with open("chess_pipeline_v2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nDone! Added 3 illustrations (6 cells total):")
print("  1. Pipeline flowchart (after Section 1 intro)")
print("  2. LogReg architecture diagram (in Section 8)")
print("  3. Training loop diagram (in Section 8)")
