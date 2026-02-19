import json

with open("chess_pipeline_v2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Find insertion point: just before "data-references-md" cell
insert_idx = None
for i, cell in enumerate(cells):
    if cell.get("id") == "data-references-md":
        insert_idx = i
        break

if insert_idx is None:
    print("ERROR: Could not find data-references-md cell")
    exit(1)

print(f"Found insertion point at cell index {insert_idx}")

new_cells = []

# --- SECTION 8: From-Scratch Logistic Regression ---

# Markdown: Section header
new_cells.append({
    "cell_type": "markdown",
    "id": "scratch-logreg-md",
    "metadata": {},
    "source": [
        "## Section 8: Logistic Regression from Scratch\n",
        "\n",
        "### Why Build It Myself?\n",
        "\n",
        "My professor requires that I implement at least one model from scratch, not just call sklearn. This is important because:\n",
        "\n",
        "1. **Understanding**: Calling `model.fit()` is a black box. Writing the gradient descent loop forces me to understand what's actually happening.\n",
        "2. **Math verification**: I can check that my implementation matches sklearn's results, which validates my understanding of the math.\n",
        "3. **Flexibility**: A from-scratch model lets me add custom loss functions, regularization, or logging that sklearn doesn't support.\n",
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
        "I also add **L2 regularization** (same as sklearn's default) which adds $\\frac{\\lambda}{2n}\\|w\\|^2$ to the loss and $\\frac{\\lambda}{n}w$ to the weight gradient."
    ]
})

# Code: LogisticRegressionScratch class
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "scratch-logreg-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "class LogisticRegressionScratch:\n",
        "    \"\"\"\n",
        "    Logistic Regression implemented from scratch using gradient descent.\n",
        "    \n",
        "    This mirrors sklearn's LogisticRegression but exposes the internals:\n",
        "    - Manual sigmoid computation\n",
        "    - Explicit gradient descent loop\n",
        "    - Loss history tracking for convergence analysis\n",
        "    - L2 regularization (equivalent to sklearn's default)\n",
        "    \"\"\"\n",
        "    \n",
        "    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=1.0, random_state=42):\n",
        "        self.learning_rate = learning_rate\n",
        "        self.n_iterations = n_iterations\n",
        "        self.lambda_reg = lambda_reg\n",
        "        self.random_state = random_state\n",
        "        self.weights = None\n",
        "        self.bias = None\n",
        "        self.loss_history = []\n",
        "        self.coef_ = None  # sklearn-compatible attribute\n",
        "    \n",
        "    def _sigmoid(self, z):\n",
        "        \"\"\"Sigmoid activation: maps any real number to (0, 1).\"\"\"\n",
        "        # Clip to avoid overflow in exp\n",
        "        z = np.clip(z, -500, 500)\n",
        "        return 1.0 / (1.0 + np.exp(-z))\n",
        "    \n",
        "    def _compute_loss(self, y_true, y_pred):\n",
        "        \"\"\"Binary cross-entropy loss with L2 regularization.\"\"\"\n",
        "        n = len(y_true)\n",
        "        # Clip predictions to avoid log(0)\n",
        "        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)\n",
        "        # Cross-entropy\n",
        "        ce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))\n",
        "        # L2 regularization term\n",
        "        l2_term = (self.lambda_reg / (2 * n)) * np.sum(self.weights ** 2)\n",
        "        return ce_loss + l2_term\n",
        "    \n",
        "    def fit(self, X, y, verbose=True):\n",
        "        \"\"\"\n",
        "        Train the model using gradient descent.\n",
        "        \n",
        "        Args:\n",
        "            X: Feature matrix (n_samples, n_features), should be scaled\n",
        "            y: Binary labels (n_samples,)\n",
        "            verbose: Print progress every 100 iterations\n",
        "        \"\"\"\n",
        "        np.random.seed(self.random_state)\n",
        "        n_samples, n_features = X.shape\n",
        "        \n",
        "        # Initialize weights to small random values\n",
        "        self.weights = np.random.randn(n_features) * 0.01\n",
        "        self.bias = 0.0\n",
        "        self.loss_history = []\n",
        "        \n",
        "        for i in range(self.n_iterations):\n",
        "            # Forward pass\n",
        "            z = X @ self.weights + self.bias\n",
        "            y_pred = self._sigmoid(z)\n",
        "            \n",
        "            # Compute loss\n",
        "            loss = self._compute_loss(y, y_pred)\n",
        "            self.loss_history.append(loss)\n",
        "            \n",
        "            # Compute gradients\n",
        "            error = y_pred - y\n",
        "            dw = (1 / n_samples) * (X.T @ error) + (self.lambda_reg / n_samples) * self.weights\n",
        "            db = (1 / n_samples) * np.sum(error)\n",
        "            \n",
        "            # Update parameters\n",
        "            self.weights -= self.learning_rate * dw\n",
        "            self.bias -= self.learning_rate * db\n",
        "            \n",
        "            if verbose and (i + 1) % 200 == 0:\n",
        "                print(f\"  Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}\")\n",
        "        \n",
        "        # Store in sklearn-compatible format\n",
        "        self.coef_ = np.array([self.weights])\n",
        "        \n",
        "        if verbose:\n",
        "            print(f\"  Training complete. Final loss: {self.loss_history[-1]:.4f}\")\n",
        "        \n",
        "        return self\n",
        "    \n",
        "    def predict_proba(self, X):\n",
        "        \"\"\"Return probability estimates (sklearn-compatible: returns [P(0), P(1)]).\"\"\"\n",
        "        z = X @ self.weights + self.bias\n",
        "        prob_1 = self._sigmoid(z)\n",
        "        prob_0 = 1 - prob_1\n",
        "        return np.column_stack([prob_0, prob_1])\n",
        "    \n",
        "    def predict(self, X, threshold=0.5):\n",
        "        \"\"\"Return binary predictions.\"\"\"\n",
        "        proba = self.predict_proba(X)[:, 1]\n",
        "        return (proba >= threshold).astype(int)\n",
        "\n",
        "\n",
        "print(\"LogisticRegressionScratch class defined.\")"
    ]
})

# Markdown: Training the scratch model
new_cells.append({
    "cell_type": "markdown",
    "id": "scratch-training-md",
    "metadata": {},
    "source": [
        "### Training My From-Scratch Model\n",
        "\n",
        "I'll train it on the same scaled training data and compare results with sklearn. The key thing to watch is the **loss convergence curve** — it should decrease smoothly and plateau, confirming that gradient descent is working correctly."
    ]
})

# Code: Train scratch model + loss curve
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "scratch-training-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train the from-scratch model\n",
        "print(\"Training LogisticRegressionScratch...\")\n",
        "scratch_model = LogisticRegressionScratch(\n",
        "    learning_rate=0.1,\n",
        "    n_iterations=1000,\n",
        "    lambda_reg=1.0,\n",
        "    random_state=RANDOM_SEED\n",
        ")\n",
        "scratch_model.fit(X_train_scaled, y_train)\n",
        "\n",
        "# Plot loss convergence\n",
        "fig, ax = plt.subplots(figsize=(10, 5))\n",
        "ax.plot(scratch_model.loss_history, color='steelblue', linewidth=1.5)\n",
        "ax.set_xlabel('Iteration')\n",
        "ax.set_ylabel('Loss (Binary Cross-Entropy + L2)')\n",
        "ax.set_title('Loss Convergence: From-Scratch Logistic Regression')\n",
        "ax.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f\"\\nInitial loss: {scratch_model.loss_history[0]:.4f}\")\n",
        "print(f\"Final loss:   {scratch_model.loss_history[-1]:.4f}\")\n",
        "print(f\"Reduction:    {((scratch_model.loss_history[0] - scratch_model.loss_history[-1]) / scratch_model.loss_history[0] * 100):.1f}%\")"
    ]
})

# Markdown: Comparison
new_cells.append({
    "cell_type": "markdown",
    "id": "scratch-comparison-md",
    "metadata": {},
    "source": [
        "### Comparing Scratch vs sklearn\n",
        "\n",
        "If my implementation is correct, the two models should produce very similar (not identical, due to different optimization algorithms) accuracy and coefficient values."
    ]
})

# Code: Compare scratch vs sklearn
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "scratch-comparison-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Compare predictions\n",
        "scratch_pred = scratch_model.predict(X_test_scaled)\n",
        "scratch_proba = scratch_model.predict_proba(X_test_scaled)[:, 1]\n",
        "\n",
        "sklearn_pred = sklearn_model.predict(X_test_scaled)\n",
        "sklearn_proba = sklearn_model.predict_proba(X_test_scaled)[:, 1]\n",
        "\n",
        "print(\"=\" * 60)\n",
        "print(\"SCRATCH vs SKLEARN COMPARISON\")\n",
        "print(\"=\" * 60)\n",
        "print(f\"\\n{'Metric':<25} {'Scratch':>12} {'Sklearn':>12}\")\n",
        "print(\"-\" * 50)\n",
        "print(f\"{'Test Accuracy':<25} {accuracy_score(y_test, scratch_pred):>12.3f} {accuracy_score(y_test, sklearn_pred):>12.3f}\")\n",
        "\n",
        "# Compare coefficients\n",
        "print(f\"\\n{'Feature':<25} {'Scratch':>12} {'Sklearn':>12}\")\n",
        "print(\"-\" * 50)\n",
        "for feat, sc, sk in zip(FEATURE_COLS, scratch_model.weights, sklearn_model.coef_[0]):\n",
        "    print(f\"{feat:<25} {sc:>12.4f} {sk:>12.4f}\")\n",
        "print(f\"{'bias':<25} {scratch_model.bias:>12.4f} {sklearn_model.intercept_[0]:>12.4f}\")\n",
        "\n",
        "print(\"\\n→ The models should show similar patterns. Small differences are expected\")\n",
        "print(\"  because sklearn uses L-BFGS optimization while I use vanilla gradient descent.\")"
    ]
})

# Code: Scratch model evaluation with ModelEvaluator
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "scratch-eval-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Use our ModelEvaluator on the scratch model\n",
        "scratch_evaluator = ModelEvaluator(\n",
        "    model=scratch_model,\n",
        "    X_test=X_test_scaled,\n",
        "    y_test=y_test,\n",
        "    feature_names=FEATURE_COLS,\n",
        "    class_labels=['not-win', 'win']\n",
        ")\n",
        "\n",
        "print(\"=\" * 50)\n",
        "print(\"FROM-SCRATCH MODEL EVALUATION\")\n",
        "print(\"=\" * 50)\n",
        "print(f\"\\nAccuracy: {accuracy_score(y_test, scratch_evaluator.y_pred):.3f}\")\n",
        "print(f\"\\nClassification Report:\")\n",
        "print(classification_report(y_test, scratch_evaluator.y_pred, target_names=['not-win', 'win']))\n",
        "\n",
        "# Side-by-side confusion matrices\n",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
        "\n",
        "cm_sk = confusion_matrix(y_test, sklearn_pred)\n",
        "disp_sk = ConfusionMatrixDisplay(cm_sk, display_labels=['not-win', 'win'])\n",
        "disp_sk.plot(ax=axes[0], cmap='Blues')\n",
        "axes[0].set_title('sklearn LogReg')\n",
        "\n",
        "cm_sc = confusion_matrix(y_test, scratch_pred)\n",
        "disp_sc = ConfusionMatrixDisplay(cm_sc, display_labels=['not-win', 'win'])\n",
        "disp_sc.plot(ax=axes[1], cmap='Oranges')\n",
        "axes[1].set_title('From-Scratch LogReg')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# --- SECTION 9: Bayesian Perspective ---

# Markdown: Bayesian section header
new_cells.append({
    "cell_type": "markdown",
    "id": "bayesian-md",
    "metadata": {},
    "source": [
        "## Section 9: The Bayesian Perspective — Why It Matters\n",
        "\n",
        "### What's Wrong with Just Using Point Estimates?\n",
        "\n",
        "Everything I've done so far is **frequentist**: I find a single \"best\" set of weights $w^*$ that minimizes the loss. But this gives me no idea how **confident** I should be in those weights.\n",
        "\n",
        "For example, my model says `rating_diff` has a coefficient of ~1.45. But is that estimate precise? Could it be 1.2 or 1.7? With only ~1,200 games, there's real uncertainty here.\n",
        "\n",
        "### The Bayesian Framework\n",
        "\n",
        "Instead of finding a single $w^*$, Bayesian inference treats the weights as **random variables** with a distribution:\n",
        "\n",
        "$$P(w \\mid \\text{data}) = \\frac{P(\\text{data} \\mid w) \\cdot P(w)}{P(\\text{data})}$$\n",
        "\n",
        "Where:\n",
        "- **Prior** $P(w)$: What I believe about the weights *before* seeing data. I use $w \\sim \\mathcal{N}(0, \\tau^2 I)$ — I expect weights to be small.\n",
        "- **Likelihood** $P(\\text{data} \\mid w)$: How well the model explains the observed wins/losses. This is the Bernoulli likelihood from logistic regression.\n",
        "- **Posterior** $P(w \\mid \\text{data})$: My updated belief about the weights *after* seeing data. This is what I want.\n",
        "- **Evidence** $P(\\text{data})$: A normalizing constant (usually intractable, but we can work around it).\n",
        "\n",
        "### The Connection to L2 Regularization\n",
        "\n",
        "Here's the beautiful connection: **sklearn's L2-regularized logistic regression is actually computing the MAP (Maximum A Posteriori) estimate** under a Gaussian prior.\n",
        "\n",
        "$$w_{\\text{MAP}} = \\arg\\max_w \\log P(w \\mid \\text{data}) = \\arg\\min_w \\left[ -\\log P(\\text{data} \\mid w) + \\frac{\\lambda}{2}\\|w\\|^2 \\right]$$\n",
        "\n",
        "The L2 penalty $\\frac{\\lambda}{2}\\|w\\|^2$ is exactly the negative log of a Gaussian prior! So every time I use `LogisticRegression(C=1.0)`, I'm implicitly doing Bayesian inference — just taking the single most likely point instead of the full distribution.\n",
        "\n",
        "### Why Go Full Bayesian?\n",
        "\n",
        "The MAP estimate gives me a point. The full posterior gives me a **distribution**. This means:\n",
        "\n",
        "1. **Uncertainty quantification**: Instead of \"this coefficient is 1.45\", I get \"this coefficient is 1.45 ± 0.3 (95% credible interval)\"\n",
        "2. **Prediction uncertainty**: Instead of \"60% win probability\", I get \"60% ± 8% win probability\" — much more honest.\n",
        "3. **Small data robustness**: With ~1,200 games, the prior provides regularization that adapts to data size automatically.\n",
        "4. **Model comparison**: Bayesian model evidence naturally penalizes overly complex models (Occam's razor).\n",
        "\n",
        "### Practical Impact for My Chess Analysis\n",
        "\n",
        "If my model says \"early queen moves reduce win probability by 8%\" but the 95% credible interval is [-2%, 18%], then I shouldn't trust that finding. The Bayesian approach would tell me this directly, while the frequentist approach might make me overconfident in a noisy estimate.\n",
        "\n",
        "**Note:** A full Bayesian implementation (e.g., using MCMC sampling) is beyond the scope of this first pipeline. But the MAP connection shows that my current model already has Bayesian foundations, and I can extend it in future work."
    ]
})

# Code: Demonstrate MAP connection
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "bayesian-demo-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Demonstrate the MAP-regularization connection\n",
        "print(\"=\" * 60)\n",
        "print(\"BAYESIAN CONNECTION: L2 Regularization = Gaussian Prior\")\n",
        "print(\"=\" * 60)\n",
        "print()\n",
        "print(\"sklearn LogisticRegression(C=1.0) is equivalent to:\")\n",
        "print(\"  MAP estimate with Gaussian prior N(0, tau^2)\")\n",
        "print(f\"  where tau^2 = C = 1.0 (sklearn default)\")\n",
        "print(f\"  and lambda = 1/C = 1.0 (regularization strength)\")\n",
        "print()\n",
        "print(\"This means:\")\n",
        "print(\"  - Larger C → weaker prior → less regularization → more trust in data\")\n",
        "print(\"  - Smaller C → stronger prior → more regularization → more trust in prior\")\n",
        "print()\n",
        "\n",
        "# Show how different priors (C values) affect coefficients\n",
        "C_values = [0.01, 0.1, 1.0, 10.0, 100.0]\n",
        "print(f\"\\n{'C (prior width)':<18} {'rating_diff coef':>18} {'queen_move coef':>18}\")\n",
        "print(\"-\" * 55)\n",
        "\n",
        "for C in C_values:\n",
        "    model_c = LogisticRegression(C=C, random_state=RANDOM_SEED, max_iter=1000)\n",
        "    model_c.fit(X_train_scaled, y_train)\n",
        "    rd_idx = FEATURE_COLS.index('rating_diff')\n",
        "    qm_idx = FEATURE_COLS.index('my_queen_moved_by_N')\n",
        "    print(f\"{C:<18.2f} {model_c.coef_[0][rd_idx]:>18.4f} {model_c.coef_[0][qm_idx]:>18.4f}\")\n",
        "\n",
        "print(\"\\n→ As C increases (weaker prior), coefficients grow larger.\")\n",
        "print(\"  The prior 'shrinks' coefficients toward zero, preventing overfitting.\")\n",
        "print(\"  This IS Bayesian reasoning, even in a frequentist framework.\")"
    ]
})

# Markdown: Summary
new_cells.append({
    "cell_type": "markdown",
    "id": "bayesian-summary-md",
    "metadata": {},
    "source": [
        "### Key Takeaway\n",
        "\n",
        "The Bayesian perspective isn't just theoretical — it's already baked into my model through L2 regularization. What I've shown:\n",
        "\n",
        "1. **Every regularized model is implicitly Bayesian** — the regularization term corresponds to a prior distribution on the weights.\n",
        "2. **The strength of regularization (C parameter) controls how much we trust the data vs the prior.** For my small dataset (~1,200 games), strong regularization (small C) is important to avoid overfitting.\n",
        "3. **Going full Bayesian** (computing the entire posterior, not just the MAP point) would give me uncertainty estimates on every prediction — useful for knowing when to trust the model and when not to.\n",
        "\n",
        "In future iterations, I could use:\n",
        "- **Laplace approximation**: Approximate the posterior as a Gaussian around the MAP estimate (cheap and fast)\n",
        "- **MCMC sampling** (e.g., with PyMC3): Sample from the true posterior for full uncertainty quantification\n",
        "- **Variational inference**: A middle ground between MAP and full MCMC\n",
        "\n",
        "For now, knowing that my L2-regularized logistic regression is a MAP estimate gives me confidence that the model is well-founded mathematically."
    ]
})

# Insert all new cells
for i, cell in enumerate(new_cells):
    cells.insert(insert_idx + i, cell)

# Write back
with open("chess_pipeline_v2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully inserted {len(new_cells)} new cells into the notebook.")
print("Sections added: Section 8 (From-Scratch LogReg) and Section 9 (Bayesian Perspective)")
