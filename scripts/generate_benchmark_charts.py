#!/usr/bin/env python3
"""
Generate visualization graphs for benchmark report.

Creates:
- Token cost comparison chart
- Time overhead comparison
- Batch scaling analysis
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_token_comparison_chart():
    """Chart 1: Token cost per generation method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    methods = ["Single\nGeneration", "Batch (5)", "Batch (151)"]
    mcp_tokens = [20000, 100000, 3020000]  # ~4k per call * 5 calls
    direct_tokens = [0, 0, 0]

    x = np.arange(len(methods))
    width = 0.35

    # Bars
    bars1 = ax.bar(x - width / 2, [t / 1000 for t in mcp_tokens], width, label="MCP", color="#ff6b6b")
    _bars2 = ax.bar(x + width / 2, direct_tokens, width, label="Direct API", color="#4ecdc4")

    # Labels
    ax.set_xlabel("Test Scenario", fontsize=12)
    ax.set_ylabel("Token Cost (thousands)", fontsize=12)
    ax.set_title("MCP vs Direct API - Token Cost Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.0f}k",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "token_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'token_comparison.png'}")


def create_cost_savings_chart():
    """Chart 2: Cost savings by batch size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data - batch sizes vs savings
    batch_sizes = [1, 10, 50, 100, 151]
    mcp_cost = [s * 0.4 for s in batch_sizes]  # $0.40 per generation via MCP
    direct_cost = [0] * len(batch_sizes)
    savings = [mcp - direct for mcp, direct in zip(mcp_cost, direct_cost)]

    # Plot
    ax.fill_between(batch_sizes, savings, alpha=0.3, color="#ff6b6b", label="MCP Cost")
    ax.plot(batch_sizes, savings, color="#ff6b6b", linewidth=2, marker="o")
    ax.axhline(y=0, color="#4ecdc4", linestyle="--", linewidth=2, label="Direct API Cost ($0)")

    # Labels
    ax.set_xlabel("Batch Size (generations)", fontsize=12)
    ax.set_ylabel("Cost (USD)", fontsize=12)
    ax.set_title("Cost Savings with Direct API by Batch Size", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation for 151 batch
    ax.annotate(
        "$60.40 savings\nat 151 generations",
        xy=(151, 60.40),
        xytext=(100, 70),
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontsize=10,
        ha="center",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cost_savings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'cost_savings.png'}")


def create_latency_comparison_chart():
    """Chart 3: Latency overhead comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    operations = ["Template\nDiscovery", "Workflow\nGeneration", "Validation", "Queue", "Wait"]
    mcp_latency = [50, 100, 30, 50, 15000]  # in ms (wait includes actual generation)
    direct_latency = [5, 10, 5, 10, 15000]  # HTTP overhead only

    x = np.arange(len(operations))
    width = 0.35

    # Bars (use log scale for visibility)
    _bars1 = ax.bar(x - width / 2, mcp_latency, width, label="MCP", color="#ff6b6b")
    _bars2 = ax.bar(x + width / 2, direct_latency, width, label="Direct API", color="#4ecdc4")

    # Labels
    ax.set_xlabel("Operation", fontsize=12)
    ax.set_ylabel("Latency (ms, log scale)", fontsize=12)
    ax.set_title("Operation Latency: MCP vs Direct API", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both", axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'latency_comparison.png'}")


def create_repo_migration_chart():
    """Chart 4: Migration priority matrix."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data: (repo, daily_gens, token_savings_per_day)
    repos = [
        ("pokedex-generator", 151, 60.40),
        ("KDH-Automation", 25, 10.00),
        ("Goat", 15, 6.00),
        ("RobloxChristian", 30, 12.00),
    ]

    names = [r[0] for r in repos]
    gens = [r[1] for r in repos]
    savings = [r[2] for r in repos]

    # Bubble chart
    colors = ["#ff6b6b", "#f9ca24", "#6c5ce7", "#4ecdc4"]
    _scatter = ax.scatter(gens, savings, s=[g * 20 for g in gens], c=colors, alpha=0.6, edgecolors="black")

    # Labels
    for i, name in enumerate(names):
        ax.annotate(
            name, (gens[i], savings[i]), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold"
        )

    ax.set_xlabel("Daily Generations", fontsize=12)
    ax.set_ylabel("Daily Token Savings ($)", fontsize=12)
    ax.set_title("Repository Migration Priority Matrix", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add quadrant lines
    ax.axhline(y=np.mean(savings), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=np.mean(gens), color="gray", linestyle="--", alpha=0.5)

    # Quadrant labels
    ax.text(
        0.95,
        0.95,
        "High Priority\nMigrate First",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        alpha=0.7,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "migration_priority.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'migration_priority.png'}")


def create_summary_metrics_table():
    """Create a summary metrics table as an image."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Data for table
    headers = ["Metric", "MCP", "Direct API", "Improvement"]
    rows = [
        ["Token Cost per Call", "~4,000 tokens", "0 tokens", "100% reduction"],
        ["Token Cost (151 batch)", "~3,020,000 tokens", "0 tokens", "$60.40 savings"],
        ["Overhead Latency", "~150ms per op", "~10ms per op", "15x faster"],
        ["Context Window", "Consumes ~8k tools", "No tool overhead", "Unlimited"],
        ["Production Use", "Interactive only", "All batch scripts", "Recommended"],
    ]

    # Create table
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="left", loc="center", colWidths=[0.3, 0.25, 0.25, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#2c3e50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#ecf0f1")

    ax.set_title("Summary: MCP vs Direct API Comparison", fontsize=14, fontweight="bold", pad=20)

    plt.savefig(OUTPUT_DIR / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'summary_table.png'}")


def main():
    """Generate all benchmark visualizations."""
    print("Generating benchmark visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    create_token_comparison_chart()
    create_cost_savings_chart()
    create_latency_comparison_chart()
    create_repo_migration_chart()
    create_summary_metrics_table()

    print()
    print("All visualizations generated successfully!")
    print(f"Location: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
