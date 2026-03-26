import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.linewidth": 1,
    "lines.linewidth": 2.2,
})

# Data
epochs = list(range(20))

train_loss = [
6.25,5.48,5.01,4.72,4.65,4.62,4.62,4.60,4.63,4.67,
4.71,4.77,4.82,4.89,4.96,5.04,5.14,5.20,5.25,5.32
]

dev_loss = [
4.68,4.43,4.15,4.04,3.96,3.90,3.85,3.82,3.81,3.78,
3.77,3.76,3.78,3.76,3.74,3.76,3.77,3.77,3.77,3.77
]

tf_ratio = [
1.000,0.976,0.952,0.928,0.903,0.879,0.855,0.831,0.807,0.783,
0.759,0.734,0.710,0.686,0.662,0.638,0.614,0.590,0.566,0.541
]

# Create figure
fig, ax1 = plt.subplots(figsize=(8,5))

# Plot loss curves
ax1.plot(epochs, train_loss, color="#1f77b4", label="Train Loss")
ax1.plot(epochs, dev_loss, color="#d62728", linestyle="--", label="Dev Loss")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_xlim(0,19)

# Grid
ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

# Remove top/right borders
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Early stopping marker
ax1.axvline(x=19, color="gray", linestyle=":", linewidth=1.5)

# Annotation
ax1.annotate(
    "Early stopping\n(no dev improvement for 5 epochs)",
    xy=(19, dev_loss[-1]),
    xytext=(8, 4.2),  # Move text further left
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=11,
    color="gray"
)

# Secondary axis for teacher forcing ratio
ax2 = ax1.twinx()
ax2.plot(
    epochs,
    tf_ratio,
    color="#2ca02c",
    linestyle=":",
    label="Teacher Forcing Ratio"
)

ax2.set_ylabel("Teacher Forcing Ratio")
ax2.set_ylim(0,1.05)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    frameon=False
)

plt.tight_layout()

# Save figure
plt.savefig("training_loss_curve_ieee.png", dpi=600, bbox_inches="tight")

plt.show()