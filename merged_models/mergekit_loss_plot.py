import matplotlib.pyplot as plt

# Prompt-wise loss values from MergeKit evaluation
prompts = [
    "Explain federated learning",
    "What is QLoRA",
    "Use of LoRA",
    "Parameter-efficient tuning",
    "Privacy benefits of federated AI",
    "LLMs in education",
    "Instruction tuning",
    "MergeKit with adapters",
    "Model merging in federated learning",
    "Prompt-tuned adaptation"
]

losses = [
    11.4933, 11.1272, 11.6967, 11.2515, 11.7961,
    10.7768, 11.5647, 10.8475, 11.3924, 11.2186
]

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(prompts, losses)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Prompts")
plt.ylabel("Loss")
plt.title("MergeKit Prompt-wise Evaluation Loss")
plt.tight_layout()

# Annotate and save
for bar, loss in zip(bars, losses):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{loss:.2f}",
             ha='center', va='bottom', fontsize=9)

plt.savefig("eval-logs/mergekit_loss_plot.png", bbox_inches="tight")
print(" Plot saved to eval-logs/mergekit_loss_plot.png")
