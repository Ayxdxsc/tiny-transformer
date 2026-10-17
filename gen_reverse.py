import random

# Set seed for reproducibility
random.seed(42)

# Character set: a-z and A-Z
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

rows = []

# Generate strings of length 1 to 8
# 500 samples per length -> 4000 samples total
for L in range(1, 9):
    for _ in range(500):
        s = "".join(random.choice(CHARS) for _ in range(L))
        r = s[::-1]
        # Format: <u>input<a>reversed_output<eos>
        rows.append(f"<u>{s}<a>{r}<eos>")

# Shuffle the dataset
random.shuffle(rows)

# Write to data.txt
with open("data.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")

print(f"Generated {len(rows)} lines.")
print(f"Vocab size: {len(set(''.join(rows)))}")