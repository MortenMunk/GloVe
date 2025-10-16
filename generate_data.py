import json

# === Load data ===
with open("key.json", "r") as f:
    data = json.load(f)

plaintext = list(data["plaintext"])  # characters as list
ciphertext = data["ciphertext"].split()  # list of integer tokens (strings)

# === Check length consistency ===
if len(plaintext) != len(ciphertext):
    print(
        f"⚠️ Warning: plaintext ({len(plaintext)}) and ciphertext ({len(ciphertext)}) lengths differ."
    )
    n = min(len(plaintext), len(ciphertext))
    plaintext = plaintext[:n]
    ciphertext = ciphertext[:n]
else:
    print("✅ Plaintext and ciphertext lengths match.")

# === Write plaintext.txt ===
with open("plaintext.txt", "w") as f:
    f.write(" ".join(plaintext))
print("✅ Created plaintext.txt")

# === Write ciphertext.txt ===
with open("ciphertext.txt", "w") as f:
    f.write(" ".join(ciphertext))
print("✅ Created ciphertext.txt")

# === Write combined.txt (non-interleaved) ===
# First line = ciphertext, second line = plaintext
with open("combined.txt", "w") as f:
    f.write(" ".join(ciphertext) + "\n")
    f.write(" ".join(plaintext) + "\n")
print("✅ Created combined.txt (ciphertext + plaintext on separate lines)")

print("\nAll files created successfully:")
print(" - plaintext.txt  → character-level corpus")
print(" - ciphertext.txt → numeric-token corpus")
print(" - combined.txt   → both for joint-space training")
