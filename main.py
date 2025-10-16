import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Patch


# === Load GloVe vectors ===
def load_glove_vectors(path):
    vocab = []
    vectors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 2 or parts[0] == "<unk>":
                continue
            vocab.append(parts[0])
            vectors.append(np.array(parts[1:], dtype=float))
    return vocab, np.vstack(vectors)


# === Build ciphertext → plaintext mapping from JSON ===
def build_cipher_to_letter_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for letter, ids in data["key"].items():
        for cid in ids:
            mapping[str(cid)] = letter
    return mapping


# === Run t-SNE ===
def run_tsne_once(embeddings, vocab, random_state=42):
    print(f"Running t-SNE on {len(vocab)} symbols ...")
    perplexity = min(20, len(vocab) - 1)  # suitable for ~200 tokens

    try:
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            init="pca",
            perplexity=perplexity,
            max_iter=500,
            method="barnes_hut",
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            init="pca",
            perplexity=perplexity,
            n_iter=500,
            method="barnes_hut",
        )

    emb_2d = tsne.fit_transform(embeddings)
    print("t-SNE complete.")
    return emb_2d


# === Plot embeddings: plaintext vs ciphertext ===
def plot_embeddings_plain(emb_2d, vocab, cipher_to_letter, save_path="plot_plain.png"):
    plt.figure(figsize=(12, 10))

    # Determine color: teal for plaintext, red for ciphertext
    colors = ["teal" if not sym.isdigit() else "tab:red" for sym in vocab]
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=60, alpha=0.7)

    for i, sym in enumerate(vocab):
        if sym.isdigit():
            label = f"{sym}:{cipher_to_letter.get(sym, '?')}"
        else:
            label = sym  # show plaintext letter only
        plt.text(
            emb_2d[i, 0], emb_2d[i, 1], label, fontsize=7, ha="center", va="center"
        )

    legend_elements = [
        Patch(facecolor="teal", label="Plaintext"),
        Patch(facecolor="tab:red", label="Ciphertext"),
    ]
    plt.legend(handles=legend_elements, title="Token Type")
    plt.title("Plaintext & Ciphertext Embeddings (t-SNE 2D)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved {save_path}")


# === Plot vowels vs consonants coloring (for qualitative grouping) ===
def plot_embeddings_vc(emb_2d, vocab, cipher_to_letter, save_path="plot_vc.png"):
    plt.figure(figsize=(12, 10))
    vowels = set("aeiou")
    colors = []

    for sym in vocab:
        ltr = cipher_to_letter.get(sym, "?").lower()
        if ltr in vowels:
            colors.append("royalblue")
        elif ltr.isalpha():
            colors.append("darkorange")
        else:
            colors.append("gray")

    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=60, alpha=0.7)

    for i, sym in enumerate(vocab):
        if sym.isdigit():
            label = f"{sym}:{cipher_to_letter.get(sym, '?')}"
        else:
            label = sym
        plt.text(
            emb_2d[i, 0], emb_2d[i, 1], label, fontsize=7, ha="center", va="center"
        )

    legend_elements = [
        Patch(facecolor="royalblue", label="Vowels"),
        Patch(facecolor="darkorange", label="Consonants"),
        Patch(facecolor="gray", label="Other / Unknown"),
    ]
    plt.legend(handles=legend_elements, title="Letter Type")

    plt.title("Vowels vs Consonants (t-SNE 2D)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved {save_path}")


# === Main ===
if __name__ == "__main__":
    vocab, embeddings = load_glove_vectors("vectors.txt")

    cipher_to_letter = build_cipher_to_letter_from_json("key.json")

    emb_2d = run_tsne_once(embeddings, vocab)

    plot_embeddings_plain(emb_2d, vocab, cipher_to_letter)
    plot_embeddings_vc(emb_2d, vocab, cipher_to_letter)
