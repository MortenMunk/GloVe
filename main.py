import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_glove_vectors(path):
    """
    Load GloVe vectors from text file.
    Each line: symbol dim1 dim2 ... dimN
    Returns: vocab (list of symbols), embeddings (np.ndarray)
    """
    vocab = []
    vectors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 2:
                continue
            symbol = parts[0]
            vec = np.array(parts[1:], dtype=float)
            vocab.append(symbol)
            vectors.append(vec)
    return vocab, np.vstack(vectors)


def build_cipher_to_letter_from_json(json_path):
    """
    Load key.json and build mapping {cipher_id_as_str: plaintext_letter}.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for letter, ids in data["key"].items():
        for cid in ids:
            mapping[str(cid)] = letter
    return mapping


def plot_embeddings_2d(
    embeddings, vocab, cipher_to_letter=None, save_path=None, random_state=42
):
    """
    Visualize embeddings in 2D using t-SNE and matplotlib.
    """
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        perplexity=min(30, len(vocab) - 1),
    )
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))

    colors = None
    if cipher_to_letter:
        letters = [cipher_to_letter.get(sym, "?") for sym in vocab]
        unique_letters = sorted(set(letters))
        color_map = {ltr: idx for idx, ltr in enumerate(unique_letters)}
        colors = [color_map[ltr] for ltr in letters]
        scatter = plt.scatter(
            emb_2d[:, 0], emb_2d[:, 1], c=colors, cmap="tab20", s=60, alpha=0.7
        )
        # legend
        handles, _ = scatter.legend_elements()
        plt.legend(handles, unique_letters, title="Plaintext")
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=60, alpha=0.7)

    # add labels on each point
    for i, cipher_symbol in enumerate(vocab):
        label = str(cipher_symbol)
        if cipher_to_letter and cipher_symbol in cipher_to_letter:
            label = f"{cipher_symbol}:{cipher_to_letter[cipher_symbol]}"
        plt.text(
            emb_2d[i, 0], emb_2d[i, 1], label, fontsize=7, ha="center", va="center"
        )

    plt.title("Cipher Symbol Embeddings (t-SNE 2D)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    vocab, embeddings = load_glove_vectors("vectors.txt")
    cipher_to_letter = build_cipher_to_letter_from_json("key.json")
    plot_embeddings_2d(embeddings, vocab, cipher_to_letter=cipher_to_letter)
