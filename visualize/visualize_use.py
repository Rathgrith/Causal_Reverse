import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def generate_sentence_embeddings(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Check if the "completion" column exists
    if "completion_" not in df.columns:
        raise ValueError("The CSV file does not have a 'completion' column.")
    # Extract sentences from the "completion" column
    sentences = df["completion_"].tolist()
    # Load the Universal Sentence Encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # Generate embeddings
    embeddings = embed(sentences)

    return embeddings.numpy()


def visualize_embeddings(embeddings, sentences):
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Plot the reduced embeddings
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    
    # Annotate the points with the corresponding sentences
    for i, sentence in enumerate(sentences):
        plt.annotate(sentence, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)
    
    plt.title("t-SNE visualization of sentence embeddings")
    plt.show()
    plt.savefig("vis.svg")

if __name__ == "__main__":
    csv_file_path = "wandb_export_2023-10-21T21_16_45.719-05_00.csv"
    df = pd.read_csv(csv_file_path)
    print(df.info())
    sentences = df["completion_"].tolist()
    embeddings = generate_sentence_embeddings(csv_file_path)
    visualize_embeddings(embeddings, sentences)
