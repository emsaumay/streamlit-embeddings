```python
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# Function to read the research paper
def read_research_paper(file_path):
    with open(file_path, 'r') as file:
        research_paper = file.read().replace('\n', '')
    return research_paper

# Function to generate embeddings
def generate_embeddings(research_paper):
    sentences = research_paper.split('.')
    sentences = [sentence.split(' ') for sentence in sentences]
    model = Word2Vec(sentences, min_count=1)
    embeddings = model[model.wv.vocab]
    return embeddings

# Function to save embeddings to a file
def save_embeddings(embeddings, file_path):
    np.savetxt(file_path, embeddings, delimiter=',')

# File paths
research_paper_path = 'research_paper.txt'
embeddings_file_path = 'embeddings_file.txt'

# Read the research paper
research_paper = read_research_paper(research_paper_path)

# Generate embeddings
embeddings = generate_embeddings(research_paper)

# Save embeddings to a file
save_embeddings(embeddings, embeddings_file_path)
```