```python
import pandas as pd

def save_embeddings_to_csv(embeddings, filename):
    df = pd.DataFrame(embeddings)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Assuming embeddings are generated and stored in a variable named `embeddings`
    # And the filename to save the embeddings is `embeddings.csv`
    save_embeddings_to_csv(embeddings, 'embeddings.csv')
```