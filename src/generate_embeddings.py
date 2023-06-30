```python
import openai
import pandas as pd

# Initialize the OpenAI API with your secret key
openai.api_key = 'your-secret-key'

# Define the path to the research paper
research_paper_path = 'path/to/your/research/paper.txt'

# Define the path to the CSV file where the embeddings will be saved
embeddings_csv_path = 'path/to/your/embeddings.csv'

def generate_embeddings(text):
    """
    Function to generate embeddings from a given text using the ada-002 model.
    """
    model = "text-davinci-002"
    tokens = openai.Completion.create(engine=model, prompt=text, max_tokens=60)
    return tokens.choices[0].text.strip()

def save_embeddings_to_csv(embeddings, csv_path):
    """
    Function to save the generated embeddings to a CSV file.
    """
    df = pd.DataFrame(embeddings)
    df.to_csv(csv_path, index=False)

def main():
    # Load the research paper
    with open(research_paper_path, 'r') as file:
        research_paper = file.read()

    # Generate embeddings for the research paper
    embeddings = generate_embeddings(research_paper)

    # Save the embeddings to a CSV file
    save_embeddings_to_csv(embeddings, embeddings_csv_path)

if __name__ == "__main__":
    main()
```