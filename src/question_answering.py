```python
import pandas as pd
from openai import OpenAI, Completion

openai = OpenAI("your-api-key")

def load_embeddings():
    return pd.read_csv('embeddings.csv')

def generate_answer(question, embeddings):
    prompt = f"{question}\n{embeddings}"
    response = openai.Completion.create(engine="ada-002", prompt=prompt, max_tokens=100)
    return response.choices[0].text.strip()

def main():
    embeddings = load_embeddings()
    while True:
        question = input("Enter your question: ")
        answer = generate_answer(question, embeddings)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
```