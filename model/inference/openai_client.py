import argparse
import os

import openai

parser = argparse.ArgumentParser(description="Inference for LLM OpenAI")
parser.add_argument('--api_key', type=str, help='Input your OpenAI API key')
args = parser.parse_args()
api_key = args.api_key

api_key = os.getenv("OPENAI_API_KEY") if api_key is None else api_key
client = openai.OpenAI(api_key=api_key)


def get_answer(prompt, model="gpt-4o", system_message="You are a helpful assistant."):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    print(get_answer(prompt))
