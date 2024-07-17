import requests
import time
import json
from typing import Iterable, List
import logging
import os

BASE_URL = os.environ["PATH"]

HEADERS = {"Content-Type": "application/json", "User-Agent": "Test Client"}

"""
You are an Expert Senior Semiconductor Engineer.
        Your role is to answer the input in the most sophisticate, detail and technical ways.
        This answer will be read by other Senior Semiconductor Engineer.
        Your answer must be concise, precise and in-depth technical knowledge.
"""


def post_http_request(prompt: str) -> requests.Response:
    """Sending Request to SEMIKONG 8b FastAPI Endpoint

    Args:
        prompt (str): the input prompt

    Returns:
        requests.Response: the response of the request
    """
    payload = {
        "prompt": f"""
        You are an Expert Senior Semiconductor Engineer who mastered the Semiconductor Manufacturing Process.
        Your role is to use your's in-depth knowledge in the field of Semiconductor Manufacturing Process and answer the input in the most sophisticate, detail and technical ways.
        Your answer will be reviewed by other Senior Semiconductor Engineer.
        Your answer must be concise, precise and in-depth technical knowledge.
        Now here is the input: {prompt}.
        Your Answer: <your answer here>""",
        "max_tokens": 200,
        "temperature": 0,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.5,
        "top_p": 0,
        "top_k": -1,
        "min_p": 0.7,
        "n": 1,
        "best_of": 1,
        "stream": True,
    }
    response = requests.post(BASE_URL, headers=HEADERS, json=payload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\0"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]

    return output


def post_process(response_output: str):
    clean_text = response_output.split("<your answer here>")
    return clean_text[-1]


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename="data.log", encoding="utf-8", level=logging.DEBUG)

    prompt_input = str(input("Input your prompt: "))

    logger.info("Start inferencing ....")
    start = time.time()
    response = post_http_request(prompt_input)
    output = get_streaming_response(response)[0]  # only 1 output as list
    output = post_process(output)
    logger.info(f"Take {time.time() - start} seconds time finish")
    logger.info("Finish inferencing")

    with open("output.txt", "a") as file:
        file.write(f'Total tokens: {len(output.split(" "))} \n\n')
        file.write(f"Answer: {output} \n\n")
        file.write("----------------------------------- \n\n")


if __name__ == "__main__":
    main()