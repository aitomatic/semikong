import json
from datasets import load_dataset
import requests
from huggingface_hub import configure_http_backend, get_session
from huggingface_hub import snapshot_download

# Create a factory function that returns a Session with configured proxies

def backend_factory() -> requests.Session:

    session = requests.Session()

    session.proxies = {"http": "", "https": ""}

    return session

def prepare_data_for_llama_factory(dataset_name, column_name, split='train', output_file='text_data.json', debug=False, sample_size=10):
    """
    Downloads a dataset from Hugging Face, extracts a specified column, and saves it to a JSON file.

    Args:
        dataset_name (str): The name of the dataset to load.
        column_name (str): The name of the column to extract.
        split (str): The split of the dataset to load (default is 'train').
        output_file (str): The path to the output JSON file (default is 'text_data.json').
        debug (bool): Whether to save only a small sample of the data for debugging (default is False).
        sample_size (int): The number of samples to save if debug is True (default is 10).

    Returns:
        str: The path to the output JSON file.
    """
    # Load the dataset from Hugging Face
    #dataset = load_dataset(dataset_name, split=split, token="")
    dataset = load_dataset("", data_files={"train": []})
    #print(dataset)
    #snapshot_download(repo_id="",
    #                  repo_type="dataset",
    #                  local_dir=output_file)

    # # Extract the specified column and format as a list of dictionaries
    column_data = [{"text": entry[column_name]} for entry in dataset["train"]]

    # If debug is True, select a small sample of the data
    if debug:
        column_data = column_data[:sample_size]

    # Save the extracted data to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(column_data, json_file, indent=4)

    return output_file

def main():
    # Define the parameters for the dataset processing
    dataset_name = None
    column_name = None
    split = None
    output_file = None
    debug = False
    sample_size = 10
    
    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    # In practice, this is mostly done internally in `huggingface_hub`
    session = get_session()

    # Call the function with the defined parameters
    output_path = prepare_data_for_llama_factory(dataset_name, column_name, split, output_file, debug, sample_size)
    print(f"Data has been saved to: {output_path}")

if __name__ == "__main__":
    main()