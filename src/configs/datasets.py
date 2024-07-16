from dataclasses import dataclass
    
@dataclass
class alpaca_dataset:
    dataset: str = "semikong_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"