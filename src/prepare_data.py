import json
from typing import List, Dict
import os

def load_data(filepath: str) -> List[List[str]]:
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def save_data(data: List[List[str]], filepath: str):
    with open(filepath, 'w') as file:
        json.dump(data, file)

def preprocess_data(data: List[List[str]]) -> List[List[str]]:
    #TODO: migrate from old repo
    return data
