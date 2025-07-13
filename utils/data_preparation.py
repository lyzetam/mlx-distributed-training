"""Data preparation utilities for MLX training."""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries from the file
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: Union[str, Path]):
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def format_for_chat(data: List[Dict], instruction_key: str = "instruction", 
                   response_key: str = "response", context_key: Optional[str] = "context") -> List[Dict]:
    """
    Format data for chat-based training.
    
    Args:
        data: Raw data list
        instruction_key: Key for instruction/question
        response_key: Key for response/answer
        context_key: Optional key for context
        
    Returns:
        Formatted data for MLX training
    """
    formatted_data = []
    
    for item in data:
        # Build user content
        user_content = item.get(instruction_key, "")
        if context_key and context_key in item and item[context_key]:
            user_content = f"{user_content}\n\nContext: {item[context_key]}"
        
        # Create message format
        formatted_item = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item.get(response_key, "")}
            ]
        }
        formatted_data.append(formatted_item)
    
    return formatted_data


def split_dataset(data: List[Dict], val_ratio: float = 0.2, 
                 shuffle: bool = True, seed: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into training and validation sets.
    
    Args:
        data: Full dataset
        val_ratio: Ratio of data to use for validation
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data)
    """
    if seed is not None:
        random.seed(seed)
    
    data_copy = data.copy()
    
    if shuffle:
        random.shuffle(data_copy)
    
    split_idx = int(len(data_copy) * (1 - val_ratio))
    train_data = data_copy[:split_idx]
    val_data = data_copy[split_idx:]
    
    return train_data, val_data


def prepare_dataset(file_path: Union[str, Path], 
                   output_dir: Union[str, Path] = "data",
                   val_ratio: float = 0.2,
                   format_type: str = "chat",
                   shuffle: bool = True,
                   seed: Optional[int] = 42) -> Dict[str, Path]:
    """
    Prepare a dataset for training by loading, formatting, and splitting.
    
    Args:
        file_path: Path to raw data file
        output_dir: Directory to save processed data
        val_ratio: Validation set ratio
        format_type: Format type ("chat" or "raw")
        shuffle: Whether to shuffle data
        seed: Random seed
        
    Returns:
        Dictionary with paths to train and validation files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {file_path}...")
    data = load_jsonl(file_path)
    print(f"Loaded {len(data)} examples")
    
    # Format data if needed
    if format_type == "chat":
        print("Formatting data for chat training...")
        data = format_for_chat(data)
    
    # Split dataset
    print(f"Splitting dataset (validation ratio: {val_ratio})...")
    train_data, val_data = split_dataset(data, val_ratio, shuffle, seed)
    print(f"Train: {len(train_data)} examples, Validation: {len(val_data)} examples")
    
    # Save processed data
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "valid.jsonl"
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    
    print(f"Saved training data to {train_path}")
    print(f"Saved validation data to {val_path}")
    
    return {
        "train": train_path,
        "valid": val_path,
        "train_size": len(train_data),
        "valid_size": len(val_data)
    }


def validate_dataset(data_dir: Union[str, Path]) -> bool:
    """
    Validate that a dataset directory has the required files.
    
    Args:
        data_dir: Directory containing train.jsonl and valid.jsonl
        
    Returns:
        True if valid, False otherwise
    """
    data_dir = Path(data_dir)
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"
    
    if not train_file.exists():
        print(f"Training file not found: {train_file}")
        return False
    
    if not valid_file.exists():
        print(f"Validation file not found: {valid_file}")
        return False
    
    # Check if files are not empty
    try:
        train_data = load_jsonl(train_file)
        valid_data = load_jsonl(valid_file)
        
        if len(train_data) == 0:
            print("Training file is empty")
            return False
        
        if len(valid_data) == 0:
            print("Validation file is empty")
            return False
        
        print(f"Dataset validated: {len(train_data)} train, {len(valid_data)} validation examples")
        return True
        
    except Exception as e:
        print(f"Error validating dataset: {e}")
        return False


def create_sample_dataset(output_path: Union[str, Path], num_examples: int = 100):
    """
    Create a sample dataset for testing.
    
    Args:
        output_path: Path to save the sample dataset
        num_examples: Number of examples to generate
    """
    sample_data = []
    
    topics = ["science", "history", "technology", "art", "literature"]
    
    for i in range(num_examples):
        topic = random.choice(topics)
        sample_data.append({
            "instruction": f"Tell me about {topic} topic #{i}",
            "context": f"This is some context about {topic}",
            "response": f"Here's an interesting fact about {topic}: Example response #{i}",
            "category": topic
        })
    
    save_jsonl(sample_data, output_path)
    print(f"Created sample dataset with {num_examples} examples at {output_path}")
