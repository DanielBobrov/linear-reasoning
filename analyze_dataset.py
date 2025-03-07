import json
import os
from pathlib import Path
from collections import Counter

def analyze_dataset():
    print("It works")
    """Analyze the dataset structure, particularly length differences between input and target"""
    data_dir = Path("/kaggle/input/paper-data/data/comparison.1000.12.6")
    
    # Check data files
    file_types = ["train.json", "valid.json", "test.json"]
    results = {}
    
    for file_type in file_types:
        file_path = data_dir / file_type
        if not file_path.exists():
            print(f"File {file_path} does not exist. Skipping.")
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Analyze differences between input_text and target_text
        length_diffs = []
        type_counts = Counter()
        
        for item in data:
            input_len = len(item['input_text'].split("><"))
            target_len = len(item['target_text'].split("><"))
            length_diffs.append(target_len - input_len)
            type_counts[item.get('type', 'unknown')] += 1
            
            # Check if target text contains </a>
            if '</a>' in item['target_text'] and '</a>' not in item['input_text']:
                item['has_end_tag'] = True
        
        # Calculate statistics
        avg_diff = sum(length_diffs) / len(length_diffs) if length_diffs else 0
        max_diff = max(length_diffs) if length_diffs else 0
        min_diff = min(length_diffs) if length_diffs else 0
        
        # Sample a few examples
        samples = data[:3] if data else []
        
        results[file_type] = {
            "count": len(data),
            "type_counts": dict(type_counts),
            "length_difference": {
                "average": avg_diff,
                "max": max_diff,
                "min": min_diff
            },
            "samples": samples
        }
    
    # Check special tokens in vocab
    vocab_path = data_dir / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        special_tokens = [token for token in vocab if token.startswith('<') and token.endswith('>')]
        end_tokens = [token for token in vocab if token == '</a>']
        
        results["vocabulary"] = {
            "size": len(vocab),
            "special_tokens_count": len(special_tokens),
            "special_tokens_sample": special_tokens[:10],
            "has_end_tag": '</a>' in vocab,
            "end_tokens": end_tokens
        }
    
    # Print the analysis
    print("\n===== DATASET ANALYSIS =====\n")
    
    for file_type, data in results.items():
        if file_type == "vocabulary":
            print(f"\nVOCABULARY ANALYSIS:")
            print(f"  Size: {data['size']} tokens")
            print(f"  Special tokens count: {data['special_tokens_count']}")
            print(f"  Special tokens sample: {', '.join(data['special_tokens_sample'])}")
            print(f"  Has </a> end tag: {data['has_end_tag']}")
            print(f"  End tokens: {data['end_tokens']}")
        else:
            print(f"\nFILE: {file_type}")
            print(f"  Sample count: {data['count']}")
            print(f"  Type distribution: {data['type_counts']}")
            print(f"  Length difference (target - input):")
            print(f"    Average: {data['length_difference']['average']:.2f} tokens")
            print(f"    Max: {data['length_difference']['max']} tokens")
            print(f"    Min: {data['length_difference']['min']} tokens")
            
            print("\n  SAMPLES:")
            for i, sample in enumerate(data['samples']):
                print(f"    Sample {i+1}:")
                print(f"      Input:  {sample['input_text']}")
                print(f"      Target: {sample['target_text']}")
                print(f"      Type: {sample.get('type', 'unknown')}")
                print(f"      Has </a> end tag: {sample.get('has_end_tag', False)}")
                
    return results

print("?")
if __name__ == "__main__":
    print("!")
    analyze_dataset()
