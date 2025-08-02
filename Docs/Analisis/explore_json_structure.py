import json
import os

def explore_structure(data, level=0, max_level=4, path=""):
    """Explore JSON structure recursively"""
    indent = "  " * level
    
    if level > max_level:
        return
    
    if isinstance(data, dict):
        print(f"{indent}Dict with {len(data)} keys")
        for i, (key, value) in enumerate(list(data.items())[:5]):  # Show first 5 keys
            print(f"{indent}  '{key}': ", end="")
            if isinstance(value, (dict, list)):
                print()
                explore_structure(value, level + 2, max_level, path + f"['{key}']")
            else:
                print(f"{type(value).__name__} = {str(value)[:50]}...")
        if len(data) > 5:
            print(f"{indent}  ... and {len(data) - 5} more keys")
    
    elif isinstance(data, list):
        print(f"{indent}List with {len(data)} items")
        if len(data) > 0:
            print(f"{indent}  First item type: {type(data[0]).__name__}")
            if isinstance(data[0], (dict, list)):
                explore_structure(data[0], level + 2, max_level, path + "[0]")
    
    else:
        print(f"{indent}{type(data).__name__}: {str(data)[:100]}...")

def main():
    file_path = '/Users/haroldgomez/Downloads/cumulative_results_20250730_071510.json'
    
    print(f"File size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
    print("\nLoading JSON file...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("\nJSON Structure:")
    print("="*60)
    explore_structure(data)
    
    # Look specifically at results structure
    if 'results' in data:
        print("\n\nDetailed Results Structure:")
        print("="*60)
        results = data['results']
        print(f"Results type: {type(results)}")
        
        if isinstance(results, dict):
            # Show first few keys to understand model/question structure
            keys = list(results.keys())[:10]
            print(f"First 10 keys in results: {keys}")
            
            # Check if keys are models or questions
            first_key = keys[0] if keys else None
            if first_key:
                print(f"\nExploring results['{first_key}']:")
                explore_structure(results[first_key], max_level=3)

if __name__ == "__main__":
    main()