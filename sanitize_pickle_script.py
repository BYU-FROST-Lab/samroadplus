import sys
import pickle
import numpy as np

def _convert(obj):
    if isinstance(obj, dict):
        return {_convert(k): _convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert(i) for i in obj)
    elif isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
    elif hasattr(obj, 'item'):  # catches numpy scalars
        return obj.item()
    return obj

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sanitize_pickle_script.py <input.p> <output.p>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    with open(input_path, "rb") as f:
        data = pickle.load(f)
        
    clean_data = _convert(data)
    
    with open(output_path, "wb") as f:
        pickle.dump(clean_data, f)
