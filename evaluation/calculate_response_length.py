import json
import os
import numpy as np

def calculate_average_response_length(json_file_path, tokenizer=None, use_tokens=True):
    """
    Calculate the average length of predictions in a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing results
        tokenizer: AutoTokenizer instance (optional). If provided and use_tokens=True, 
                   will count tokens instead of characters
        use_tokens: Whether to use token count (True) or character count (False)
        
    Returns:
        dict with statistics about response lengths
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Handle MVSR format with problem_results
    if not results and 'problem_results' in data:
        # Extract first response from each problem in problem_results
        problem_results = data['problem_results']
        results = []
        for problem_idx, samples in problem_results.items():
            if samples and len(samples) > 0:
                # Take the first sample (sample_idx = 0)
                results.append(samples[0])
    
    if not results:
        return {
            'count': 0,
            'avg_length': 0,
            'min_length': 0,
            'max_length': 0,
            'median_length': 0
        }
    
    # Calculate lengths of all predictions
    lengths = []
    for item in results:
        prediction = item.get('prediction', '')
        
        if use_tokens and tokenizer is not None:
            # Use tokenizer to count tokens
            tokenized = tokenizer(prediction, add_special_tokens=False)
            length = len(tokenized['input_ids'])
        else:
            # Use character count
            length = len(prediction)
        
        lengths.append(length)
    
    return {
        'count': len(lengths),
        'avg_length': np.mean(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths)
    }


def main():
    # Option to use tokenizer for token-based length calculation
    use_tokenizer = True  # Set to False to use character count instead
    
    # Define your model names with their JSON file paths and tokenizer paths
    # Each model can have its own tokenizer
    models = {
        'Qwen2.5-VL-7B': {
            'json_path': 'results/qwen2.5_vl_7b/mmk12.json',
            'tokenizer_path': 'Qwen/Qwen2.5-VL-7B-Instruct'
        },
        'GRPO': {
            'json_path': 'results/7b_grpo_vision_120/mmk12.json',
            'tokenizer_path': 'checkpoints/mm-spo/7b_grpo_vision/global_step_120/actor/huggingface/'
        },
        'MSSR': {
            'json_path': 'results/7b_mssr_vision_120/mmk12.json',
            'tokenizer_path': 'checkpoints/mm-spo/7b_mssr_vision/global_step_120/actor/huggingface/'
        },
        'MVSR': {
            'json_path': 'results/7b_spo_vision_120/mmk12.json',
            'tokenizer_path': 'checkpoints/mm-spo/7b_spo_vision/global_step_120/actor/huggingface/'
        }
    }
    
    print("Response Length Statistics")
    print("=" * 80)
    
    all_stats = {}
    
    for model_name, model_config in models.items():
        json_path = model_config['json_path']
        tokenizer_path = model_config['tokenizer_path']
        
        if not os.path.exists(json_path):
            print(f"\n{model_name}: JSON file not found - {json_path}")
            continue
        
        # Load tokenizer for this specific model
        tokenizer = None
        if use_tokenizer:
            try:
                from transformers import AutoTokenizer
                print(f"\n{model_name}: Loading tokenizer from {tokenizer_path}")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                print(f"  Tokenizer loaded successfully!")
            except Exception as e:
                print(f"  Failed to load tokenizer: {e}")
                print(f"  Using character-based calculation for this model")
        
        stats = calculate_average_response_length(json_path, tokenizer=tokenizer, use_tokens=(use_tokenizer and tokenizer is not None))
        all_stats[model_name] = stats
        
        unit = "tokens" if (use_tokenizer and tokenizer is not None) else "characters"
        
        print(f"\n{model_name} Results:")
        print(f"  Number of responses: {stats['count']}")
        print(f"  Average length: {stats['avg_length']:.2f} {unit}")
        print(f"  Median length: {stats['median_length']:.2f} {unit}")
        print(f"  Min length: {stats['min_length']} {unit}")
        print(f"  Max length: {stats['max_length']} {unit}")
        print(f"  Std deviation: {stats['std_length']:.2f} {unit}")
    
    # Summary comparison
    if all_stats:
        unit = "tokens" if use_tokenizer else "characters"
        print("\n" + "=" * 80)
        print(f"Summary Comparison (Average Length):")
        print("-" * 80)
        sorted_models = sorted(all_stats.items(), key=lambda x: x[1]['avg_length'], reverse=True)
        for model_name, stats in sorted_models:
            print(f"  {model_name:20s}: {stats['avg_length']:8.2f}")


if __name__ == "__main__":
    # Example: If you want to specify files via command line or directly here
    # Update the paths in the models dictionary above
    
    # Or you can specify them here:
    import sys
    
    if len(sys.argv) > 1:
        # If JSON files are provided as command line arguments
        use_tokenizer = True
        tokenizer = None
        
        if use_tokenizer:
            try:
                from transformers import AutoTokenizer
                tokenizer_path = "/apdcephfs_gy2/share_302625456/model/pretrain/Qwen/Qwen2.5-VL-7B-Instruct"
                print(f"Loading tokenizer from: {tokenizer_path}")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                print("Tokenizer loaded successfully!")
                print("Using TOKEN-based length calculation\n")
            except Exception as e:
                print(f"Failed to load tokenizer: {e}")
                print("Falling back to character-based length calculation\n")
                use_tokenizer = False
        
        print("Response Length Statistics")
        print("=" * 80)
        unit = "tokens" if use_tokenizer and tokenizer else "characters"
        
        for i, json_path in enumerate(sys.argv[1:], 1):
            model_name = os.path.basename(json_path).replace('.json', '')
            
            if not os.path.exists(json_path):
                print(f"\nModel {i} ({model_name}): File not found - {json_path}")
                continue
            
            stats = calculate_average_response_length(json_path, tokenizer=tokenizer, use_tokens=use_tokenizer)
            
            print(f"\nModel {i} ({model_name}):")
            print(f"  Number of responses: {stats['count']}")
            print(f"  Average length: {stats['avg_length']:.2f} {unit}")
            print(f"  Median length: {stats['median_length']:.2f} {unit}")
            print(f"  Min length: {stats['min_length']} {unit}")
            print(f"  Max length: {stats['max_length']} {unit}")
            print(f"  Std deviation: {stats['std_length']:.2f} {unit}")
    else:
        main()
