import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import openai
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
from utils import load_config, load_prompts, load_test_data, save_results, format_user_prompt, get_data_fields

def main():
    parser = argparse.ArgumentParser(description='Run stance detection inference')
    parser.add_argument('--method', type=str, choices=['baseline', 'joa-icl-roberta', 'joa-icl-oracle'], 
                       required=True, help='Method to use: baseline, joa-icl-roberta, or joa-icl-oracle')
    parser.add_argument('--icl', type=str, choices=['zero-shot', 'zero-shot_cot', '6-shot', '6-shot_cot'],
                       default='zero-shot', help='ICL method: zero-shot, zero-shot_cot, 6-shot, 6-shot_cot')
    args = parser.parse_args()
    
    data_config = load_config("../../../configs/data_config.yaml")
    gen_config = load_config("../../../configs/generation_config.yaml")
    
    system_prompt_name = "instruction_baseline" if args.method == "baseline" else "instruction_joa_icl"
    
    if args.method == "baseline":
        field_key = "baseline"
    elif args.method == "joa-icl-roberta":
        field_key = "joa_icl_roberta"
    elif args.method == "joa-icl-oracle":
        field_key = "joa_icl_oracle"
    
    system_prompt, user_template = load_prompts(
        data_config['prompts']['prompt_dir'],
        args.icl,
        system_prompt_name
    )
    
    test_data = load_test_data(data_config['data']['test_data_path'])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(data_config['output']['results_dir'], f"gpt_{args.method}_{args.icl}_{timestamp}.json")

    load_dotenv(dotenv_path="../../../../.env")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = []
    
    for record in tqdm(test_data, desc="Processing"):
        data_fields = get_data_fields(data_config, record, field_key)
        
        user_prompt = format_user_prompt(
            user_template, 
            data_fields['issue'], 
            data_fields['news_headline'], 
            data_fields['news_article']
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = client.chat.completions.create(
                model=gen_config['model']['name'],
                messages=messages,
                max_tokens=gen_config['generation']['max_tokens'],
                temperature=gen_config['generation']['temperature'],
            )
            raw_response = response.choices[0].message.content
            
        except Exception as e:
            print(f"Error processing record {record.get('id')}: {e}")
            raw_response = ""
        
        result_record = {
            "id": record.get("id"),
            "gold": data_fields['ground_truth'],
            "pred": raw_response.strip(),
            "raw": raw_response
        }
        results.append(result_record)
    
    save_results(results, output_file)
    print(f"Inference completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()