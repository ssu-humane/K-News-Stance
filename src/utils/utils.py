import os
import json
import yaml

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_prompts(prompt_dir, prompt_type, system_prompt_name):
    """Load system and user prompts from separate files"""
    system_prompt_path = os.path.join(prompt_dir, "system", f"{system_prompt_name}.txt")
    user_prompt_path = os.path.join(prompt_dir, "user", f"{prompt_type}.txt")
    
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    
    with open(user_prompt_path, 'r', encoding='utf-8') as f:
        user_template = f.read().strip()
    
    return system_prompt, user_template

def load_test_data(data_path, split_field="split_99", split_value="test"):
    """Load and filter test data"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item for item in data if item.get(split_field) == split_value]

def save_results(results_list, output_path):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)

def format_user_prompt(template, issue, news_headline, news_article, examples=None):
    """Format user prompt with actual data"""
    format_dict = {
        'issue': issue,
        'news_headline': news_headline, 
        'news_article': news_article
    }
    
    if examples is not None:
        format_dict['examples'] = examples
    
    return template.format(**format_dict)

def get_data_fields(config, record, field_key):
    """Extract data fields based on field key configuration"""
    field_mapping = config['data']['fields'][field_key]
    
    return {
        'issue': record.get(field_mapping['event_name'], ""),
        'news_headline': record.get(field_mapping['title'], ""),
        'news_article': record.get(field_mapping['news_content'], ""),
        'ground_truth': record.get(field_mapping['ground_truth'], "")
    }