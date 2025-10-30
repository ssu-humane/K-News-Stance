import json
import jsonlines
import os
from typing import Dict, List

def load_test_data(data_path: str = 'data/k-news-stance-test.json') -> List[Dict]:
    """Load test dataset from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results(segment_results: List[Dict], article_results: List[Dict] = None, output_dir: str = 'results'):
    """Save prediction results to JSONL files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save segment results
    with jsonlines.open(f'{output_dir}/segment_predictions.jsonl', 'w') as writer:
        for result in segment_results:
            writer.write(result)
    print(f"Segment results saved to: {output_dir}/segment_predictions.jsonl")
    
    # Save article results if provided
    if article_results:
        with jsonlines.open(f'{output_dir}/article_predictions.jsonl', 'w') as writer:
            for result in article_results:
                writer.write(result)
        print(f"Article results saved to: {output_dir}/article_predictions.jsonl")

def add_xml_tags(article: str, headline_stance: str, lead_stance: str, 
                 quotations_stance: List[str], conclusion_stance: str,
                 headline: str, lead: str, quotations: List[Dict], conclusion: str) -> str:
    """Add XML tags with stance information to the article"""
    # English to Korean mapping for XML tags
    stance_mapping = {
        "supportive": "지지적",
        "neutral": "중립적",
        "oppositional": "비판적"
    }
    
    tagged_article = article
    
    # Add headline tags
    if headline and headline in tagged_article:
        korean_stance = stance_mapping.get(headline_stance, headline_stance)
        tagged_article = tagged_article.replace(
            headline,
            f'<제목 입장="{korean_stance}">{headline}</제목>',
            1
        )
    
    # Add lead tags
    if lead and tagged_article.startswith(lead):
        korean_stance = stance_mapping.get(lead_stance, lead_stance)
        tagged_article = tagged_article.replace(
            lead,
            f'<도입부 입장="{korean_stance}">{lead}</도입부>',
            1
        )
    
    # Add conclusion tags
    if conclusion and tagged_article.endswith(conclusion):
        korean_stance = stance_mapping.get(conclusion_stance, conclusion_stance)
        tagged_article = tagged_article.replace(
            conclusion,
            f'<결론부 입장="{korean_stance}">{conclusion}</결론부>',
            1
        )
    
    # Add quotation tags
    for i, quote in enumerate(quotations):
        quote_text = quote.get('quotation', '').strip()
        if quote_text and i < len(quotations_stance):
            stance = quotations_stance[i]
            korean_stance = stance_mapping.get(stance, stance)
            if quote_text in tagged_article:
                tagged_article = tagged_article.replace(
                    quote_text,
                    f'<직접 인용구 입장="{korean_stance}">{quote_text}</직접 인용구>',
                    1
                )
    
    return tagged_article