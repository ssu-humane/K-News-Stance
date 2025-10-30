import torch
import google.generativeai as genai
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import add_xml_tags, load_test_data, save_results

# Configure Gemini API
genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")

class StanceDetector:
    def __init__(self):
        # Label mappings
        self.LABEL_ID2NAME = {
            0: "supportive",
            1: "neutral", 
            2: "oppositional"
        }
        
        # Load segment model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = self.load_segment_model()
        
        # Load prompt and initialize Gemini model
        with open('prompt/joa_icl.txt', 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    
    def load_segment_model(self):
        """Load the segment-level stance detection model"""
        model_dir = "humane-lab/klue-roberta-large-JoA-segment"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(self.device)
        model.eval()
        return tokenizer, model
    
    def predict_segment_stance(self, issue: str, text: str) -> str:
        """Predict stance for a single segment"""
        encoded = self.tokenizer(
            text,
            issue,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
        
        english_label = self.LABEL_ID2NAME[pred_id]
        return english_label
    
    def predict_article_stance(self, issue: str, headline: str, tagged_article: str, show_input: bool = False) -> str:
        """Predict article-level stance using Gemini"""
        prompt = self.prompt_template.format(
            issue=issue,
            headline=headline,
            article=tagged_article
        )
        
        if show_input:
            print("="*80)
            print("ARTICLE-LEVEL STANCE PREDICTION Prompt:")
            print(prompt)
            print("="*80)
        
        try:
            response = self.gemini_model.generate_content(prompt)
            prediction = response.text.strip()
            
            if show_input:
                print(f"Gemini Response: {prediction}")
                print("="*80)
            
            return prediction.strip()
        except Exception as e:
            print(f"Error in Gemini prediction: {e}")
            return "neutral"  # fallback
    
    def process_dataset(self):
        """Process the test dataset"""
        # Load test data
        test_data = load_test_data()
        
        segment_results = []
        article_results = []
        
        print(f"Processing {len(test_data)} samples...")
        
        for i, sample in enumerate(tqdm(test_data, desc="Processing samples")):
            
            # Extract data
            sample_id = sample['id']
            issue = sample['issue']
            headline = sample['headline']
            lead = sample['lead']
            quotations = sample.get('quotations', [])
            conclusion = sample.get('conclusion', '')
            article = sample['article']
            
            # Segment-level predictions
            headline_stance = self.predict_segment_stance(issue, headline)
            lead_stance = self.predict_segment_stance(issue, lead)
            conclusion_stance = self.predict_segment_stance(issue, conclusion) if conclusion else ""
            
            quotations_stance = []
            for quote in quotations:
                quote_text = quote.get('quotation', '')
                if quote_text:
                    stance = self.predict_segment_stance(issue, quote_text)
                    quotations_stance.append(stance)
            
            # Save segment results
            segment_result = {
                "id": sample_id,
                "headline_stance": headline_stance,
                "lead_stance": lead_stance,
                "quotations_stance": quotations_stance,
                "conclusion_stance": conclusion_stance
            }
            segment_results.append(segment_result)
            
            # Create tagged article
            tagged_article = add_xml_tags(
                article, headline_stance, lead_stance, quotations_stance, 
                conclusion_stance, headline, lead, quotations, conclusion
            )
            
            # Article-level prediction (show input for first sample)
            show_input = (i == 0)
            article_stance = self.predict_article_stance(issue, headline, tagged_article, show_input)
            
            # Save article result
            article_result = {
                "id": sample_id,
                "prediction": article_stance,
                "gold": sample.get('article_stance', '')
            }
            article_results.append(article_result)
        
        # Save results
        save_results(segment_results, article_results)

def main():
    detector = StanceDetector()
    detector.process_dataset()

if __name__ == "__main__":
    main()