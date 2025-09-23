from collections import Counter
import spacy
import json
import numpy as np
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm  # For progress bars

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spite_ai.config import Config

config = Config.from_env()

CORPUS_PATH = config.CORPUS_PATH
EMBEDDINGS_PATH = config.EMBEDDINGS_PATH
METADATA_PATH = config.METADATA_PATH
STYLE_PROFILE_PATH = config.STYLE_PROFILE_PATH
SYSTEM_PROMPT_PATH = config.SYSTEM_PROMPT_PATH

# Load spacy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading corpus...")
with open(CORPUS_PATH) as f:
    corpus = json.load(f)
print(f"Loaded {len(corpus)} documents")

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove special characters but keep punctuation we care about
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    return text.strip()

def analyze_corpus_style():
    print("\nAnalyzing corpus style...")
    
    print("Cleaning texts...")
    cleaned_corpus = [clean_text(text) for text in tqdm(corpus)]
    
    # Initialize metrics
    metrics = {
        'sentence_lengths': [],
        'sentiment_scores': [],
        'common_phrases': Counter(),
        'tone_patterns': Counter(),
        'topics': [],
        'response_styles': Counter()
    }
    
    print("\nAnalyzing individual posts...")
    post_redundancy_checker = set()
    for text in tqdm(cleaned_corpus):
        if text in post_redundancy_checker:
            continue
        post_redundancy_checker.add(text)

        doc = nlp(text)
        
        
        # Sentence length analysis
        sentences = [sent.text.strip() for sent in doc.sents]
        metrics['sentence_lengths'].extend([len(sent.split()) for sent in sentences])
        
        # Sentiment analysis
        blob = TextBlob(text)
        metrics['sentiment_scores'].append(blob.sentiment.polarity)
        
        # Capture writing style patterns
        if '?' in text:
            metrics['response_styles']['questioning'] += 1
        if '!' in text:
            metrics['response_styles']['emphatic'] += 1
        if any(word in text.lower() for word in ['actually', 'literally']):
            metrics['response_styles']['ironic'] += 1
        if len(text.split()) < 20:
            metrics['response_styles']['brief'] += 1
        
        # Extract characteristic phrases (2-3 words)
        redundancy_checker = []
        words = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
        for i in range(len(words)-1):
            # Skip repeat words
            if words[i] == words[i+1]:
                continue
            # Skip short words
            elif len(words[i]) < 3 or len(words[i+1]) < 3:
                continue
            elif any(word in words[i] for word in ['the', 'of', 'in', 'to', 'from', 'at', 'on', 'with', 'as', 'but', 'by', 'for', 'at', 'as', 'until', 'while']):
                continue
            # Skip repeated phrases
            elif f"{words[i]} {words[i+1]}" in redundancy_checker:
                continue

            redundancy_checker.append(f"{words[i]} {words[i+1]}")
            metrics['common_phrases'][f"{words[i]} {words[i+1]}"] += 1
            if i < len(words)-2:
                # Skip repeated words
                if words[i] == words[i+2]:
                    continue
                # Skip repeated phrases
                elif f"{words[i]} {words[i+1]} {words[i+2]}" in redundancy_checker:
                    continue

                redundancy_checker.append(f"{words[i]} {words[i+1]} {words[i+2]}")
                metrics['common_phrases'][f"{words[i]} {words[i+1]} {words[i+2]}"] += 1

    print("\nPerforming topic modeling...")
    # Topic modeling
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    print("Vectorizing documents...")
    doc_term_matrix = vectorizer.fit_transform(cleaned_corpus)
    
    print("Fitting LDA model...")
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Get top words for each topic
    print("Extracting topics...")
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
        metrics['topics'].append(top_words)

    print("\nCalculating summary statistics...")
    # Calculate summary statistics. JSON can't serialize numpy types or NaN values,
    # so we convert to primitive floats and fall back to None when data is missing.
    def safe_stats(values):
        if not values:
            return None, None
        mean = float(np.mean(values))
        std = float(np.std(values))
        return mean, std

    avg_sentence_length, sentence_std = safe_stats(metrics['sentence_lengths'])
    sentiment_mean, sentiment_std = safe_stats(metrics['sentiment_scores'])

    style_profile = {
        'avg_sentence_length': avg_sentence_length,
        'sentence_std': sentence_std,
        'sentiment_mean': sentiment_mean,
        'sentiment_std': sentiment_std,
        'common_phrases': metrics['common_phrases'].most_common(20),
        'response_styles': dict(metrics['response_styles']),
        'topics': metrics['topics']
    }
    
    return style_profile

def generate_style_prompt(style_profile):
    """Convert style profile into a natural language prompt"""
    print("\nGenerating style prompt...")

    avg_sentence_length = style_profile.get('avg_sentence_length')
    sentence_std = style_profile.get('sentence_std')
    sentiment_mean = style_profile.get('sentiment_mean')
    sentiment_std = style_profile.get('sentiment_std')

    def format_num(value):
        return f"{value:.1f}" if value is not None else "N/A"

    if sentiment_mean is None:
        sentiment_descriptor = "unknown"
    else:
        sentiment_descriptor = (
            "neutral" if abs(sentiment_mean) < 0.1
            else "positive" if sentiment_mean > 0
            else "negative"
        )

    prompt = f"""You are a contributor to Spite Magazine. Your responses should naturally embody these characteristics:

Voice & Tone:
- Typical response length: {format_num(avg_sentence_length)} words
- Response length standard deviation: {format_num(sentence_std)} words
- Overall sentiment: {sentiment_descriptor}
- Sentiment standard deviation: {format_num(sentiment_std)}
- Common response styles: {', '.join(f'{k}: {v}' for k, v in style_profile['response_styles'].items() if v > len(corpus)/10)}

Frequent topics and themes:
{', '.join(', '.join(topic) for topic in style_profile['topics'])}

Characteristic phrases:
{', '.join(f'"{phrase}"' for phrase, _ in style_profile['common_phrases'])}

Remember: This is your natural voice - don't force it or reference these patterns explicitly."""

    return prompt

if __name__ == "__main__":
    print("\n=== Starting Spite Style Analysis ===\n")
    style_profile = analyze_corpus_style()
    prompt = generate_style_prompt(style_profile)
    
    print("\nSaving results...")
    # Save results
    with open(STYLE_PROFILE_PATH, "w") as f:
        json.dump(style_profile, f, indent=2)
    print(f"✓ Style profile saved to: {STYLE_PROFILE_PATH}")
    
    with open(SYSTEM_PROMPT_PATH, "w") as f:
        f.write(prompt)
    print(f"✓ System prompt saved to: {SYSTEM_PROMPT_PATH}")
    
    print("\n=== Analysis Complete! ===")
    print("\nGenerated system prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
