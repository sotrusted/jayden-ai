#!/usr/bin/env python3
"""
Quick script to create minimal metadata for testing without full re-embedding.
This creates dummy metadata that matches the existing corpus length.
"""

import json
import os

def create_test_metadata():
    """Create test metadata file that matches existing corpus."""
    
    # Load existing corpus to get the length
    corpus_path = "data/spite_corpus.json"
    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found. Run embed_posts.py first.")
        return
    
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    # Create dummy metadata for each corpus entry
    metadata = []
    for i in range(len(corpus)):
        # Alternate between posts and comments for testing
        if i % 3 == 0:  # Every 3rd entry is a post
            metadata.append({
                "type": "post",
                "id": 1000 + i,
                "title": f"Test Post {i}",
                "author": None,
                "anon_uuid": f"test-uuid-{i}",
                "display_name": f"User{i % 10}"
            })
        else:  # Others are comments
            metadata.append({
                "type": "comment", 
                "id": 2000 + i,
                "name": f"Commenter{i % 5}" if i % 2 else "",
                "post_id": 1000 + (i // 3) * 3  # Link to parent post
            })
    
    # Save test metadata
    metadata_path = "data/spite_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"Created test metadata: {len(metadata)} entries")
    print(f"Posts: {len([m for m in metadata if m['type'] == 'post'])}")
    print(f"Comments: {len([m for m in metadata if m['type'] == 'comment'])}")
    print(f"Saved to: {metadata_path}")
    print("\nNOTE: This is test data. Run 'python src/embed_posts.py' for real metadata.")

if __name__ == "__main__":
    create_test_metadata()
