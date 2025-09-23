from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import pandas as pd 
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="data/spite_dump.json")
parser.add_argument("--corpus_file", type=str, default="data/spite_corpus.json")
parser.add_argument("--metadata_file", type=str, default="data/spite_metadata.json")
parser.add_argument("--embeddings_file", type=str, default="data/spite_embeddings.npy")
args = parser.parse_args()

model = SentenceTransformer("all-mpnet-base-v2")
print(model.device)

# Load your posts
with open(args.input_file) as f:
    print(f"Loading {args.input_file}...")
    posts = json.load(f)

# Flatten into text chunks with metadata
corpus = []
metadata = []
for object in tqdm(posts):
    obj_model = object["model"]
    obj_id = object["pk"]
    
    if obj_model == 'blog.post':
        post = object["fields"]
        data = post["title"]
        post_content = post["content"]
        if post_content:
            data += f'\n{post_content}'
        
        # Store metadata for this post
        metadata.append({
            "type": "post",
            "id": obj_id,
            "title": post["title"],
            "author": post.get("author"),
            "anon_uuid": post.get("anon_uuid"),
            "display_name": post.get("display_name", "")
        })

    elif obj_model == 'blog.comment':
        comment = object["fields"]
        name = comment["name"]
        content = comment["content"]
        if name:
            data = f'{name}\n{content}'
        else:
            data = content
        
        # Store metadata for this comment
        metadata.append({
            "type": "comment",
            "id": obj_id,
            "name": comment.get("name", ""),
            "post_id": comment.get("post")  # Reference to parent post
        })
        

    corpus.append(data)

corpus_df = pd.DataFrame(corpus)
print(corpus_df.shape)
print(corpus_df.head())
# Embed with progress tracking and batch processing
print(f"Embedding {len(corpus)} entries...")
batch_size = 1000  # Process in smaller batches to avoid memory issues

all_embeddings = []
for i in range(0, len(corpus), batch_size):
    batch = corpus[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}/{(len(corpus) + batch_size - 1)//batch_size} ({len(batch)} entries)")
    
    try:
        batch_embeddings = model.encode(batch, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)
        print(f"Batch {i//batch_size + 1} completed. Shape: {batch_embeddings.shape}")
    except Exception as e:
        print(f"Error in batch {i//batch_size + 1}: {e}")
        raise

# Combine all embeddings
print("Combining batches...")
embeddings = np.vstack(all_embeddings)
print(f"Final embeddings shape: {embeddings.shape}")

# Sample outputs for verification
print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")

# Save with FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
np.save(args.embeddings_file, embeddings)
with open(args.corpus_file, "w+") as f:
    json.dump(corpus, f)
with open(args.metadata_file, "w+") as f:
    json.dump(metadata, f)

print(f"Saved {len(corpus)} entries with metadata")
print(f"Posts: {len([m for m in metadata if m['type'] == 'post'])}")
print(f"Comments: {len([m for m in metadata if m['type'] == 'comment'])}")