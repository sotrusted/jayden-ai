from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import pandas as pd 

model = SentenceTransformer("all-mpnet-base-v2")

# Load your posts
with open("spite_dump.json") as f:
    posts = json.load(f)

# Flatten into text chunks
corpus =  []
for object in posts:
    obj_model = object["model"]
    if obj_model == 'blog.post':
        post = object["fields"]
        data = post["title"]
        post_content = post["content"]
        if post_content:
            data += f'\n{post_content}'

    elif obj_model == 'blog.comment':
        comment = object["fields"]
        name = comment["name"]
        content = comment["content"]
        if name:
            data = f'{name}\n{content}'
        else:
            data = content
        

    corpus.append(data)

corpus_df = pd.DataFrame(corpus)
print(corpus_df.shape)
print(corpus_df.head())
# Embed
embeddings = model.encode(corpus)

# Save with FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
np.save("spite_embeddings.npy", embeddings)
with open("spite_corpus.json", "w+") as f:
    json.dump(corpus, f)