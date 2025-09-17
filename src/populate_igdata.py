import pandas as pd
import json

df = pd.read_csv('spite-igdata.csv')

posts_df = pd.read_csv('spite-posts.csv')

posts = json.load(open('spite_dump.json'))

for index, row in df.iterrows():
    post_id = row['Post ID']
    if pd.isna(post_id):
        continue

    if ',' in post_id:
        post_id = int(post_id.split(',')[0])
    elif '/' in post_id:
        post_id = int(post_id.split('/')[0])
    elif '-' in post_id:
        post_id = int(post_id.split('-')[0])
    else:
        try:
            post_id = int(post_id)
        except ValueError:
            continue

    print(post_id)
    post = posts[post_id]
    df.at[index, 'Title'] = post['fields']['title']
    df.at[index, 'Content'] = post['fields']['content']
    df.at[index, 'Display Name'] = post['fields']['display_name']

df.to_csv('spite-igdata-populated.csv', index=False)