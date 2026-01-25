from search import connect_chromadb, load_embeddings, create_chromadb, read_csv

df = read_csv()

df_text = df['name'] + " " + df['categories']
df_image = df['imageURLs']
df_review = df['reviews.title'] + " " + df['reviews.text']
df_review = df_review.fillna("")
df_recommend = df['reviews.doRecommend']

df_text = (
    df_text
    .astype(str)
    .str.lower()
    .str.replace("&", " and ")
    .str.replace("/", " ")
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df_review = (
    df_review
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

embeddings_text, embeddings_images = load_embeddings()
chroma_client = connect_chromadb()
create_chromadb(chroma_client, df, df_text, df_image, embeddings_text, embeddings_images)
print("finish")