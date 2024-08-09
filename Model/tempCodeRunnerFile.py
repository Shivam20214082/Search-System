import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # For handling malformed queries
from langdetect import detect  # For language detection
from deep_translator import GoogleTranslator  # For translation
import gradio as gr
from functools import lru_cache

# Load the dataset
df = pd.read_csv("C:\\Users\\shiva\\Desktop\\ZEPTO\\Database\\cleaned_data.csv")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Define a normalization function for the combined features
def normalize_text_simple(text):
    if pd.isna(text):  # Handle NaNs
        return ''
    text = str(text)  # Convert to string
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Create the combined features with normalization
df['combined_features'] = (
    df['product_name'].apply(normalize_text_simple) + ' ' +
    df['product_category_tree'].apply(normalize_text_simple) + ' ' +
    df['description'].apply(normalize_text_simple) + ' ' +
    df['brand'].apply(normalize_text_simple) + ' ' +
    df['extracted_specifications'].apply(normalize_text_simple)
)

# Initialize the TF-IDF Vectorizer with reduced features
tfidf = TfidfVectorizer(max_features=3000)  # Limiting to top 3000 features for better performance

# Fit and transform the combined features
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

@lru_cache(maxsize=100)
def process_query(query):
    # Detect and translate the query if it's not in English
    detected_lang = detect(query)
    if detected_lang != 'en':
        query = GoogleTranslator(source=detected_lang, target='en').translate(query)

    # Normalize the query
    query = normalize_text_simple(query)
    
    return query

# Define a function to get the query vector
def get_query_vector(query, vectorizer):
    query = process_query(query)  # Process the query
    return vectorizer.transform([query])

# Define a function to get relevant products
def get_relevant_products(query, tfidf_matrix, vectorizer, top_n=12):
    if not query.strip():  # Handle empty queries
        return []  # Return an empty list if query is empty
    query_vector = get_query_vector(query, vectorizer)
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get indices of top_n products
    results = df.iloc[top_indices]

    products = []
    for _, row in results.iterrows():
        # Ensure that 'image' column contains publicly accessible URLs
        try:
            image_urls = eval(row['image'])  # Convert the string to a list of URLs
        except:
            image_urls = []  # Handle cases where eval fails
        if image_urls:  # If the list is not empty
            products.append({
                "image": image_urls[0],  # Use the first image URL
                "name": row['product_name'],
                "brand": row['brand'],
                "price": row['discounted_price'],
                "url": row['product_url']
            })
    return products

# Gradio interface function
def search_products(query):
    results = get_relevant_products(query, tfidf_matrix, tfidf)
    if not results:
        return "No results found for the given query."
    
    # Create the output HTML string with vertical layout (one product per row)
    output_html = """
    <div style="display: flex; flex-direction: column; align-items: center;">
    """
    for product in results:
        output_html += f"""
        <div style="width: 80%; margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 10px; background-color: #f9f9f9; text-align: center;">
            <img src="{product['image']}" width="150" style="object-fit: cover; border-radius: 10px;">
            <div style="font-size: 18px; font-weight: bold; color: #333; margin-top: 10px;">{product['name']}</div>
            <div style="font-size: 16px; color: #555; margin-top: 5px;">Brand: {product['brand']}</div>
            <div style="font-size: 16px; font-weight: bold; color: #e74c3c; margin-top: 5px;">Price: {product['price']}</div>
            <div style="margin-top: 10px;">
                <a href="{product['url']}" style="display: inline-block; padding: 10px 20px; background-color: #3498db; color: white; text-decoration: none; border-radius: 5px;">View Product</a>
            </div>
        </div>
        """
    output_html += "</div>"
    return output_html

# Gradio interface setup
interface = gr.Interface(
    fn=search_products,
    inputs="text",
    outputs="html",
    title="Product Search System",
    description="Enter a search term to find relevant products.",
    examples=["Shirt", "Smartphone", "Shoes"],
    allow_flagging="never"  # Disable flagging
)

# Launch the Gradio interface
interface.launch(share=True)
