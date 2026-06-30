import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch

# --- 1. Load Model and Tokenizer (Cached for performance) ---
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        st.success(f"Successfully loaded model: {model_name}")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        st.warning("Please ensure the model name is correct and accessible.")
        return None, None

tokenizer, model = load_model_and_tokenizer()

# --- 2. Streamlit UI Elements ---
st.title("Sentence Tokenization and Embedding Generator")
st.markdown("Enter a sentence below to get its tokens, token IDs, and BERT embeddings.")

input_sentence = st.text_area(
    "Enter your sentence here:",
    "The quick brown fox jumps over the lazy dog.",
    height=100
)

if st.button("Generate Info"):
    if tokenizer and model:
        if input_sentence:
            st.subheader("Processing Results:")

            # --- Tokenization and Token IDs ---
            st.markdown("#### 1. Tokens and Token IDs")
            encoded_input = tokenizer(input_sentence, return_tensors='pt')
            tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
            token_ids = encoded_input['input_ids'][0].tolist()

            st.write("**Tokens:**", tokens)
            st.write("**Token IDs:**", token_ids)

            # --- Generating Embeddings ---
            st.markdown("#### 2. Embeddings")
            with torch.no_grad():
                outputs = model(**encoded_input)

            last_hidden_states = outputs.last_hidden_state
            # For sentence-transformers/all-MiniLM-L6-v2, it's common to average token embeddings
            # to get a sentence embedding, or use the [CLS] token if available and appropriate.
            # Here, we'll continue using the [CLS] token for consistency with BERT-base.
            # For more specific 'all-MiniLM' sentence embeddings, you might want to consider
            # the SentenceTransformer library's output or mean pooling.
            sentence_embedding = last_hidden_states[0][0] # [CLS] token embedding

            st.write(f"**Shape of all token embeddings:** {last_hidden_states.shape}")
            st.write(f"**Shape of [CLS] token (sentence) embedding:** {sentence_embedding.shape}")
            st.write("**First 5 dimensions of the [CLS] token embedding:**", sentence_embedding[:5].tolist())

            st.info("The full token embeddings are large and are typically used for downstream tasks. The [CLS] token embedding (first token) is often used as a sentence-level representation, though for 'all-MiniLM' models, mean pooling across all tokens is also a common approach for sentence embeddings.")
        else:
            st.warning("Please enter a sentence to process.")
    else:
        st.error("Model and Tokenizer could not be loaded. Please check the model name and your internet connection.")
