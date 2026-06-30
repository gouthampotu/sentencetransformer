input_sentence = "The quick brown fox jumps over the lazy dog."
print(f"Input Sentence: {input_sentence}")

if tokenizer and model:
    print("\n--- Tokenization and Token IDs ---")
    # Tokenize the input sentence and get token IDs
    # `return_tensors='pt'` returns PyTorch tensors
    encoded_input = tokenizer(input_sentence, return_tensors='pt')

    # Extract tokens and token IDs
    tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
    token_ids = encoded_input['input_ids'][0].tolist()

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    print("\n--- Generating Embeddings ---")
    # Get the model's output (embeddings)
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(**encoded_input)

    # The last hidden state contains the token embeddings
    last_hidden_states = outputs.last_hidden_state

    # We can take the embedding of the first token ([CLS]) as a sentence embedding
    # Or iterate through `last_hidden_states[0]` for individual token embeddings
    sentence_embedding = last_hidden_states[0][0] # Embedding for the [CLS] token

    print(f"Shape of all token embeddings: {last_hidden_states.shape}")
    print(f"Shape of [CLS] token (sentence) embedding: {sentence_embedding.shape}")
    print("First 5 dimensions of the [CLS] token embedding:", sentence_embedding[:5].tolist())
else:
    print("Cannot process sentence: Model or Tokenizer not loaded successfully due to previous error.")
