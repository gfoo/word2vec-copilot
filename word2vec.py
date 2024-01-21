from gensim.models import Word2Vec

# Define a simple corpus with each sentence as a list of words
corpus = [['apple', 'banana', 'orange', 'grape'], ['fruit', 'salad', 'delicious', 'healthy']]

# Initialize model
model = Word2Vec(vector_size=100, min_count=1)

# Build vocab
model.build_vocab(corpus)

# Train model
model.train(corpus, total_examples=model.corpus_count, epochs=10)

# Print vocabulary
print('Vocabulary:', list(model.wv.key_to_index.keys()))

# Print a summary of the model
print('Model:', model)

print("Word vector for apple")
print(model.wv['apple'])
