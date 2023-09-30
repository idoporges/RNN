import os
import pickle
from nltk.corpus import gutenberg
from train import preprocess_text


# Load tokens from a pickle file
def load_tokens(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Check if a word exists in the vocabulary
def word_exists(word, vocab):
    return word in vocab


def calc_perplexity_batch(model, n, test_set, vocab, batch_size=50):
    tokens = preprocess_text(test_set)
    perplexities = []
    context_OOV_cnt = 0
    word_OOV_cnt = 0
    for i in range(0, len(tokens) - n, batch_size):
        batch_tokens = tokens[i:i + batch_size]
        batch_perplexity = 1

        for j in range(len(batch_tokens) - n):
            context = tuple(batch_tokens[j: j + n - 1])
            word = batch_tokens[j + n - 1]

            if context not in model:
                probability = model[('<OOV>',)]['<OOV>']
                context_OOV_cnt += 1
            else:
                if word not in model[context]:
                    probability = model[('<OOV>',)]['<OOV>']
                    word_OOV_cnt += 1
                else:
                    probability = model[context][word]

            batch_perplexity = batch_perplexity / probability

        batch_perplexity = batch_perplexity ** (1.0 / (len(batch_tokens) - n))
        perplexities.append(batch_perplexity)

    if word_OOV_cnt + context_OOV_cnt != 0:
        print("Num of OOV: ", context_OOV_cnt + word_OOV_cnt)
        print("Num of context OOV: ", context_OOV_cnt)
        print("Num of word OOV: ", word_OOV_cnt)
    average_perplexity = sum(perplexities) / len(perplexities)
    return average_perplexity


def main():
    # Test sets.
    test_set_austen = gutenberg.raw("austen-sense.txt")
    test_set = test_set_austen
    vocab = load_tokens('models/Vocab/vocab.pkl')
    # Loop over all files in the 'models' directory
    for filename in os.listdir('models'):
        if filename.endswith('.pkl'):
            filepath = os.path.join('models', filename)

            # Load the model
            with open(filepath, 'rb') as f:
                print("filepath:", filepath)
                loaded_info = pickle.load(f)
                Ngram_model = loaded_info['model']
                n_value = loaded_info['n_value']

            # Calculate the perplexity
            perplexity = calc_perplexity_batch(Ngram_model, n_value, test_set, vocab)

            # Update the perplexity placeholder
            loaded_info['perplexity'] = perplexity

            # Save the updated info back to the file
            with open(filepath, 'wb') as f:
                pickle.dump(loaded_info, f)

            print(f"Updated perplexity for {filename} to {perplexity}")


if __name__ == "__main__":
    main()
