import os
import random
import matplotlib.pyplot as plt
import pickle
from itertools import islice


def print_first_n_items(model, n):
    for context, word_probs in islice(model.items(), n):
        print(f"{context}: {word_probs}")


def get_num_of_all_words_from_ngram(model):
    words = set()

    for context, word_probs in model.items():
        words.update(word_probs.keys())

    return len(words)


def generate_text(model, n, max_length=100):
    # Choose a random starting context that starts at a beginning of a sentence.
    filtered_keys = [key for key in model.keys() if key[0] == '<S>']
    starting_context = random.choice(filtered_keys)
    generated_text = list(starting_context)

    for i in range(max_length):
        # Extract the last n-1 tokens from the generated text as the context
        context = tuple(generated_text[-(n - 1):])

        # Get the probabilities for the next word based on the context
        word_probs = model[context]

        if not word_probs:
            break  # If there are no more words to predict, stop generating

        # Choose the next word based on the probabilities
        next_word = random.choices(list(word_probs.keys()), weights=list(word_probs.values()))[0]
        while next_word == '<OOV>':
            next_word = random.choices(list(word_probs.keys()), weights=list(word_probs.values()))[0]

        # Append the chosen word to the generated text
        generated_text.append(next_word)

    return ' '.join(generated_text)


def main():
    model_files = [f for f in os.listdir("models") if f.endswith('.pkl')]
    perplexity_values = []
    n_values = []

    for model_file in model_files:
        with open(f'models/{model_file}', 'rb') as f:
            model_info = pickle.load(f)

        n_value = model_info['n_value']
        perplexity = model_info['perplexity']

        n_values.append(n_value)
        perplexity_values.append(perplexity)

    # Sort by n_values for plotting
    sorted_indices = sorted(range(len(n_values)), key=lambda k: n_values[k])
    perplexity_values = [perplexity_values[i] for i in sorted_indices]
    n_values = [n_values[i] for i in sorted_indices]

    # Debugging
    print("n_values: ", n_values)
    print("perplexity_values: ", perplexity_values)

    # Plotting code
    plt.plot(n_values, perplexity_values)
    plt.xlabel('n (Order of the Language Model)')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs. n')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
