import perplexity
import plots
import train


default_probability = 0.01

if __name__ == "__main__":
    train.main(default_probability)
    perplexity.main()
    plots.main()
