# Evolving-Titles

`pip install -r requirements.txt`

Evolving titles is a model designed to generate book titles. The components of the model represent fundamentals of human creative thinking, such as free association performed through a semantic network of concepts and evaluation of ideas based on former experience. The model uses WordNet, an open source semantic graph of English words to generate a whole population of new title ideas, which are then improved by evolutionary algorithms. Fitness of the individual ideas in a population is evaluated by a neural network, previously trained to differentiate real book titles from rudimentary machine-generated pieces of text. Results show a significant improvement in the ability of the model to create sensible and even original titles.

The original idea of combining a neural network with evolutionary algorithms to solve a creative task is adopted from [Robert Levy](https://github.com/rplevy/poevolve/blob/master/thesis/doc/levythesis2000.pdf).
