# Generative Graph Markov State Model
We propose a Generative Graph Neural Network model able to learn to propagate proteins molecular dynamic. Our net is composed by a deterministic encoder,
which will maps from high-dimensional configuration space to a small-sized vector, a probabilistic decoder which will generate the next set of coordinates. We imple-
mented a regularization loss in order to generate realistic configurations and a flexible message-parsing layer for the graph convolution
