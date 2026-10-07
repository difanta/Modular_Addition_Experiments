The contents of the lib folder are the classes and functions required to define, train and analyze the model.
train_main is the code that trains from scratch a new model and saves it to /data. (keyboard input required: 'y' saves, any other input cancels save)
embedding_fourier contains the Fourier visualizations for embedding, unembedding, query, key, value matrices.
metrics_main contains code to analyze the true features of the model in many parts of it, for example attention inputs, outputs, scores, etc.
metrics_main also contains the code that compares the predicted output frequencies for the attention head 1 with the true outputs.

Many visualizations are commented, just uncomment them if needed.
