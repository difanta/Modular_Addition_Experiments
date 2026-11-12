The contents of the lib folder are the classes and functions required to define, train and analyze the model.

**train_main** is the code that trains from scratch a new model and _saves_ it to /data. (keyboard input _required_ to save: 'y' saves, any other input, including enter/newline cancels save).

**weight_fft_analysis** contains the Fourier visualizations for embedding, unembedding, query, key, value weight matrices.

**activation_fft_analysis** contains code to analyze the activations of the model in many parts of it, for example attention inputs, outputs, scores, etc. The analysis does the fft of the activation matrix, calculates FVE of some activations given others (to check whether residual connections are used)

**attn_out_frequency_analysis** contains the code that compares the predicted output frequencies for the attention head 1 with the true outputs, both by just selecting the predicted frequencies and trying to predict the fourier coefficients exactly.

**svd_analysis** contains the code to discover input and output space bases of each layer, and computes PK alignment between different kinds of layers. It contains the code for aligning known ordered bases (DFT, etc.) with the disordered lowrank bases produced by the lowrank SVD.

Many visualizations are commented because they open a lot of windows, uncomment them if needed.
