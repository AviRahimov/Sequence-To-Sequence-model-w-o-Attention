# Advanced Models for Language Understanding - Assignment 1: Seq2Seq Model for Sentence Unshuffling

This repository contains the implementation for Assignment 1, focusing on building a Sequence-to-Sequence (Seq2Seq) model to reconstruct original sentences from their shuffled versions. The project progresses from a basic Seq2Seq model to an enhanced version incorporating an attention mechanism.

## 🔹 Overview

### 🎯 Goal
The primary objective is to implement a Seq2Seq neural network using PyTorch. Given a sentence with randomly shuffled words, the model reconstructs the original, correctly ordered sentence.

Example:
Input (shuffled): `mat the on sat cat The`
Output (original): `The cat sat on the mat`

### 🧠 Why this task?
This task demonstrates how language models learn word order and syntactic structures, engaging working memory, syntax, and semantics.

## 📚 PyTorch Workflow Summary
The notebook follows a standard PyTorch workflow:
1.  **Data Preparation**: Defining `Dataset`, `DataLoader`, and tokenization.
2.  **Model Definition**: Subclassing `nn.Module` for model components.
3.  **Training**: Implementing loss functions, optimizers, and the training loop.

---

## ⚙️ Step 0: One-time Preparations

### Step 0.1: Install Python Dependencies
The first code cell installs necessary libraries: `torch`, `pandas`, and `scikit-learn` using `%pip install`.

### File Check (Kaggle Environment)
The second code cell lists files in the `/kaggle/input` directory to ensure dataset accessibility. It uses `os.walk`.

---

## 📊 Step 1: Prepare Data

### Load Dataset and Initial Setup
This cell loads the training data (`train.csv`) using `pandas`, sets random seeds for reproducibility (`random.seed`, `torch.manual_seed`), and displays the first few rows of the DataFrame.

### Train-Dev Split
The dataset is split into training (90%) and development (10%) sets using `sklearn.model_selection.train_test_split`. The sizes of these sets are printed.

### Handle Vocabulary and Tokenization
This section processes sentences into numerical inputs:
-   `tokenize(sentence)`: A function to convert a sentence string to a list of lowercase tokens.
-   `collections.Counter`: Builds a word frequency distribution.
-   Special Tokens: `PAD`, `SOS`, `EOS`, `UNK` are defined.
-   Vocabulary Construction: `word2idx` (word to index) and `idx2word` (index to word) mappings are created.
-   `encode_sentence(sentence, word2idx, max_len)`: A function to convert a sentence string into a fixed-length list of token IDs, including padding and EOS/UNK handling.

### Custom Dataset and DataLoader
-   `SentenceDataset(torch.utils.data.Dataset)`: A custom dataset class.
    -   `__init__`: Stores sentences, `word2idx`, and `max_len`.
    -   `__len__`: Returns the dataset size.
    -   `__getitem__`: Encodes and returns a source-target sentence pair as tensors.
-   `torch.utils.data.DataLoader`: Instances (`train_loader`, `dev_loader`) are created for batching and shuffling data. `batch_size` and `max_len` are defined.

### Verify DataLoader Output
A batch is fetched from `train_loader` using `next(iter(train_loader))`, and the shapes of the source and target tensors are printed to verify.

---

## 🧠 Step 2: Define Model (Basic Seq2Seq)

This section defines the components of the basic Seq2Seq model.

### Encoder and Decoder Implementation
-   `EncoderRNN(nn.Module)`:
    -   `__init__`: Initializes `nn.Embedding` and `nn.GRU` layers.
    -   `forward(input, hidden)`: Defines the forward pass (embedding -> GRU).
    -   `init_hidden(batch_size)`: Utility to create an initial zero hidden state.
-   `DecoderRNN(nn.Module)`:
    -   `__init__`: Initializes `nn.Embedding`, `nn.GRU`, and an `nn.Linear` output layer (`self.out`).
    -   `forward(input, hidden)`: Defines the one-step decoding process (embedding -> GRU -> linear output).

### Instantiate Encoder & Decoder and Sanity Checks
Instances of `EncoderRNN` and `DecoderRNN` are created. `embedding_size` and `hidden_size` are defined. A sanity check is performed by passing a sample batch through the encoder and one step of the decoder, printing their output shapes.

### Seq2Seq Model Wrapper
-   `Seq2Seq(nn.Module)`: Combines the encoder and decoder.
    -   `__init__`: Stores encoder, decoder, `sos_idx`, and `device`.
    -   `forward(src, trg)`: Implements the full sequence-to-sequence process, including initializing the encoder, passing the encoder's final hidden state to the decoder, and performing iterative decoding with teacher forcing.

---

## 🏋️ Step 3: Training (Basic Seq2Seq)

### Training Loop Function
-   `train_model(model, train_loader, dev_loader, optimizer, criterion, device, num_epochs)`:
    -   Iterates through epochs.
    -   In each epoch, iterates through `train_loader` for training (forward pass, loss calculation, backward pass, optimizer step).
    -   Evaluates the model on `dev_loader` after each training epoch.
    -   Prints training and development losses and stores them.

### Instantiate Model and Run Training
The `Seq2Seq` model is instantiated. `optim.Adam` is chosen as the optimizer, and `nn.CrossEntropyLoss` (ignoring `PAD` tokens) as the criterion. The `train_model` function is called to train the model for 10 epochs.

### Loss Visualization
-   `plot_losses(train_losses, dev_losses)`: A helper function using `matplotlib.pyplot` to plot training and development losses over epochs. The plot is displayed.

---

## 🔮 Step 4: Inference (Basic Seq2Seq)

### Inference Function
-   `infer_sentences(shuffled_sentences, model, word2idx, idx2word, device, max_len)`:
    -   Sets the model to `eval()` mode.
    -   Iterates through input sentences.
    -   For each sentence:
        -   Encodes the input sentence using `encode_sentence`.
        -   Passes it through the encoder.
        -   Performs greedy decoding token by token using the decoder, starting with `<SOS>` and using the encoder's final hidden state.
        -   Stops decoding at `<EOS>` or `max_len`.
        -   Converts predicted token IDs back to words.
    -   Returns a list of predicted sentence strings.

### Evaluation Function
-   `evaluate_sentence_predictions(predictions, targets)`: Computes exact match accuracy (case-insensitive and ignoring leading/trailing whitespace) between predicted and target sentences.

### Evaluate on Dev Dataset
The `infer_sentences` function is used to get predictions on the `dev_df`, and `evaluate_sentence_predictions` calculates the accuracy.

### Saving Predictions (Basic Model)
Predictions for the test set (`/kaggle/input/test-no-target/test_no_target.csv`) are generated using `infer_sentences` and saved to `/kaggle/working/seq2seq_predictions.csv` using `pandas`.

---

## ✨ Stage 2: Add Attention to Your Seq2Seq Model

This stage enhances the Seq2Seq model by incorporating an attention mechanism.

### Decoder with Attention
-   `DecoderRNNWithAttention(nn.Module)`: A modified decoder.
    -   `__init__`: The GRU's input size (`self.gru_input_size`) is now `embedding_size + encoder_hidden_size` to accommodate the concatenated embedding and context vector.
    -   `forward(input_token, hidden, context_vector)`: The context vector is concatenated with the embedded input token before being fed to the GRU.

### Seq2Seq Model with Attention
-   `Seq2SeqWithAttention(nn.Module)`:
    -   `__init__`: Stores encoder, the new `DecoderRNNWithAttention` instance, `sos_idx`, and `device`.
    -   `compute_attention(decoder_hidden, encoder_outputs)`: Implements dot-product attention. It calculates attention scores (energy) by taking the dot product of the current decoder hidden state (query) and all encoder outputs (keys/values), applies softmax to get attention weights, and computes the context vector as a weighted sum of encoder outputs.
    -   `forward(src, trg)`: Similar to the basic `Seq2Seq` model, but at each decoding step, it first calls `compute_attention` to get a context vector, then passes this context vector to the `decoder.forward` method along with the input token and decoder hidden state.

### Training Function with Early Stopping
-   `train_model_with_early_stopping(model, train_loader, dev_loader, optimizer, criterion, device, num_epochs, patience)`:
    -   A new training function that incorporates early stopping.
    -   It monitors the development loss and stops training if the loss doesn't improve for a specified `patience` number of epochs.
    -   It saves the state of the model (`best_model_state`) that achieved the best development loss and loads it back before returning. `copy.deepcopy` is used for saving the model state.

### Inference Function for Attention Model
-   `infer_sentences_with_attention(input_sentences, model, word2idx, idx2word, device, max_len)`:
    -   Similar to `infer_sentences` but adapted for the attention model.
    -   In the decoding loop, it calls `model.compute_attention` at each step to get the `context_vector`.
    -   This `context_vector` is then passed to `model.decoder.forward`.

### Instantiate and Train Attention Model
New hyperparameters (`embedding_size`, `encoder_hidden_size`, `decoder_hidden_size`) are set for the attention model. Instances of `EncoderRNN` (can be reused or a new one), `DecoderRNNWithAttention`, and `Seq2SeqWithAttention` are created. The model is trained using `train_model_with_early_stopping`.

### Plot Losses (Attention Model)
The `plot_losses` function is used to visualize the training and development losses for the attention model, showing the effect of early stopping.

### Evaluate Attention Model on Dev Set
The `infer_sentences_with_attention` function is used to get predictions on `dev_df`, and `evaluate_sentence_predictions` calculates the accuracy for the attention-based model.

### Saving Predictions (Attention Model)
Predictions for the test set (`/kaggle/input/test-no-target/test_no_target.csv`) are generated using `infer_sentences_with_attention` and saved to `/kaggle/working/seq2seq_with_attention_predictions.csv`.
