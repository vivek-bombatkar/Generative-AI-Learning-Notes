
## 01 Deep Learning
- Deep learning is a type of machine learning that uses algorithmic structures called neural networks. 
- In deep learning models, we use software modules called nodes to simulate the behavior of neurons.
- Deep neural networks comprise layers of nodes, including an input layer, several hidden layers, and an output layer of nodes.
- Every `node` in the neural network autonomously assigns `weights` to each `feature`.
- Information flows through the network in a forward direction from input to output.
- During training, the `difference` between the `predicted` output and the `actual` output is then calculated.
- The `weights` of the neurons are repeatedly adjusted to minimize the error.
- CNN - Image processing
- RNN - Natural Language processing
- KNN - K-Nearest neighbour, slow in inference. As most of the computations happens at infereence stage. 
- 

## Generative AI
- Generative AI is accomplished by using deep learning models that are pre-trained on extremely large datasets containing strings of text or, in AI terms, sequences.
- They use `transformer` neural networks, which change an input sequence, in Gen AI known as prompt, into an output sequence, which is the response to your prompt.
- Neural networks process the elements of a sequence `sequentially` one word at a time.
- Transformers process the sequence in `parallel`, which speeds up the training and allows much bigger datasets to be used.

## Data drift
- Data drift is when there are significant changes to the data distribution compared to the data used for training. 
- Concept drift is when the properties of the target variables change.

## Accuracy
- Accuracy measures how close the predicted class values are to the actual values.
- The formula for accuracy is the number of true positives plus true negatives divided by the total number of predictions.

## Precision
- Precision measures how well an algorithm predicts true positives out of all the positives that it identifies.
- The formula is the number of true positives divided by the number of true positives, plus the number of false positives.

## Recall
- Recall, is also known as sensitivity or the true positive rate.
- However, if recall and precision are both important to us, the F1 score balances precision and recall by combining them in a single metric

## ROUGE 
- Recall-Oriented Understudy for Gisting Evaluation, is primarily employed to assess the quality of automatically-generated summaries by comparing them to human-generated reference summaries

## BLEU
- Bilingual Evaluation Understudy, is an algorithm designed to evaluate the quality of machine-translated texts by comparing it to human-generated translations.

