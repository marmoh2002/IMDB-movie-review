# Custom DistilBERT Model for IMDB Movie Reviews

<p align="center">
<img src="readme_imgs/IMDb_BrandBanner_1920x425.jpg" width="1920" height="250">
</p>

[Dataset Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
[Download Model Here](https://drive.google.com/file/d/1OYmK3vtIiUbh_hZD_hKLfRDq8PDm79xt/view?usp=sharing)


This project uses a custom PyTorch model based on the DistilBERT architecture for sentiment analysis on the IMDB movie reviews dataset.

## Model Architecture

The `MyTaskSpecificCustomModel` class is the core of the model. It inherits from `nn.Module` and has the following components:

1. **DistilBERT Base Model**: The model starts with the pre-trained DistilBERT base model, which is loaded using `AutoModel.from_pretrained()`. This provides the base transformer and encoder layers.

2. **Dropout Layer**: A dropout layer with a rate of 0.1 is added after the base model's output to introduce regularization.

3. **Classification Head**: A custom linear layer is added on top of the base model's output to perform the final classification. The input to this layer is the CLS token representation (first element of the output sequence), which is flattened to a 768-dimensional vector.

The forward pass of the model is as follows:

```python
def forward(self, input_ids=None, attention_mask=None, labels=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs[0]
    sequence_outputs = self.dropouts(last_hidden_state)
    logits = self.classifier(sequence_outputs[:, 0, :].view(-1, 768))
    
    loss = None
    if labels is not None:
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions
    )
```

The model takes in `input_ids`, `attention_mask`, and optionally `labels` as input. It outputs the classification loss (if labels are provided), logits, hidden states, and attentions.

## Key Features

1. **Pre-trained DistilBERT Base**: The model leverages the pre-trained DistilBERT base model, which is a smaller and faster version of the original BERT model. This provides a strong starting point for the sentiment analysis task.

2. **Custom Classification Head**: The model adds a custom linear layer on top of the DistilBERT base model to perform the final sentiment classification. This allows the model to be fine-tuned for the specific task at hand.

3. **Dropout Regularization**: A dropout layer is added to the model to introduce regularization and prevent overfitting.

4. **Compatibility with Transformers Library**: The model is designed to be compatible with the Transformers library, allowing seamless integration with other tools and utilities provided by the library.

## Training and Deployment

The provided scripts handle the training, exporting, and deployment of the custom DistilBERT model:

1. `model-export.py`: This script saves the trained model, tokenizer, and requirements in a format that can be easily loaded and used elsewhere.
2. `model-inference.py`: This script demonstrates how to load the saved model and use it for inference on new data.
3. `flask-deployment.py`: This script creates a Flask web application that exposes an API endpoint for making sentiment predictions using the custom DistilBERT model.

To use this model, follow the instructions in the respective scripts.

## Dependencies

The project requires the following dependencies:

- PyTorch
- Transformers
- NumPy
- Flask (for deployment)

The exact versions of these dependencies are specified in the `requirements.txt` file, which is saved alongside the model.

## Example Usage
<p align="center">
<img src="readme_imgs/Screenshot 2024-10-28 195054.png" alt="PSO minimizing Easom benchmark function" width="720" height="400">
</p>


## License

This project is licensed under the [MIT License](LICENSE).
