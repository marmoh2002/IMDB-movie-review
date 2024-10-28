import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

class MyTaskSpecificCustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(MyTaskSpecificCustomModel, self).__init__()
        self.num_labels = num_labels
        
        self.model = AutoModel.from_pretrained(checkpoint, 
                                             config=AutoConfig.from_pretrained(checkpoint,
                                                                            output_attention=True,
                                                                            output_hidden_state=True))
        self.dropouts = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        
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

class SentimentPredictor:
    def __init__(self, model_directory):
        """
        Initialize the predictor with a saved model
        """
        # Load the saved model configuration and weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = f"{model_directory}/model_state.pt"
        model_config = torch.load(model_path,map_location=device, weights_only=True)
        
        # Initialize model
        self.model = MyTaskSpecificCustomModel(
            checkpoint=model_config['checkpoint'],
            num_labels=model_config['num_labels']
        )
        self.model.load_state_dict(model_config['state_dict'])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text, max_length=512):
        """
        Predict sentiment for a given text
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1)
        
        # Convert prediction to label
        label_map = {0: "negative", 1: "positive"}
        result = {
            "label": label_map[predicted_class.item()],
            "confidence": predictions[0][predicted_class.item()].item()
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = SentimentPredictor("./material")
    
    # Get user input for predictions
    while True:
        text = input("Enter text for sentiment prediction (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        result = predictor.predict(text)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
