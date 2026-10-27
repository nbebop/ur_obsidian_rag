# Trainer

The Trainer class provides a complete training loop for PyTorch models. Configure the
learning rate, batch size and number of epochs with TrainingArguments, then call
trainer.train(). Gradient accumulation lets you simulate a larger batch size.
