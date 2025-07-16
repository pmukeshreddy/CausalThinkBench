class ModelTrainer:
    """
    Trainer for the reasoning model
    """
    def __init__(self, model: ReasoningModel, learning_rate: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Loss functions for different heads
        self.confidence_loss = nn.MSELoss()
        self.emotion_loss = nn.CrossEntropyLoss()
        self.mode_loss = nn.CrossEntropyLoss()
        self.memory_loss = nn.BCELoss()
    
    def train_step(self, batch):
        """
        Single training step with multi-task learning
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            memory_context=batch.get('memory_context')
        )
        
        # Calculate losses for each head
        total_loss = 0
        losses = {}
        
        if 'confidence_labels' in batch:
            conf_loss = self.confidence_loss(
                outputs['confidence'].squeeze(), 
                batch['confidence_labels'].float()
            )
            total_loss += conf_loss
            losses['confidence'] = conf_loss.item()
        
        if 'emotion_labels' in batch:
            emo_loss = self.emotion_loss(
                outputs['emotions'], 
                batch['emotion_labels']
            )
            total_loss += emo_loss
            losses['emotion'] = emo_loss.item()
        
        if 'mode_labels' in batch:
            mode_loss = self.mode_loss(
                outputs['reasoning_mode'], 
                batch['mode_labels']
            )
            total_loss += mode_loss
            losses['mode'] = mode_loss.item()
        
        if 'memory_labels' in batch and outputs['memory_relevance'] is not None:
            mem_loss = self.memory_loss(
                outputs['memory_relevance'].squeeze(), 
                batch['memory_labels'].float()
            )
            total_loss += mem_loss
            losses['memory'] = mem_loss.item()
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        losses['total'] = total_loss.item()
        return losses
