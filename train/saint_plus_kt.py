"""
Canonical SAINT+ Knowledge Tracing Implementation

This implementation follows the official SAINT+ architecture with:
- Transformer encoder for processing exercise sequences
- Transformer decoder with masked self-attention AND cross-attention
- Layer-by-layer information fusion between encoder and decoder
- Proper causal masking in the decoder to prevent future response leakage
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import gc
from database_utils import DatabaseManager


class Config:
    """Basic configuration."""
    # Data
    MAX_SEQ = 574  # Updated based on 100k user analysis - provides 95% coverage
    NUM_QUESTIONS = 13169
    
    # Model
    EMBEDDING_SIZE = 256
    NUM_BLOCKS = 4
    NUM_ATTN_HEADS = 8
    DROPOUT = 0.2
    MAX_TIME_ELAPSED = 100000  # 100 seconds for elapsed times (captures 98%+ based on analysis)
    MAX_TIME_LAG = 75415000    # 75,415 seconds for lag times (captures 98%+ based on analysis)
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 1
    LEARNING_RATE = 0.001
    
    # Device
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"


class KnowledgeTracingDataset(Dataset):
    """Simple dataset for SAINT+ training."""
    
    def __init__(self, sequences, max_seq_len):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Get sequence data
        questions = sequence['questions'][:self.max_seq_len]
        responses = sequence['responses'][:self.max_seq_len]
        elapsed_times = sequence['elapsed_times'][:self.max_seq_len]  # Time spent on each question
        lag_times = sequence['lag_times'][:self.max_seq_len]  # Time between consecutive questions
        
        # Pad sequences
        seq_len = len(questions)
        questions += [0] * (self.max_seq_len - seq_len)
        responses += [0] * (self.max_seq_len - seq_len)
        elapsed_times += [0] * (self.max_seq_len - seq_len)
        lag_times += [0] * (self.max_seq_len - seq_len)
        
        # Create attention mask
        attention_mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)
        
        return {
            'questions': torch.tensor(questions, dtype=torch.long),
            'responses': torch.tensor(responses, dtype=torch.long),
            'elapsed_time': torch.tensor(elapsed_times, dtype=torch.float),  # Time spent on each question
            'lag_time': torch.tensor(lag_times, dtype=torch.float),  # Time between consecutive questions
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool)
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ResponseEmbedding(nn.Module):
    """Embedding for responses and time features."""
    
    def __init__(self, embedding_size, max_time_elapsed, max_time_lag):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_time_elapsed = max_time_elapsed  # Separate max time for elapsed times
        self.max_time_lag = max_time_lag          # Separate max time for lag times
        
        # Response embedding
        self.response_embed = nn.Embedding(2, embedding_size)  # Creates a lookup table that maps response (0 or 1) to a learnable vector of size embedding_size.
        
        # Time embeddings
        self.time_projection = nn.Linear(2, embedding_size)  # Maps two time features (elapsed and lag) to a vector of size embedding_size using a linear transformation.
        
    def forward(self, responses, elapsed_time, lag_time):
        # Response embedding
        response_emb = self.response_embed(responses)  # Looks up the embedding vector for each response (0 or 1).
        
        # Time features - now with separate normalization
        elapsed_norm = torch.clamp(elapsed_time / self.max_time_elapsed, 0, 1)  # Divides elapsed_time by max_time_elapsed and clamps the result between 0 and 1.
        lag_norm = torch.clamp(lag_time / self.max_time_lag, 0, 1)  # Divides lag_time by max_time_lag and clamps the result between 0 and 1.
        time_features = torch.stack([elapsed_norm, lag_norm], dim=-1)  # Combines elapsed_norm and lag_norm into a single tensor along a new last dimension.
        time_emb = self.time_projection(time_features)  # Projects the 2D time features into the embedding space using a linear layer.
        
        return response_emb + time_emb  # Adds the response and time embeddings together for each position.


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block for processing exercises."""
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or d_model * 4
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        # Self attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=~attention_mask if attention_mask is not None else None)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class SAINTDecoderBlock(nn.Module):
    """SAINT+ decoder block with masked self-attention and cross-attention."""
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or d_model * 4
        
        # 1. Masked self-attention for responses (causal masking)
        self.masked_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # 2. Cross-attention to encoder output (exercises)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model)
        )
        
        # Layer normalization for each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, attention_mask=None):
        """
        Forward pass through SAINT+ decoder block.
        
        Args:
            x: Response embeddings (Query for both attention mechanisms)
            encoder_output: Output from encoder (Key/Value for cross-attention)
            attention_mask: Padding mask for valid positions
        """
        batch_size, seq_len = x.shape[:2]
        
        # 1. Masked self-attention on response sequence
        # Create causal mask to prevent looking at future responses
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Apply masked self-attention
        masked_attn_out, _ = self.masked_self_attn(
            x, x, x, 
            attn_mask=causal_mask,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        x = self.norm1(x + self.dropout(masked_attn_out))
        
        # 2. Cross-attention: Query from decoder, Key/Value from encoder
        # This allows the decoder to attend to all exercise information
        cross_attn_out, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # 3. Feed forward network
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x, cross_attn_weights  # Return attention weights for interpretability


class SAINTPlus(nn.Module):
    """Canonical SAINT+ model with proper encoder-decoder architecture."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.question_embed = nn.Embedding(config.NUM_QUESTIONS + 1, config.EMBEDDING_SIZE, padding_idx=0)  # Creates a lookup table that maps question IDs to learnable vectors.
        self.response_embed = ResponseEmbedding(config.EMBEDDING_SIZE, config.MAX_TIME_ELAPSED, config.MAX_TIME_LAG)  # Handles embedding of responses and time features with separate max times.
        self.pos_encoding = PositionalEncoding(config.EMBEDDING_SIZE, config.MAX_SEQ)  # Adds positional information to embeddings so the model knows the order of items.
        
        # Encoder (for exercises) - processes exercise sequences
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(config.EMBEDDING_SIZE, config.NUM_ATTN_HEADS, dropout=config.DROPOUT)  # Standard transformer encoder blocks for processing exercise information.
            for _ in range(config.NUM_BLOCKS)
        ])
        
        # Decoder (for responses) - processes responses with cross-attention to exercises
        self.decoder_blocks = nn.ModuleList([
            SAINTDecoderBlock(config.EMBEDDING_SIZE, config.NUM_ATTN_HEADS, dropout=config.DROPOUT)  # SAINT+ decoder blocks with masked self-attention and cross-attention to encoder.
            for _ in range(config.NUM_BLOCKS)
        ])
        
        # Output projection - now only uses decoder output since cross-attention already incorporates encoder information
        self.prediction_head = nn.Sequential(
            nn.Linear(config.EMBEDDING_SIZE, config.EMBEDDING_SIZE),  # Projects decoder output (which already includes encoder info via cross-attention) to prediction space.
            nn.ReLU(),  # Applies a non-linear activation function.
            nn.Dropout(config.DROPOUT),  # Randomly sets some elements to zero during training to prevent overfitting.
            nn.Linear(config.EMBEDDING_SIZE, 1),  # Maps to a single output value (probability).
            nn.Sigmoid()  # Squashes output to range [0, 1] for binary prediction.
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, questions, responses, elapsed_time, lag_time, attention_mask=None):
        batch_size, seq_len = questions.shape
        
        # 1. Embeddings
        question_emb = self.question_embed(questions)  # Embed question IDs to vectors
        question_emb = self.pos_encoding(question_emb.transpose(0, 1)).transpose(0, 1)  # Add positional encoding
        
        response_emb = self.response_embed(responses, elapsed_time, lag_time)  # Embed responses with time features
        response_emb = self.pos_encoding(response_emb.transpose(0, 1)).transpose(0, 1)  # Add positional encoding
        
        # 2. Encoder: Process exercise sequence
        encoder_out = question_emb
        for encoder_block in self.encoder_blocks:
            encoder_out = encoder_block(encoder_out, attention_mask)  # Self-attention on exercise sequence
        
        # 3. Decoder: Process response sequence with cross-attention to exercises
        decoder_out = response_emb
        cross_attention_weights = []
        
        for decoder_block in self.decoder_blocks:
            # Each decoder block uses masked self-attention on responses AND cross-attention to encoder output
            decoder_out, cross_attn_weights = decoder_block(decoder_out, encoder_out, attention_mask)
            cross_attention_weights.append(cross_attn_weights)  # Store for potential analysis
        
        # 4. Prediction: Use decoder output (which already incorporates encoder info via cross-attention)
        predictions = self.prediction_head(decoder_out)
        
        return predictions.squeeze(-1)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Training"):
        questions = batch['questions'].to(device)
        responses = batch['responses'].to(device)
        elapsed_time = batch['elapsed_time'].to(device)
        lag_time = batch['lag_time'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Create targets (predict next response)
        targets = responses[:, 1:].float()  # Shift responses
        input_responses = responses[:, :-1]  # Input responses
        input_elapsed = elapsed_time[:, :-1]
        input_lag = lag_time[:, :-1]
        input_questions = questions[:, :-1]
        input_mask = attention_mask[:, :-1]
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(input_questions, input_responses, input_elapsed, input_lag, input_mask)
        
        # Calculate loss (only on valid positions)
        # The targets should align with predictions - both should have same shape
        target_mask = attention_mask[:, 1:]  # Mask for target positions
        loss = criterion(predictions[target_mask], targets[target_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for AUC
        with torch.no_grad():
            valid_preds = predictions[target_mask].cpu().numpy()
            valid_targets = targets[target_mask].cpu().numpy()
            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.5
    
    return {'loss': avg_loss, 'auc': auc}


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            questions = batch['questions'].to(device)
            responses = batch['responses'].to(device)
            elapsed_time = batch['elapsed_time'].to(device)
            lag_time = batch['lag_time'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Create targets
            targets = responses[:, 1:].float()
            input_responses = responses[:, :-1]
            input_elapsed = elapsed_time[:, :-1]
            input_lag = lag_time[:, :-1]
            input_questions = questions[:, :-1]
            input_mask = attention_mask[:, :-1]
            
            # Forward pass
            predictions = model(input_questions, input_responses, input_elapsed, input_lag, input_mask)
            
            # Calculate loss
            target_mask = attention_mask[:, 1:]
            loss = criterion(predictions[target_mask], targets[target_mask])
            total_loss += loss.item()
            
            # Collect predictions
            valid_preds = predictions[target_mask].cpu().numpy()
            valid_targets = targets[target_mask].cpu().numpy()
            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.5
    
    return {'loss': avg_loss, 'auc': auc}


def main():
    """Main training function."""
    print("=== Canonical SAINT+ Training ===")
    print("Architecture: Encoder-Decoder with Cross-Attention")
    
    # Config
    config = Config()
    print(f"Device: {config.DEVICE}")
    
    # Database
    db_manager = DatabaseManager()
    
    # Model
    model = SAINTPlus(config).to(config.DEVICE)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{config.NUM_EPOCHS} ===")
        
        # Get training data by split
        print("Loading training data (split='train')...")
        train_sequences = db_manager.get_users_by_split('train')
        print(f"Loaded {len(train_sequences)} training sequences")
        
        if not train_sequences:
            print("No training data available!")
            break
        
        # Create dataset and dataloader
        train_dataset = KnowledgeTracingDataset(train_sequences, config.MAX_SEQ)
        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        
        # Train
        train_metrics = train_epoch(model, train_dataloader, optimizer, criterion, config.DEVICE)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"saint_plus_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'config': config
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Clean up memory
        del train_sequences, train_dataset, train_dataloader
        gc.collect()
        if config.DEVICE == "mps":
            torch.mps.empty_cache()
        elif config.DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
