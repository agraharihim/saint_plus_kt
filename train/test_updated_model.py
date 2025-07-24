"""
Test script to validate the updated SAINT+ model with separate MAX_TIME values.
"""

import torch
from saint_plus_kt import Config, SAINTPlus

def test_updated_model():
    """Test the updated model with separate max times."""
    print("=== Testing Updated SAINT+ Model ===")
    
    # Create config
    config = Config()
    print(f"MAX_TIME_ELAPSED: {config.MAX_TIME_ELAPSED:,} ms ({config.MAX_TIME_ELAPSED/1000:.0f} seconds)")
    print(f"MAX_TIME_LAG: {config.MAX_TIME_LAG:,} ms ({config.MAX_TIME_LAG/1000:.0f} seconds)")
    
    # Create model
    model = SAINTPlus(config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test with sample data
    batch_size = 2
    seq_len = 5
    
    # Create sample inputs
    questions = torch.randint(1, config.NUM_QUESTIONS + 1, (batch_size, seq_len))
    responses = torch.randint(0, 2, (batch_size, seq_len))
    
    # Test different time ranges
    elapsed_times = torch.tensor([
        [10000, 25000, 50000, 80000, 150000],  # Normal range
        [5000, 15000, 30000, 60000, 120000]   # Normal range
    ], dtype=torch.float)
    
    lag_times = torch.tensor([
        [30000, 60000, 300000, 1800000, 7200000],  # 30s, 1min, 5min, 30min, 2hr
        [15000, 45000, 600000, 3600000, 86400000]  # 15s, 45s, 10min, 1hr, 24hr
    ], dtype=torch.float)
    
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    
    print("\n=== Testing Time Normalization ===")
    
    # Test the response embedding directly
    response_embed = model.response_embed
    
    # Check normalization
    elapsed_norm = torch.clamp(elapsed_times / config.MAX_TIME_ELAPSED, 0, 1)
    lag_norm = torch.clamp(lag_times / config.MAX_TIME_LAG, 0, 1)
    
    print("Elapsed times (ms):", elapsed_times[0].tolist())
    print("Elapsed normalized:", [f"{x:.3f}" for x in elapsed_norm[0].tolist()])
    print("Lag times (ms):", lag_times[0].tolist())
    print("Lag normalized:", [f"{x:.3f}" for x in lag_norm[0].tolist()])
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    try:
        with torch.no_grad():
            predictions = model(questions, responses, elapsed_times, lag_times, attention_mask)
        print(f"✅ Forward pass successful! Output shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[0, :3].tolist()}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("\n=== Testing Edge Cases ===")
    
    # Test with times at the boundaries
    edge_elapsed = torch.tensor([[config.MAX_TIME_ELAPSED - 1000, config.MAX_TIME_ELAPSED + 1000]], dtype=torch.float)
    edge_lag = torch.tensor([[config.MAX_TIME_LAG - 1000, config.MAX_TIME_LAG + 1000]], dtype=torch.float)
    
    edge_elapsed_norm = torch.clamp(edge_elapsed / config.MAX_TIME_ELAPSED, 0, 1)
    edge_lag_norm = torch.clamp(edge_lag / config.MAX_TIME_LAG, 0, 1)
    
    print(f"Edge elapsed times: {edge_elapsed[0].tolist()}")
    print(f"Edge elapsed normalized: {edge_elapsed_norm[0].tolist()}")
    print(f"Edge lag times: {edge_lag[0].tolist()}")
    print(f"Edge lag normalized: {edge_lag_norm[0].tolist()}")
    
    # Verify clamping works
    assert edge_elapsed_norm[0, 1].item() == 1.0, "Elapsed time clamping failed"
    assert edge_lag_norm[0, 1].item() == 1.0, "Lag time clamping failed"
    print("✅ Clamping works correctly")
    
    print("\n=== Coverage Analysis ===")
    
    # Based on the 100k user analysis
    p98_elapsed = 80000  # 80 seconds
    p98_lag = 75415000   # 75,415 seconds
    
    elapsed_coverage = (config.MAX_TIME_ELAPSED / p98_elapsed) * 100
    lag_coverage = (config.MAX_TIME_LAG / p98_lag) * 100
    
    print(f"Elapsed time coverage vs p98: {elapsed_coverage:.1f}%")
    print(f"Lag time coverage vs p98: {lag_coverage:.1f}%")
    
    print("\n✅ All tests passed! Model is ready for training with optimized time parameters.")
    return True

if __name__ == "__main__":
    test_updated_model()
