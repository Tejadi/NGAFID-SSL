#!/usr/bin/env python3
"""
Calculate memory requirements for the optimized BERT model with seq_len=10000.
"""

def calculate_memory_usage():
    print("🧮 Memory Usage Calculation for Optimized BERT (seq_len=10000)")
    print("=" * 70)

    # Model parameters
    batch_size = 1
    seq_len = 10000
    feat_dim = 44
    hidden_size = 1024  # Reduced from 1536
    encoder_layers = 8  # Reduced from 12
    decoder_layers = 6  # Reduced from 8
    num_heads = 16

    print(f"📊 Model Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len:,}")
    print(f"   Feature dimension: {feat_dim}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Encoder layers: {encoder_layers}")
    print(f"   Decoder layers: {decoder_layers}")
    print()

    # 1. Model Parameters
    # Encoder: BERT with custom projection
    projection_params = feat_dim * hidden_size

    # BERT encoder parameters (approximate)
    attention_params_per_layer = 4 * (hidden_size * hidden_size)  # Q, K, V, O
    ffn_params_per_layer = 2 * (hidden_size * hidden_size * 4)  # Up and down projections
    layer_norm_params_per_layer = 2 * hidden_size  # 2 layer norms per layer
    encoder_params = encoder_layers * (attention_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer)

    # Position embeddings
    position_params = seq_len * hidden_size

    # Decoder parameters (simplified - depends on exact architecture)
    decoder_params = 0
    current_dim = hidden_size
    for i in range(decoder_layers):
        if i == decoder_layers - 1:
            next_dim = feat_dim
        elif i == 0:
            next_dim = hidden_size * 2
        elif i == 1:
            next_dim = hidden_size
        elif i == 2:
            next_dim = hidden_size // 2
        else:
            next_dim = hidden_size // 4

        decoder_params += current_dim * next_dim
        if i < decoder_layers - 1:
            decoder_params += next_dim  # Layer norm
        current_dim = next_dim

    total_params = projection_params + encoder_params + position_params + decoder_params
    model_memory_gb = total_params * 4 / 1e9  # FP32

    print(f"🏗️  Model Memory:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model weights (FP32): {model_memory_gb:.2f} GB")

    # 2. Optimizer Memory (with 8-bit AdamW)
    # Standard AdamW needs 2x model params for momentum and variance
    # 8-bit AdamW reduces this significantly
    optimizer_memory_gb = model_memory_gb * 0.5  # ~50% reduction with 8-bit
    print(f"   8-bit AdamW optimizer: {optimizer_memory_gb:.2f} GB")

    # 3. Activation Memory (with gradient checkpointing and mixed precision)
    # This is the critical part for long sequences!

    # Attention activations: batch_size * num_heads * seq_len * seq_len
    attention_memory_per_layer = batch_size * num_heads * seq_len * seq_len * 2 / 1e9  # FP16
    total_attention_memory = encoder_layers * attention_memory_per_layer

    # Linear layer activations: batch_size * seq_len * hidden_size per layer
    linear_activation_per_layer = batch_size * seq_len * hidden_size * 2 / 1e9  # FP16
    total_linear_memory = (encoder_layers * 4 + decoder_layers * 2) * linear_activation_per_layer  # Multiple linear layers per transformer layer

    # FFN intermediate activations (4x hidden size)
    ffn_memory_per_layer = batch_size * seq_len * (hidden_size * 4) * 2 / 1e9  # FP16
    total_ffn_memory = encoder_layers * ffn_memory_per_layer

    base_activation_memory = total_attention_memory + total_linear_memory + total_ffn_memory

    # Gradient checkpointing saves most intermediate activations but attention is still large
    # For seq_len=10000, attention matrices are massive (10000x10000)
    checkpointing_reduction = 0.4  # Keep 40% (attention matrices can't be fully checkpointed)

    optimized_activation_memory = base_activation_memory * checkpointing_reduction

    print(f"   Attention matrices: {total_attention_memory:.2f} GB")
    print(f"   Linear activations: {total_linear_memory:.2f} GB")
    print(f"   FFN activations: {total_ffn_memory:.2f} GB")
    print(f"   Total base activations: {base_activation_memory:.2f} GB")
    print(f"   With checkpointing (FP16): {optimized_activation_memory:.2f} GB")

    # 4. Gradient Memory (with mixed precision)
    gradient_memory = model_memory_gb * 0.5  # FP16 gradients
    print(f"   Gradients (FP16): {gradient_memory:.2f} GB")

    # 5. Data Memory (batch on GPU)
    data_memory = batch_size * seq_len * feat_dim * 3 * 4 / 1e9  # x_masked, x_original, mask
    print(f"   Batch data: {data_memory:.2f} GB")

    # 6. CUDA overhead and misc
    cuda_overhead = 1.0  # GB
    print(f"   CUDA overhead: {cuda_overhead:.2f} GB")

    # Total memory
    total_memory = (model_memory_gb + optimizer_memory_gb + optimized_activation_memory +
                   gradient_memory + data_memory + cuda_overhead)

    print(f"\n📈 Total Estimated Memory Usage:")
    print(f"   Total: {total_memory:.2f} GB")
    print(f"   Available: 24.0 GB (RTX 6000)")
    print(f"   Margin: {24.0 - total_memory:.2f} GB")

    if total_memory < 24.0:
        print(f"✅ SHOULD FIT! Memory margin: {24.0 - total_memory:.2f} GB")
        if 24.0 - total_memory < 2.0:
            print("⚠️  Tight fit - monitor memory usage during training")
        else:
            print("🎉 Comfortable memory margin")
    else:
        print(f"❌ WILL NOT FIT! Exceeds by: {total_memory - 24.0:.2f} GB")
        print("\n🔧 Additional optimizations needed:")
        if total_memory - 24.0 < 2.0:
            print("   - Reduce batch size to 1")
            print("   - Increase gradient accumulation to 8")
        else:
            print("   - Reduce hidden_size further (768 or 512)")
            print("   - Reduce encoder layers further (6 or 4)")

    print(f"\n💡 Key optimizations applied:")
    print(f"   - Mixed precision training (FP16): ~50% activation memory reduction")
    print(f"   - Gradient checkpointing: ~70% activation memory reduction")
    print(f"   - 8-bit optimizer: ~50% optimizer memory reduction")
    print(f"   - Reduced model size: ~40% parameter reduction")

    return total_memory < 24.0, total_memory


if __name__ == "__main__":
    will_fit, total_memory = calculate_memory_usage()

    if not will_fit:
        print(f"\n🚨 URGENT: Need additional {total_memory - 24.0:.2f} GB reduction!")
        print("Recommended immediate changes:")
        print("1. batch_size = 1 (saves ~2-3 GB)")
        print("2. gradient_accumulation_steps = 8 (maintains effective batch size)")
        print("3. If still not enough, reduce hidden_size to 768")