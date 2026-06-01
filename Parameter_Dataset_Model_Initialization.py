#MODEL PARAMETER DEFINITION
# ═══ HYPERPARAMETERS ════════════════════════════════════════
VOCAB_SIZE   = 26       # a-z
D_MODEL      = 64       # Embedding dimension
N_HEADS      = 4        # Attention heads per layer
N_LAYERS     = 2        # Transformer blocks
D_FF         = 128      # Feed-forward hidden size
PATTERN_LEN  = 5        # Length of base pattern
NUM_REPEATS  = 4        # Times pattern repeats
NUM_SAMPLES  = 10000    # Training sequences
BATCH_SIZE   = 64       # Batch size
EPOCHS       = 50       # Training epochs
LR           = 3e-4     # Learning rate

MAX_SEQ = PATTERN_LEN * NUM_REPEATS
RANDOM_LOSS = -np.log(1.0 / VOCAB_SIZE)  # ≈ 3.258

print("Hyperparameters:")
print(f"  Pattern:      {PATTERN_LEN} chars × {NUM_REPEATS} repeats = {MAX_SEQ} tokens")
print(f"  Model:        {N_LAYERS} layers × {N_HEADS} heads, d_model={D_MODEL}")
print(f"  Training:     {NUM_SAMPLES} samples, {EPOCHS} epochs, lr={LR}")
print(f"  Random loss:  {RANDOM_LOSS:.3f}")

#════════════════════════════════════════════════════════════════════════════════
#DATASET & MODEL DEFINITION
train_dataset=RepeatedPatternDataset(
    num_samples=NUM_SAMPLES,
    pattern_len=PATTERN_LEN,
    num_repeats=NUM_REPEATS,
    vocab_size=VOCAB_SIZE,
    seed=42
)

train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True)

#MODEL initilization
model=MiniTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_seq=MAX_SEQ

).to(DEVICE)

criterion=nn.CrossEntropy()
optimizer=nn.optim.Adam(model.parameters(), lr=LR)
