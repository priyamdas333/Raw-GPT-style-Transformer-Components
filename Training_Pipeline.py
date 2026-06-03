#@title 3.3 — Training Loop (Run This Cell)

epoch_losses = []
epoch_accuracies = []

print("Training...")
print(f"{'Epoch':>7} | {'Loss':>9} | {'Accuracy':>9} | {'Status'}")
print("-" * 50)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        logits = model(batch_x)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE),
                         batch_y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch_y).sum().item()
        total_tokens += batch_y.numel()

    avg_loss = total_loss / len(train_dataset)
    accuracy = total_correct / total_tokens * 100
    epoch_losses.append(avg_loss)
    epoch_accuracies.append(accuracy)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        status = "learning..." if accuracy < 95 else "✓ converged"
        print(f"{epoch+1:>7} | {avg_loss:>9.4f} | {accuracy:>8.1f}% | {status}")

print()
print(f"Final — Loss: {epoch_losses[-1]:.4f}, Accuracy: {epoch_accuracies[-1]:.1f}%")