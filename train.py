from Models.Rtdetr import *


#region Optimizer:
optimizer = torch.optim.AdamW(model.parameters() , lr=1e-5)
#endregion

#region Fine Tune model On new dataset:


EPOCHS = 100

save_dir = Path("/content/best_rtdetr_voc")
save_dir.mkdir(exist_ok=True)

best_val_loss = float("inf")

train_history = []
val_history = []

for epoch in range(EPOCHS):

    # =====================
    # Train
    # =====================

    model.train()

    train_loss = 0.0

    for batch in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    ):

        pixel_values = batch["pixel_values"].to(device)

        labels = [
            {
                k: v.to(device)
                for k, v in target.items()
            }
            for target in batch["labels"]
        ]

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # =====================
    # Validation
    # =====================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            pixel_values = batch["pixel_values"].to(device)

            labels = [
                {
                    k: v.to(device)
                    for k, v in target.items()
                }
                for target in batch["labels"]
            ]

            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            val_loss += outputs.loss.item()

    val_loss /= len(val_loader)

    train_history.append(train_loss)
    val_history.append(val_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # =====================
    # Save Best Model
    # =====================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

        print(
            f"Best model saved "
            f"(val_loss={val_loss:.4f})"
        )

print("Training finished.")

#endregion

#region Plotting training and testing error curves

def plot_train_test_curves() ->None:
    plt.figure(figsize=(10,5))

    plt.plot(train_history, label="Train Loss")
    plt.plot(val_history, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RT-DETR Fine-Tuning on Pascal VOC")

    plt.legend()
    plt.grid()

    plt.show()
#endregion