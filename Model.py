#region libraries
from Data import download_dataset , classes
import torch
from transformers import RTDetrForObjectDetection , RTDetrImageProcessor
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
#endregion

#region Load pretrained model:
model_name = "PekingU/rtdetr_r50vd" #pip install transformers accelerate

processor = RTDetrImageProcessor.from_pretrained(model_name)
model = RTDetrForObjectDetection.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
#endregion

#region Dataset_class
class VOCRTDETRDataset(Dataset):
    """Dataset class for change the format of dataset on this model , the uniq dataset for model"""
    
    def __init__(self, voc_dataset, processor):
        self.voc_dataset = voc_dataset
        self.processor = processor

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):

        image, target = self.voc_dataset[idx]

        pil_image = F.to_pil_image(image)

        objects = target["annotation"]["object"]

        if not isinstance(objects, list):
            objects = [objects]

        boxes = []
        class_labels = []

        for obj in objects:

            bbox = obj["bndbox"]

            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])

            boxes.append(
                [xmin, ymin, xmax, ymax]
            )

            class_labels.append(
                classes.label2id[obj["name"]]
            )

        annotations = {
            "image_id": idx,
            "annotations": [
                {
                    "bbox": [
                        box[0],
                        box[1],
                        box[2] - box[0],
                        box[3] - box[1]
                    ],
                    "category_id": label,
                    "area": (
                        (box[2]-box[0]) *
                        (box[3]-box[1])
                    ),
                    "iscrowd": 0
                }
                for box, label in zip(
                    boxes,
                    class_labels
                )
            ]
        }

        encoding = processor(
            images=pil_image,
            annotations=annotations,
            return_tensors="pt"
        )

        pixel_values = encoding["pixel_values"].squeeze()

        labels = encoding["labels"][0]

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
#endregion

#region making dataset:
train_dataset , val_dataset = download_dataset()
train_subset = Subset(
    train_dataset,
    range(5000)
)

val_subset = Subset(
    val_dataset,
    range(500)
)

train_rtdetr = VOCRTDETRDataset(
    train_subset,
    processor
)

val_rtdetr = VOCRTDETRDataset(
    val_subset,
    processor
)
#endregion

#region dataloader:


def collate_fn(batch) -> dict:
    """Making batch data"""
    
    pixel_values = torch.stack(
        [x["pixel_values"] for x in batch]
    )

    labels = [
        x["labels"]
        for x in batch
    ]

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


train_loader = DataLoader(
    train_rtdetr,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2
)

val_loader = DataLoader(
    val_rtdetr,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2
)
#endregion

#region Head changed
label2id , id2label ,VOC_CLASSES = classes()

model = RTDetrForObjectDetection.from_pretrained(
    model_name,
    num_labels=len(VOC_CLASSES),
    ignore_mismatched_sizes=True
)


model.config.id2label = id2label
model.config.label2id = label2id

model.to(device)
#endregion

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