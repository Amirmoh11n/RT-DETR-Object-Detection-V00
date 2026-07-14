from torchvision.datasets import VOCDetection
from torchvision import transforms

def download_dataset():
    """Make dataset"""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])


    train_dataset = VOCDetection(
        root="./VOC",
        year="2012",
        image_set="train",
        download=True,
        transform=transform
    )


    val_dataset = VOCDetection(
        root="./VOC",
        year="2012",
        image_set="val",
        download=True,
        transform=transform
    )
    
    return train_dataset , val_dataset

def classes():
    """Label mapping"""
    VOC_CLASSES = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    ]

    label2id = {
        name: idx
        for idx, name in enumerate(VOC_CLASSES)
    }

    id2label = {
        idx: name
        for idx, name in enumerate(VOC_CLASSES)
    }

    return label2id , id2label , VOC_CLASSES