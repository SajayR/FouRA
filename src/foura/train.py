import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import classification_report
import wandb

from .wrappers import FouRAConfig, get_foura_model
from .utils import print_trainable_parameters
from .evaluate import evaluate_model


def train_model(
    model_name = "google/vit-base-patch16-224-in21k",
    num_labels = 100,
    batch_size = 64,
    num_epochs = 1,
    learning_rate = 1e-4,
    weight_decay = 1e-4,
    rank=32,
    foura_alpha=32,
    transform_type="dct",
    target_modules=["query", "value"],   
    use_gate=False,
    run_name = "FouRA-32-Rank-NoGate-FFT",  
    use_wandb = False
):

    if use_wandb:
        wandb.init(
            project="foura-cifar100",
            name=run_name,
            config={
            "model_name": model_name,
            "num_labels": num_labels,
            "rank": rank,
            "foura_alpha": foura_alpha,
            "transform_type": transform_type,
            "target_modules": target_modules,   
            "use_gate": use_gate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay
        }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    cfg = FouRAConfig(
        rank=rank,
        foura_alpha=foura_alpha,
        transform_type=transform_type,
        target_modules=target_modules,   
        use_gate=use_gate
    )
    model = get_foura_model(model, cfg)
    print_trainable_parameters(model)
    model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(pixel_values=imgs).logits
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(
                loss=running_loss/total, acc=correct/total
            )

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, val_acc, _ = evaluate_model(model, test_loader, device)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
        print(f"Epoch {epoch} Results: Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Saved best model with Val Acc: {best_acc:.4f}")
    print("Final best val accuracy: ", best_acc)
    print("Performing final evaluation...")
    _, final_val_acc, final_report = evaluate_model(model, test_loader, device)
    print(f"Final Best Val Accuracy (from training epochs): {best_acc:.4f}")
    print(f"Final Val Accuracy (after full training): {final_val_acc:.4f}")

    if use_wandb:
        wandb.log({
                "best_val_accuracy": best_acc,
                "final_val_accuracy_after_training": final_val_acc,
                "final_weighted_precision": final_report['weighted avg']['precision'],
                "final_weighted_recall": final_report['weighted avg']['recall'],
                "final_weighted_f1": final_report['weighted avg']['f1-score']
        })
        wandb.finish()
    print("Final best val accuracy (from training): ", best_acc)

if __name__ == '__main__':
    # This allows running the training directly, e.g., for testing
    # You can set parameters here or use argparse for command-line arguments
    train_model(num_epochs=1, use_wandb=False) 