import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import classification_report

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test dataset."""
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    criterion = CrossEntropyLoss() # Define criterion locally for evaluation

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(pixel_values=imgs).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / total
    val_acc = correct / total
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    print(f"Evaluation Results: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    # print(classification_report(all_labels, all_preds, zero_division=0)) # Optional: print full report

    return avg_val_loss, val_acc, report 