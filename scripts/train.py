"""
Training and validation functions.
"""
import torch
from tqdm.auto import tqdm


def train(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    tloss, tacc = 0.0, 0.0
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    for i, (frames, phonemes) in enumerate(dataloader):
        optimizer.zero_grad()
        frames = frames.to(device)
        phonemes = phonemes.to(device)

        if scaler is not None:
            with torch.autocast(device_type=device.type if hasattr(device, "type") else "cuda", dtype=torch.float16):
                logits = model(frames)
                loss = criterion(logits, phonemes)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames)
            loss = criterion(logits, phonemes)
            loss.backward()
            optimizer.step()

        tloss += loss.item()
        tacc += (torch.argmax(logits, dim=1) == phonemes).float().mean().item()

        batch_bar.set_postfix(
            loss=f"{tloss / (i + 1):.04f}",
            acc=f"{tacc * 100 / (i + 1):.04f}%",
        )
        batch_bar.update()

        del frames, phonemes, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    batch_bar.close()
    tloss /= len(dataloader)
    tacc /= len(dataloader)
    return tloss, tacc


def eval(model, dataloader, criterion, device):
    model.eval()
    vloss, vacc = 0.0, 0.0
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    for i, (frames, phonemes) in enumerate(dataloader):
        frames = frames.to(device)
        phonemes = phonemes.to(device)

        with torch.inference_mode():
            logits = model(frames)
            loss = criterion(logits, phonemes)

        vloss += loss.item()
        vacc += (torch.argmax(logits, dim=1) == phonemes).float().mean().item()

        batch_bar.set_postfix(
            loss=f"{vloss / (i + 1):.04f}",
            acc=f"{vacc * 100 / (i + 1):.04f}%",
        )
        batch_bar.update()

        del frames, phonemes, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    batch_bar.close()
    vloss /= len(dataloader)
    vacc /= len(dataloader)
    return vloss, vacc
