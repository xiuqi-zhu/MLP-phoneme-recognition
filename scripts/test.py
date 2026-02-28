"""
Test-set inference and submission.csv generation.
"""
import torch
from tqdm.auto import tqdm


def test(model, test_loader, phonemes, device, output_path="./submission.csv"):
    model.eval()
    test_predictions = []

    with torch.no_grad():
        for mfccs in tqdm(test_loader, desc="Test"):
            mfccs = mfccs.to(device)
            logits = model(mfccs)
            predicted_phonemes = torch.argmax(logits, dim=1)
            test_phonemes = [phonemes[p] for p in predicted_phonemes.cpu().tolist()]
            test_predictions.extend(test_phonemes)

    sample = test_predictions[:10]
    if sample and not isinstance(sample[0], str):
        print(f"ERROR: Predictions should be phoneme STRINGS, not {type(sample[0]).__name__}!")
    else:
        print("Sample predictions:", sample)
        print("Predictions generated successfully!")

    with open(output_path, "w+") as f:
        f.write("id,label\n")
        for i in range(len(test_predictions)):
            f.write(f"{i},{test_predictions[i]}\n")
    print(f"submission.csv saved to {output_path}")
    return test_predictions
