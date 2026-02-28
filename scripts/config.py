"""
Configuration and constants: hyperparameters, phoneme list, etc.
"""

# Phoneme list (aligned with assignment spec)
PHONEMES = [
    "[SIL]", "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW",
    "V", "W", "Y", "Z", "ZH", "[SOS]", "[EOS]",
]


def get_config(
    name="Xiuqi Zhu",
    subset=1.0,
    context=25,
    archetype="diamond",
    activations="GELU",
    learning_rate=0.0005,
    dropout=0.2,
    optimizers="SGD",
    scheduler="ReduceLROnPlateau",
    epochs=30,
    batch_size=8192,
    weight_decay=0.05,
    weight_initialization="kaiming_normal",
    augmentations="Both",
    freq_mask_param=6,
    time_mask_param=8,
    **kwargs,
):
    """Return experiment config dict; any key can be overridden via kwargs."""
    return {
        "Name": name,
        "subset": subset,
        "context": context,
        "archetype": archetype,
        "activations": activations,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "optimizers": optimizers,
        "scheduler": scheduler,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "weight_initialization": weight_initialization,
        "augmentations": augmentations,
        "freq_mask_param": freq_mask_param,
        "time_mask_param": time_mask_param,
        **kwargs,
    }
