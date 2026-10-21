"""Temporal interface: Z+2 context sections and Z target sections."""
def context_sections(lr, target_count=9):
    """Convert augmented Z+4 cached sections to Z+2 context sections."""
    if lr.ndim != 4 or lr.shape[1] != target_count + 4:
        raise ValueError(f"Expected [C,{target_count + 4},H,W], got {tuple(lr.shape)}")
    return lr[:, 1:-1]

def target_sections(lr_context):
    """Select target sections from the BWA context."""
    if lr_context.ndim != 5 or lr_context.shape[2] < 3:
        raise ValueError("Expected [B,C,Z+2,H,W]")
    return lr_context[:, :, 1:-1]
