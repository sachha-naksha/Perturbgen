import pytorch_lightning as pl
import torch
import torch.ao.quantization


class QuantizedLightningModule(pl.LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model  # Base model before quantization

    def forward(self, x):
        return self.model(x)

    def load_unquantized_state_dict(self, checkpoint_path):
        """Load the state dict from an unquantized checkpoint."""
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict)

    def apply_quantization(self):
        """Apply dynamic quantization to the model."""
        self.model = torch.ao.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Quantize only Linear layers
            dtype=torch.qint8,
        )

    def on_load_checkpoint(self, checkpoint):
        """Override on_load_checkpoint to integrate quantization logic."""
        # Load unquantized state dict
        self.load_unquantized_state_dict(checkpoint['state_dict'])

        # Apply quantization after loading the checkpoint
        self.apply_quantization()
