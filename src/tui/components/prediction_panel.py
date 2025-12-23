"""
Prediction Panel Component - Displays ML predictions and confidence.
"""
from textual.widgets import Static, ProgressBar
from textual.containers import Vertical, Horizontal
from textual.app import ComposeResult
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Prediction:
    """A single prediction."""
    label: str
    value: float
    confidence: float
    details: Optional[str] = None


class PredictionPanel(Static):
    """Panel showing predictions with confidence indicators."""

    DEFAULT_CSS = """
    PredictionPanel {
        height: auto;
        padding: 1;
        border: solid $primary;
        background: $surface;
        margin: 1;
    }
    
    .panel-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .prediction-item {
        height: auto;
        margin-bottom: 1;
    }
    
    .prediction-header {
        height: 1;
    }
    
    .prediction-label {
        width: 1fr;
    }
    
    .prediction-value {
        width: auto;
        text-style: bold;
    }
    
    .confidence-bar {
        height: 1;
        margin-top: 0;
    }
    
    .confidence-high {
        color: #27ae60;
    }
    
    .confidence-medium {
        color: #f39c12;
    }
    
    .confidence-low {
        color: #e74c3c;
    }
    
    .prediction-details {
        color: $text-muted;
        text-style: italic;
    }
    
    .model-info {
        margin-top: 1;
        padding-top: 1;
        border-top: solid $primary-darken-2;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        title: str = "Predictions",
        predictions: Optional[List[Prediction]] = None,
        model_name: Optional[str] = None,
        model_accuracy: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.predictions = predictions or []
        self.model_name = model_name
        self.model_accuracy = model_accuracy

    def compose(self) -> ComposeResult:
        """Compose the prediction panel."""
        with Vertical():
            yield Static(f"📊 {self.title}", classes="panel-title")
            
            for pred in self.predictions:
                with Vertical(classes="prediction-item"):
                    with Horizontal(classes="prediction-header"):
                        yield Static(pred.label, classes="prediction-label")
                        
                        # Determine confidence class
                        if pred.confidence >= 0.7:
                            conf_class = "confidence-high"
                        elif pred.confidence >= 0.55:
                            conf_class = "confidence-medium"
                        else:
                            conf_class = "confidence-low"
                        
                        yield Static(
                            f"{pred.value:.1f} ({pred.confidence*100:.0f}%)",
                            classes=f"prediction-value {conf_class}"
                        )
                    
                    if pred.details:
                        yield Static(pred.details, classes="prediction-details")
            
            # Model info
            if self.model_name:
                with Horizontal(classes="model-info"):
                    info_text = f"🤖 Model: {self.model_name}"
                    if self.model_accuracy:
                        info_text += f" • Accuracy: {self.model_accuracy*100:.1f}%"
                    yield Static(info_text)

    def update_predictions(self, predictions: List[Prediction]) -> None:
        """Update predictions and refresh display."""
        self.predictions = predictions
        self.refresh()
