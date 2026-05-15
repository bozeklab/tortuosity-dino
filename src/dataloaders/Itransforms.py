from abc import ABC, abstractmethod

from torchvision import transforms


class ITransforms(ABC):

    @abstractmethod
    def compose(self) -> transforms.Compose:
        """Apply the transformation to the data."""
        pass
