from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.utils import draw_bounding_boxes
import numpy as np
import torch
from typing import Dict, List, Tuple
from python_template.preprocessing import deserialize_compose_transformation
from torchvision.transforms import v2


class Detector:
    def __init__(self, configuration: Dict):
        """Initialize the detector

        Parameters
        ----------
        configuration : Dict
            Configuration contains information used for initializing the detector and
            pre- and post-processors.
        """

        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights, box_score_thresh=0.9
        )
        self.model.eval()

        # Get pre-processor configuration
        preprocessor_configuration = configuration["Preprocessor"]

        self.preprocessor = v2.Compose(
            [
                deserialize_compose_transformation(preprocessor_configuration),
                self.weights.transforms(),
            ]
        )

    def preprocess(self, image: np.ndarray) -> torch.tensor:
        """Preprocess an image.

        Parameters
        ----------
        image : np.ndarray
            Input image to be preprocessed

        Returns
        -------
        torch.tensor
            Preprocessed image
        """
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        preprocessed = self.preprocessor(image_tensor).unsqueeze(0)
        return preprocessed

    def process(self, image: torch.Tensor) -> any:
        """Calculates predictions for the given image.

        Parameters
        ----------
        image : torch.tensor
            Image for which the predictions are calculated for.

        Returns
        -------
        any
            Predictions from the network
        """
        predictions = self.model(image)
        return predictions

    def postprocess(self, predictions: any) -> Tuple[torch.Tensor, List]:
        """Postprocess the predictions so that these can be used.

        Parameters
        ----------
        predictions : any
            Predictions from the network

        Returns
        -------
        Tuple[torch.Tensor, List]
            - Bounding boxes
            - Classes
        """
        labels = [self.weights.meta["categories"][i] for i in predictions["labels"]]
        bbox = predictions["boxes"]

        return bbox, labels

    def visualize(
        self, image: torch.Tensor, results: Tuple[torch.Tensor, List]
    ) -> torch.Tensor:
        """Visualizes results after they have been post-processed.

        Parameters
        ----------
        image : torch.Tensor
            Input image where the bounding boxes and classes are drawn.
        results : Tuple[torch.Tensor, List]
            Results from the post-processed

        Returns
        -------
        torch.Tensor
            Image with bboxes and classes
        """
        bbox, labels = results

        box = draw_bounding_boxes(
            image, boxes=bbox, labels=labels, colors="red", width=4, font_size=30
        ).permute(1, 2, 0)

        return box.detach()
