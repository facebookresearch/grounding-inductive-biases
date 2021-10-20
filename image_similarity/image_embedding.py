"""
Generate embedding vectors from pre-trained ResNets
"""

import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms


class Embedding:
    def __init__(
        self,
        cuda: bool = False,
        model_name: str = "resnet18",
    ):
        """
        Loads pre-trained model for generating embeddings
        Args:
            model (str): string name for model.  Options model -> output_size
                "resnet18": 512,
                "resnet34": 512,
                "resnet50": 2048,
                "resnet101": 2048,
                "resnet152": 2048,
                "resnet18_no_aug": 512 (no_aug = trained on augmentation used during inference)
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model_name = model_name

        self.model = self.get_model_without_last_layer(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def get_model_without_last_layer(self, model_name) -> torch.nn.Sequential:
        if "no_aug" in model_name:
            return self.get_resnet_no_aug(model_name)
        model = getattr(models, model_name)(pretrained=True)
        embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        return embedding_model

    def get_resnet_no_aug(
        self, model_name, path=None, num_classes=1000
    ) -> torch.nn.Sequential:
        """Returns resnet using best pre-trained model with no augmentation"""
        if path is None:
            user = os.getenv("USER")
            cluster_path = f""
            local_path = f""
            if os.path.isfile(cluster_path):
                path = cluster_path
            else:
                path = local_path
        best_checkpoint = torch.load(path, map_location=self.device)
        original_model_name = model_name.replace("_no_aug", "")
        model = torch.nn.DataParallel(
            getattr(models, original_model_name)(num_classes=num_classes)
        )
        model.load_state_dict(best_checkpoint["state_dict"])
        model.eval()
        return model

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        if abs(images.mean()) > 0.1:
            images = self.normalize(images)
        with torch.no_grad():
            embeddings = self.model(images).squeeze()
        return embeddings
