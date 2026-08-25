"""nnU-Net trainer entry point for the encoder-only MG-SSM baseline."""

from torch import nn

from nnunetv2.nets.nnunet_mgssm import get_nnunet_mgssm_from_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


class nnUNetTrainerMGSSMBaseline(nnUNetTrainer):
    """Default nnU-Net training recipe with only the network changed."""

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return get_nnunet_mgssm_from_plans(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=configuration_manager,
            num_input_channels=num_input_channels,
            deep_supervision=enable_deep_supervision,
        )

