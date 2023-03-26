# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import poptorch
from optimum.utils import logging
from scipy.stats import truncnorm
from transformers import (
   SwinForMaskedImageModeling
)
from transformers.models.swin.modeling_swin import SwinMaskedImageModelingOutput

from ...modeling_utils import (
    OnehotGather,
    PipelineMixin,
    SerializedEmbedding,
    SerializedLinear,
    get_layer_ipu,
    outline_attribute,
    recomputation_checkpoint,
    register,
)

logger = logging.get_logger(__name__)


@register(SwinForMaskedImageModeling)
class PipelinedSwinForMaskedImageModeling(SwinForMaskedImageModeling, PipelineMixin):
    """
    SwinForMaskedImageModeling transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = SwinForMaskedImageModeling(config).parallelize().half().train()
    ```
    """

    def __init__(self, config):
        super().__init__(config)
    
    def _get_layer_ipu(self):
        ## temporary for Swin-Tiny
        layer_ipu: List[int] = []
        number_layers = len(self.swin.encoder.layers)
        if (number_layers == 4) :
            layer_ipu = [0,1,2,3]
        else:
            pass
        return layer_ipu

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Adds recomputation checkpoints
        """
        super().parallelize()

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding --> IPU 0")
        self.swin.embeddings = poptorch.BeginBlock(self.swin.embeddings, "Embedding", ipu_id=0)


        layer_ipu = self._get_layer_ipu()
        for index, layer in enumerate(self.swin.encoder.layers):
            ipu = layer_ipu[index]
            # if self.ipu_config.recompute_checkpoint_every_layer:
            #     h = recomputation_checkpoint(layer)
            #     self._hooks.append(h)
            self.swin.encoder.layers[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        logger.info("Classifier --> IPU 0")
        self.swin.layernorm = poptorch.BeginBlock(self.swin.layernorm, "Classifier", ipu_id=0)
        logger.info("-----------------------------------------------------------")

        ##Comment following as there is pooling layer after layernorm in 
        # logger.info("Pooler --> IPU 0")
        # self.swin.pooler = poptorch.BeginBlock(self.swin.pooler, "Pooler", ipu_id=0)

        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()
        return self

    def _init_weights(self, module):
        """Initialize the weights"""

        def truncated_normal_(tensor, mean=0, std=1):
            """
            Truncated Normal distribution, truncated at 2 sigma
            """
            r = torch.tensor(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape))
            tensor.data.copy_(r)

        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SwinMaskedImageModelingOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, SwinForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-simmim-window6-192")
        >>> model = SwinForMaskedImageModeling.from_pretrained("microsoft/swin-base-simmim-window6-192")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.logits
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 192, 192]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print (f"xiachsh debug {type(pixel_values)}, {pixel_values.shape}")
        outputs = self.swin(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output.transpose(1, 2)
        batch_size, num_channels, sequence_length = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)

            dim_1 = bool_masked_pos.shape[1]
            dim_2 = bool_masked_pos.shape[2]
            output_size_1 = dim_1 * self.config.patch_size
            output_size_2 = dim_2 * self.config.patch_size
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1, output_size = output_size_1)
                .repeat_interleave(self.config.patch_size, 2, output_size = output_size_2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="mean")
            masked_im_loss = poptorch.identity_loss( (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels, reduction="none" )


        print (f"xiachsh debug  class :{type(masked_im_loss)}")
        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[2:]
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output


        return SwinMaskedImageModelingOutput(
            loss=masked_im_loss,
            logits=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
