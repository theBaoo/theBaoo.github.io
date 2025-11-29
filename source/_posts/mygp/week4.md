---
title: week3
date: 2025-11-25 16:11:11
tags:
---

## 毕设第四周: TODO, SUMMARY

### TODO

上一周补全了SmolVLA需要的RL逻辑, 主要是`sample_mean_var_val`等一系列方法的实现; 同时重写了`sample_action`和`forward`方法. 接下来需要迁移SmolVLA的输入/输出处理函数.
问题的关键在于RLinf和lerobot的observation分别具有什么格式.

#### RLinf's input

#### SmolVLA's transform

从SmolVLA的测试文件中可以观察到完整的数据流动:

```Python
# lerobot/tests/policies/test_smolvla_rtc.py:135
batch = {
    "observation.state": torch.randn(1, 14, dtype=torch.float32, device=device),
    "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
    "task": ["Pick up the object"],
}
batch = preprocessor(batch) # from make_pre_post_processors
# ...other codes...
actions_with_rtc = policy.predict_action_chunk(
    batch,
    noise=noise.clone(),
    prev_chunk_left_over=prev_chunk,
    inference_delay=4,
    execution_horizon=10,
)

# lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:310
batch = self._prepare_batch(batch)
actions = self._get_action_chunk(batch, noise, **kwargs)

# lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:273
images, img_masks = self.prepare_images(batch)
state = self.prepare_state(batch)
lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

actions = self.model.sample_actions(
    images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
)

# 后续进入RL逻辑
```

其实应该直接关注sample_action如何获取元数据并处理, 而不是关注数据流动的全过程.
对于图像:

```Python
def prepare_images(self, batch):
    """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
    convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
    """
    images = []
    img_masks = []
    present_img_keys = [key for key in self.config.image_features if key in batch]
    missing_img_keys = [key for key in self.config.image_features if key not in batch]

    if len(present_img_keys) == 0:
        raise ValueError(
            f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
        )
    # Preprocess image features present in the batch
    for key in present_img_keys:
        img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
        if self.config.resize_imgs_with_padding is not None:
            img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

        # Normalize from range [0,1] to [-1,1] as expacted by siglip
        img = img * 2.0 - 1.0

        bsize = img.shape[0]
        device = img.device
        if f"{key}_padding_mask" in batch:
            mask = batch[f"{key}_padding_mask"].bool()
        else:
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
        images.append(img)
        img_masks.append(mask)

    # Create image features not present in the batch
    # as fully 0 padded images.
    for num_empty_cameras in range(len(missing_img_keys)):
        if num_empty_cameras >= self.config.empty_cameras:
            break
        img = torch.ones_like(img) * -1
        mask = torch.zeros_like(mask)
        images.append(img)
        img_masks.append(mask)
    return images, img_masks
```

#### OpenPI: 模型加载

```Python
import openpi.shared.download as download
import openpi.transforms as transforms
import safetensors
from openpi.training import checkpoints as _checkpoints

from .embodiment.openpi import get_openpi_config
from .embodiment.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)

# config
simulator_type = getattr(cfg.openpi, "simulator_type", "libero")
if simulator_type == "libero":
    if getattr(cfg.openpi, "pi05", False):
        actor_train_config = get_openpi_config("pi05_libero")
    else:
        actor_train_config = get_openpi_config("pi0_libero")
elif simulator_type == "metaworld":
    if getattr(cfg.openpi, "pi05", False):
        actor_train_config = get_openpi_config("pi05_metaworld")
    else:
        actor_train_config = get_openpi_config("pi0_metaworld")
else:
    raise ValueError(f"Invalid simulator type: {simulator_type}")
actor_model_config = actor_train_config.model
actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
override_config_kwargs = cfg.openpi
if override_config_kwargs is not None:
    for key, val in override_config_kwargs.items():
        actor_model_config.__dict__[key] = val
# load model
checkpoint_dir = download.maybe_download(str(model_path))
weight_path = os.path.join(checkpoint_dir, "model.safetensors")
model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
    actor_model_config
)
# train expert only
if actor_model_config.train_expert_only:
    model.freeze_vlm()
safetensors.torch.load_model(model, weight_path, strict=False)
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
# fsdp replace
# model.paligemma_with_expert.replace_gemma_decoder_layers()
# load data stats
data_config = actor_train_config.data.create(
    actor_train_config.assets_dirs, actor_model_config
)
norm_stats = None
if norm_stats is None:
    # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
    # that the policy is using the same normalization stats as the original training process.
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(
        checkpoint_dir, data_config.asset_id
    )
# wrappers
repack_transforms = transforms.Group()
default_prompt = None
model.setup_wrappers(
    transforms=[
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        transforms.Normalize(
            norm_stats, use_quantiles=data_config.use_quantile_norm
        ),
        *data_config.model_transforms.inputs,
    ],
    output_transforms=[
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(
            norm_stats, use_quantiles=data_config.use_quantile_norm
        ),
        *data_config.data_transforms.outputs,
        *repack_transforms.outputs,
    ],
)
```
