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
