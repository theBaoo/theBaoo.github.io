我先在代码库中搜索包含 `libero` 和 `observation` 的定义与使用位置，找出环境返回的 observation 字段与模型输入是如何连接的。

搜索文本 `libero|observation|wrist_image|observation/image|observation/state`，200 个结果

我将打开 io_struct.py 中 `prepare_observations` 函数，查看 libero 情况下返回的 obs 字段结构。

读取 [](file:///root/codes/python/RLinf/rlinf/data/io_struct.py#1040-1040)，行 1040 到 1235

搜索文本 `predict_action_batch(`，9 个结果

读取 [](file:///root/codes/python/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py)

搜索文本 `EnvOutput(`，5 个结果

读取 [](file:///root/codes/python/RLinf/rlinf/workers/env/env_worker.py)

搜索文本 `def _preprocess_observation`，无结果

搜索文本 `_preprocess_observation(`，2 个结果

我会说明 LIBERO 环境的 observation 长什么样、在数据流中如何被打包（batch 化），以及模型端如何接收和处理这些字段（从 environment → EnvOutput → Rollout Worker → model.predict_action_batch → model.input_transform 的完整路径）。

**总体流程（概览）**

- 环境（`LiberoEnv`）每一次 step/采样产生原始观测 `extracted_obs`（通常包含图像、wrist 图像、物理 state、task 描述等）。
- `EnvOutput.prepare_observations` 将 `extracted_obs` 转换成统一的字典：`{"images", "wrist_images", "states", "task_descriptions"}`（并把图像从 HWC 转为 CHW）。
- `EnvWorker` 会把多个环境的单步观测按 batch 合并并通过 channel 发送给 Rollout Worker（因此模型收到的是带 batch 维度的数据）。
- Rollout Worker 调用模型的 `predict_action_batch(env_obs, ...)`，模型再通过 `input_transform` / `Observation.from_dict` / `_preprocess_observation` 把这些张量转换为模型需要的 embed/token/mask 格式。

**`prepare_observations` 在 LIBERO 下返回了哪些字段（精确）**

- `images`:
  - 由 `obs["images_and_states"]["full_image"]` 的每个条目 `value` 处理得到：
  - 单个样本（未 batch）形状：`[num_cam, 3, H, W]`，代码通过
    `torch.stack([value.clone().permute(2,0,1) for value in full_image])`
    生成（HWC -> CHW 并把不同 camera 的帧沿第 0 维堆叠）。
  - 经过 EnvWorker 的 batch 化后（传到 model），通常为：`[B, num_cam, 3, H, W]`。
- `wrist_images`:
  - 可选，存在于 `obs["images_and_states"]["wrist_image"]` 时同样处理为单样本 `[num_wrist, 3, H, W]`，batch 后为 `[B, num_wrist, 3, H, W]`。若不存在则为 `None`。
- `states`:
  - 物理/关节/任务相关的状态向量，来自 `obs["images_and_states"]["state"]` 或 `obs["state"]`。
  - 单样本形状通常为 `[state_dim]` 或 `[T, state_dim]`（取决 env 提供格式），batch 后一般为 `[B, state_dim]`。
- `task_descriptions`:
  - 如果存在，返回一个长度为 batch 的 `list[str]`（每个样本的自然语言 prompt 或任务描述）。

（实现位置参考：`rlinf/data/io_struct.py::EnvOutput.prepare_observations`）

**模型端如何接收这些字段（关键代码路径）**

- Rollout Worker（huggingface_worker.py）在 `generate()` / `predict()` 中接收 `env_output["obs"]` 并把它传给模型的 `predict_action_batch(env_obs, ...)`。
- `OpenPi0ForRLActionPrediction.predict_action_batch`（示例模型实现）会先调用 `input_processor(env_obs)`：
  - `input_processor` 把 `env_processed_obs` 映射为 model 要求的键名：
    - `"observation/image"` ← `env_processed_obs["images"]`
    - `"observation/state"` ← `env_processed_obs["states"]`
    - 若 `simulator_type == "libero"` 会额外添加 `"observation/wrist_image"` ← `env_processed_obs["wrist_images"]`
    - 并把这些张量转到模型 device（`.to(device)`）且确保 contiguous。
- 然后 `predict_action_batch` 内部会把处理后的字典传给 `self.sample_actions(observation, ...)`（或 `self.forward`），而 `input_transform`（定义在模型里）会进一步做视觉/语言的 transforms（resize/normalize/tokenize 等），最后通过 `Observation.from_dict` 和 `_preprocess_observation` 产出：
  - `images, img_masks, lang_tokens, lang_masks, state` 之类的结构，供模型编码器/decoder 使用。

**典型形状（便于理解）**

- 在模型 `input_processor` 完成 device 转移之后（batch）：
  - `env_obs["images"]` → `[B, num_cam, 3, H, W]`
  - `env_obs["wrist_images"]` → `[B, num_wrist, 3, H, W]` 或 `None`
  - `env_obs["states"]` → `[B, state_dim]`
  - `env_obs["task_descriptions"]` → `List[str]` 长度 `B`
- 在 `input_transform` / tokenizer 之后：
  - `lang_tokens` → `[B, seq_len]`
  - `lang_masks` → `[B, seq_len]`
  - `images` 可能会被拆成 `List[Tensor]`（每个 camera 一项）并做 padding/mask 成 `[B, N_tokens, feat_dim]`（取决具体 VLM 实现）。

**LIBERO 特殊点 / 注意项**

- LIBERO 常用多个相机（main view + wrist view）；代码中把 main view 的 `full_image`（可能包含多个 camera）与 `wrist_image` 区分开来，模型可选择是否使用 wrist 图像（配置项 `use_wrist_image` 在 config files 中出现）。
- 图像维度与 tokenization：`prepare_observations` 做了 HWC→CHW，但后续 `input_transform` 可能会 resize/normalize/patchify（视 VLM 实现）。
- Batch化：`EnvWorker.send_env_batch` 会把单样本 obs 列表合并成 batch（list → tensor 或 tensor 已有 batch 维通过 chunk），因此模型端看到的输入第 0 维始终是 batch 维。
- 对齐：模型需要知道 camera 数、图像尺寸等元信息；如果你在新环境/配置下运行，确认 config（`examples/embodiment/config/...`）中 `use_wrist_image`、`simulator_type` 等保持一致。

**如何快速验证你拿到的 observation（调试建议）**
在 rollout 过程中，临时打印或断言检查：

- 打印 `env_obs["images"].shape`、`env_obs["wrist_images"]` 是否为 None、`env_obs["states"].shape`。
- 在 model 的 `input_processor` 或 `input_transform` 里打印 `batch_size = next(v.shape[0] ...)` 是否与你期望的 `B` 一致。
- 如果模型报 shape/mask 不匹配，优先检查 camera 数 (num_cam) 与模型期望是否一致（代码里很多地方按固定 camera 分段做处理）。

---

如果你想，我可以：

- 在当前环境里运行一个“小型 dry-run”打印真实的 `env_obs` shapes（需要启动 EnvWorker/EnvManager，这会启动 simulator）；或者
- 在模型的 `input_processor` 添加一个短的 shape-assert/日志语句并提交 patch，方便后续调试。你希望我做哪种？
