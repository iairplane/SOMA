# SOMA 推理 Pipeline 使用说明（`SOMA/src`）

## 0. 怎么跑起来
### 0.1 启动 SAM3 服务（另一个终端）
```bash
python /home/lizhuoran/SOMA/src/sam3_service.py \
  --host 0.0.0.0 \
  --port 5001 \
  --device cuda \
  --sam3_weight_path /mnt/disk1/shared_data/lzy/models/sam/sam3.pt
```

健康检查：
```bash
curl http://127.0.0.1:5001/health | cat
```

### 0.2 运行推理/评测入口 `soma_eval.py`
```bash
export SOMA_SAM3_URL=http://127.0.0.1:5001
# 可选：启用 memory（如果你已有 memory 数据）
export SOMA_ENABLE_MEMORY=0

python /home/lizhuoran/SOMA/src/soma_eval.py \
  --policy.path <你的policy路径或hub id> \
  --env.type libero \
  --eval.batch_size 1 \
  --eval.n_episodes 1 \
  --policy.device cuda \
  --output_dir outputs/soma_eval_run
```

---

## 1. 目录结构与每个文件的作用

### 1.1 推理入口与核心胶水
- **`soma_eval.py`**
  - **入口脚本**：创建 env + policy + pre/post processor + SOMA perception
  - 在 rollout 循环中：
    - 定期调用 `PerceptionModule.process_frame()` 得到 `processed_img/refined_task/control_flags`
    - 将 `processed_img` 写回 `observation[visual_key]`（policy 视觉输入）
    - 将 `refined_task` 写入 `add_envs_task(..., task_desc=...)`（policy 文本输入）
    - 消费控制流 `control_flags`，实现：
      - `encore`：回退/复位
      - `key_step_retry`：关键步骤重做（按历史成功步数阈值触发）
      - `task_decompose`：子任务拆分（成功后复位再推进）

- **`soma_control_flow.py`**
  - 控制流状态机实现：
    - `RollbackState`：reverse + buffer 的动作执行器
    - `TaskDecomposeState`：子任务列表/索引管理
    - `KeyStepRetryState`：保留控制流参数（当前 `soma_eval.py` 主要用 memory 的 `success_max_step/key_frame_start` 触发）

### 1.2 在线“编排器”
- **`soma_perception.py`**
  - `PerceptionModule`：在线编排器
  - 输入：`image(HWC uint8) + current_task + step + rag_context(可选)`
  - 输出：`processed_image + refined_task + control_flags`
  - 内部做：
    1. 调 `soma_vlm.py` 的 `orchestrate_perception()` 得到 plan JSON（tool_chain/params/refined_task/task_plan）
    2. 对视觉类 tool 调 `soma_tools.py`（HTTP→SAM3 服务）
    3. 对控制流类 tool 产出 `control_flags`（给 `soma_eval.py` 做状态机）

### 1.3 MCP Tools（视觉类，走 SAM3 服务）
- **`soma_tools.py`**
  - `MCPTools`：视觉 MCP 工具客户端（**不加载 SAM3**）
  - 通过 `Sam3HttpClient` 调用 `sam3_service.py`：
    - `visual_overlay`：高亮目标物体
    - `remove_distractor`：擦除干扰物（inpaint）
    - `replace_texture`：相似物体替代（贴图替换）
    - `replace_background`：背景替换（地面/桌面纹理）

- **`sam3_service.py`**
  - Flask 服务端，加载 SAM3 模型
  - 对外提供 HTTP API，输入 base64 PNG，输出处理后 base64 PNG
  - 端点：
    - `GET /health`
    - `POST /visual_overlay`
    - `POST /remove_distractor`
    - `POST /replace_texture`
    - `POST /replace_background`

### 1.4 VLM（策略编排决策）
- **`soma_vlm.py`**
  - `Qwen3VLAPIClient`
  - 核心：
    - `orchestrate_perception(image, task_desc, rag_context, rag_hints)`：输出严格 JSON plan
    - `generate_failure_report(...)`：日志/复盘时的失败归因（给 logger 用）
    - `generate_success_description(...)`：成功总结

依赖环境变量：
```bash
export SOMA_VLM_API_KEY=...
export SOMA_VLM_BASE_URL=...
export SOMA_VLM_MODEL=qwen3vl
```

### 1.5 Memory / Encoder / Logger（可选）
- **`soma_encoder.py`**
  - `AdvancedEmbeddingEncoder`：CLIP 图像向量 + 文本向量 + hash 拼接
  - 用于 RAG 检索 / 经验入库向量化

- **`soma_memory.py`**
  - `MemoryBank`：本地向量库（success/failure 分区）
  - 保存 `metadata.jsonl` + `vectors.npy`
  - `info.task_plan` 里存控制流统计（供 `soma_eval.py` 读取）：
    - `key_frame_range.start`
    - `success_max_step`
    - `rollback.reverse_steps/buffer_steps`
    -（可扩展）`subtask_success_max_steps`

- **`soma_logger.py`**
  - `ExperienceLogger`：采样视频关键帧、调用 VLM 归因、生成 embedding、写入 memory

- **`soma_agent.py`**
  - Facade：把 VLM/Encoder/Memory/Perception/Logger 封装成一个对象
  - 目前 `soma_eval.py` 没走这个 facade（直接手动拼 pipeline），但未来可以用它简化接入。

### 1.6 测试脚本
- **`test_sam3_mcp_tools.py`**
  - 最小可行测试：验证 sam3_service + MCPTools 的 4 个视觉工具端到端能跑，并生成输出图片。

---

## 2. 完整 pipeline 如何协作（数据流）
每个 step 的关键链路：

1. `env.step()` → 得到 `observation`
2. `preprocess_observation(observation)` → torch 化/规范化
3. 每 N 步调用 perception（默认每 10 步）：
   - 取出视觉 key（包含 `image` 或 `rgb`）
   - tensor → uint8 HWC → `PerceptionModule.process_frame(...)`
   - 得到 `processed_img/refined_task/control_flags`
   - `processed_img` 写回 `observation[visual_key]`
4. `add_envs_task(..., task_desc=refined_task)` → 注入任务文本
5. `preprocessor(observation)` → 对齐 device/rename 等
6. `policy.select_action(observation)` → 输出 action
7. `postprocessor(action)` → 后处理
8. `env.step(action_numpy)` → 推进环境
9. `soma_eval.py` 消费 `control_flags` 做控制流（rollback / 子任务切换 / 关键步骤重做）

---

## 3. 控制流 MCP tool（不走 SAM3）当前实现
### 3.1 `encore`
- 语义：立即请求回退/复位
- 执行：`soma_eval.py` 对当前累计动作求和并 reverse（reverse_steps + buffer_steps）

### 3.2 `key_step_retry`
- 语义：超过历史成功最大步数仍未成功，则回退到关键帧起点
- 当前触发阈值来源：
  - memory 的 `info.task_plan.success_max_step`
- 当前回退窗口：
  - `sum(action_hist[key_frame_start : now])`
- 执行：
  - `reverse_action = -window_sum / reverse_steps` 执行 reverse_steps，再 buffer

### 3.3 `task_decompose`
- 语义：将长程任务拆成子任务列表 `subtasks`
- 推进规则：仍按 `success_any=True` 推进，但**推进前先复位**
- 复位窗口：
  - `sum(action_hist[subtask_start_step : now])`
- 执行顺序：
  1. reverse + buffer
  2. `subtask_idx += 1`
  3. 更新 `subtask_start_step`

---

## 4. 依赖与安装建议
最小依赖（推理侧）：
- `numpy`
- `torch`
- `Pillow`
- `requests`
- `gymnasium`
- `einops`
- `tqdm`

SAM3 服务端额外依赖：
- `flask`
- `flask_cors`
- `sam3`（你的 SAM3 repo/安装方式）

VLM 额外依赖：
- `openai`（兼容 OpenAI API 格式服务）
- 网络可达的 VLM endpoint

---

## 5. 视觉 MCP tools 快速验证
```bash
python /home/lizhuoran/SOMA/src/test_sam3_mcp_tools.py \
  --sam3_url http://127.0.0.1:5001 \
  --image /home/lizhuoran/picture/with_tomato_sauce.png \
  --target "ketchup" \
  --distractor "tomato sauce" \
  --floor_prompt "floor" \
  --texture /home/lizhuoran/picture/tile_grigia_caldera_porcelain_floor.png \
  --out_dir /home/lizhuoran/soma_mcp_test
```

输出目录包含：
- `0_input.png`
- `1_overlay.png`
- `2_remove_distractor.png`
- `3_replace_texture.png`
- `4_replace_background.png`

---

## 6. 常见问题
- **看不到输出图片**
  - 确认 `--out_dir` 指向的目录是否正确
  - `ls -lah <out_dir>`
- **视觉工具没效果但 HTTP 200**
  - 多数是 prompt 不匹配（mask 为空）
  - prompt 建议 `.strip()`；尝试更泛化 prompt：`bottle/floor/table`
- **VLM 不可用**
  - 临时让 `tool_chain=[]` 也能跑 eval（只是不做编排与工具）
- **控制流触发看不出来**
  - 看日志里的 `[CTRL] Rollback start reason=...`，以及 tqdm postfix 的 `rollback=True/False`

