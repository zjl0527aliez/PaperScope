# PaperScope · 文献研究助手

PaperScope 是一个“从一篇种子论文出发”的文献研究工具。用户输入论文标题、DOI、arXiv 编号或链接后，系统调用后端 API 完成论文识别、结构化解读、相关工作发现、证据核验、研究脉络整理和交互式可视化。

当前版本为 **v0.7.4 可核验成本版**：在材料降级透明、开放全文身份校验和分阶段模型缓存的基础上，新增显著数值事实锚定、模型阶段耗时、DeepSeek Prompt Cache Token 细分、端到端/材料准备/服务端耗时拆分，以及摘要版与自动全文版的可审计对比报告。详细模式没有正文时仍会先询问用户，不会静默降级。

用户仍可在页面内选择模型供应商、填写 API Key、Base URL 和模型名称，测试成功后立即激活。未激活模型时使用内置 EchoSight 演示；激活模型后可输入任意可被 OpenAlex 识别的论文，执行真实检索并由模型生成结构化解读。

## 运行

核心服务只使用 Python 标准库，不需要创建虚拟环境。直接读取 PDF 需要本机提供 `pdftotext`；当前开发机器已经安装，TXT、Markdown 和粘贴文本不依赖它。

```bash
cd PaperScope
python3 run.py
```

浏览器打开：<http://127.0.0.1:8765>

可选参数：

```bash
python3 run.py --host 127.0.0.1 --port 8765
```

启动后点击页面右上角“API 配置”：

1. 选择智谱 AI、OpenAI、DeepSeek、通义千问或其他 OpenAI 兼容接口；
2. 填写 Base URL、模型名称和 API Key；
3. 点击“测试并激活”。

密钥通过当前页面提交到本机后端，只保存在 Python 进程内存中。页面不会把密钥写入 `localStorage`、`sessionStorage` 或项目文件，也不会从后端重新读取密钥；关闭服务后配置自动消失。

实时研究完成后，服务会在 `runtime/research-runs/` 中最多保留 24 份最近报告，便于复盘耗时、Token 和材料范围。归档只包含生成后的结构化报告，会递归移除 API Key、原始 PDF/全文和请求中的 `sourceText`；`runtime/` 已由 `.gitignore` 排除。可通过 `GET /api/research/runs` 查看安全摘要。

模型调用凭证来自供应商每次请求的响应，不依赖供应商控制台的“最近使用”状态。不同供应商返回字段并不完全一致，因此 Request ID 或 Response ID 可能只有一项；PaperScope 会保留可用项并汇总多次质量修复调用的 Token。凭证和导出报告都不会包含 API Key。

## 当前可以体验什么

- 输入标题、DOI、arXiv 或 URL 的统一入口。
- 标准泛读与深度解读两种模式；深度模式优先使用用户提供的全文或重点章节。
- 标准模式只生成摘要有证据支持的核心分析；实验设置、发现和精读指南不会用模板句伪装完整。
- 深度模式把种子论文拆为核心分析与实验证据两个独立模型阶段，失败和缓存均可分阶段处理。
- PDF/正文按章节筛选后进入对应模型阶段，参考文献区不会作为论文事实材料重复输入。
- 支持粘贴最多 80,000 字符的论文材料，或读取本地 PDF、TXT、Markdown 文件。
- 本地 PDF 文本提取与摘要、引言、相关工作、方法、实验、局限、结论等章节识别；原始 PDF 不落盘。
- 页面内配置 OpenAlex 免费 API Key，论文检索和引用网络共用，关闭服务后自动清除。
- 展示 OpenAlex 额度、重置时间、实际请求与缓存命中；重复查询使用 15 分钟内存缓存。
- 429 错误区分“当日额度耗尽”和“瞬时请求过快”，并给出对应等待提示。
- 模型侧 401/403/429/5xx、网络超时和格式异常分别处理；额度错误零重试、拥堵只重试一次，避免无效消耗。
- 失败时保留实际研究阶段和建议等待时间；保持输入不变即可点击“从失败阶段继续”。
- 综合洞察新增相关工作横向对比矩阵。
- 深度报告新增关键概念、技术机制、实验设置、主要发现和精读指南。
- 相关工作卡片新增为何相关、关键差异和证据边界，不再只有一句摘要。
- 所有具备公开摘要的相关论文都会生成面向种子论文的逐篇详解；中文摘要也不会跳过关系与差异分析。
- 相关工作卡片明确标注“模型详解 / 公开摘要 / 仅元数据”，便于判断当前说明的证据深度。
- 六阶段研究管线和逐步运行反馈。
- 种子论文的研究问题、贡献、方法、证据和局限展示。
- 相关工作列表、时间线和交互式关系网络。
- 研究簇、演进判断、研究空白和建议阅读顺序。
- JSON 与 Markdown 报告导出。
- 页面内模型 API 配置、真实连接测试、激活和停用。
- 默认启用 OpenAlex 匿名免费数据源，并可在页面内测试连接。
- DOI 精确识别、题名检索、OpenAlex 相关工作推荐。
- 稀疏记录自动回退：相关工作少于 3 篇时，最多补做 1 次摘要语义检索。
- 相关工作按 OpenAlex ID、DOI 和规范化题名三重去重，同题不同版本不会伪装成多篇相关工作。
- 模型输出经过中文、占位内容和相关论文覆盖校验；首次不合格时自动修复一次。
- 百分比、模型规模和两位以上显著数值必须在当前阶段摘要或正文材料中出现；无依据数值会触发修复，不会直接进入报告。
- 模型调用按种子解读、相关工作和综合判断分阶段执行；各阶段只重试自己的小型 JSON，并在调用凭证中记录 stage 与 stageAttempt。
- 每个通过校验的模型阶段可独立缓存；后续运行会显示缓存命中，并只为尚未成功的阶段产生新 token。
- 深度列表在严格校验和一次模型修复均失败后，允许用明确的“材料不足/需原文核验”边界补足结构；报告会记录补足字段，基准评分不会把这些边界句计为有效深度内容。
- 相关工作说明被模型漏掉时采用同样的透明降级策略：保留检索关系和来源，但明确提示中文简介未通过校验，且基准评分会扣除对应质量分。
- “来源与证据”页展示模型供应商、模型名称、Request/Response ID、调用次数、Token 用量和质量修复次数；导出的 JSON 与 Markdown 报告也包含同一份凭证。
- 成本审计进一步拆分 DeepSeek Prompt Cache 命中/未命中 Token、推理 Token、逐阶段模型耗时、服务端耗时、材料准备耗时和端到端耗时。
- 有公开摘要的相关论文会生成中文简介与相关性说明，缺少摘要时保留明确提示。
- 参考文献回溯和后续引用扩展；真实引用与语义相关分开标注。
- Crossref 公共 API 作为 DOI 元数据回退。
- 智谱、OpenAI、DeepSeek、通义千问和其他 OpenAI 兼容接口的统一适配。
- DeepSeek V4 使用 `deepseek-flash` 预设并关闭思考模式，适合结构化论文整理且减少无效推理消耗。
- DeepSeek 激活仅对 429、5xx 或网络抖动有限重试一次；认证、余额和参数错误不会重复请求。
- API 激活失败时 Key 只暂留在当前输入框便于重试，激活成功、刷新页面或关闭服务后即清除。
- 已激活模型基于实时检索材料生成论文问题、贡献、方法、证据、局限与综合判断。

未激活模型时报告使用内置 EchoSight 数据。激活模型后，输入标题、DOI、arXiv 编号或论文链接会先由 OpenAlex 识别论文，再扩展相关工作和引用关系，最后调用当前模型完成结构化解读。免费模式不自动下载 PDF，因此具体实验数字和全文细节仍需回到原文核验。

### 深度解读建议

1. 输入题名、DOI、arXiv 编号或论文链接；
2. 将“分析深度”切换为“深度解读”；
3. 展开“添加论文全文或重点摘录”，直接读取 PDF、TXT、Markdown，或粘贴摘要、引言、方法、实验与局限章节；
4. 启动研究后，在“技术细读”和“相关工作详解”中查看扩展结果。

PDF 由本机 `pdftotext` 在内存管道中解析，不创建临时文件；扫描版 PDF 仍需要先进行 OCR。论文材料不会原样进入导出报告，报告只保留模型据此生成的结构化分析和材料范围说明。

## API 接入位置

1. 模型密钥通过页面提交，并在连接测试成功后保存在后端进程内存中。不要把真实密钥写入源码、`.env.example` 或提交到 Git。
2. `backend/integrations.py` 已实现通用 `OpenAICompatibleLLMClient`，以及 DeepSeek、智谱专用预设，通过标准库调用 `/chat/completions`，无需安装 SDK。结构化重组任务默认关闭深度思考，减少无效推理消耗。
3. `OpenAlexClient` 同时实现 `PaperSearchClient` 与 `CitationGraphClient`，默认使用匿名免费额度。
4. `CrossrefClient` 只在 DOI 无法由 OpenAlex 返回时提供基础元数据回退。
5. 实时检索结果会映射到与演示数据一致的稳定报告结构，再交给模型生成分析字段。

环境变量入口仍作为高级启动方式保留，但不是默认交互：

- `PAPERSCOPE_LLM_PROVIDER`
- `PAPERSCOPE_LLM_API_KEY`
- `PAPERSCOPE_LLM_BASE_URL`
- `PAPERSCOPE_LLM_MODEL`
- `PAPERSCOPE_OPENALEX_API_KEY`（推荐：同时用于论文检索和引用网络）
- `PAPERSCOPE_PAPER_SEARCH_PROVIDER`（默认 `openalex-free`）
- `PAPERSCOPE_PAPER_SEARCH_API_KEY`（匿名模式留空）
- `PAPERSCOPE_CITATION_PROVIDER`（默认 `openalex-free`）
- `PAPERSCOPE_CITATION_API_KEY`（匿名模式留空）

后端的 `/api/config` 只返回是否配置、provider 和 model，不返回密钥。

## API 草案

- `GET /api/health`：服务状态。
- `GET /api/config`：安全的配置状态。
- `GET /api/pipeline`：研究管线定义。
- `POST /api/research/start`：创建后台研究任务。
- `GET /api/research/status?jobId=...`：读取真实研究进度和最终报告。
- `GET /api/research/runs`：查看最近安全归档的报告摘要、耗时与 Token。
- `POST /api/research`：同步创建研究报告，保留给脚本和兼容调用。
- `GET /api/cache`：读取模型阶段缓存数量与占用空间。
- `DELETE /api/cache`：清除本地模型阶段缓存。
- `POST /api/llm/test`：测试当前模型配置，响应中不包含密钥。
- `POST /api/data/test`：测试 OpenAlex 免费数据源。
- `POST /api/config/openalex`：测试并在进程内存中激活 OpenAlex 免费 Key。
- `DELETE /api/config/openalex`：清除 OpenAlex Key 并恢复匿名模式。
- `POST /api/extract/pdf`：在本机内存中提取 PDF 文本与章节信息。
- `POST /api/fulltext/open-access`：由 OpenAlex 定位开放 PDF，在内存中下载、校验论文身份并提取正文。
- `POST /api/config/llm`：测试并激活页面提交的模型配置。
- `DELETE /api/config/llm`：从后端进程内存中清除模型配置。

请求示例：

```json
{
  "query": "EchoSight: Advancing Visual-Language Models with Wiki Knowledge",
  "demo": false,
  "depth": "deep",
  "sourceText": "可选：本次研究使用的论文全文或重点章节，最多 80,000 字符",
  "sourceInfo": {"kind": "user-file", "label": "paper.pdf"}
}
```

设置 `demo: false` 时，需要先在页面激活模型；论文数据源已默认启用。`depth` 可取 `standard` 或 `deep`。`sourceText` 不会写入文件，也不会原样进入导出报告；`sourceInfo` 只保留可展示的材料类型和名称。详细模式没有 `sourceText` 时，前端会询问是否调用 `/api/fulltext/open-access`；失败时由用户决定是否回退到摘要，不会偷偷降级。

## 验证

```bash
python3 -m unittest discover -s tests -v
node --check web/app.js
```

### 固定论文质量基准

功能测试继续使用模拟数据和 mock；质量优化改用此前未参与开发的 ReASC 论文作为固定盲测基准。基准输入为 DOI `10.18653/v1/2026.findings-acl.1085`，深度模式使用同一份 ACL Anthology 正式 PDF。评分锚点仅在报告生成后读取，不会传给模型。

先启动 PaperScope，并在页面激活模型与 OpenAlex。然后运行：

```bash
python3 scripts/benchmark.py capture \
  --label v0.5.1-baseline \
  --source /path/to/2026.findings-acl.1085.pdf
```

每次优化后用相同模型、相同 PDF 和相同配置再次捕获，再比较两次运行：

```bash
python3 scripts/benchmark.py compare \
  evaluation/runs/reasc/v0.5.1-baseline/report.json \
  evaluation/runs/reasc/v0.5.2-candidate/report.json
```

正式判断稳定性时连续运行三次；单次失败不会中止后续运行，最终会输出成功率以及评分、Token、耗时、局部修复和结构降级的统计：

```bash
python3 scripts/benchmark.py repeat \
  --label v0.6.3-api-guard \
  --runs 3 \
  --source /path/to/2026.findings-acl.1085.pdf
```

v0.7.4 评测可直接成对运行“ReASC 摘要版”和“OpenAlex 自动开放全文详细版”，不再需要手动下载 PDF：

```bash
python3 scripts/benchmark.py pair --label v0.7.4-deepseek
python3 scripts/benchmark.py pair --label v0.7.4-deepseek-reverse --order fulltext-first
```

该命令分别生成 `-abstract` 与 `-fulltext` 两份运行目录，并在 `evaluation/comparisons/` 写入质量、事实锚点、耗时和 Token 对比。建议用相反顺序再运行一次，以识别 OpenAlex 缓存和模型 Prompt Cache 带来的冷/热启动偏差；对比报告会记录顺序与 OpenAlex 缓存状态。单独捕获时也可使用 `--material abstract`、`--material auto-oa`，或配合 `--source` 使用 `--material source`。

评测总分为 100，覆盖论文识别、结构完整性、深度解读、原文事实锚点、相关工作和可追溯性；同时保留模型、Token、调用次数、修复次数与运行时长，避免只看文字长度。真实质量判断仍需结合评分报告中的未通过项和人工抽查，不能把单一总分当作最终结论。

## 目录结构

```text
PaperScope/
├── backend/
│   ├── evaluation.py      # 固定基准的确定性评分与前后比较
│   ├── config.py          # 页面内存配置与免费数据源默认值
│   ├── integrations.py    # 模型、OpenAlex 与 Crossref 适配器
│   ├── mock_data.py       # EchoSight 演示报告
│   ├── orchestrator.py    # 六阶段研究工作流
│   ├── stage_cache.py     # 经过校验的模型阶段缓存与自动失效
│   └── server.py          # 本地 API 与静态文件服务
├── tests/
├── evaluation/
│   ├── benchmarks/       # 固定输入、正式来源与事后评分锚点
│   └── runs/             # 各版本运行报告与评分
├── scripts/
│   └── benchmark.py      # 捕获、评分和比较命令
├── web/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── runtime/               # 本地运行缓存，已由 .gitignore 排除
├── .env.example
├── run.py
└── README.md
```

## 免费模式边界

- 不需要 OpenAlex 或 Crossref Key，但匿名额度较低，适合当前单人低频测试。
- 每次实时研究尽量只做一次论文识别、一次相关工作批量查询、一次参考文献批量查询和一次后续引用查询；只有稀疏记录会增加 1 次语义检索。
- 语义回退每次只发出 1 个请求，计入 OpenAlex 匿名免费额度；不自动下载 PDF，也不批量抓取全文。
- 后续需要更高额度时，可申请 OpenAlex 免费 Key，再保留当前接口直接升级。

