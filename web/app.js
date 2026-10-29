const state = {
  config: null,
  pipeline: [],
  report: null,
  activeView: "overview",
  selectedNode: null,
  running: false,
  sourceMaterialInfo: null,
  sourceInfo: null,
  cache: null,
  lastFailure: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function safeUrl(value = "") {
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url.href : "";
  } catch {
    return "";
  }
}

function formatProviderFailure(error) {
  const failure = error.payload?.failure || {};
  const label = {
    authentication: "API Key 无效或认证失败",
    permission: "当前账号没有模型访问权限",
    quota: "模型额度或余额不足",
    congestion: "模型当前拥堵",
    service: "模型服务暂时异常",
    network: "无法连接模型服务",
    timeout: "模型响应超时",
    "invalid-response": "模型返回格式无效",
  }[failure.category];
  const retryAfter = Number(failure.retryAfter);
  const wait = Number.isFinite(retryAfter) && retryAfter > 0 ? `，建议至少等待 ${Math.ceil(retryAfter)} 秒` : "";
  return `${label ? `${label}：` : ""}${error.message}${wait}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `请求失败（${response.status}）`);
    error.payload = payload;
    throw error;
  }
  return payload;
}

async function bootstrap() {
  try {
    const [config, pipeline, cache] = await Promise.all([api("/api/config"), api("/api/pipeline"), api("/api/cache")]);
    state.config = config;
    state.pipeline = pipeline.steps;
    state.cache = cache;
    renderConfig();
    renderCacheStatus();
    renderWorkflow();
    bindEvents();
    await runResearch(true, true);
  } catch (error) {
    showToast(`初始化失败：${error.message}`);
  }
}

function bindEvents() {
  $("#research-form").addEventListener("submit", event => {
    event.preventDefault();
    runResearch(state.config?.mode === "demo");
  });
  $$('[data-example]').forEach(button => button.addEventListener("click", () => {
    $("#paper-query").value = button.dataset.example;
    $("#paper-query").focus();
  }));
  $$(".view-tabs button").forEach(button => button.addEventListener("click", () => switchView(button.dataset.view)));
  $("#related-search").addEventListener("input", renderRelatedPapers);
  $("#relation-filter").addEventListener("change", renderGraph);
  $("#reset-graph").addEventListener("click", () => { state.selectedNode = null; $("#relation-filter").value = "all"; renderGraph(); });
  $("#export-json").addEventListener("click", exportJson);
  $("#export-md").addEventListener("click", exportMarkdown);
  $("#open-api").addEventListener("click", () => {
    $("#api-dialog").showModal();
    refreshCacheStatus();
  });
  $("#test-llm").addEventListener("click", testLlmConnection);
  $("#test-data").addEventListener("click", testDataConnection);
  $("#deactivate-llm").addEventListener("click", deactivateLlm);
  $("#llm-config-form").addEventListener("submit", activateLlm);
  $("#llm-provider").addEventListener("change", () => applyProviderPreset(true));
  $("#openalex-config-form").addEventListener("submit", activateOpenAlex);
  $("#deactivate-openalex").addEventListener("click", deactivateOpenAlex);
  $("#clear-stage-cache").addEventListener("click", clearStageCache);
  $("#paper-query").addEventListener("input", clearRetryState);
  $("#analysis-depth").addEventListener("change", clearRetryState);
  $("#source-material").addEventListener("input", () => {
    state.sourceMaterialInfo = null;
    state.sourceInfo = null;
    updateMaterialCount();
    clearRetryState();
  });
  $("#source-file").addEventListener("change", loadSourceFile);
  $("#clear-material").addEventListener("click", () => {
    $("#source-material").value = "";
    $("#source-file").value = "";
    state.sourceMaterialInfo = null;
    state.sourceInfo = null;
    updateMaterialCount();
    clearRetryState();
  });
}

function clearRetryState() {
  state.lastFailure = null;
  if (!state.running) $("#run-research").querySelector("span").textContent = "启动研究";
}

function updateMaterialCount() {
  const length = $("#source-material").value.length;
  const detail = state.sourceMaterialInfo ? ` · ${state.sourceMaterialInfo}` : "";
  $("#material-count").textContent = `${length.toLocaleString("zh-CN")} / 80,000 字符${detail}`;
  if (length > 0) $("#analysis-depth").value = "deep";
}

async function loadSourceFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  clearRetryState();
  const pdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  if (pdf && file.size > 12 * 1024 * 1024) {
    showToast("PDF 最大支持 12 MB，请只保留论文正文页面。", 4200);
    event.target.value = "";
    return;
  }
  if (!pdf && file.size > 300_000) {
    showToast("文本文件过大，请保留摘要、方法、实验和局限等关键章节。", 4200);
    event.target.value = "";
    return;
  }
  try {
    if (pdf) {
      showToast("正在本地提取 PDF 文本…", 6000);
      const data = arrayBufferToBase64(await file.arrayBuffer());
      const result = await api("/api/extract/pdf", { method: "POST", body: JSON.stringify({ data }) });
      $("#source-material").value = result.text;
      const sections = result.sections?.length ? `；识别 ${result.sections.join("、")}` : "";
      state.sourceMaterialInfo = `PDF ${result.pages} 页${sections}${result.truncated ? "；已截取前 80,000 字符" : ""}`;
      state.sourceInfo = { kind: "user-file", label: file.name };
    } else {
      const text = await file.text();
      $("#source-material").value = text.slice(0, 80_000);
      state.sourceMaterialInfo = file.name;
      state.sourceInfo = { kind: "user-file", label: file.name };
    }
    updateMaterialCount();
    showToast(`已读取 ${file.name}；材料只用于本次请求。`, 3600);
  } catch (error) {
    state.sourceMaterialInfo = null;
    state.sourceInfo = null;
    showToast(`无法读取文件：${error.message}`, 5200);
  }
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  }
  return btoa(binary);
}

function renderConfig() {
  const live = state.config?.mode === "live";
  const assisted = state.config?.mode === "assisted";
  const mode = $("#mode-status");
  mode.className = `status-pill ${live || assisted ? "status-live" : "status-demo"}`;
  mode.innerHTML = `<i></i>${live ? "实时模式" : assisted ? "模型测试" : "演示模式"}`;
  const entries = [
    ["LLM", "论文解读与综合", state.config?.llm],
    ["SEA", "论文元数据与语义检索", state.config?.paperSearch],
    ["CIT", "参考文献与引用网络", state.config?.citationGraph],
  ];
  $("#integration-list").innerHTML = entries.map(([icon, label, item]) => `
    <div class="integration-item">
      <span class="integration-icon">${icon}</span>
      <div><strong>${escapeHtml(label)}</strong><small>${escapeHtml(item?.providerLabel || item?.provider || "尚未配置")}${item?.model ? ` · ${escapeHtml(item.model)}` : ""}</small></div>
      <span class="config-state ${item?.configured ? "ready" : ""}">${item?.configured ? (item?.mode === "anonymous" ? "免费启用" : "已配置") : "待接入"}</span>
    </div>`).join("");
  renderProviderForm();
  const testButton = $("#test-llm");
  testButton.disabled = !state.config?.llm?.configured;
  testButton.textContent = state.config?.llm?.configured ? "测试模型连接" : "模型尚未配置";
  $("#deactivate-llm").disabled = !state.config?.llm?.configured;
  renderOpenAlexConfig();
}

function renderOpenAlexConfig() {
  const data = state.config?.paperSearch || {};
  const keyed = data.mode === "api-key";
  const badge = $("#openalex-key-state");
  badge.textContent = keyed ? "免费 Key 已启用" : "匿名模式";
  badge.className = `config-state ${keyed ? "ready" : ""}`;
  $("#deactivate-openalex").disabled = !keyed;
  const quota = data.rateLimit || {};
  $("#openalex-quota").textContent = formatOpenAlexQuota(quota, keyed);
}

function formatOpenAlexQuota(quota, keyed = false) {
  if (quota.remaining !== undefined || quota.limit !== undefined) {
    const remaining = quota.remaining ?? "—";
    const limit = quota.limit ?? "—";
    const reset = formatOpenAlexReset(quota);
    return `今日剩余额度 ${remaining} / ${limit}${reset ? ` · ${reset}` : ""}`;
  }
  return keyed ? "Key 已激活；执行一次连接测试或研究后显示剩余额度。" : "尚未读取个人额度；匿名模式仍可继续使用。";
}

function formatOpenAlexReset(quota) {
  if (quota.resetsAt) {
    const date = new Date(quota.resetsAt);
    if (!Number.isNaN(date.getTime())) return `${date.toLocaleString("zh-CN", { hour12: false })} 重置`;
  }
  const seconds = Number(quota.resetSeconds);
  if (Number.isFinite(seconds) && seconds >= 0) {
    const hours = Math.floor(seconds / 3600), minutes = Math.ceil((seconds % 3600) / 60);
    return `${hours}小时${minutes}分钟后重置`;
  }
  return "";
}

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function renderCacheStatus() {
  const cache = state.cache || {};
  const entries = Number(cache.entries) || 0;
  const stale = Number(cache.staleEntries) || 0;
  $("#stage-cache-status").textContent = entries
    ? `${entries} 个可复用阶段 · ${formatBytes(cache.sizeBytes)}${stale ? `；${stale} 个旧缓存已失效` : ""}`
    : stale ? `${stale} 个旧缓存已自动失效` : "暂无可复用阶段";
  $("#clear-stage-cache").disabled = entries + stale === 0;
}

async function refreshCacheStatus() {
  try {
    state.cache = await api("/api/cache");
    renderCacheStatus();
  } catch (error) {
    $("#stage-cache-status").textContent = `状态读取失败：${error.message}`;
  }
}

async function clearStageCache() {
  const button = $("#clear-stage-cache");
  button.disabled = true;
  try {
    const result = await api("/api/cache", { method: "DELETE" });
    state.cache = result;
    renderCacheStatus();
    showToast(`已清除 ${result.removed || 0} 个模型阶段缓存。`, 3600);
  } catch (error) {
    showToast(`缓存清理失败：${error.message}`, 4200);
  } finally {
    renderCacheStatus();
  }
}

function renderProviderForm() {
  const select = $("#llm-provider");
  const providers = state.config?.llmProviders || [];
  const selected = state.config?.llm?.provider || select.value || providers[0]?.id;
  select.innerHTML = providers.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`).join("");
  if (selected && providers.some(item => item.id === selected)) select.value = selected;
  if (state.config?.llm?.configured) {
    $("#llm-base-url").value = state.config.llm.baseUrl || "";
    $("#llm-model").value = state.config.llm.model || "";
  } else {
    applyProviderPreset(false);
  }
}

function applyProviderPreset(force) {
  const preset = (state.config?.llmProviders || []).find(item => item.id === $("#llm-provider").value);
  if (!preset) return;
  if (force || !$("#llm-base-url").value) $("#llm-base-url").value = preset.baseUrl || "";
  if (force || !$("#llm-model").value) $("#llm-model").value = preset.defaultModel || "";
}

async function activateLlm(event) {
  event.preventDefault();
  const button = $("#activate-llm");
  const apiKey = $("#llm-api-key");
  const payload = {
    provider: $("#llm-provider").value,
    baseUrl: $("#llm-base-url").value.trim(),
    model: $("#llm-model").value.trim(),
    apiKey: apiKey.value.trim(),
  };
  if (!payload.apiKey || !payload.baseUrl || !payload.model) {
    showToast("请完整填写 Base URL、模型名称和 API Key。", 3600);
    return;
  }
  button.disabled = true;
  button.textContent = "正在测试连接…";
  let activated = false;
  try {
    const result = await api("/api/config/llm", { method: "POST", body: JSON.stringify(payload) });
    state.config = result.config;
    activated = true;
    renderConfig();
    showToast(`已激活：${result.config.llm.providerLabel} · ${result.connection.model}`, 4000);
  } catch (error) {
    showToast(`激活失败：${formatProviderFailure(error)}`, 6200);
  } finally {
    if (activated) apiKey.value = "";
    button.disabled = false;
    button.textContent = "测试并激活";
  }
}

async function deactivateLlm() {
  const button = $("#deactivate-llm");
  button.disabled = true;
  try {
    const result = await api("/api/config/llm", { method: "DELETE" });
    state.config = result.config;
    $("#llm-api-key").value = "";
    renderConfig();
    showToast("模型配置已从后端进程内存中清除。", 3600);
  } catch (error) {
    showToast(`停用失败：${error.message}`, 4200);
  }
}

async function activateOpenAlex(event) {
  event.preventDefault();
  const apiKey = $("#openalex-api-key");
  const button = $("#activate-openalex");
  const value = apiKey.value.trim();
  if (!value) { showToast("请输入 OpenAlex API Key。", 3600); return; }
  button.disabled = true;
  button.textContent = "正在验证额度…";
  try {
    const result = await api("/api/config/openalex", { method: "POST", body: JSON.stringify({ apiKey: value }) });
    state.config = result.config;
    renderConfig();
    showToast(`OpenAlex 免费 Key 已激活。${formatOpenAlexQuota(result.connection.rateLimit || {}, true)}`, 5200);
  } catch (error) {
    showToast(`OpenAlex Key 激活失败：${error.message}`, 5200);
  } finally {
    apiKey.value = "";
    button.disabled = false;
    button.textContent = "测试并激活";
  }
}

async function deactivateOpenAlex() {
  const button = $("#deactivate-openalex");
  button.disabled = true;
  try {
    const result = await api("/api/config/openalex", { method: "DELETE" });
    state.config = result.config;
    $("#openalex-api-key").value = "";
    renderConfig();
    showToast("OpenAlex Key 已从内存清除，当前恢复匿名模式。", 4200);
  } catch (error) {
    showToast(`停用失败：${error.message}`, 4200);
  }
}

async function testLlmConnection() {
  const button = $("#test-llm");
  button.disabled = true;
  button.textContent = "正在连接…";
  try {
    const result = await api("/api/llm/test", { method: "POST", body: "{}" });
    showToast(result.ok ? `连接成功：${result.model}` : "模型已响应，但校验短语不匹配。", 3600);
  } catch (error) {
    showToast(`连接失败：${formatProviderFailure(error)}`, 5600);
  } finally {
    button.disabled = !state.config?.llm?.configured;
    button.textContent = state.config?.llm?.configured ? "测试模型连接" : "模型尚未配置";
  }
}

async function testDataConnection() {
  const button = $("#test-data");
  button.disabled = true;
  button.textContent = "正在连接…";
  try {
    const result = await api("/api/data/test", { method: "POST", body: "{}" });
    if (result.rateLimit) {
      state.config.paperSearch.rateLimit = result.rateLimit;
      state.config.citationGraph.rateLimit = result.rateLimit;
      renderOpenAlexConfig();
    }
    showToast(result.ok ? `OpenAlex 连接成功；${formatOpenAlexQuota(result.rateLimit || {}, result.mode === "api-key")}` : "数据源已响应，但校验未通过。", 4600);
  } catch (error) {
    showToast(`数据源连接失败：${error.message}`, 4600);
  } finally {
    button.disabled = false;
    button.textContent = "测试免费论文数据源";
  }
}

function renderWorkflow(activeIndex = -1, activeDetail = "") {
  $("#workflow-count").textContent = `${state.pipeline.length || 6} 阶段`;
  $("#workflow-list").innerHTML = state.pipeline.map((step, index) => `
    <li class="workflow-item ${index < activeIndex ? "done" : index === activeIndex ? "active" : ""}" data-step="${escapeHtml(step.id)}">
      <span class="workflow-dot">${index < activeIndex ? "✓" : String(index + 1).padStart(2, "0")}</span>
      <div class="workflow-text"><strong>${escapeHtml(step.label)}</strong><p>${escapeHtml(index === activeIndex && activeDetail ? activeDetail : step.description)}</p></div>
    </li>`).join("");
}

function renderResearchJob(job, button) {
  const activeIndex = state.pipeline.findIndex(step => step.id === job.stepId);
  renderWorkflow(activeIndex >= 0 ? activeIndex : 0, job.detail || "正在研究…");
  if (job.status === "failed") $(".workflow-item.active")?.classList.add("failed");
  button.querySelector("span").textContent = job.detail || "正在研究…";
}

async function waitForResearchJob(initialJob, button) {
  let job = initialJob;
  while (job.status === "queued" || job.status === "running") {
    renderResearchJob(job, button);
    await delay(700);
    job = await api(`/api/research/status?jobId=${encodeURIComponent(job.jobId)}`);
  }
  renderResearchJob(job, button);
  if (job.status === "failed") {
    const error = new Error(job.error || "研究任务失败，请重试。");
    error.job = job;
    throw error;
  }
  if (job.status !== "completed" || !job.report) throw new Error("研究任务返回了无效状态，请重试。");
  return job.report;
}

async function runResearch(demo = true, quiet = false) {
  if (state.running) return;
  const query = $("#paper-query").value.trim();
  if (!query) { showToast("请先输入一篇论文。"); return; }
  state.running = true;
  const button = $("#run-research");
  button.disabled = true;
  const providerLabel = state.config?.llm?.providerLabel || "模型";
  button.querySelector("span").textContent = demo ? "生成演示报告…" : state.config?.mode === "assisted" ? `调用${providerLabel}…` : "正在研究…";
  const depth = $("#analysis-depth").value;
  let sourceText = $("#source-material").value.trim();
  let sourceInfo = state.sourceInfo;
  try {
    if (!demo && depth === "deep" && !sourceText) {
      const choice = await chooseFulltextMode();
      if (choice === "cancel") return;
      if (choice === "summary") {
        sourceInfo = { kind: "abstract-fallback", label: "用户选择仅使用 OpenAlex 摘要", reason: "user-choice" };
      }
      if (choice === "find") {
        button.querySelector("span").textContent = "查找开放全文…";
        try {
          const result = await api("/api/fulltext/open-access", {
            method: "POST",
            body: JSON.stringify({ query }),
          });
          $("#source-material").value = result.text;
          const sections = result.sections?.length ? `；识别 ${result.sections.join("、")}` : "";
          state.sourceMaterialInfo = `开放 PDF ${result.pages} 页${sections}${result.truncated ? "；已截取前 80,000 字符" : ""}`;
          state.sourceInfo = result.source;
          sourceText = result.text;
          sourceInfo = result.source;
          updateMaterialCount();
          showToast("已匹配并读取开放全文；PDF 只在本次进程内存中处理。", 4600);
        } catch (error) {
          const fallback = await chooseFulltextFallback(error.message);
          if (fallback !== "summary") return;
          sourceInfo = {
            kind: "abstract-fallback",
            label: "开放全文不可用，已回退到 OpenAlex 摘要",
            reason: error.message,
          };
        }
      }
    }
    const job = await api("/api/research/start", {
      method: "POST",
      body: JSON.stringify({ query, demo, depth, sourceText, sourceInfo }),
    });
    const report = await waitForResearchJob(job, button);
    state.lastFailure = null;
    state.report = report;
    if (report.meta?.openAlex && state.config?.paperSearch) {
      state.config.paperSearch.rateLimit = report.meta.openAlex;
      state.config.citationGraph.rateLimit = report.meta.openAlex;
      renderOpenAlexConfig();
    }
    renderWorkflow(state.pipeline.length);
    renderReport();
    await refreshCacheStatus();
    const cacheHits = Number(report.meta?.modelCall?.cacheHits) || 0;
    if (!quiet) showToast(
      demo ? "演示报告已生成；接入 API 后会替换为实时结果。" : cacheHits ? `研究报告已生成；复用了 ${cacheHits} 个模型阶段缓存。` : "研究报告已生成。",
      3600,
    );
  } catch (error) {
    if (error.job) {
      state.lastFailure = { job: error.job, query, depth, sourceText, sourceInfo };
      renderResearchJob(error.job, button);
      const retryAfter = Number(error.job.failure?.retryAfter);
      const wait = Number.isFinite(retryAfter) && retryAfter > 0 ? `；建议至少等待 ${Math.ceil(retryAfter)} 秒` : "";
      showToast(`${error.job.detail}：${error.message}${wait}`, 6500);
    } else {
      renderWorkflow();
      showToast(error.message, 4200);
    }
  } finally {
    state.running = false;
    button.disabled = false;
    if (state.lastFailure) {
      const failedStep = state.pipeline.find(step => step.id === state.lastFailure.job.stepId)?.label || "失败阶段";
      button.querySelector("span").textContent = state.lastFailure.job.canResume ? `从${failedStep}继续` : "修复配置后重试";
    } else {
      button.querySelector("span").textContent = "启动研究";
    }
  }
}

function chooseFulltextMode() {
  const dialog = $("#fulltext-dialog");
  return new Promise(resolve => {
    const onClose = () => {
      dialog.removeEventListener("close", onClose);
      resolve(dialog.returnValue || "cancel");
    };
    dialog.addEventListener("close", onClose);
    dialog.showModal();
  });
}

function chooseFulltextFallback(reason) {
  const dialog = $("#fulltext-fallback-dialog");
  $("#fulltext-fallback-reason").textContent = reason || "开放全文暂时不可用。";
  return new Promise(resolve => {
    const onClose = () => {
      dialog.removeEventListener("close", onClose);
      resolve(dialog.returnValue || "cancel");
    };
    dialog.addEventListener("close", onClose);
    dialog.showModal();
  });
}

function delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

function renderReport() {
  const report = state.report;
  if (!report) return;
  $("#demo-notice").textContent = report.meta.notice || "报告已生成。";
  $("#demo-banner").classList.toggle("live", report.meta.mode !== "demo");
  renderPaperHero();
  renderOverview();
  renderRelatedPapers();
  renderGraph();
  renderTimeline();
  renderSynthesis();
  renderSources();
}

function renderPaperHero() {
  const paper = state.report.paper;
  const url = safeUrl(paper.sourceUrl);
  $("#paper-hero").innerHTML = `
    <div class="hero-meta"><span class="seed-badge">种子论文</span><span>${escapeHtml(paper.authors)}</span><span>·</span><span>${paper.year}</span><span>·</span><span>${escapeHtml(paper.venue)}</span></div>
    <h2>${escapeHtml(paper.title)}</h2>
    <p>${escapeHtml(paper.oneSentence)}</p>
    <div class="hero-foot">
      <div class="tag-row">${paper.tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>
      ${url ? `<a class="source-link" href="${url}" target="_blank" rel="noreferrer">查看原始来源 <svg viewBox="0 0 24 24"><path d="M8 16 16 8m-6 0h6v6"/><path d="M16 14v5H5V8h5"/></svg></a>` : ""}
    </div>`;
}

function renderOverview() {
  const { analysis, relatedPapers, relations, paper } = state.report;
  const discoveredPapers = relatedPapers.filter(item => item.id !== paper.id);
  const yearSpan = `${Math.min(...relatedPapers.map(p => p.year))}—${Math.max(...relatedPapers.map(p => p.year))}`;
  $("#report-metrics").innerHTML = [
    [discoveredPapers.length, "纳入相关工作"],
    [relations.length, "关系判断"],
    [new Set(relatedPapers.map(p => p.cluster)).size, "研究主题簇"],
    [yearSpan, "时间覆盖"],
  ].map(([value, label]) => `<div class="metric"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("");
  $("#paper-problem").textContent = analysis.problem;
  $("#paper-contributions").innerHTML = analysis.contributions.map(item => `<li>${escapeHtml(item)}</li>`).join("");
  $("#method-flow").innerHTML = analysis.methodSteps.map((item, index) => `<div class="method-step"><span>STEP ${String(index + 1).padStart(2, "0")}</span><strong>${escapeHtml(item.label)}</strong><p>${escapeHtml(item.text)}</p></div>`).join("");
  $("#paper-evidence").innerHTML = analysis.evidence.map(item => `<li>${escapeHtml(item)}</li>`).join("");
  $("#paper-limitations").innerHTML = analysis.limitations.map(item => `<li>${escapeHtml(item)}</li>`).join("");
  renderDeepDive(analysis.deepDive || {});
}

function renderDeepDive(deepDive) {
  const concepts = Array.isArray(deepDive.keyConcepts) ? deepDive.keyConcepts : [];
  const sections = [
    ["#technical-details", deepDive.technicalDetails],
    ["#experimental-setup", deepDive.experimentalSetup],
    ["#detailed-findings", deepDive.findings],
    ["#reading-guide", deepDive.readingGuide],
  ];
  $("#material-scope").textContent = state.report.meta.materialScope || "基于当前可用材料";
  $("#key-concepts").innerHTML = concepts.length
    ? concepts.map(item => `<article><strong>${escapeHtml(item.term)}</strong><p>${escapeHtml(item.explanation)}</p></article>`).join("")
    : `<p class="deep-empty">当前报告未生成关键概念解释；选择“深度解读”并补充全文材料可获得更完整结果。</p>`;
  sections.forEach(([selector, items]) => {
    const values = Array.isArray(items) ? items : [];
    $(selector).innerHTML = values.length
      ? values.map(item => `<li>${escapeHtml(item)}</li>`).join("")
      : `<li class="deep-empty">当前材料不足，暂未生成。</li>`;
  });
}

function renderRelatedPapers() {
  if (!state.report) return;
  const query = $("#related-search").value.trim().toLowerCase();
  const papers = state.report.relatedPapers.filter(paper => paper.id !== state.report.paper.id && [paper.title, paper.shortTitle, paper.summary, paper.relation, paper.cluster].join(" ").toLowerCase().includes(query));
  $("#related-grid").innerHTML = papers.map(paper => {
    const url = safeUrl(paper.sourceUrl);
    const detail = paper.why || paper.difference || paper.evidenceBoundary ? `<dl class="related-detail">
      <dt>为何相关</dt><dd>${escapeHtml(paper.why || "尚待核验")}</dd>
      <dt>关键差异</dt><dd>${escapeHtml(paper.difference || "当前材料不足以可靠判断")}</dd>
      <dt>证据边界</dt><dd>${escapeHtml(paper.evidenceBoundary || "依据公开摘要或元数据整理")}</dd>
    </dl>` : "";
    return `<article class="related-item ${paper.id === state.report.paper.id ? "seed" : ""}">
      <div class="related-item-top"><span class="relation-label">${escapeHtml(paper.relation)}</span><span>${escapeHtml(paper.analysisLevel || "公开材料")} · ${paper.year} · ${escapeHtml(paper.venue)}</span></div>
      <h4>${escapeHtml(paper.shortTitle)}</h4><p>${escapeHtml(paper.summary)}</p>
      ${detail}
      ${url ? `<a href="${url}" target="_blank" rel="noreferrer">原始来源 ↗</a>` : ""}
    </article>`;
  }).join("") || `<p class="lead-copy">当前数据源未发现可核验的相关工作；系统不会用种子论文充数。</p>`;
}

function graphKind(paper) {
  if (paper.id === state.report.paper.id) return "seed";
  if (paper.relation === "范式对照") return "contrast";
  return paper.year < state.report.paper.year ? "foundation" : "follow";
}

function renderGraph() {
  if (!state.report) return;
  const filter = $("#relation-filter").value;
  const seedYear = state.report.paper.year;
  const allPapers = state.report.relatedPapers;
  const papers = allPapers.filter(p => filter === "all" || (filter === "earlier" ? p.year <= seedYear : p.year >= seedYear));
  const ids = new Set(papers.map(p => p.id));
  const relations = state.report.relations.filter(r => ids.has(r.from) && ids.has(r.to));
  const years = [...new Set(allPapers.map(p => p.year))].sort();
  const minYear = Math.min(...years), maxYear = Math.max(...years);
  const yByCluster = new Map([...new Set(allPapers.map(p => p.cluster))].map((cluster, index, list) => [cluster, 90 + index * (340 / Math.max(1, list.length - 1))]));
  const positions = {};
  papers.forEach((paper, index) => {
    const same = papers.filter(p => p.year === paper.year && p.cluster === paper.cluster);
    const sameIndex = same.findIndex(p => p.id === paper.id);
    positions[paper.id] = {
      x: 78 + ((paper.year - minYear) / Math.max(1, maxYear - minYear)) * 820,
      y: yByCluster.get(paper.cluster) + (sameIndex - (same.length - 1) / 2) * 50,
    };
  });
  const colors = { seed: "#3267e3", foundation: "#1b8c82", follow: "#e36f51", contrast: "#ba7a13" };
  const lines = relations.map(relation => {
    const a = positions[relation.from], b = positions[relation.to];
    const contrast = relation.type.includes("对照") ? "contrast" : "";
    return `<path class="graph-edge ${contrast}" d="M ${a.x} ${a.y} C ${(a.x + b.x) / 2} ${a.y}, ${(a.x + b.x) / 2} ${b.y}, ${b.x} ${b.y}"><title>${escapeHtml(relation.label)}</title></path>`;
  }).join("");
  const yearLabels = years.map(year => `<text class="graph-year" x="${78 + ((year - minYear) / Math.max(1, maxYear - minYear)) * 820}" y="492">${year}</text>`).join("");
  const nodes = papers.map(paper => {
    const pos = positions[paper.id], kind = graphKind(paper);
    return `<g class="graph-node ${state.selectedNode === paper.id ? "selected" : ""}" data-node="${escapeHtml(paper.id)}" transform="translate(${pos.x} ${pos.y})" role="button" tabindex="0">
      <circle r="${kind === "seed" ? 18 : 13}" fill="${colors[kind]}"><title>${escapeHtml(paper.title)}</title></circle>
      <text y="${kind === "seed" ? 34 : 29}">${escapeHtml(paper.shortTitle)}</text>
    </g>`;
  }).join("");
  $("#graph-canvas").innerHTML = `<svg viewBox="0 0 980 520" aria-label="相关论文关系图"><g>${lines}</g><g>${nodes}</g><g>${yearLabels}</g></svg>`;
  $$('[data-node]', $("#graph-canvas")).forEach(node => {
    const select = () => { state.selectedNode = node.dataset.node; renderGraph(); renderNodeDetail(); };
    node.addEventListener("click", select);
    node.addEventListener("keydown", event => { if (["Enter", " "].includes(event.key)) select(); });
  });
  renderNodeDetail();
}

function renderNodeDetail() {
  const paper = state.report?.relatedPapers.find(item => item.id === state.selectedNode);
  if (!paper) {
    $("#node-detail").innerHTML = `<span class="empty-kicker">选择节点</span><h4>点击论文查看关系说明</h4><p>网络中的连线表示方法演进、任务基础或编者对照，不自动等同于引用关系。</p>`;
    return;
  }
  const url = safeUrl(paper.sourceUrl);
  $("#node-detail").innerHTML = `<span class="empty-kicker">${escapeHtml(paper.relation)}</span><h4>${escapeHtml(paper.shortTitle)}</h4><p>${escapeHtml(paper.summary)}</p><dl><dt>发表信息</dt><dd>${paper.year} · ${escapeHtml(paper.venue)}</dd><dt>为何相关</dt><dd>${escapeHtml(paper.why)}</dd><dt>关系置信度</dt><dd>${escapeHtml(paper.confidence)}</dd></dl>${url ? `<a href="${url}" target="_blank" rel="noreferrer">打开原始来源 ↗</a>` : ""}`;
}

function renderTimeline() {
  const groups = [...state.report.relatedPapers]
    .sort((a, b) => a.year - b.year)
    .reduce((result, item) => {
      (result[item.year] ??= []).push(item);
      return result;
    }, {});
  $("#timeline").innerHTML = Object.entries(groups).map(([year, papers]) => `<section class="timeline-group"><div class="timeline-year">${year}</div><div class="timeline-papers">${papers.map(paper => `<article class="timeline-paper ${paper.id === state.report.paper.id ? "seed" : ""}"><div class="timeline-paper-head"><strong>${escapeHtml(paper.shortTitle)}</strong><span>${escapeHtml(paper.relation)}</span></div><p>${escapeHtml(paper.summary)}</p></article>`).join("")}</div></section>`).join("");
}

function renderSynthesis() {
  const { synthesis, searchPlan } = state.report;
  $("#synthesis-thesis").textContent = synthesis.thesis;
  renderComparisonMatrix();
  $("#cluster-grid").innerHTML = synthesis.clusters.map((cluster, index) => `<article class="cluster-card"><span class="cluster-no">0${index + 1}</span><h3>${escapeHtml(cluster.name)}</h3><p>${escapeHtml(cluster.insight)}</p><div class="paper-chips">${cluster.papers.map(paper => `<span>${escapeHtml(paper)}</span>`).join("")}</div></article>`).join("");
  $("#gap-list").innerHTML = synthesis.gaps.map(gap => `<li>${escapeHtml(gap)}</li>`).join("");
  $("#reading-path").innerHTML = synthesis.readingOrder.map(paper => `<span>${escapeHtml(paper)}</span>`).join("");
  const columns = [["检索词", searchPlan.queries], ["发现通道", searchPlan.channels], ["纳入规则", searchPlan.rules]];
  $("#search-strategy").innerHTML = columns.map(([title, items]) => `<div class="strategy-column"><h4>${escapeHtml(title)}</h4><ul>${items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>`).join("");
}

function renderComparisonMatrix() {
  const papers = state.report.relatedPapers.filter(item => item.id !== state.report.paper.id);
  $("#comparison-matrix").innerHTML = papers.map(paper => `<tr>
    <th>${escapeHtml(paper.shortTitle)}</th>
    <td>${escapeHtml(String(paper.year || "—"))}</td>
    <td><span class="relation-label">${escapeHtml(paper.relation)}</span></td>
    <td>${escapeHtml(paper.summary)}</td>
    <td>${escapeHtml(paper.difference || paper.why || "当前材料不足以可靠判断")}</td>
    <td>${escapeHtml(paper.evidenceBoundary || "依据公开摘要或元数据整理")}</td>
  </tr>`).join("") || `<tr><td colspan="6">当前没有可比较的相关工作。</td></tr>`;
}

function renderSources() {
  renderModelAudit();
  renderDataAudit();
  const header = `<div class="provenance-row header"><span>项目</span><span>类型</span><span>来源</span><span>状态</span></div>`;
  $("#provenance").innerHTML = header + state.report.provenance.map(item => {
    const url = safeUrl(item.url);
    return `<div class="provenance-row"><strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(item.kind)}</span><span>${url ? `<a href="${url}" target="_blank" rel="noreferrer">打开来源 ↗</a>` : "—"}</span><span class="provenance-status">${escapeHtml(item.status)}</span></div>`;
  }).join("");
}

function renderDataAudit() {
  const data = state.report?.meta?.openAlex;
  const status = $("#data-audit-status");
  if (!data) {
    status.textContent = "无实时数据";
    status.className = "audit-status muted";
    $("#data-audit").innerHTML = `<div class="audit-empty"><strong>当前报告没有 OpenAlex 额度记录</strong><p>演示报告不会请求外部论文数据源。</p></div>`;
    return;
  }
  const keyed = data.mode === "api-key";
  status.textContent = keyed ? "免费 Key 模式" : "匿名模式";
  status.className = `audit-status ${keyed ? "verified" : "muted"}`;
  const fields = [
    ["访问模式", keyed ? "免费个人 API Key" : "匿名共享额度"],
    ["今日剩余额度", data.remaining ?? "供应商未返回"],
    ["每日额度上限", data.limit ?? "供应商未返回"],
    ["本次请求消耗", data.creditsUsedByRequest ?? "供应商未返回"],
    ["缓存命中", data.cacheHits ?? 0],
    ["实际请求", data.cacheMisses ?? 0],
    ["额度重置", formatOpenAlexReset(data) || "供应商未返回"],
  ];
  $("#data-audit").innerHTML = `<div class="audit-grid">${fields.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`).join("")}</div>`;
}

function renderModelAudit() {
  const meta = state.report?.meta || {};
  const audit = meta.modelCall;
  const status = $("#model-audit-status");
  if (!audit) {
    status.textContent = "未调用模型";
    status.className = "audit-status muted";
    $("#model-audit").innerHTML = `<div class="audit-empty"><strong>当前报告没有模型调用凭证</strong><p>${meta.mode === "demo" ? "这是内置演示报告，未请求外部模型。" : "当前供应商没有返回可核验的调用信息。"}</p></div>`;
    return;
  }
  status.textContent = audit.cacheHits && !(audit.usage?.total_tokens) ? "缓存已校验" : audit.verified ? "服务端已响应" : "信息不完整";
  status.className = `audit-status ${audit.verified ? "verified" : "muted"}`;
  const usage = audit.usage || {};
  const timestamp = formatAuditTime(audit.providerCreatedAt || meta.generatedAt);
  const fields = [
    ["供应商 / 模型", `${audit.providerLabel || audit.provider || "—"} · ${audit.model || "—"}`],
    ["阶段记录数", String(audit.callCount || 1)],
    ["输入 tokens", String(usage.prompt_tokens ?? "—")],
    ["输出 tokens", String(usage.completion_tokens ?? "—")],
    ["总 tokens", String(usage.total_tokens ?? "—")],
    ["质量修复次数", String(audit.validationRetries || 0)],
    ["传输重试次数", String(audit.transportRetries || 0)],
    ["复用阶段缓存", String(audit.cacheHits || 0)],
    ["结构降级字段", String(audit.schemaFallbackCount || 0)],
    ["服务端耗时", Number.isFinite(Number(meta.elapsedSeconds)) ? `${Number(meta.elapsedSeconds).toFixed(2)} 秒` : "—"],
    ["服务端时间", timestamp],
  ];
  const identifiers = [
    ["Request ID", audit.requestId],
    ["Response ID", audit.responseId],
  ];
  const calls = Array.isArray(audit.calls) ? audit.calls : [];
  $("#model-audit").innerHTML = `
    <div class="audit-grid">${fields.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}</div>
    <div class="audit-identifiers">${identifiers.map(([label, value]) => `<div><span>${label}</span><code>${escapeHtml(value || "供应商未返回")}</code></div>`).join("")}</div>
    ${calls.length > 1 ? `<details class="audit-calls"><summary>查看 ${calls.length} 个模型阶段明细</summary>${calls.map(call => `<div><strong>${call.cacheHit ? "缓存" : `第 ${call.sequence} 次`}</strong><span>${escapeHtml(call.stage || call.model || audit.model || "—")}</span><code>${escapeHtml(call.cacheHit ? "已通过本地校验缓存复用" : call.requestId || call.responseId || "无请求标识")}</code><span>${call.cacheHit ? "0 新 tokens" : `${escapeHtml(String(call.usage?.total_tokens ?? "—"))} tokens`}</span></div>`).join("")}</details>` : ""}`;
}

function formatAuditTime(value) {
  if (!value) return "—";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
}

function switchView(view) {
  state.activeView = view;
  $$(".view-tabs button").forEach(button => button.classList.toggle("active", button.dataset.view === view));
  $$(".report-view").forEach(panel => panel.classList.toggle("active", panel.dataset.panel === view));
}

function reportMarkdown() {
  const { paper, analysis, relatedPapers, synthesis, meta } = state.report;
  const deepDive = analysis.deepDive || {};
  const deepSection = `\n## 技术细读\n\n### 关键概念\n\n${(deepDive.keyConcepts || []).map(item => `- **${item.term}**：${item.explanation}`).join("\n") || "- 当前材料不足"}\n\n### 机制与实现\n\n${(deepDive.technicalDetails || []).map(item => `- ${item}`).join("\n") || "- 当前材料不足"}\n\n### 实验设置\n\n${(deepDive.experimentalSetup || []).map(item => `- ${item}`).join("\n") || "- 当前材料不足"}\n\n### 主要发现\n\n${(deepDive.findings || []).map(item => `- ${item}`).join("\n") || "- 当前材料不足"}\n\n### 精读指南\n\n${(deepDive.readingGuide || []).map((item, i) => `${i + 1}. ${item}`).join("\n") || "1. 当前材料不足"}\n`;
  const audit = meta.modelCall;
  const auditSection = audit ? `\n## 模型调用凭证\n\n- 供应商：${audit.providerLabel || audit.provider}\n- 模型：${audit.model}\n- Request ID：${audit.requestId || "供应商未返回"}\n- Response ID：${audit.responseId || "供应商未返回"}\n- 阶段记录数：${audit.callCount || 1}\n- 本地缓存命中：${audit.cacheHits || 0}\n- 输入 tokens：${audit.usage?.prompt_tokens ?? "—"}\n- 输出 tokens：${audit.usage?.completion_tokens ?? "—"}\n- 总 tokens：${audit.usage?.total_tokens ?? "—"}\n- 服务端耗时：${Number.isFinite(Number(meta.elapsedSeconds)) ? `${Number(meta.elapsedSeconds).toFixed(2)} 秒` : "—"}\n- 质量修复次数：${audit.validationRetries || 0}\n` : "";
  return `# ${paper.title}\n\n> ${paper.oneSentence}\n\n- 作者：${paper.authors}\n- 年份：${paper.year}\n- 场所：${paper.venue}\n- 模式：${meta.mode === "demo" ? "演示数据" : "实时研究"}\n- 材料范围：${meta.materialScope || "公开摘要与元数据"}\n- 材料质量：${meta.materialQuality?.label || "未单独评估"}\n\n## 研究问题\n\n${analysis.problem}\n\n## 核心贡献\n\n${analysis.contributions.map((item, i) => `${i + 1}. ${item}`).join("\n")}\n\n## 方法流程\n\n${analysis.methodSteps.map((item, i) => `${i + 1}. **${item.label}**：${item.text}`).join("\n")}\n${deepSection}\n## 相关工作\n\n${relatedPapers.map(item => `- **${item.shortTitle}**（${item.year}，${item.venue}）：${item.summary}${item.difference ? `；与种子论文的差异：${item.difference}` : ""}`).join("\n")}\n\n## 综合判断\n\n${synthesis.thesis}\n\n## 研究空白\n\n${synthesis.gaps.map(item => `- ${item}`).join("\n")}\n${auditSection}`;
}

function download(name, content, type) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([content], { type }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 500);
}

function exportJson() {
  if (!state.report) return;
  download(`PaperScope-${state.report.paper.shortTitle}.json`, JSON.stringify(state.report, null, 2), "application/json");
  showToast("JSON 报告已导出。")
}

function exportMarkdown() {
  if (!state.report) return;
  download(`PaperScope-${state.report.paper.shortTitle}.md`, reportMarkdown(), "text/markdown");
  showToast("Markdown 报告已导出。")
}

let toastTimer;
function showToast(message, duration = 2400) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), duration);
}

bootstrap();

