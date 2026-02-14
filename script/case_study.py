# python ./script/case_study.py --data output/memory.json --host 0.0.0.0 --port 8080 --title "Case Study Viewer"


import argparse
import json
import os
from typing import Any, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from mimetypes import guess_type
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Case Study Viewer Service")


class State:
    data_path: str = ""
    data_files: List[str] = []
    title: str = "Case Study Viewer"
    cases: List[List[dict]] = []
    static_roots: List[str] = []


STATE = State()


def load_cases(path: str) -> List[List[dict]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    def is_step(d: Any) -> bool:
        return isinstance(d, dict) and ("step" in d) and ("memory" in d)

    def is_case(obj: Any) -> bool:
        return isinstance(obj, list) and all(is_step(x) for x in obj)

    def normalize_case(obj: Any) -> List[dict] | None:
        if is_case(obj):
            return obj
        if isinstance(obj, dict):
            for k in ("steps", "case"):
                if k in obj and is_case(obj[k]):
                    return obj[k]
        return None

    def unwrap_container(obj: Any) -> Any:
        if isinstance(obj, dict):
            for k in (
                "cases",
                "data",
                "items",
                "dataset",
                "results",
                "records",
                "list",
            ):
                if k in obj:
                    return obj[k]
        return obj

    txt = open(path, "r", encoding="utf-8").read().strip()
    try:
        obj = json.loads(txt)
        obj = unwrap_container(obj)

        c = normalize_case(obj)
        if c is not None:
            return [c]

        if isinstance(obj, list):
            out: List[List[dict]] = []
            for elem in obj:
                elem = unwrap_container(elem)
                c = normalize_case(elem)
                if c is None:
                    raise ValueError(
                        "Dataset element is not a valid case; expected a list of {step,memory} or an object with 'steps'."
                    )
                out.append(c)
            if out:
                return out
    except Exception:
        pass

    out: List[List[dict]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Line {i} invalid JSON: {e}")
            obj = unwrap_container(obj)
            c = normalize_case(obj)
            if c is None:
                raise ValueError(
                    f"Line {i} is not a valid case (expect steps list or object with 'steps')."
                )
            out.append(c)
    if not out:
        raise ValueError("No valid cases found.")
    return out


def _estimate_case_count_from_steps(steps: List[dict]) -> int:
    max_len = 1
    for st in steps:
        mem = st.get("memory", {}) if isinstance(st, dict) else {}
        if isinstance(mem, dict):
            for v in mem.values():
                if isinstance(v, list):
                    max_len = max(max_len, len(v))
    return max_len


def _slice_case_by_index(steps: List[dict], idx: int) -> List[dict]:
    out_steps: List[dict] = []
    for st in steps:
        step_name = st.get("step")
        mem = st.get("memory", {})
        new_mem = {}
        if isinstance(mem, dict):
            for k, v in mem.items():
                if isinstance(v, list):
                    new_mem[k] = v[idx] if 0 <= idx < len(v) else None
                else:
                    new_mem[k] = v
        out_steps.append({"step": step_name, "memory": new_mem})
    return out_steps


def _expand_cases_if_needed(cases: List[List[dict]]) -> List[List[dict]]:
    expanded: List[List[dict]] = []
    for steps in cases:
        n = _estimate_case_count_from_steps(steps)
        if n <= 1:
            expanded.append(steps)
        else:
            for i in range(n):
                expanded.append(_slice_case_by_index(steps, i))
    return expanded


# Helper: collect image directories
def _collect_image_dirs(cases: List[List[dict]]) -> List[str]:
    """Scan cases to find directories that contain image paths so we can auto-serve them (supports nested arrays)."""
    exts = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
    dirs: set[str] = set()

    def add_if_image_path(v: Any):
        if isinstance(v, str) and v.lower().endswith(exts):
            d = os.path.dirname(v)
            if d:
                dirs.add(d)

    def walk(x: Any):
        if x is None:
            return
        if isinstance(x, list):
            for it in x:
                walk(it)
        elif isinstance(x, dict):
            for vv in x.values():
                walk(vv)
        else:
            add_if_image_path(x)

    for steps in cases or []:
        for st in steps or []:
            if isinstance(st, dict):
                mem = st.get("memory", {})
                walk(mem)

    return sorted(dirs)


def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


CSS = r"""
:root {
  --bg-0: #f7f7f8; --bg-1: #ffffff; --bg-2: #efefef; --bg-3: #e5e5e5;
  --text-0: #1a1a1a; --text-1: #374151; --text-2: #6b7280;
  --border: #e3e3e3; --border-h: #d1d1d1;
  --accent: #2563eb; --accent-d: rgba(37,99,235,.08);
  --green: #059669; --green-d: rgba(5,150,105,.08);
  --r: 8px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{height:100%}
body{min-height:100%;background:var(--bg-0);color:var(--text-0);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif;
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.topbar{position:sticky;top:0;z-index:50;
  background:rgba(255,255,255,.88);backdrop-filter:saturate(180%) blur(14px);
  -webkit-backdrop-filter:saturate(180%) blur(14px);
  border-bottom:1px solid var(--border);padding:0 24px}
.topbar-row{display:flex;align-items:center;gap:10px;height:48px}
.topbar-title{font-size:15px;font-weight:700;color:var(--text-0);
  display:flex;align-items:center;gap:8px}
.topbar-icon{width:20px;height:20px;border-radius:5px;
  background:linear-gradient(135deg,#58a6ff,#3fb950)}
.topbar-badge{margin-left:auto;font-size:12px;font-weight:500;color:var(--text-2);
  background:var(--bg-2);border:1px solid var(--border);padding:2px 10px;border-radius:999px}
.topbar-sep{width:1px;height:20px;background:var(--border)}
.btn{appearance:none;border:1px solid var(--border);background:var(--bg-2);color:var(--text-1);
  padding:4px 12px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;
  transition:all .12s;white-space:nowrap;line-height:20px}
.btn:hover:not(:disabled){background:var(--bg-3);border-color:var(--border-h);color:var(--text-0)}
.btn:active:not(:disabled){background:var(--bg-1)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-group{display:inline-flex}
.btn-group .btn{border-radius:0;margin-left:-1px}
.btn-group .btn:first-child{border-radius:6px 0 0 6px;margin-left:0}
.btn-group .btn:last-child{border-radius:0 6px 6px 0}
select.file-select{background:var(--bg-2);border:1px solid var(--border);color:var(--text-0);
  padding:4px 10px;border-radius:6px;font-size:12px;font-weight:500;cursor:pointer;
  max-width:320px;line-height:20px;transition:border-color .12s}
select.file-select:hover{border-color:var(--border-h)}
select.file-select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-d)}
select.file-select option{background:#fff;color:var(--text-0)}
.container{max-width:960px;margin:0 auto;padding:20px 20px 80px}
.overview{background:var(--bg-1);border:1px solid var(--border);border-radius:var(--r);
  padding:14px 16px;margin-bottom:12px}
.overview-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.overview-label{font-size:12px;font-weight:600;color:var(--text-2);text-transform:uppercase;letter-spacing:.4px}
.overview-count{font-size:12px;color:var(--text-2)}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:4px}
.flow-chip{font-size:12px;font-weight:500;padding:2px 8px;border-radius:6px;
  background:var(--accent-d);color:var(--accent);border:1px solid rgba(56,139,253,.15)}
.flow-arr{color:var(--text-2);font-size:10px;padding:0 2px}
.step-card{background:var(--bg-1);border:1px solid var(--border);border-radius:var(--r);
  margin-bottom:8px;overflow:hidden;transition:border-color .12s}
.step-card:hover{border-color:var(--border-h)}
.step-header{display:flex;align-items:center;gap:12px;padding:12px 16px;cursor:pointer;user-select:none}
.step-num{width:28px;height:28px;display:flex;align-items:center;justify-content:center;
  border-radius:6px;flex-shrink:0;background:var(--accent-d);color:var(--accent);
  font-size:12px;font-weight:700}
.step-info{flex:1;min-width:0}
.step-title{font-size:13px;font-weight:600;color:var(--text-0);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.step-sub{font-size:11px;color:var(--text-2);margin-top:1px}
.step-chev{color:var(--text-2);font-size:10px;transition:transform .2s ease;flex-shrink:0}
.step-chev.open{transform:rotate(90deg)}
.step-body{display:grid;grid-template-rows:0fr;transition:grid-template-rows .25s ease}
.step-body.open{grid-template-rows:1fr}
.step-body-inner{overflow:hidden}
.step-body.open .step-body-inner{padding:0 16px 14px;border-top:1px solid var(--border)}
.mem-item{margin-top:12px}
.mem-item:first-child{margin-top:10px}
.mem-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.mem-key{font-size:12px;font-weight:600;color:var(--green);
  font-family:'SF Mono','Cascadia Code',Consolas,monospace}
.copy-btn{appearance:none;border:1px solid var(--border);background:transparent;color:var(--text-2);
  padding:1px 8px;border-radius:4px;font-size:11px;cursor:pointer;transition:all .12s}
.copy-btn:hover{border-color:var(--border-h);color:var(--text-0);background:var(--bg-3)}
.mem-val{margin:0;padding:10px 12px;background:var(--bg-0);border:1px solid var(--border);
  border-radius:6px;font-family:'SF Mono','Cascadia Code',Consolas,monospace;
  font-size:12px;line-height:1.6;color:var(--text-0);
  white-space:pre-wrap;word-wrap:break-word;overflow:auto;max-height:420px}
img.zoomable{cursor:zoom-in;border-radius:6px;display:block}
.img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:6px}
.img-wrap{background:var(--bg-0);border:1px solid var(--border);border-radius:6px;padding:4px;overflow:hidden}
.img-wrap img{width:100%;height:120px;object-fit:cover;border-radius:4px;display:block}
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.5);
  display:none;align-items:center;justify-content:center;padding:40px;z-index:999}
.lightbox.open{display:flex}
.lightbox img{max-width:min(92vw,1200px);max-height:90vh;width:auto;height:auto;
  border-radius:10px;box-shadow:0 24px 64px rgba(0,0,0,.25);object-fit:contain}
.lightbox-close{position:absolute;top:20px;right:20px;
  background:rgba(255,255,255,.92);border:1px solid var(--border);color:var(--text-0);
  border-radius:6px;font-size:12px;font-weight:600;padding:4px 14px;cursor:pointer}
.lightbox-close:hover{border-color:var(--accent)}
body.lightbox-open{overflow:hidden}
.footer{text-align:center;padding:20px;font-size:12px;color:var(--text-2)}
.footer a{color:var(--accent);text-decoration:none}
.footer a:hover{text-decoration:underline}
.empty{text-align:center;padding:60px 20px;color:var(--text-2)}
"""

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>{css}</style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-row">
      <div class="topbar-title"><div class="topbar-icon"></div>{title}</div>
      <select id="file-select" class="file-select" title="选择数据文件"></select>
      <div class="topbar-sep"></div>
      <div class="btn-group">
        <button id="prev" class="btn">← 上一条</button>
        <button id="next" class="btn">下一条 →</button>
      </div>
      <button id="toggle-all" class="btn">全部展开</button>
      <div class="topbar-badge" id="counter">Case 1 / 1</div>
    </div>
  </div>

  <div class="container">
    <div id="cases"></div>
  </div>

  <div id="lightbox" class="lightbox" aria-hidden="true">
    <img id="lightbox-img" alt="" />
    <button id="lightbox-close" class="lightbox-close" type="button">关闭 ✕</button>
  </div>

  <div class="footer">
    键盘 ← → 切换 Case &#183; 点击步骤展开/收起 &#183; 顶部切换数据文件 &#183; <a href="/api/reload">/api/reload</a> 热加载
  </div>

<script>
const state = { idx: 0, cases: [], files: [], currentFile: '' };
let lightboxEl = null;
let lightboxImgEl = null;
let lightboxCloseBtn = null;
let lightboxReady = false;

function $(sel, root=document){ return root.querySelector(sel); }
function $all(sel, root=document){ return Array.from(root.querySelectorAll(sel)); }

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function closeLightbox() {
  if (!lightboxEl || !lightboxImgEl) return;
  lightboxEl.classList.remove("open");
  if (document.body) document.body.classList.remove("lightbox-open");
  lightboxImgEl.src = "";
}

function setupLightbox() {
  if (lightboxReady) return;
  lightboxEl = $("#lightbox");
  lightboxImgEl = $("#lightbox-img");
  lightboxCloseBtn = $("#lightbox-close");
  if (!lightboxEl || !lightboxImgEl || !lightboxCloseBtn) return;
  lightboxReady = true;
  lightboxCloseBtn.addEventListener("click", closeLightbox);
  lightboxEl.addEventListener("click", (e) => {
    if (e.target === lightboxEl) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightboxEl.classList.contains("open")) {
      closeLightbox();
    }
  });
}

function openLightbox(src, altText="") {
  setupLightbox();
  if (!lightboxImgEl || !lightboxEl) return;
  lightboxImgEl.src = src;
  lightboxImgEl.alt = altText;
  lightboxEl.classList.add("open");
  if (document.body) document.body.classList.add("lightbox-open");
}

function attachImageZoom(img, altLabel) {
  if (!img) return;
  img.classList.add("zoomable");
  img.addEventListener("click", () => {
    openLightbox(img.dataset.fullSrc || img.src, altLabel || img.alt || "");
  });
}

async function fetchCases() {
  const res = await fetch('/api/cases');
  if (!res.ok) throw new Error('Failed to load cases');
  const data = await res.json();
  state.cases = data.cases || [];
}

async function fetchFiles() {
  try {
    const res = await fetch('/api/files');
    if (!res.ok) return;
    const data = await res.json();
    state.files = data.files || [];
    state.currentFile = data.current || '';
    const sel = $('#file-select');
    if (!sel) return;
    sel.innerHTML = '';
    state.files.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.path;
      opt.textContent = f.name;
      if (f.path === state.currentFile) opt.selected = true;
      sel.appendChild(opt);
    });
    if (state.files.length <= 1) sel.style.display = 'none';
    else sel.style.display = '';
  } catch (e) {
    console.warn('fetchFiles error:', e);
  }
}

async function switchFile(filePath) {
  const sel = $('#file-select');
  if (sel) sel.disabled = true;
  try {
    const res = await fetch('/api/switch?file=' + encodeURIComponent(filePath));
    if (!res.ok) { alert('切换文件失败'); return; }
    state.currentFile = filePath;
    await fetchCases();
    state.idx = 0;
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  } finally {
    if (sel) sel.disabled = false;
  }
}

function parseHash() {
  if (location.hash.startsWith("#case-")) {
    const n = parseInt(location.hash.replace("#case-", ""), 10);
    if (!Number.isNaN(n)) state.idx = n;
  }
}

function setHash() {
  history.replaceState(null, "", `#case-${state.idx}`);
}

function render() {
  var container = $("#cases");
  var idx = state.idx;
  var casesCount = Array.isArray(state.cases) ? state.cases.length : 0;

  if (casesCount === 0) {
    $("#counter").textContent = "Case 0 / 0";
    container.innerHTML = '<div class="empty">没有加载到任何 case</div>';
    $("#prev").disabled = true;
    $("#next").disabled = true;
    return;
  }

  if (idx < 0) idx = 0;
  if (idx > casesCount - 1) idx = casesCount - 1;
  state.idx = idx;
  $("#counter").textContent = "Case " + (idx + 1) + " / " + casesCount;

  var steps = state.cases[idx] || [];
  var frag = document.createDocumentFragment();

  /* ---- Overview card ---- */
  var ov = document.createElement("div"); ov.className = "overview";
  var names = steps.map(function(s){ return (s && s.step) ? s.step : "unknown"; });
  ov.innerHTML = '<div class="overview-head"><span class="overview-label">Pipeline</span>' +
    '<span class="overview-count">' + steps.length + ' steps</span></div>' +
    '<div class="flow">' +
    names.map(function(n){ return '<span class="flow-chip">' + escapeHtml(n) + '</span>'; })
         .join('<span class="flow-arr">\u2192</span>') + '</div>';
  frag.appendChild(ov);

  /* ---- Helpers ---- */
  function isImagePath(p) {
    if (typeof p !== "string") return false;
    var s = p.toLowerCase();
    return s.endsWith(".png")||s.endsWith(".jpg")||s.endsWith(".jpeg")||
           s.endsWith(".gif")||s.endsWith(".webp")||s.endsWith(".bmp")||s.endsWith(".svg");
  }
  function collectImagePaths(v) {
    var out = [];
    (function w(x){ if(x==null)return; if(Array.isArray(x)){x.forEach(w);}
      else if(typeof x==="object"){Object.values(x).forEach(w);}
      else if(typeof x==="string"&&isImagePath(x)){out.push(x);} })(v);
    return out;
  }
  function isImagesOnly(v) {
    var ok=true;
    (function w(x){ if(!ok)return; if(x==null)return; if(Array.isArray(x)){x.forEach(w);}
      else if(typeof x==="object"){Object.values(x).forEach(w);}
      else if(typeof x==="string"){if(!isImagePath(x))ok=false;} else{ok=false;} })(v);
    return ok;
  }

  /* ---- Step cards ---- */
  steps.forEach(function(st, i) {
    var card = document.createElement("div"); card.className = "step-card";
    var stepName = String((st && st.step) != null ? st.step : "unknown");
    var memory = (st && st.memory) != null ? st.memory : {};
    var memKeys = Object.keys(memory);
    var isOpen = i < 2;

    var hdr = document.createElement("div"); hdr.className = "step-header";
    hdr.innerHTML = '<div class="step-num">' + (i+1) + '</div>' +
      '<div class="step-info"><div class="step-title">' + escapeHtml(stepName) + '</div>' +
      '<div class="step-sub">' + memKeys.length + ' memory fields</div></div>' +
      '<div class="step-chev' + (isOpen ? ' open' : '') + '">\u25B6</div>';

    var body = document.createElement("div");
    body.className = "step-body" + (isOpen ? " open" : "");
    var inner = document.createElement("div"); inner.className = "step-body-inner";

    if (memKeys.length === 0) {
      inner.innerHTML = '<div class="mem-item" style="color:var(--text-2);font-size:12px">No memory</div>';
    } else {
      memKeys.forEach(function(k) {
        var val = memory[k];
        var item = document.createElement("div"); item.className = "mem-item";
        var mh = document.createElement("div"); mh.className = "mem-head";
        var ks = document.createElement("span"); ks.className = "mem-key"; ks.textContent = k;
        var cb = document.createElement("button"); cb.className = "copy-btn"; cb.textContent = "\u590D\u5236";
        mh.appendChild(ks); mh.appendChild(cb); item.appendChild(mh);

        var copyText = "";
        var imgs = collectImagePaths(val);
        var imagesOnly = isImagesOnly(val);

        if (typeof val === "string" && isImagePath(val)) {
          copyText = val;
          var img = document.createElement("img");
          var imgSrc = "/file?path=" + encodeURIComponent(val);
          img.src = imgSrc; img.dataset.fullSrc = imgSrc; img.alt = k;
          img.style = "max-width:100%;border-radius:6px;display:block;margin-top:4px";
          img.loading = "lazy"; attachImageZoom(img, k); item.appendChild(img);
        } else if (imagesOnly && imgs.length > 0) {
          copyText = imgs.join("\n");
          var grid = document.createElement("div"); grid.className = "img-grid";
          imgs.forEach(function(p) {
            var w = document.createElement("div"); w.className = "img-wrap";
            var img = document.createElement("img");
            var imgSrc = "/file?path=" + encodeURIComponent(p);
            img.src = imgSrc; img.dataset.fullSrc = imgSrc; img.alt = k; img.loading = "lazy";
            attachImageZoom(img, k); w.appendChild(img); grid.appendChild(w);
          });
          item.appendChild(grid);
        } else {
          var pre = document.createElement("pre"); pre.className = "mem-val";
          var pt;
          if (typeof val === "string") { pt = val; }
          else if (Array.isArray(val) && val.every(function(x){ return typeof x === "string"; })) { pt = val.join("\n"); }
          else { try { pt = JSON.stringify(val, null, 2); } catch(e) { pt = String(val); } }
          copyText = pt; pre.textContent = pt; item.appendChild(pre);
        }

        cb.addEventListener("click", function() {
          navigator.clipboard.writeText(copyText || "").then(function() {
            cb.textContent = "\u5DF2\u590D\u5236";
            setTimeout(function(){ cb.textContent = "\u590D\u5236"; }, 1200);
          });
        });
        inner.appendChild(item);
      });
    }

    body.appendChild(inner);
    hdr.addEventListener("click", function() {
      body.classList.toggle("open");
      hdr.querySelector(".step-chev").classList.toggle("open");
    });
    card.appendChild(hdr); card.appendChild(body); frag.appendChild(card);
  });

  container.innerHTML = "";
  container.appendChild(frag);
  setHash();
  $("#prev").disabled = (state.idx === 0);
  $("#next").disabled = (state.idx >= casesCount - 1);
}

function toggleAll() {
  var container = $("#cases");
  if (!container) return;
  var bodies = $all(".step-body", container);
  var chevs = $all(".step-chev", container);
  var btn = $("#toggle-all");
  var allOpen = bodies.length > 0 && bodies.every(function(c){ return c.classList.contains("open"); });
  bodies.forEach(function(c){ if (allOpen) c.classList.remove("open"); else c.classList.add("open"); });
  chevs.forEach(function(c){ if (allOpen) c.classList.remove("open"); else c.classList.add("open"); });
  if (btn) btn.textContent = allOpen ? "\u5168\u90E8\u5C55\u5F00" : "\u5168\u90E8\u6536\u8D77";
}

function goto(delta) {
  const total = Array.isArray(state.cases) ? state.cases.length : 0;
  if (total === 0) return;
  let n = state.idx + delta;
  if (n < 0) n = 0;
  if (n > total - 1) n = total - 1;
  if (n === state.idx) return;
  state.idx = n;
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function main() {
  setupLightbox();
  await fetchFiles();
  await fetchCases();
  parseHash();
  if (!Array.isArray(state.cases)) state.cases = [];
  if (state.cases.length === 0) {
    // 空状态
  } else {
    if (state.idx < 0) state.idx = 0;
    if (state.idx > state.cases.length - 1) state.idx = state.cases.length - 1;
    history.replaceState(null, "", `#case-${state.idx}`);
  }
  $("#prev").addEventListener("click", () => goto(-1));
  $("#next").addEventListener("click", () => goto(1));
  var fileSel = $("#file-select");
  if (fileSel) fileSel.addEventListener("change", (e) => switchFile(e.target.value));
  $("#toggle-all").addEventListener("click", toggleAll);
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") goto(-1);
    if (e.key === "ArrowRight") goto(1);
  });
  render();
}

window.addEventListener("DOMContentLoaded", main);
window.addEventListener("hashchange", () => { parseHash(); render(); });
</script>
</body>
</html>
""".replace(
    "{css}", CSS
)


@app.get("/", response_class=HTMLResponse)
def index():
    html = INDEX_HTML.replace("{title}", escape_html(STATE.title))
    return HTMLResponse(html)


@app.get("/api/cases")
def api_cases():
    return JSONResponse({"count": len(STATE.cases), "cases": STATE.cases})


@app.get("/api/reload")
def api_reload():
    try:
        STATE.cases = load_cases(STATE.data_path)
        # auto-detect static roots again
        auto_roots = _collect_image_dirs(STATE.cases)
        if auto_roots:
            STATE.static_roots = auto_roots
        return JSONResponse({"ok": True, "count": len(STATE.cases), "msg": "reloaded"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/files")
def api_files():
    """Return the list of available data files."""
    files_info = []
    for f in STATE.data_files:
        files_info.append({"path": f, "name": os.path.basename(f)})
    return JSONResponse({"files": files_info, "current": STATE.data_path})


@app.get("/api/switch")
def api_switch(file: str):
    """Switch to a different data file and reload cases."""
    real_file = os.path.realpath(file)
    allowed = {os.path.realpath(f) for f in STATE.data_files}
    if real_file not in allowed:
        return JSONResponse(
            {"ok": False, "error": "File not in allowed list"}, status_code=403
        )
    try:
        STATE.data_path = file
        STATE.cases = _expand_cases_if_needed(load_cases(file))
        auto_roots = _collect_image_dirs(STATE.cases)
        if auto_roots:
            STATE.static_roots = auto_roots
        return JSONResponse({"ok": True, "count": len(STATE.cases), "file": file})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# Serve local files (images) under allowed roots
@app.get("/file")
def get_file(path: str):
    # Security: only serve files under configured static roots (if provided / detected)
    real = os.path.realpath(path)
    if STATE.static_roots:
        allowed = False
        for root in STATE.static_roots:
            r = os.path.realpath(root)
            if real == r or real.startswith(r + os.sep):
                allowed = True
                break
        if not allowed:
            return PlainTextResponse("Forbidden", status_code=403)
    if not os.path.exists(real):
        return PlainTextResponse("Not Found", status_code=404)
    mime = guess_type(real)[0] or "application/octet-stream"
    return FileResponse(real, media_type=mime)


@app.get("/health")
def health():
    return PlainTextResponse("ok")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a local web service to view case study data."
    )
    p.add_argument(
        "--data",
        "-d",
        required=True,
        help="数据文件路径（.json / .jsonl / 单个 case 的 JSON）",
    )
    p.add_argument("--host", default="127.0.0.1", help="绑定地址，默认 127.0.0.1")
    p.add_argument("--port", type=int, default=8080, help="端口，默认 8080")
    p.add_argument("--title", "-t", default="Case Study Viewer", help="页面标题")
    p.add_argument("--static-root", action="append", default=[], help="本地静态资源根目录，可多次指定，用于图片/文件直出")
    return p.parse_args()


def main():
    args = parse_args()
    STATE.data_path = args.data
    STATE.data_files = [args.data]
    STATE.title = args.title
    try:
        STATE.cases = _expand_cases_if_needed(load_cases(STATE.data_path))
    except Exception as e:
        raise SystemExit(f"!!! 加载数据失败: {e}")

    # Configure static roots (manual > auto-detect)
    if args.static_root:
        # de-duplicate while preserving order
        seen = set()
        roots = []
        for r in args.static_root:
            r = os.path.realpath(r)
            if r not in seen:
                roots.append(r)
                seen.add(r)
        STATE.static_roots = roots
    else:
        STATE.static_roots = _collect_image_dirs(STATE.cases)

    if STATE.static_roots:
        print("[OK] Static roots:")
        for r in STATE.static_roots:
            print("     -", r)
    else:
        print("[Info] No static roots detected; image paths will be shown as text.")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    print(f"[OK] Loaded cases: {len(STATE.cases)} from {STATE.data_path}")
    print(f"[OK] Open: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
