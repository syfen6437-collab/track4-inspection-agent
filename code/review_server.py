from __future__ import annotations

"""Local, dependency-free browser for reviewing training labels.

This tool intentionally writes only to ``qa/review_annotations.jsonl``. It
never edits the training manifest or the competition result file.
"""

import argparse
import json
import mimetypes
import threading
import webbrowser
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TRAIN_MANIFEST = ROOT / "data" / "processed" / "train_manifest.jsonl"
LEXICON = ROOT / "data" / "processed" / "label_lexicon.json"
CALIBRATION_FILES = (
    (ROOT / "logs" / "qwen3_vl_4b_calibration.json", "Qwen3-VL-4B"),
    (ROOT / "logs" / "calibration_records.json", "Qwen3-VL-2B"),
)
QA_DIR = ROOT / "qa"
QA_FILE = QA_DIR / "review_annotations.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_state() -> dict[str, dict]:
    state: dict[str, dict] = {}
    for row in read_jsonl(QA_FILE):
        if row.get("id"):
            state[str(row["id"])] = row
    return state


def save_state(state: dict[str, dict]) -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    with QA_FILE.open("w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(state, key=str.lower):
            handle.write(json.dumps(state[key], ensure_ascii=False, separators=(",", ":")) + "\n")


def load_items() -> list[dict]:
    if not TRAIN_MANIFEST.exists():
        raise FileNotFoundError(f"缺少训练清单: {TRAIN_MANIFEST}")
    calibration = {}
    calibration_model = ""
    for calibration_path, model_name in CALIBRATION_FILES:
        if not calibration_path.exists():
            continue
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        records = payload.get("records", []) if isinstance(payload, dict) else payload
        for record in records:
            truth = record.get("truth", {})
            if truth.get("image"):
                prediction = record.get("prediction", {})
                calibration[str(truth["image"])] = {
                    "prediction": prediction.get("defectType", ""),
                    "confidence": prediction.get("confidence", 0.0),
                    "exact_match": truth.get("defectType", "") == prediction.get("defectType", ""),
                    "model": model_name,
                }
                calibration_model = model_name
        if calibration:
            break
    state = load_state()
    items = []
    with TRAIN_MANIFEST.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row["image"])
            model = calibration.get(item_id, {})
            review = state.get(item_id, {})
            item = {
                "index": index,
                "id": item_id,
                "image": item_id,
                "questionCategory": row.get("questionCategory", ""),
                "bridgeName": row.get("bridgeName", ""),
                "filename": row.get("filename", ""),
                "defectType": row.get("defectType", ""),
                "defectDescription": row.get("defectDescription", ""),
                "ratingScale": row.get("ratingScale(1-5)", ""),
                "modelPrediction": model.get("prediction", ""),
                "confidence": model.get("confidence", ""),
                "exactMatch": model.get("exact_match", ""),
                "modelName": model.get("model", calibration_model),
                "reviewStatus": review.get("status", "unreviewed"),
                "reviewedLabel": review.get("reviewedLabel", ""),
                "reviewNote": review.get("note", ""),
                "reviewedAt": review.get("reviewedAt", ""),
            }
            item["hard"] = bool(model and (not model.get("exact_match") or float(model.get("confidence", 0) or 0) < 0.65))
            items.append(item)
    return items


def safe_image_path(relative: str) -> Path:
    candidate = (RAW_DIR / unquote(relative)).resolve()
    if RAW_DIR.resolve() not in candidate.parents or not candidate.is_file():
        raise FileNotFoundError(relative)
    return candidate


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>赛道四训练标签审核</title>
<style>
:root{color-scheme:light;--bg:#f4f6f8;--panel:#fff;--line:#d9dee5;--text:#18212b;--muted:#667383;--accent:#1565c0;--ok:#147d4d;--warn:#a45b00}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif}
header{height:58px;background:#18212b;color:#fff;display:flex;align-items:center;padding:0 20px;gap:18px}header h1{font-size:17px;margin:0;font-weight:650}header span{color:#b9c5d1;font-size:12px}
.layout{display:grid;grid-template-columns:310px minmax(460px,1fr) 350px;gap:12px;max-width:1600px;margin:12px auto;padding:0 12px;min-height:calc(100vh - 82px)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden}.panel h2{font-size:14px;margin:0;padding:13px 14px;border-bottom:1px solid var(--line)}
.filters{padding:12px}.filters label{display:block;font-size:12px;color:var(--muted);margin:9px 0 4px}.filters input,.filters select,.filters textarea{width:100%;border:1px solid #c6ced8;border-radius:4px;padding:8px;background:#fff;color:var(--text)}.filters textarea{min-height:86px;resize:vertical}
.checks{display:flex;gap:10px;flex-wrap:wrap;margin-top:9px}.checks label{margin:0;color:var(--text);display:flex;align-items:center;gap:5px}.checks input{width:auto}
button{border:1px solid #bbc5d0;background:#fff;border-radius:4px;padding:8px 11px;cursor:pointer;color:var(--text)}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:hover{filter:brightness(.97)}.actions{display:flex;gap:8px;margin-top:12px}
.list{height:calc(100vh - 142px);overflow:auto}.row{padding:10px 12px;border-bottom:1px solid #edf0f3;cursor:pointer}.row:hover,.row.active{background:#edf4fc}.row strong{font-size:13px;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.row small{display:block;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:3px}.badge{display:inline-block;border-radius:3px;padding:1px 5px;font-size:11px;margin-left:5px;background:#e9edf2;color:#4d5a68}.badge.hard{background:#fff1dc;color:var(--warn)}.badge.ok{background:#e2f3eb;color:var(--ok)}
.viewer{display:flex;flex-direction:column;min-height:0}.image-wrap{background:#111;display:flex;justify-content:center;align-items:center;min-height:420px;height:calc(100vh - 310px);overflow:hidden}.image-wrap img{max-width:100%;max-height:100%;object-fit:contain}.empty{color:#aab4bf}.nav{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border-bottom:1px solid var(--line)}.nav strong{font-size:12px;color:var(--muted)}
.meta{padding:13px 14px;overflow:auto}.meta dl{display:grid;grid-template-columns:92px 1fr;margin:0;gap:7px 8px}.meta dt{color:var(--muted)}.meta dd{margin:0;word-break:break-word}.meta .label{font-weight:650;color:#0d4c8e}.meta .pred{color:#7b4c00}.meta .match{color:var(--ok)}.meta .mismatch{color:#b12828}.savebar{position:sticky;bottom:0;background:#fff;border-top:1px solid var(--line);padding:11px 14px}.hint{font-size:12px;color:var(--muted);margin-top:10px}
@media(max-width:1050px){.layout{grid-template-columns:260px 1fr}.right{grid-column:1/-1}.image-wrap{height:55vh}}@media(max-width:700px){.layout{display:block}.panel{margin-bottom:12px}.list{height:260px}.image-wrap{height:50vh}}
</style></head><body>
<header><h1>赛道四 · 训练标签审核</h1><span>仅写入 qa/review_annotations.jsonl，不修改训练清单和测试结果</span></header>
<main class="layout"><section class="panel"><h2>筛选 <span id="count"></span></h2><div class="filters">
<label>搜索文件名 / 桥梁名</label><input id="q" placeholder="例如 DJI_ 或支座">
<label>问题类型</label><select id="category"><option value="">全部</option><option>桥梁</option><option>轨道</option></select>
<label>训练标签</label><select id="label"><option value="">全部标签</option></select>
<div class="checks"><label><input id="hard" type="checkbox">难例/低置信度</label><label><input id="unreviewed" type="checkbox">未确认</label></div>
<div class="actions"><button class="primary" id="apply">应用筛选</button><button id="reset">重置</button></div>
<p class="hint">难例来自已有校准记录：模型预测错误或置信度低于 0.65。页面优先使用 4B 校准记录；没有校准记录的样本仍可浏览。</p></div><div class="list" id="list"></div></section>
<section class="panel viewer"><div class="nav"><button id="prev">上一张</button><strong id="position">未选择</strong><button id="next">下一张</button></div><div class="image-wrap" id="imageWrap"><span class="empty">从左侧选择图片</span></div></section>
<section class="panel right"><h2>标签与人工确认</h2><div class="meta" id="meta"><span class="empty">从左侧选择图片</span></div><div class="savebar"><label>审核状态</label><select id="status"><option value="unreviewed">未确认</option><option value="confirmed">已确认</option><option value="needs_review">需复核</option></select><label style="display:block;margin-top:9px">确认标签（可留空，原始标签不会改变）</label><input id="reviewedLabel" placeholder="仅在你明确认为原标签不对时填写"><label style="display:block;margin-top:9px">备注</label><textarea id="note" placeholder="记录证据、疑点或建议"></textarea><div class="actions"><button class="primary" id="save">保存审核记录</button></div><p class="hint">人工记录用于后续优化提示词/词典，不会自动写入 result.json。</p></div></section></main>
<script>
let all=[], filtered=[], current=-1;
const $=id=>document.getElementById(id);
async function load(){const r=await fetch('/api/items');all=await r.json();const labels=[...new Set(all.map(x=>x.defectType))].sort();$('label').innerHTML='<option value="">全部标签</option>'+labels.map(x=>`<option>${esc(x)}</option>`).join('');apply();}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function apply(){const q=$('q').value.trim().toLowerCase(),cat=$('category').value,lab=$('label').value,hard=$('hard').checked,un=$('unreviewed').checked;filtered=all.filter(x=>(!q||`${x.filename} ${x.bridgeName} ${x.id}`.toLowerCase().includes(q))&&(!cat||x.questionCategory===cat)&&(!lab||x.defectType===lab)&&(!hard||x.hard)&&(!un||x.reviewStatus==='unreviewed'));filtered.sort((a,b)=>(b.hard-a.hard)||a.id.localeCompare(b.id));$('count').textContent=`${filtered.length}/${all.length}`;renderList();if(current<0&&filtered.length)select(0);else if(current>=filtered.length)select(filtered.length-1);}
function renderList(){$('list').innerHTML=filtered.map((x,i)=>`<div class="row ${i===current?'active':''}" data-i="${i}"><strong>${esc(x.filename)} ${x.hard?'<span class="badge hard">难例</span>':''}${x.reviewStatus==='confirmed'?'<span class="badge ok">已确认</span>':''}</strong><small>${esc(x.questionCategory)} · ${esc(x.bridgeName||'轨道')} · ${esc(x.defectType)}</small></div>`).join('')||'<div class="meta">没有匹配项</div>';document.querySelectorAll('.row').forEach(e=>e.onclick=()=>select(Number(e.dataset.i)));}
function select(i){if(i<0||i>=filtered.length){current=-1;$('position').textContent='未选择';$('imageWrap').innerHTML='<span class="empty">从左侧选择图片</span>';return}current=i;const x=filtered[i];renderList();$('position').textContent=`${i+1}/${filtered.length}`;$('imageWrap').innerHTML=`<img src="/api/image?path=${encodeURIComponent(x.image)}" alt="${esc(x.filename)}">`;$('meta').innerHTML=`<dl><dt>文件</dt><dd>${esc(x.filename)}</dd><dt>类别</dt><dd>${esc(x.questionCategory)}</dd><dt>桥梁</dt><dd>${esc(x.bridgeName||'轨道')}</dd><dt>原始标签</dt><dd class="label">${esc(x.defectType)}</dd><dt>原始描述</dt><dd>${esc(x.defectDescription)}</dd><dt>原始评分</dt><dd>${esc(x.ratingScale)||'（空）'}</dd><dt>模型预测</dt><dd class="${x.exactMatch===true?'match':x.exactMatch===false?'mismatch':'pred'}">${esc(x.modelPrediction)||'暂无校准记录'}${x.modelName?' · '+esc(x.modelName):''}${x.confidence!==''?' · 置信度 '+Number(x.confidence).toFixed(3):''}</dd><dt>审核状态</dt><dd>${esc(x.reviewStatus)}</dd><dt>审核标签</dt><dd>${esc(x.reviewedLabel)||'（未填写）'}</dd><dt>审核备注</dt><dd>${esc(x.reviewNote)||'（未填写）'}</dd></dl>`;$('status').value=x.reviewStatus;$('reviewedLabel').value=x.reviewedLabel;$('note').value=x.reviewNote;}
async function save(){if(current<0)return;const x=filtered[current],payload={id:x.id,status:$('status').value,reviewedLabel:$('reviewedLabel').value.trim(),note:$('note').value.trim()};const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!r.ok){alert(await r.text());return}const updated=await r.json();const idx=all.findIndex(y=>y.id===x.id);if(idx>=0)all[idx]={...all[idx],...updated};apply();const newIndex=filtered.findIndex(y=>y.id===x.id);if(newIndex>=0)select(newIndex);}
$('apply').onclick=apply;$('reset').onclick=()=>{$('q').value='';$('category').value='';$('label').value='';$('hard').checked=false;$('unreviewed').checked=false;apply()};$('prev').onclick=()=>select(current-1);$('next').onclick=()=>select(current+1);$('save').onclick=save;document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')select(current-1);if(e.key==='ArrowRight')select(current+1);});load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "Track4Review/1.0"

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/items":
            try:
                body = json.dumps(load_items(), ensure_ascii=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
            except Exception as exc:
                self._send(str(exc).encode("utf-8"), "text/plain; charset=utf-8", HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/image":
            values = parse_qs(parsed.query).get("path", [])
            try:
                path = safe_image_path(values[0])
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self._send(path.read_bytes(), mime)
            except Exception:
                self._send(b"image not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
            return
        self._send(b"not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/review":
            self._send(b"not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            item_id = str(payload.get("id", ""))
            valid_ids = {item["id"] for item in load_items()}
            if item_id not in valid_ids:
                raise ValueError("unknown training item")
            status = str(payload.get("status", "unreviewed"))
            if status not in {"unreviewed", "confirmed", "needs_review"}:
                raise ValueError("invalid status")
            state = load_state()
            state[item_id] = {
                "id": item_id,
                "status": status,
                "reviewedLabel": str(payload.get("reviewedLabel", "")).strip()[:120],
                "note": str(payload.get("note", "")).strip()[:1000],
                "reviewedAt": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            }
            save_state(state)
            self._send(json.dumps(state[item_id], ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        except Exception as exc:
            self._send(str(exc).encode("utf-8"), "text/plain; charset=utf-8", HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="赛道四训练标签本地浏览与审核工具")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    load_items()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[review] {url}", flush=True)
    print(f"[review] QA records: {QA_FILE}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
