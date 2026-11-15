"""The single-page query console (plain HTML/CSS/JS, no build step)."""

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini SQL 查询台</title>
<style>
:root{
  --bg:#0f1420; --panel:#171e2e; --panel2:#1f2940; --line:#2a3650;
  --txt:#e6ebf5; --muted:#8b98b3; --accent:#4f8cff; --ok:#36c98f;
  --warn:#f0a93b; --err:#f0616d; --mono:"Cascadia Code",Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
  font:14px/1.55 "Segoe UI",system-ui,sans-serif}
header{padding:14px 22px;background:linear-gradient(90deg,#16203a,#101728);
  border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
header h1{font-size:17px;margin:0;font-weight:600}
header .sub{color:var(--muted);font-size:12px}
.wrap{display:grid;grid-template-columns:330px 1fr;gap:16px;padding:16px 22px;
  max-width:1500px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:16px;margin-bottom:16px}
.card h2{font-size:13px;margin:0 0 12px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted)}
label{display:block;font-size:12px;color:var(--muted);margin:8px 0 4px}
input,textarea,select{width:100%;background:var(--panel2);border:1px solid var(--line);
  color:var(--txt);border-radius:8px;padding:9px 10px;font-family:var(--mono);
  font-size:13px;outline:none}
input:focus,textarea:focus{border-color:var(--accent)}
textarea{resize:vertical;min-height:150px;line-height:1.5}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;
  padding:9px 16px;font-size:13px;font-weight:600;cursor:pointer;margin-top:10px}
button.secondary{background:var(--panel2);border:1px solid var(--line)}
button.danger{background:var(--err)}
button:disabled{opacity:.45;cursor:not-allowed}
.row{display:flex;gap:8px;align-items:center}
.row>*{flex:1}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);
  font-family:var(--mono);white-space:nowrap}
th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--panel)}
.src{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
  padding:10px;margin-bottom:8px;font-size:12.5px}
.src .nm{font-weight:600;font-family:var(--mono)}
.src .meta{color:var(--muted);font-size:11.5px;margin-top:3px;
  word-break:break-all}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;
  font-weight:600}
.pill.run{background:#1d3a2c;color:var(--ok)}
.pill.err{background:#3d1f24;color:var(--err)}
.pill.wait{background:#3a3019;color:var(--warn)}
.pill.cancel{background:#33253d;color:#c79bff}
pre.plan{background:#0c1120;border:1px solid var(--line);border-radius:8px;
  padding:12px;overflow:auto;font-family:var(--mono);font-size:12.5px;
  color:#b9c7e4;margin:0;max-height:340px}
.opnode{border-left:3px solid var(--accent);background:var(--panel2);
  border-radius:0 8px 8px 0;padding:9px 12px;margin:7px 0;font-size:12.5px}
.opnode.spill{border-left-color:var(--warn)}
.opnode .head{font-weight:600;font-family:var(--mono)}
.opnode .nums{color:var(--muted);font-family:var(--mono);font-size:11.5px;margin-top:3px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:12.5px;
  font-family:var(--mono)}
.kv .k{color:var(--muted)}
.scroll{max-height:380px;overflow:auto;border:1px solid var(--line);border-radius:8px}
.hint{color:var(--muted);font-size:12px;margin-top:6px}
.bar{height:6px;background:var(--panel2);border-radius:99px;overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;background:var(--accent)}
.bar.warn>i{background:var(--warn)}
.null{color:var(--muted);font-style:italic}
.tag{font-size:10.5px;padding:1px 6px;border-radius:5px;background:var(--panel2);
  border:1px solid var(--line);color:var(--muted);margin-left:6px}
#toast{position:fixed;right:20px;bottom:20px;background:var(--panel2);
  border:1px solid var(--line);padding:12px 16px;border-radius:10px;display:none;
  max-width:380px;font-size:13px}
#toast.show{display:block}
#toast.err{border-color:var(--err)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:1000px){.wrap{grid-template-columns:1fr}.grid2{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>⚡ Mini SQL 查询台</h1>
  <span class="sub">单机列存批向量化引擎 · numpy + 标准库 · 无数据库/数据框依赖</span>
</header>

<div class="wrap">
  <!-- LEFT COLUMN -->
  <div>
    <div class="card">
      <h2>① 数据源挂载</h2>
      <label>表名</label>
      <input id="m_name" placeholder="orders">
      <label>CSV 路径</label>
      <input id="m_path" placeholder="D:\data\orders.csv">
      <div class="row" style="margin-top:8px">
        <div>
          <label>分隔符</label>
          <input id="m_delim" value="|" maxlength="1">
        </div>
        <div>
          <label>编码</label>
          <input id="m_enc" value="utf-8">
        </div>
      </div>
      <div class="row">
        <button onclick="mountSource()">挂载并猜类型</button>
        <button class="secondary" onclick="loadSources()">刷新</button>
      </div>
      <div id="sources" style="margin-top:12px"></div>
    </div>

    <div class="card">
      <h2>引擎参数</h2>
      <div class="kv">
        <span class="k">内存上限</span><span id="p_budget">-</span>
        <span class="k">解析并行</span><span id="p_parallel">-</span>
        <span class="k">溢写阈值</span><span id="p_threshold">-</span>
      </div>
      <div class="hint">排序 / 哈希聚合超阈值自动溢写到临时目录，查询结束即清理。</div>
    </div>
  </div>

  <!-- RIGHT COLUMN -->
  <div>
    <div class="card">
      <h2>② SQL 编辑</h2>
      <textarea id="sql" spellcheck="false">SELECT 1</textarea>
      <div class="row" style="margin-top:10px">
        <div style="flex:0 0 130px">
          <label>返回行数上限</label>
          <input id="maxrows" type="number" value="1000" min="1">
        </div>
        <div style="flex:1"></div>
        <button class="secondary" style="flex:0 0 auto" onclick="explain()">查看执行计划</button>
        <button style="flex:0 0 auto" id="btn_run" onclick="runQuery()">▶ 执行</button>
        <button class="danger" style="flex:0 0 auto" id="btn_cancel"
                onclick="cancelQuery()" disabled>■ 取消</button>
      </div>
      <div id="status_pill" style="margin-top:10px"></div>
    </div>

    <div class="card">
      <h2>③ 执行计划（含下推与每算子进出行数）</h2>
      <pre class="plan" id="plan_tree">尚未执行。</pre>
      <div id="op_tree" style="margin-top:12px"></div>
    </div>

    <div class="grid2">
      <div class="card">
        <h2>④ 算子耗时 / 内存 / 溢写</h2>
        <div id="timing"></div>
      </div>
      <div class="card">
        <h2>解析统计</h2>
        <div id="parse_stats"></div>
      </div>
    </div>

    <div class="card">
      <h2>⑤ 查询结果 <span id="result_meta" class="tag"></span>
        <button class="secondary" style="float:right;margin:0;padding:6px 12px"
                onclick="downloadResult()">下载 CSV</button>
      </h2>
      <div class="scroll">
        <table id="result_table"><thead></thead><tbody></tbody></table>
      </div>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
let currentJob = null, pollTimer = null;

function toast(msg, isErr){
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'show' + (isErr ? ' err' : '');
  setTimeout(()=>t.className='', 4000);
}
async function api(path, opts){
  const r = await fetch(path, opts);
  const data = await r.json();
  if(!r.ok || data.error) throw new Error(data.error || ('HTTP '+r.status));
  return data;
}
function fmtBytes(n){
  if(n==null) return '-';
  const u=['B','KB','MB','GB','TB']; let i=0; let v=n;
  while(v>=1024 && i<u.length-1){v/=1024;i++;}
  return v.toFixed(v<10?2:1)+' '+u[i];
}
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

async function loadSources(){
  const data = await api('/api/sources');
  const box = document.getElementById('sources');
  const names = Object.keys(data.sources);
  if(!names.length){box.innerHTML='<div class="hint">尚未挂载任何数据源。</div>';return;}
  box.innerHTML = names.map(n=>{
    const s=data.sources[n];
    const cols = s.schema.map(c=>`${esc(c.name)}<span class="tag">${c.type}</span>`).join(' ');
    return `<div class="src">
      <div class="nm">${esc(n)}</div>
      <div class="meta">${esc(s.path)}</div>
      <div class="meta">列: ${cols}</div>
      <div class="meta">扫描 ${s.rows_seen} 行 · 有效 ${s.rows_ok} · 脏行跳过 <b style="color:var(--warn)">${s.rows_dirty}</b>
        <button class="secondary" style="float:right;margin:0;padding:3px 9px"
          onclick="unmount('${esc(n)}')">卸载</button></div>
    </div>`;
  }).join('');
}
async function mountSource(){
  const body = {
    name: document.getElementById('m_name').value.trim(),
    path: document.getElementById('m_path').value.trim(),
    delimiter: document.getElementById('m_delim').value || '|',
    encoding: document.getElementById('m_enc').value.trim() || 'utf-8',
  };
  if(!body.name || !body.path){toast('表名和路径必填', true);return;}
  try{
    const d = await api('/api/mount',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    toast('挂载成功: '+body.name); loadSources();
  }catch(e){toast(e.message, true);}
}
async function unmount(name){
  await api('/api/unmount',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})});
  loadSources();
}
async function explain(){
  try{
    const d = await api('/api/explain',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sql:document.getElementById('sql').value})});
    document.getElementById('plan_tree').textContent = d.plan;
  }catch(e){toast(e.message, true);}
}
async function runQuery(){
  const sql = document.getElementById('sql').value;
  const maxRows = parseInt(document.getElementById('maxrows').value||'1000',10);
  document.getElementById('btn_run').disabled = true;
  document.getElementById('btn_cancel').disabled = false;
  setPill('running');
  try{
    const d = await api('/api/query',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sql, max_rows:maxRows})});
    currentJob = d.job_id;
    clearInterval(pollTimer);
    pollTimer = setInterval(pollJob, 400);
    pollJob();
  }catch(e){toast(e.message,true); resetButtons();}
}
async function cancelQuery(){
  if(!currentJob) return;
  await api('/api/cancel',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({job_id:currentJob})});
  toast('已请求取消…');
}
function resetButtons(){
  document.getElementById('btn_run').disabled=false;
  document.getElementById('btn_cancel').disabled=true;
  clearInterval(pollTimer);
}
function setPill(status, extra){
  const map={running:['wait','运行中…'],done:['run','完成'],error:['err','出错'],
    cancelled:['cancel','已取消'],queued:['wait','排队中']};
  const [cls,txt]=map[status]||['wait',status];
  document.getElementById('status_pill').innerHTML =
    `<span class="pill ${cls}">${txt}</span> ${extra||''}`;
}
async function pollJob(){
  if(!currentJob) return;
  let d;
  try{ d = await api('/api/job/'+currentJob); }catch(e){return;}
  if(d.status==='running'||d.status==='queued'){
    setPill(d.status, d.elapsed_ms?`${d.elapsed_ms} ms`:'');
    return;
  }
  resetButtons();
  if(d.status==='error'){ setPill('error'); toast(d.error,true); return; }
  if(d.status==='cancelled'){ setPill('cancelled');
    document.getElementById('plan_tree').textContent='查询已取消，临时文件已清理。';
    return; }
  setPill('done', `${d.elapsed_ms} ms · ${d.row_count} 行`);
  renderResult(d); renderPlan(d); renderTiming(d); renderParse(d);
}
function renderResult(d){
  document.getElementById('result_meta').textContent =
    `${d.row_count} 行 × ${d.columns.length} 列`;
  const th = d.columns.map((c,i)=>`<th>${esc(c)}<span class="tag">${d.types[i]}</span></th>`).join('');
  document.querySelector('#result_table thead').innerHTML = '<tr>'+th+'</tr>';
  document.querySelector('#result_table tbody').innerHTML = d.rows.map(r=>
    '<tr>'+r.map(v=>v===null?'<td class="null">NULL</td>':`<td>${esc(v)}</td>`).join('')+'</tr>'
  ).join('');
}
function renderPlan(d){
  document.getElementById('plan_tree').textContent = d.plan_tree;
  const root = d.stats.plan;
  const walk = (n, depth)=>`
    <div class="opnode ${n.spilled?'spill':''}" style="margin-left:${depth*18}px">
      <div class="head">${esc(n.node)} ${n.spilled?'<span class="tag" style="color:var(--warn)">溢写</span>':''}</div>
      <div class="meta" style="font-size:11.5px;color:var(--muted);margin-top:2px">${esc(n.detail)}</div>
      <div class="nums">进 ${n.rows_in} 行 / 出 ${n.rows_out} 行 ·
        批次 ${n.batches_in}→${n.batches_out} · ${n.elapsed_ms} ms
        ${n.spilled?`· 溢写段 ${n.spill_runs} / ${fmtBytes(n.spill_bytes)}`:''}</div>
    </div>
    ${(n.children||[]).map(c=>walk(c,depth+1)).join('')}`;
  document.getElementById('op_tree').innerHTML = walk(root,0);
}
function renderTiming(d){
  const m = d.stats.memory;
  const pct = Math.min(100, m.peak_bytes/m.budget_bytes*100);
  const flat=[];
  const collect=n=>{flat.push(n); (n.children||[]).forEach(collect);};
  collect(d.stats.plan);
  const maxMs = Math.max(...flat.map(n=>n.elapsed_ms),1);
  const rows = flat.map(n=>`
    <div style="margin:7px 0">
      <div class="row" style="gap:10px">
        <span style="flex:0 0 150px;font-family:var(--mono);font-size:12px">${esc(n.node)}</span>
        <div class="bar ${n.elapsed_ms>maxMs*0.5?'warn':''}" style="flex:1">
          <i style="width:${Math.max(2,n.elapsed_ms/maxMs*100)}%"></i></div>
        <span style="flex:0 0 72px;text-align:right;font-family:var(--mono);font-size:12px">${n.elapsed_ms}ms</span>
      </div>
    </div>`).join('');
  document.getElementById('timing').innerHTML = `
    <div class="kv">
      <span class="k">总耗时</span><span>${d.stats.elapsed_ms} ms</span>
      <span class="k">内存峰值</span><span>${fmtBytes(m.peak_bytes)} / ${fmtBytes(m.budget_bytes)}</span>
      <span class="k">溢写阈值</span><span>${fmtBytes(m.spill_threshold_bytes)}</span>
      <span class="k">溢写写盘</span><span>${fmtBytes(d.stats.spill_bytes_written)}</span>
      <span class="k">溢写读盘</span><span>${fmtBytes(d.stats.spill_bytes_read)}</span>
    </div>
    <div class="bar ${pct>75?'warn':''}" style="margin:10px 0"><i style="width:${pct}%"></i></div>
    ${rows}`;
}
function renderParse(d){
  const s=d.stats.sources||{};
  document.getElementById('parse_stats').innerHTML = Object.keys(s).length?
    '<table><thead><tr><th>表</th><th>扫描</th><th>有效</th><th>脏行</th></tr></thead><tbody>'+
    Object.entries(s).map(([n,v])=>`<tr><td>${esc(n)}</td><td>${v.rows_seen}</td>
      <td>${v.rows_ok}</td><td style="color:var(--warn)">${v.rows_dirty}</td></tr>`).join('')+
    '</tbody></table>' : '<div class="hint">无</div>';
}
function downloadResult(){
  if(!currentJob){toast('没有可下载的结果',true);return;}
  window.open('/api/job/'+currentJob+'/download','_blank');
}
loadSources();
fetch('/api/sources').then(()=>{
  document.getElementById('p_budget').textContent='见任务内存面板（默认 2GB）';
  document.getElementById('p_parallel').textContent='启动时配置';
  document.getElementById('p_threshold').textContent='预算的 75%';
});
</script>
</body>
</html>
"""
