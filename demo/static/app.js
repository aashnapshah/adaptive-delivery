"use strict";
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const post = (url, body) => fetch(url, {method:"POST", headers:{"Content-Type":"application/json"},
  body: JSON.stringify(body||{})}).then(r => r.json());
const getj = url => fetch(url).then(r => r.ok ? r.json() : null);
const esc = s => String(s||"").replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const TIP = t => `<span class="info" data-tip="${esc(t)}">ⓘ</span>`;   // hover-explained metric
const CLOCK = `<svg class="clk" viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v4.7l3 1.8"/></svg>`;

/* ---------- study logging: idempotent (event_id deduped server-side), retried once ---------- */
const uuid = () => (self.crypto && crypto.randomUUID) ? crypto.randomUUID()
                 : "e-" + Date.now() + "-" + Math.random().toString(16).slice(2);
async function logEvent(url, payload){
  const body = {event_id: uuid(), ...payload};
  for (let i = 0; i < 2; i++){
    try { const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
      if (r.ok) return; } catch(e){}
  }
  console.warn("study log failed:", url, body);
}

/* ---------- consent gate ---------- */
{
  const cBtn = $("#consentBtn");
  if (cBtn) cBtn.onclick = async () => {
    cBtn.disabled = true;
    await fetch("/api/consent", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
    const m = $("#consentModal"); if (m) m.remove();
  };
}

let model = null;                                   // Benchmark uses the LLM gatekeeper for every case
let curModel = null;                                // replay tabs: which recorded model's runs to show
const mparam = () => curModel ? `&model=${encodeURIComponent(curModel)}` : "";
let currentCaseId = null, currentMode = "ehr", caseLoaded = false;
const recCache = {};                 // design -> recording for the current case
let evalDesign = "maidxo", deployDesign = "maidxo";
// Three designs on a complexity axis: 1 plain call -> 1 call playing every role -> many calls talking.
const DESIGNS = [{id:"single",label:"LLM",abbr:"LLM"},
                 {id:"maidxo",label:"LLM-Roles",abbr:"LLM-Roles"},
                 {id:"debate",label:"LLM-Multi",abbr:"LLM-Multi"}];
const ABBR = {single:"LLM", maidxo:"LLM-Roles", debate:"LLM-Multi"};
let visibleDesigns = ["single", "maidxo", "debate"];   // show all three; the bottom summary compares them

/* ---------- tabs ---------- */
$$(".tabs button").forEach(b => b.onclick = () => {
  $$(".tabs button").forEach(x => x.classList.toggle("on", x === b));
  $$("section.tab").forEach(s => s.classList.toggle("on", s.dataset.tab === b.dataset.tab));
  const t = b.dataset.tab;
  if (t === "score")  { grPager.sync(currentCaseId); renderEvaluate(); }
  if (t === "deliver") renderDeploy();
  if (t === "stats")   renderStats();
  if (t === "benchmark") clPager.sync(currentCaseId);
});

/* ---------- RESULTS: one cohesive chart system ---------- */
// Design tokens — single source of truth so every chart reads as one system.
const CHART = { ink:"#1b2430", sub:"#5b6472", mute:"#9aa3af", grid:"#eceef1", axis:"#d5d9df", surface:"#ffffff",
                t_title:12, t_tick:9.5, t_axis:10, t_val:9.5 };
// Categorical palette for the 3 designs — FIXED order, CVD-safe (Okabe-Ito derived).
const DCOLOR = { single:"#6e7681", maidxo:"#e69f00", debate:"#0072b2" };
const DNAME  = { single:"LLM", maidxo:"LLM-Roles", debate:"LLM-Multi" };
const DORDER = { single:0, maidxo:1, debate:2 };
// Prettify a raw model id for display: "llama3.1:8b" -> "Llama 3.1 8B", "llama-3.3-70b" -> "Llama 3.3 70B".
function prettyModel(s){
  return String(s).replace(/[:_-]+/g, " ").replace(/([a-z])(\d)/gi, "$1 $2")
    .split(/\s+/).filter(Boolean)
    .map(p => /^\d+(\.\d+)?b$/i.test(p) ? p.toUpperCase()
            : /^\d/.test(p) ? p
            : p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}
// One structured, full-width note per section: Method / Result / Interpretation.
function rbubble(method, result, interp){
  const row = (k, v) => v ? `<div class="rc-k">${k}</div><div class="rc-v">${v}</div>` : "";
  return `<div class="rcard">${row("Method", method)}${row("Result", result)}${row("Interpretation", interp)}</div>`;
}
// Pareto-optimal = minimize x (cost/effort), maximize y (accuracy): no other point is both
// cheaper-or-equal AND more-accurate-or-equal (and strictly better in one).
function paretoFront(pts, xk){
  return pts.filter(p => !pts.some(q => q!==p && q[xk]<=p[xk] && q.accuracy>=p.accuracy && (q[xk]<p[xk] || q.accuracy>p.accuracy)));
}
function paretoPlot(pts, xk, xLabel, title, multiModel){
  const W=480,H=320, mL=52,mR=16,mT=34,mB=44;
  const maxX = Math.max(1, ...pts.map(p=>p[xk])) * 1.18;
  const X=v=>mL+(W-mL-mR)*v/maxX, Y=v=>H-mB-(H-mB-mT)*v/100;
  let s=`<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" font-family="${CHART.font}">`;
  s+=`<text x="${mL}" y="20" font-size="${CHART.t_title}" fill="${CHART.ink}" font-weight="600">${title}</text>`;
  [0,25,50,75,100].forEach(v=>{ s+=`<line x1="${mL}" y1="${Y(v)}" x2="${W-mR}" y2="${Y(v)}" stroke="${CHART.grid}"/>`+
    `<text x="${mL-8}" y="${Y(v)+3}" text-anchor="end" font-size="${CHART.t_tick}" fill="${CHART.mute}">${v}</text>`; });
  [0,maxX/2,maxX].forEach(v=>{ s+=`<text x="${X(v)}" y="${H-mB+16}" text-anchor="middle" font-size="${CHART.t_tick}" fill="${CHART.mute}">${Math.round(v)}</text>`; });
  s+=`<line x1="${mL}" y1="${H-mB}" x2="${W-mR}" y2="${H-mB}" stroke="${CHART.axis}"/>`;
  s+=`<text x="${(mL+W-mR)/2}" y="${H-6}" text-anchor="middle" font-size="${CHART.t_axis}" fill="${CHART.sub}">${xLabel}</text>`;
  s+=`<text x="14" y="${(mT+H-mB)/2}" text-anchor="middle" font-size="${CHART.t_axis}" fill="${CHART.sub}" transform="rotate(-90 14 ${(mT+H-mB)/2})">Accuracy (%)</text>`;
  const front = paretoFront(pts, xk).sort((a,b)=>a[xk]-b[xk]);
  const fset = new Set(front);
  if (front.length>1){
    const poly = front.map(p=>`${X(p[xk]).toFixed(1)},${Y(p.accuracy).toFixed(1)}`).join(" ");
    s+=`<polyline points="${poly}" fill="none" stroke="${CHART.sub}" stroke-width="1.5" stroke-dasharray="5 4" opacity=".5"/>`;
  }
  pts.forEach(p=>{
    const on = fset.has(p), r = on?6:4.5, col = DCOLOR[p.design]||"#777";
    const ax=X(p[xk]), ay=Y(p.accuracy), yci=p.accuracy_ci||0;
    if(yci){                                                     // vertical 95% CI on accuracy, with caps
      const yh=Y(Math.min(100,p.accuracy+yci)), yl=Y(Math.max(0,p.accuracy-yci));
      s+=`<line x1="${ax}" y1="${yh}" x2="${ax}" y2="${yl}" stroke="${col}" stroke-width="1.2" opacity=".55"/>`+
         `<line x1="${ax-3}" y1="${yh}" x2="${ax+3}" y2="${yh}" stroke="${col}" stroke-width="1.2" opacity=".55"/>`+
         `<line x1="${ax-3}" y1="${yl}" x2="${ax+3}" y2="${yl}" stroke="${col}" stroke-width="1.2" opacity=".55"/>`;
    }
    s+=`<circle cx="${ax}" cy="${ay}" r="${r}" fill="${col}" ${on?'':'fill-opacity=".5"'} stroke="${CHART.surface}" stroke-width="1.5"/>`;
    const tag = (p.model_short.match(/\d+\.?\d*\s*b/i)||[p.model_short.slice(0,5)])[0].replace(/\s/g,"").toUpperCase();
    const lab = multiModel ? `${p.abbr} ${tag}` : p.abbr;            // colour=design, short model tag
    s+=`<text x="${ax+9}" y="${ay+3.5}" font-size="${CHART.t_tick}" fill="${on?CHART.ink:CHART.mute}">${esc(lab)}</text>`;
  });
  return s+`</svg>`;
}
// One small bar chart: value of `key` per design (one bar each), for the compute-cost panel.
// Answers "how much more does each design cost in tokens / time / $ / effort?" at a glance.
function barChart(pts, key, title, fmt, multi){
  const W=320,H=240, mL=46,mR=12,mT=30,mB=multi?52:38;
  const rows = pts.filter(p => p[key] != null);
  const head = `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" font-family="${CHART.font}">`+
    `<text x="${mL}" y="18" font-size="${CHART.t_title}" fill="${CHART.ink}" font-weight="600">${title}</text>`;
  if(!rows.length)
    return `<div class="chartcard">${head}<text x="${W/2}" y="${H/2}" text-anchor="middle" `+
      `font-size="${CHART.t_axis}" fill="${CHART.mute}">not recorded yet</text></svg></div>`;
  const ci = p => 1.96 * (p[key + "_sem"] || 0);                 // 95% CI half-width
  const maxV = Math.max(...rows.map(p => p[key] + ci(p))) * 1.14 || 1;
  const n = rows.length, gap = 18, bw = Math.min(64, (W-mL-mR-(n-1)*gap)/n);
  const startX = mL + ((W-mL-mR) - (n*bw+(n-1)*gap))/2;          // centre the bar group in the plot
  const Y = v => H-mB-(H-mB-mT)*v/maxV;
  let s = head;
  [0,.5,1].forEach(f => { const v = maxV*f; s += `<line x1="${mL}" y1="${Y(v)}" x2="${W-mR}" y2="${Y(v)}" stroke="${CHART.grid}"/>`+
    `<text x="${mL-6}" y="${Y(v)+3}" text-anchor="end" font-size="${CHART.t_tick}" fill="${CHART.mute}">${fmt(v)}</text>`; });
  s += `<line x1="${mL}" y1="${H-mB}" x2="${W-mR}" y2="${H-mB}" stroke="${CHART.axis}"/>`;
  rows.forEach((p, i) => {
    const cx = startX+i*(bw+gap)+bw/2, x = cx-bw/2, y = Y(p[key]), h = Math.max(1, (H-mB)-y);
    s += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="4" fill="${DCOLOR[p.design]||'#777'}"/>`;
    let topY = y; const c = ci(p);
    if (c > 0){                                                  // 95% CI whisker with caps
      const yh = Y(Math.min(maxV, p[key]+c)), yl = Y(Math.max(0, p[key]-c)); topY = yh;
      s += `<line x1="${cx}" y1="${yh}" x2="${cx}" y2="${yl}" stroke="${CHART.sub}" stroke-width="1.4"/>`+
           `<line x1="${cx-4}" y1="${yh}" x2="${cx+4}" y2="${yh}" stroke="${CHART.sub}" stroke-width="1.4"/>`+
           `<line x1="${cx-4}" y1="${yl}" x2="${cx+4}" y2="${yl}" stroke="${CHART.sub}" stroke-width="1.4"/>`;
    }
    s += `<text x="${cx}" y="${topY-6}" text-anchor="middle" font-size="${CHART.t_val}" fill="${CHART.ink}" font-weight="600">${fmt(p[key])}</text>`;
    s += `<text x="${cx}" y="${H-mB+14}" text-anchor="middle" font-size="${CHART.t_tick}" fill="${CHART.sub}">${esc(p.abbr)}</text>`;
    if(multi) s += `<text x="${x+bw/2}" y="${H-mB+25}" text-anchor="middle" font-size="8" fill="${CHART.mute}">${esc((p.model_short||'').slice(0,10))}</text>`;
  });
  return `<div class="chartcard">${s}</svg></div>`;
}
const _FMT = {
  tokens: v => v>=1000 ? (v/1000).toFixed(v<10000?1:0)+"k" : String(Math.round(v)),
  secs:   v => v.toFixed(v<10?1:0)+"s",
  cost:   v => "$"+Math.round(v),
  turns:  v => v.toFixed(1),
  tests:  v => v.toFixed(1),
  recall: v => Math.round(v*100)+"%",
  precision: v => Math.round(v*100)+"%",
};
// The compute-cost panel: the five things the user is comparing across designs.
function computeBars(pts, multi){
  const core = [                                                // the five per-case cost axes — one clean row
    ["tokens", "Tokens"], ["secs", "Time (s)"], ["cost", "Workup cost ($)"],
    ["turns", "Turns"], ["tests", "Tests"],
  ];
  const sorted = pts.slice().sort((a,b) => (DORDER[a.design]??9)-(DORDER[b.design]??9) || a.model_short.localeCompare(b.model_short));
  let html = `<div class="stat-plots cols-5">`+
    core.map(([k,t]) => barChart(sorted, k, t, _FMT[k], multi)).join("")+`</div>`;
  if (pts.some(p => p.recall != null))                          // MIMIC: real-order concordance, its own row
    html += `<div class="stat-plots cols-2">`+
      [["recall","Order concordance · recall"],["precision","Order concordance · precision"]]
        .map(([k,t]) => barChart(sorted, k, t, _FMT[k], multi)).join("")+`</div>`;
  return html + statLegend([...new Set(sorted.map(p => p.design))]);
}
function statLegend(present){
  const ids = (present && present.length ? present.filter(id => DNAME[id])
                                         : Object.keys(DNAME)).sort((a,b)=>DORDER[a]-DORDER[b]);
  const items = ids.map(id => `<span class="lg-item"><i style="background:${DCOLOR[id]}"></i>${DNAME[id]}</span>`).join("");
  return `<div class="stat-legend">${items}<span class="lg-item lg-pareto"><i class="lg-dash"></i>Pareto-optimal</span></div>`;
}
// Pull the headline quantitative figures straight from the benchmark rows so the
// summary can't drift from the plots below it. Robust slice = the model with the
// largest single-config n; scale contrast = best config on any other model.
function keyFigures(rows){
  if (!rows || !rows.length) return null;
  const byModel = {};
  rows.forEach(p => { (byModel[p.model_short] = byModel[p.model_short] || []).push(p); });
  let robust = null, robustN = -1;
  for (const [m, ps] of Object.entries(byModel)){
    const nmax = Math.max(...ps.map(p => p.n));
    if (nmax > robustN){ robustN = nmax; robust = m; }
  }
  const rps = byModel[robust];
  const single = rps.find(p => p.design === "single");
  const best = rps.slice().sort((a, b) => b.accuracy - a.accuracy)[0];
  const leanest = rps.slice().sort((a, b) => a.turns - b.turns)[0];
  let scale = null;
  for (const [m, ps] of Object.entries(byModel)){
    if (m === robust) continue;
    const top = ps.slice().sort((a, b) => b.accuracy - a.accuracy)[0];
    if (!scale || top.accuracy > scale.accuracy) scale = top;
  }
  if (!single || !best) return null;
  return {model: robust, n: robustN, single, best, leanest, scale};
}
function headlineBand(kf){
  if (!kf) return "";
  const {model, n, single, best, leanest, scale} = kf;
  const liftTxt = single.accuracy > 0 ? `${(best.accuracy / single.accuracy).toFixed(1)}×`
                                      : `+${best.accuracy - single.accuracy} pts`;
  const stat = (big, lab) => `<div class="hl-stat"><div class="hl-n">${big}</div><div class="hl-lab">${lab}</div></div>`;
  return `<div class="keyband">`+
    `<div class="hl-tag">Key result</div>`+
    `<div class="hl-claim">On the sequential CPC benchmark, agent scaffolding lifts diagnostic accuracy `+
      `<b>${liftTxt}</b> over a single LLM (${esc(prettyModel(model))}, n=${n}); scaling the base model lifts it further.</div>`+
    `<div class="hl-stats">`+
      stat(`${single.accuracy}% → ${best.accuracy}%`, `Accuracy: single LLM → ${esc(best.label)}<br><i>${esc(prettyModel(model))}, n=${n}</i>`)+
      stat(`${single.turns} → ${leanest.turns}`, `Clinician turns per case<br><i>single LLM → ${esc(leanest.label)}</i>`)+
      (scale ? stat(`${scale.accuracy}%`, `Best design on ${esc(prettyModel(scale.model_short))}<br><i>n=${scale.n}, preliminary</i>`) : "")+
    `</div></div>`;
}
let statMode = "cpc";                                   // remembered CPC/MIMIC/All selection
async function renderStats(){
  const box = $("#statsBox");
  const s = await getj("/api/stats");
  if (!s || !s.modes){ box.innerHTML = `<div class="empty-hint">No recordings to summarize yet: run the benchmark first.</div>`; return; }
  const multi = (s.n_models||1) > 1;
  const cpc = s.modes.cpc || [];
  const byModel = {};
  cpc.forEach(p => { (byModel[p.model_short] = byModel[p.model_short] || []).push(p); });
  const ranges = Object.entries(byModel).map(([m, ps]) => {
    const a = ps.map(p => p.accuracy);
    return `<b>${esc(m)}</b> ${Math.min(...a)}–${Math.max(...a)}% (n=${Math.max(...ps.map(p => p.n))})`;
  }).join("; ");
  const kf = keyFigures(cpc);
  const resultTxt = kf
    ? `On the robust slice (${esc(prettyModel(kf.model))}, n=${kf.n}), Judge accuracy rises from `+
      `${kf.single.accuracy}% (single LLM) to ${kf.best.accuracy}% (${esc(kf.best.label)}), while interaction turns `+
      `fall from ${kf.single.turns} to ${kf.leanest.turns}.` +
      (kf.scale ? ` On ${esc(prettyModel(kf.scale.model_short))} (n=${kf.scale.n}) the best design reaches ${kf.scale.accuracy}%.` : "")
    : (ranges ? `Accuracy was ${ranges}.` : "");
  const liftTxt = kf && kf.single.accuracy > 0 ? `${(kf.best.accuracy / kf.single.accuracy).toFixed(1)}×` : "a clear";
  box.innerHTML =
    headlineBand(kf)+
    `<div class="an-title rtop">Main finding · accuracy vs cost &amp; effort`+
    `<div class="seg" id="statMode" style="margin-left:auto">`+
    ["cpc","mimic","all"].map(m=>`<button data-m="${m}" class="${statMode===m?'on':''}">${m==="all"?"All":m.toUpperCase()}</button>`).join("")+`</div></div>`+
    rbubble(
      "Each design works every case through an information gatekeeper (a finding is revealed only when the agent asks, at a cost) and is graded 1–5 by an LLM judge (≥4 = correct). Points nearer the top-left — more accurate, cheaper, or fewer turns — are better; whiskers are 95% CIs.",
      resultTxt,
      `On this slice, agent scaffolding gives a ${liftTxt} accuracy change over a single LLM; with small n the confidence intervals are wide, so read the direction, not the decimals.`
    )+
    `<div id="statBody"></div>`+
    `<div class="an-title rtop">Cost of complexity · per-case compute &amp; workup</div>`+
    rbubble(
      "For every design we log the LLM tokens, wall-clock time to a delivered recommendation, dollar cost of the tests ordered, interaction turns, and number of tests — averaged per case; whiskers are 95% CIs. LLM-Multi issues several LLM calls per turn; the single-call designs issue one.",
      "Compare bar heights across designs: the token and time bars show what the added scaffolding actually costs to run.",
      "If LLM-Roles' (or LLM-Multi's) tokens and latency rise without a matching accuracy gain above, the extra machinery isn't buying its keep for lab-test recommendation."
    )+
    `<div id="statCompute"></div>`;
  const draw = m => {
    const pts = (s.modes[m]||[]).filter(p => p.n > 0);
    if (!pts.length){
      $("#statBody").innerHTML = `<div class="empty-hint">No ${m.toUpperCase()} cases recorded yet.</div>`;
      $("#statCompute").innerHTML = ""; return;
    }
    $("#statCompute").innerHTML = computeBars(pts, multi);
    let html = `<div class="stat-plots cols-3">`+
      `<div class="chartcard">${paretoPlot(pts, "cost", "Workup cost ($)", "Accuracy vs cost", multi)}</div>`+
      `<div class="chartcard">${paretoPlot(pts, "turns", "Turns", "Accuracy vs turns", multi)}</div>`+
      `<div class="chartcard">${paretoPlot(pts, "tests", "Tests", "Accuracy vs tests", multi)}</div></div>`+
      statLegend([...new Set(pts.map(p => p.design))])+
      `<table class="qr-table"><thead><tr><th>Model</th><th>Agent</th>`+
      `<th class="num">n</th><th class="num">Accuracy</th><th class="num">Judge</th>`+
      `<th class="num">Tokens</th><th class="num">Latency</th>`+
      `<th class="num">Cost</th><th class="num">Turns</th><th class="num">Tests</th></tr></thead><tbody>`;
    const front = new Set(paretoFront(pts, "cost"));
    const fmtTok = v => v==null ? "–" : (v>=1000 ? (v/1000).toFixed(v<10000?1:0)+"k" : Math.round(v));
    pts.slice().sort((a,b)=>b.accuracy-a.accuracy).forEach(p => {
      html += `<tr${front.has(p)?' class="pareto"':''}><td>${esc(prettyModel(p.model_short))}</td>`+
        `<td class="agent"><span class="ddot" style="background:${DCOLOR[p.design]||'#777'}"></span>${p.label}</td>`+
        `<td class="num">${p.n}</td><td class="num">${p.accuracy}%</td><td class="num">${p.judge}</td>`+
        `<td class="num">${fmtTok(p.tokens)}</td><td class="num">${p.secs==null?"–":p.secs+"s"}</td>`+
        `<td class="num">$${p.cost}</td><td class="num">${p.turns}</td><td class="num">${p.tests}</td></tr>`; });
    $("#statBody").innerHTML = html + `</tbody></table>`;
  };
  draw(statMode);
  box.querySelectorAll("#statMode button").forEach(b => b.onclick = () => {
    statMode = b.dataset.m;
    box.querySelectorAll("#statMode button").forEach(x => x.classList.toggle("on", x === b)); draw(statMode); });
  await renderBudget(box);
  await renderAblation(box);
  await renderAnalysis(box);
}
// Auto-refresh the Results tab while a benchmark writes recordings, so it updates without a reload.
setInterval(() => {
  const tab = document.querySelector('.tab[data-tab="stats"]');
  if (tab && tab.classList.contains("on")) renderStats();
}, 30000);

/* ---------- Results: panel role ablation (which panelist matters?) ---------- */
async function renderAblation(box){
  const a = await getj("/api/ablation");
  const head = `<div class="analysis"><div class="an-title">Role ablation<span class="an-sub">which perspective matters?</span></div>`+
    rbubble("Drop one perspective from the LLM-Roles prompt at a time (outcomes / cost / burden / workload / checklist) and re-measure accuracy — isolates which role actually drives performance.", "", "");
  if (!a || !a.models || !a.models.length){
    box.insertAdjacentHTML("beforeend", head + `<div class="empty-hint">Not collected yet — run the role-ablation sweep.</div></div>`);
    return;
  }
  let h = head;
  a.models.forEach(m => {
    h += `<div class="abl-model"><span class="hint">${esc(prettyModel(m.model_short))}</span><table class="qr-table"><thead><tr>`+
      `<th>Variant</th><th class="num">n</th><th class="num">Accuracy</th><th class="num">Δ vs full</th></tr></thead><tbody>`;
    m.variants.forEach(v => {
      const full = v.key === "full";
      const dcls = v.delta < 0 ? "abl-dn" : v.delta > 0 ? "abl-up" : "abl-flat";
      h += `<tr${full ? ' class="pareto"' : ''}><td>${full ? "<b>" + esc(v.label) + "</b>" : esc(v.label)}</td>`+
        `<td class="num">${v.n}</td><td class="num">${v.accuracy}%</td>`+
        `<td class="num ${full ? '' : dcls}">${full ? "-" : (v.delta > 0 ? "+" : "") + v.delta + " pt"}</td></tr>`;
    });
    h += `</tbody></table></div>`;
  });
  h += `</div>`;
  const div = document.createElement("div"); div.innerHTML = h; box.appendChild(div);
}

/* ---------- Results: cost-budget trade-off curve (SDBench sweep) ---------- */
function budgetCurve(m){                                // x = avg actual cost, y = accuracy; line per design
  const all = m.designs.flatMap(d => d.points);
  const W=480, H=300, pad=48;
  const maxX = Math.max(1, ...all.map(p => p.cost)) * 1.15;
  const X=v=>pad+(W-2*pad)*v/maxX, Y=v=>H-pad-(H-2*pad)*v/100;
  let s=`<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
  [0,25,50,75,100].forEach(v=>{ s+=`<line x1="${pad}" y1="${Y(v)}" x2="${W-pad}" y2="${Y(v)}" stroke="#eceef1"/>`+
    `<text x="${pad-8}" y="${Y(v)+3}" text-anchor="end" font-size="9" fill="#9aa3af">${v}</text>`; });
  [0,maxX/2,maxX].forEach(v=>{ s+=`<text x="${X(v)}" y="${H-pad+15}" text-anchor="middle" font-size="9" fill="#9aa3af">${Math.round(v)}</text>`; });
  s+=`<line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="#d5d9df"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H-pad}" stroke="#d5d9df"/>`;
  s+=`<text x="${pad}" y="18" font-size="11.5" fill="#1b2430" font-weight="600">${esc(prettyModel(m.model_short))}</text>`;
  s+=`<text x="${W/2}" y="${H-5}" text-anchor="middle" font-size="10" fill="#9aa3af">Cost per case ($)</text>`;
  s+=`<text x="13" y="${H/2}" text-anchor="middle" font-size="10" fill="#9aa3af" transform="rotate(-90 13 ${H/2})">Accuracy (%)</text>`;
  m.designs.forEach(d => {
    const col = DCOLOR[d.id] || "#777";
    if (d.points.length > 1)
      s += `<polyline points="${d.points.map(p=>`${X(p.cost).toFixed(1)},${Y(p.accuracy).toFixed(1)}`).join(" ")}" fill="none" stroke="${col}" stroke-width="2"/>`;
    d.points.forEach(p => { s += `<circle cx="${X(p.cost)}" cy="${Y(p.accuracy)}" r="4" fill="${col}"/>`+
      `<text x="${X(p.cost)}" y="${Y(p.accuracy)-7}" font-size="8" fill="#9aa3af" text-anchor="middle">${p.cap==='inf'?'∞':'$'+p.cap}</text>`; });
  });
  return s+`</svg>`;
}
async function renderBudget(box){
  const b = await getj("/api/budget");
  const head = `<div class="analysis"><div class="an-title">Cost &amp; budget trade-off<span class="an-sub">accuracy vs a per-case spend cap</span></div>`+
    rbubble("Cap the per-case test spend at several levels and re-run each design — does giving the agent a bigger budget buy accuracy, or is the limit the model?", "", "");
  if (!b || !b.models || !b.models.length){
    box.insertAdjacentHTML("beforeend", head + `<div class="empty-hint">Not collected yet — run the budget sweep.</div></div>`);
    return;
  }
  let h = head + `<div class="stat-plots">`;
  b.models.forEach(m => { h += `<div class="chartcard">${budgetCurve(m)}</div>`; });
  h += `</div></div>`;
  box.insertAdjacentHTML("beforeend", h);
}

/* ---------- Results: SDBench-style breakdowns (error profile / cost / specialty) ---------- */
const ERR_C = ["#c0392b", "#e07b39", "#e8b84b", "#7cb342", "#2e7d32"];
const ERR_L = ["1 · wrong system", "2 · superficial", "3 · right category", "4 · correct", "5 · exact"];
// Cost categories — CVD-safe (Okabe-Ito), fixed order, distinct from the design palette.
const COST_C = {Visits:"#6e7681", Labs:"#0072b2", "Micro/Serol":"#009e73", Imaging:"#e69f00", Procedure:"#d55e00", "Genetic/Path":"#cc79a7"};
async function renderAnalysis(box){
  const a = await getj("/api/analysis");
  const n = a && a.designs && a.designs[0] ? a.designs[0].n || 0 : 0;
  const head = `<div class="analysis"><div class="an-title">Diagnostic breakdowns<span class="an-sub">error profile · cost composition · by specialty${n?` · n≈${n}`:""}</span></div>`;
  if (!a || !a.designs || !a.designs.length){
    box.insertAdjacentHTML("beforeend", head +
      rbubble("Per-design breakdowns: the Judge 1–5 error profile, where the workup dollars go (labs / imaging / procedures / …), and accuracy split by disease specialty (infectious, onc/heme, cardiovascular, …).", "", "")+
      `<div class="empty-hint">Not collected yet — populates from the benchmark run.</div></div>`);
    return;
  }
  let h = head;

  // 1. error profile — stacked Judge 1–5 per design
  h += `<h3 class="an-h">Error Profile</h3>`;
  h += rbubble("We bin each final diagnosis by LLM-judge score (1 to 5).",
               "Most errors are Judge 1 (wrong organ system).",
               "Failures are gross, not near-misses.");
  a.designs.forEach(d => {
    h += `<div class="an-row"><span class="an-label">${esc(d.label)}</span><div class="an-bar">`+
      d.errors.map((p, i) => p ? `<span style="width:${p}%;background:${ERR_C[i]}" title="${ERR_L[i]}: ${p}%"></span>` : "").join("")+
      `</div></div>`;
  });
  h += `<div class="an-legend">`+ERR_L.map((l, i) => `<span><i style="background:${ERR_C[i]}"></i>${l}</span>`).join("")+`</div>`;

  // 2. cost composition — stacked $ by category per design
  h += `<h3 class="an-h">Cost Composition</h3>`;
  h += rbubble("We split mean per-case spend by test category.",
               "Imaging and procedures account for most spend across all designs.",
               "Spend concentrates in high-cost modalities rather than cheap labs.");
  const cats = [...new Set(a.designs.flatMap(d => Object.keys(d.cost)))];
  const maxC = Math.max(1, ...a.designs.map(d => Object.values(d.cost).reduce((x, y) => x + y, 0)));
  a.designs.forEach(d => {
    const tot = Object.values(d.cost).reduce((x, y) => x + y, 0);
    h += `<div class="an-row"><span class="an-label">${esc(d.label)}</span><div class="an-bar track">`+
      cats.map(c => d.cost[c] ? `<span style="width:${100*d.cost[c]/maxC}%;background:${COST_C[c]||'#888'}" title="${c}: $${d.cost[c]}"></span>` : "").join("")+
      `</div><span class="an-tot">$${tot}</span></div>`;
  });
  h += `<div class="an-legend">`+cats.map(c => `<span><i style="background:${COST_C[c]||'#888'}"></i>${c}</span>`).join("")+`</div>`;

  // 3. accuracy by specialty — rows = specialty, cols = design
  h += `<h3 class="an-h">Accuracy by Specialty</h3>`;
  h += rbubble("We stratify accuracy by a specialty label inferred from the gold diagnosis.",
               "Per-specialty counts are small (e.g. Other 56, Infectious 16).",
               "Differences across specialties are not reliable at these counts.");
  const sm = {};
  a.designs.forEach(d => d.specialty.forEach(s => { (sm[s.name] = sm[s.name] || {n: s.n, acc: {}}).acc[d.id] = s.acc; }));
  const rows = Object.entries(sm).sort((x, y) => y[1].n - x[1].n);
  if (rows.length){
    h += `<table class="qr-table"><thead><tr><th>Specialty</th><th class="num">n</th>`+
      a.designs.map(d => `<th class="num">${esc(d.label)}</th>`).join("")+`</tr></thead><tbody>`;
    rows.forEach(([name, v]) => {
      h += `<tr><td>${esc(name)}</td><td class="num">${v.n}</td>`+
        a.designs.map(d => `<td class="num">${v.acc[d.id] != null ? v.acc[d.id] + "%" : "-"}</td>`).join("")+`</tr>`;
    });
    h += `</tbody></table>`;
  }
  h += `</div>`;
  const div = document.createElement("div"); div.innerHTML = h; box.appendChild(div);
}

/* ---------- collapsible reasoning: click the meta line to unfold ---------- */
document.addEventListener("click", e => {
  const tog = e.target.closest(".rfold .rtoggle");
  if (tog) tog.parentElement.classList.toggle("open");
});

/* ---------- shared ‹ x/total › pager + CPC/EHR/All dataset filter ---------- */
let recAll = [], recMode = "cpc", recList = [];
function rebuildRecList(){                            // separates the tabular EHR cases from the CPC narratives
  const avail = new Set([...$("#caseSel").options].map(o => o.value));
  const modes = window.__MODES__ || {};
  recList = recAll.filter(id => avail.has(id) && (recMode === "all" || modes[id] === recMode));
}
// Friendly, stable case label ("CPC Case 7") from its position within its dataset, keeping the raw id.
function caseLabel(id, mode){
  const modes = window.__MODES__ || {};
  const m = modes[id] || mode || "cpc";
  const n = (recAll || []).filter(x => (modes[x] || "cpc") === m).indexOf(id) + 1;
  return n > 0 ? `${m === "ehr" ? "EHR" : "CPC"} Case ${n}` : id;
}
function ctxTitle(id, mode){ return `<span title="${esc(id)}">${esc(caseLabel(id, mode))}</span>`; }
function wirePager(prefix, onLoad){
  const num = $(`#${prefix}Num`), tot = $(`#${prefix}Tot`);
  let idx = 0;
  const go = i => { if (!recList.length) return; idx = (i % recList.length + recList.length) % recList.length;
                    num.value = idx + 1; onLoad(recList[idx]); };
  const refresh = () => { if (tot) tot.textContent = recList.length; idx = 0; if (num) num.value = recList.length ? 1 : 0; };
  refresh();
  if ($(`#${prefix}Prev`)){
    $(`#${prefix}Prev`).onclick = () => go(idx - 1);
    $(`#${prefix}Next`).onclick = () => go(idx + 1);
    const jump = () => { const n = parseInt(num.value, 10); if (n >= 1 && n <= recList.length) go(n - 1); else num.value = idx + 1; };
    num.onchange = jump; num.onkeydown = e => { if (e.key === "Enter") jump(); };
  }
  return { go, refresh, sync: cid => { const i = recList.indexOf(cid); if (i >= 0){ idx = i; num.value = i + 1; } } };
}
recAll = (window.__RECORDED__ || []);
rebuildRecList();

// Recommend: page loads an example's agent comparison
$("#caseSel").onchange = () => { if ($("#caseSel").value) startCase(); };
const recPager = wirePager("pg", cid => { $("#caseSel").value = cid; startCase(); });

// Benchmark: page picks which case the clinician works up live (LLM gatekeeper for every case)
const clPager = wirePager("cl", cid => loadClinician(cid));

// Grading: page picks which recorded case to grade
const grPager = wirePager("gr", cid => { currentCaseId = cid; for (const k in recCache) delete recCache[k]; renderEvaluate(); });

// CPC / EHR / All dataset filter — one toggle, mirrored on every tab, reloads the active tab
$$(".modeseg button").forEach(b => b.onclick = () => {
  recMode = b.dataset.m;
  $$(".modeseg button").forEach(x => x.classList.toggle("on", x.dataset.m === recMode));   // keep all instances in sync
  rebuildRecList();
  recPager.refresh(); grPager.refresh(); clPager.refresh();
  const active = $(".tabs button.on")?.dataset.tab;
  if (active === "benchmark") clPager.go(0);
  else if (active === "score") grPager.go(0);
  else if (recList.length) recPager.go(0);
});
if (recList.length) recPager.go(0);                                  // auto-load the first example

/* ---------- model picker: swap which recorded model the replay tabs show ---------- */
async function initModelPicker(){
  const sel = $("#modelSel");
  if (!sel) return;
  const d = await getj("/api/models");
  const models = (d && d.models) || [];
  if (!models.length){ if (sel.parentElement) sel.parentElement.style.display = "none"; return; }
  sel.innerHTML = models.map(m => `<option value="${esc(m.slug)}">${esc(prettyModel(m.slug))} · ${m.n_cases} cases</option>`).join("");
  const bySlug = Object.fromEntries(models.map(m => [m.slug, m]));
  const apply = slug => {
    const m = bySlug[slug]; if (!m) return;
    curModel = slug;
    window.__RECORDED__ = m.cases; window.__MODES__ = m.modes; recAll = m.cases;
    for (const k in recCache) delete recCache[k];                    // recordings are per-model
    rebuildRecList();
    recPager.refresh(); grPager.refresh(); clPager.refresh();
    const active = $(".tabs button.on")?.dataset.tab;
    if (active === "score")        grPager.go(0);
    else if (active === "benchmark") clPager.go(0);
    else if (recList.length)       recPager.go(0);
  };
  sel.onchange = () => apply(sel.value);
  apply(models[0].slug);                                            // default = best-covered model
}
initModelPicker();

$("#vignetteToggle").onclick = () => {
  const hidden = $("#ctxFull").classList.toggle("hidden");
  $("#vignetteToggle").textContent = hidden ? "View full case" : "Hide full case";
};

function fillContext(id, mode, presentation, full){
  currentMode = mode;
  $("#caseCtx").classList.remove("hidden");
  $("#ctxId").innerHTML = ctxTitle(id, mode);
  // Show the FULL presentation the model received at turn 0 (history + exam), not just a one-liner.
  $("#ctxAbs").textContent = presentation;
  $("#ctxAbs").classList.add("ctx-presentation");
  $("#ctxFull").textContent = full;
  $("#ctxFull").classList.add("hidden");
  $("#vignetteToggle").textContent = "View full case";
}

async function startCase(){                          // Recommend tab: the agent comparison
  const id = $("#caseSel").value; if (!id) return;
  currentCaseId = id; caseLoaded = true;
  for (const k in recCache) delete recCache[k];
  $("#recHint").style.display = "none";
  $("#panels").classList.remove("hidden");
  $("#vignetteToggle").style.display = "";          // agents only see the abstract; reviewer may inspect
  await loadAgentPanels(id);
}

// Benchmark tab: the clinician works the case up live (human baseline). Anti-cheat: abstract only.
let clinTurnStart = null;
async function loadClinician(id){
  if (!id) return;
  currentCaseId = id;
  const r = await post("/api/new", {case_id:id, design:"single", model, gatekeeper:"llm"});
  $("#clinCtx").classList.remove("hidden");
  $("#clinCtxId").innerHTML = ctxTitle(id, r.mode);
  $("#clinCtxAbs").textContent = r.abstract;
  $("#clinHint").style.display = "none";
  $("#chat").classList.remove("hidden");
  $("#composer").style.display = "flex";
  $("#chat").innerHTML = `<div class="empty-hint">Your move: ask, order a test, or commit a diagnosis below.</div>`;
  clinTurnStart = Date.now();                        // start the decision-time clock
}

/* ---------- time + metrics ---------- */
function fmtClock(min){ const h=Math.floor(min/60)%24, m=Math.round(min)%60;
  return String(h).padStart(2,"0")+":"+String(m).padStart(2,"0"); }
function durationFor(t){
  if (t.action === "ask") return 5;
  if (t.action === "diagnose") return 2;
  const n = (t.query||"").toLowerCase();
  if (/ct|x-ray|radiograph|echo|angiogram|mri|ultrasound/.test(n)) return 55;
  if (/culture|micro/.test(n)) return 120;
  return 30;
}
function totalMinutes(rec){ let m=0; rec.turns.forEach(t=>m+=durationFor(t)); return m; }
function fmtDur(m){ const h=Math.floor(m/60), mm=Math.round(m)%60; return h?`${h}h ${mm}m`:`${mm}m`; }

/* ---------- shared conversation renderer ---------- */
function bubble(box, side, avatar, inner){
  const el = document.createElement("div"); el.className = "msg " + side;
  el.innerHTML = `<div class="avatar">${avatar}</div><div class="bubble">${inner}</div>`;
  box.appendChild(el); return el;
}
function timeLabel(box, txt){ const d=document.createElement("div"); d.className="time"; d.textContent=txt; box.appendChild(d); }
function judgeBubble(box, t){                         // the Judge speaks as its own bubble
  const ok = t.correct, sc = t.judge_score != null ? `${t.judge_score}/5 · ` : "";
  let inner = `<div class="act">judge</div><div class="jverdict ${ok?"ok":"no"}">${sc}${ok?"✓ Correct":"✗ Incorrect"}</div>`;
  if (t.judge_reason) inner += `<div class="jreason">${esc(t.judge_reason)}</div>`;
  bubble(box, "judge", "J", inner);
}

function thinkTime(t){                              // synthesised "time thought before responding"
  const base = t.action === "diagnose" ? 1.6 : t.action === "ask" ? 0.8 : 1.2;
  const len = ((t.reasoning || "").length + JSON.stringify(t.differential || []).length) / 120;
  return (base + len).toFixed(1);
}
function metaBlock(t){                              // think-time + a › arrow that unfolds the reasoning
  const think = `${CLOCK}${thinkTime(t)}s`;
  if (!t.reasoning) return `<div class="rtoggle solo">${think}</div>`;
  return `<div class="rfold"><div class="rtoggle"><span>${think}</span>`+
         `<span class="rlink" title="show reasoning"></span></div>`+
         `<div class="reason">“${esc(t.reasoning)}”</div></div>`;
}
// Long gatekeeper results (lab panels, imaging reports) collapse to one line with a reveal.
function gkFinding(text){
  const s = String(text || "");
  if (!s.includes("\n") && s.length <= 100) return esc(s);
  return `<div class="gkfold"><span class="gk-short">${esc(s.split("\n")[0].slice(0, 96))}… </span>`+
    `<span class="gk-full hidden">${esc(s)}</span>`+
    `<button class="gk-more">show result</button></div>`;
}
document.addEventListener("click", e => {
  const m = e.target.closest(".gkfold .gk-more"); if (!m) return;
  const f = m.closest(".gkfold");
  const open = f.querySelector(".gk-full").classList.toggle("hidden") === false;
  f.querySelector(".gk-short").classList.toggle("hidden", open);
  m.textContent = open ? "hide" : "show result";
});
function renderConvo(box, rec, opts){
  opts = opts || {}; const mode = opts.mode || "plain", abbr = opts.abbr || "AI";
  box.innerHTML = ""; let clock = 9*60;
  rec.turns.forEach(t => {
    clock += durationFor(t); timeLabel(box, fmtClock(clock));
    if (t.action === "diagnose"){
      let inner = `<div class="act">diagnosis</div><div class="headline">${esc(t.query)}</div>` + metaBlock(t);
      bubble(box, "agent", abbr, inner);
      if (mode !== "plain") judgeBubble(box, t);     // panels use the header badge + bottom review instead
    } else if (t.action === "ask"){
      let inner = `<div class="act">asks</div><div class="headline">${esc(t.query)}</div>` + metaBlock(t);
      bubble(box, "agent", abbr, inner);
      bubble(box, "gk", "GK", gkFinding(t.finding));
    } else { // order (may batch several tests)
      let inner = `<div class="act">recommends</div><div class="headline">${esc(t.query)}</div>` + metaBlock(t);
      if (mode !== "plain" && t.differential && t.differential.length)   // keep panels clean
        inner += `<div class="diff"><span class="diff-lab">differential</span>`+
          t.differential.slice(0,3).map(d=>`<span class="chip">${esc(d.dx)}${d.p!=null?` · ${Math.round(d.p*100)}%`:""}</span>`).join("")+`</div>`;
      bubble(box, "agent", abbr, inner);
      const orders = t.orders || [{name:t.query, finding:t.finding, cost:t.cost}];
      orders.forEach(o => bubble(box, "gk", "GK", gkFinding(o.finding)));
    }
  });
}

/* ---------- RECOMMEND: agents side by side (LLM + Multi-agent by default) ---------- */
function renderPanelBar(id){
  const bar = $("#panelBar");
  const hidden = DESIGNS.filter(d => !visibleDesigns.includes(d.id));
  bar.classList.remove("hidden");
  if (!hidden.length){ bar.innerHTML = ""; return; }
  bar.innerHTML = `<div class="addmenu"><button class="addbtn" id="addBtn">+ Add agent</button>`+
    `<div class="addlist" id="addList">${hidden.map(d=>`<button data-add="${d.id}">${d.label}</button>`).join("")}</div></div>`;
  $("#addBtn").onclick = () => $("#addList").classList.toggle("open");
  $$("#panelBar [data-add]").forEach(b => b.onclick = () => { visibleDesigns = [...visibleDesigns, b.dataset.add]; loadAgentPanels(id); });
}
async function loadAgentPanels(id){
  $("#quadReview").classList.add("hidden");
  renderPanelBar(id);
  $("#panels").style.gridTemplateColumns = `repeat(${visibleDesigns.length}, 1fr)`;
  $("#panels").innerHTML = visibleDesigns.map(did => {
    const d = DESIGNS.find(x => x.id === did);
    return `<div class="apanel"><div class="apanel-head">`+
      `<span class="apanel-name">${d.label}</span>`+
      `<span class="jbadge" id="badge-${d.id}"></span>`+
      `<button class="apanel-x" data-rm="${d.id}" title="remove">×</button></div>`+
      `<button class="apanel-toggle" data-tog="${d.id}">Hide workup ▴</button>`+
      `<div class="apanel-body" id="body-${d.id}"><div class="empty-hint">loading…</div></div></div>`;
  }).join("");
  $$("#panels [data-rm]").forEach(b => b.onclick = () => {
    if (visibleDesigns.length <= 1) return;                 // keep at least one
    visibleDesigns = visibleDesigns.filter(x => x !== b.dataset.rm); loadAgentPanels(id);
  });
  $$("#panels [data-tog]").forEach(b => b.onclick = () => {
    const open = !$("#body-"+b.dataset.tog).classList.toggle("collapsed");
    b.textContent = open ? "Hide workup ▴" : "View workup ▾";
  });

  // Show the presentation the model was given right away — independent of any recording.
  let gotCtx = false;
  try {
    const c = await getj(`/api/case?case=${encodeURIComponent(id)}`);
    if (c && c.presentation){ gotCtx = true; fillContext(id, c.mode, c.presentation, c.full_case); }
  } catch (e) {}
  await Promise.all(visibleDesigns.map(async did => {
    const d = DESIGNS.find(x => x.id === did);
    const rec = await recordingFor(did);
    const body = $("#body-"+d.id);
    if (!rec){ body.innerHTML = `<div class="empty-hint">not recorded yet: run the benchmark for this case</div>`; return; }
    if (!gotCtx){ gotCtx = true; fillContext(id, rec.mode, rec.presentation || rec.abstract, rec.full_case); }
    renderConvo(body, rec, {mode:"plain", abbr:d.abbr});
    const dxT = rec.turns.find(t => t.action === "diagnose") || {};
    const badge = $("#badge-"+d.id);             // verdict, always visible (no scroll)
    badge.className = "jbadge " + (dxT.correct ? "good" : "bad");
    badge.textContent = (dxT.correct ? "✓ " : "✗ ") + (dxT.judge_score != null ? `Judge ${dxT.judge_score}/5` : "");
  }));
  renderQuadAim();
}

/* ---------- Quadruple-Aim review (one shared section, below the panels) ---------- */
function quadStats(rec){
  const last = rec.turns[rec.turns.length-1] || {};
  const dxT = rec.turns.find(t => t.action === "diagnose") || {};
  const mins = totalMinutes(rec);
  const tests = last.n_tests || 0;
  const burden = tests + mins/60;                 // workup load on the patient
  return {correct: !!dxT.correct, score: dxT.judge_score, dx: dxT.query || "-",
          reason: dxT.judge_reason || "-",
          turns: rec.turns.length, tests, cost: last.total_cost || 0, mins,
          burden: burden < 1.5 ? "Low" : burden < 3.5 ? "Moderate" : "High"};
}
function renderQuadAim(){
  const rows = visibleDesigns.map(did => ({d: DESIGNS.find(x => x.id === did), rec: recCache[did]})).filter(x => x.rec);
  const box = $("#quadReview");
  if (rows.length < 2){ box.classList.add("hidden"); return; }   // need ≥2 agents to compare
  const stats = rows.map(x => ({label: x.d.label, ...quadStats(x.rec)}));
  const min = k => Math.min(...stats.map(s => s[k]));
  const best = {turns: min("turns"), tests: min("tests"), cost: min("cost"), mins: min("mins")};
  const cell = (v, isBest) => `<td class="num${isBest ? " best" : ""}">${v}</td>`;
  const wins = stats.filter(s => s.correct);                     // best overall = correct diagnosis at lowest cost
  const bestOverall = wins.length ? wins.reduce((a, b) => b.cost < a.cost ? b : a) : null;
  const gold = rows[0].rec.true_dx || "";
  let h = `<div class="qr-head"><h3>Quadruple-Aim Review</h3>`+
    `<span class="qr-sub">across four aims: better health · clinician effort · patient burden · resource use</span></div>`+
    (gold ? `<div class="qr-gold"><span>Gold diagnosis</span>${esc(gold)}</div>` : "")+
    `<table class="qr-table"><thead><tr><th>Agent</th>`+
    `<th>Diagnosis</th>`+
    `<th>Judge ${TIP("LLM judge (SDBench rubric): scores the final diagnosis 1–5 vs the case's gold diagnosis. ≥4 = correct.")}</th>`+
    `<th class="num">Turns</th>`+
    `<th class="num">Tests</th>`+
    `<th class="num">Cost ${TIP("SDBench costing: $300 per physician visit (a burst of questions) + each test priced by category from a standard price table (labs ~$30, culture ~$80, CT ~$300, MRI ~$600, biopsy/marrow ~$600, PCR/genetic ~$400), summed.")}</th>`+
    `<th class="num">Est. time ${TIP("Synthetic estimate (not from data): question 5 min · diagnosis 2 min · imaging 55 min · culture 120 min · other test 30 min, summed over turns.")}</th>`+
    `<th>Patient exp. ${TIP("Derived burden = tests + (est. time in hours): Low / Moderate / High.")}</th></tr></thead><tbody>`;
  stats.forEach(s => {
    const sc = s.score != null ? `${s.score}/5 ` : "";
    h += `<tr${s === bestOverall ? ' class="qr-win"' : ''}><td class="agent">${s === bestOverall ? '★ ' : ''}${s.label}</td>`+
      `<td>${esc(s.dx)}</td>`+
      `<td class="${s.correct ? "ok" : "no"}">${sc}${s.correct ? "✓" : "✗"}`+
        `<span class="info" data-tip="${esc(s.reason)}">ⓘ</span></td>`+
      cell(s.turns, s.turns === best.turns) +
      cell(s.tests, s.tests === best.tests) +
      `<td class="num${s.cost === best.cost ? " best" : ""}">$${s.cost}</td>`+
      `<td class="num${s.mins === best.mins ? " best" : ""}">${fmtDur(s.mins)}</td>`+
      `<td><span class="burden ${s.burden.toLowerCase()}">${s.burden}</span></td></tr>`;
  });
  box.innerHTML = h + `</tbody></table>`;
  box.classList.remove("hidden");
}

/* ---------- Benchmark: clinician live turns (with decision-time capture) ---------- */
$("#sendBtn").onclick = async () => {
  const q = $("#driveInput").value.trim(); if (!q) return;
  const action = $("#actSel").value;
  const latency_ms = clinTurnStart ? Date.now() - clinTurnStart : null;
  const h = $("#chat .empty-hint"); if (h) h.remove();
  const t = await post("/api/manual", {action, query:q});
  if (t.error){ bubble($("#chat"),"gk","GK","⚠ "+t.error); return; }
  $("#driveInput").value = "";
  // record this clinician turn for the study (action, decision time, outcome)
  logEvent("/api/clin_event", {case_id:currentCaseId, action, query:q, latency_ms,
                               turn:t.turn, total_cost:t.total_cost, correct:t.correct, judge_score:t.judge_score});
  clinTurnStart = Date.now();
  // render this single turn into the live chat
  if (t.action === "diagnose"){
    bubble($("#chat"),"human","You",`<div class="act">diagnosis</div><div class="headline">${esc(t.query)}</div>`);
    judgeBubble($("#chat"), t);
  } else {
    bubble($("#chat"),"human","You",`<div class="act">${t.action==="ask"?"asks":"orders"}</div><div class="headline">${esc(t.query)}</div>`);
    bubble($("#chat"),"gk","GK",esc(t.finding));
  }
  $("#chat").lastChild.scrollIntoView({behavior:"smooth", block:"end"});
};
$("#driveInput").onkeydown = e => { if (e.key==="Enter") $("#sendBtn").click(); };

async function recordingFor(design){
  if (recCache[design]) return recCache[design];
  recCache[design] = await getj(`/api/recording?case=${currentCaseId}&design=${design}${mparam()}`);
  return recCache[design];
}
function designPicker(sel, onPick){
  return `<div class="dpick">Agent: ${DESIGNS.map(d=>`<button data-d="${d.id}" class="${d.id===sel?'on':''}">${d.label}</button>`).join("")}</div>`;
}

/* ---------- GRADING — clinician grades each recommendation, blind to the auto-scorer ---------- */
const NOHARM_LEGEND =
  `<details class="rubric-box"><summary>Grading Rubric: NOHARM (Appropriateness + Harm)</summary>`+
  `<p class="rubric-intro">Adapted from NOHARM (Wu et al., 2025): a modified RAND/UCLA appropriateness rating combined with WHO harm severity. Each recommended action is graded on two independent axes. Note: omission harm (needed actions the agent did <em>not</em> recommend) is out of scope here, since only recommended actions are shown.</p>`+
  `<div class="rubric-band a"><div class="rb-h"><b>Appropriateness</b><span>is this action appropriate for this patient?</span></div>`+
    `<div class="rb-rows"><span><b>Appropriate</b>: benefit outweighs harm</span><span><b>Inappropriate</b>: harm outweighs benefit</span></div></div>`+
  `<div class="rubric-band i"><div class="rb-h"><b>Harm if performed</b><span>WHO severity, rated only when inappropriate</span></div>`+
    `<div class="rb-rows"><span><b>Mild</b></span><span><b>Moderate</b></span><span><b>Severe</b></span></div></div>`+
  `</details>`;

// Blind two-axis grade (no auto-scorer shown → no anchoring). Appropriateness first; harm severity
// only if inappropriate (commission harm). A derived 1–9 keeps the old numeric column populated.
function gradeCard(box, o, onGraded){
  const card = document.createElement("div"); card.className = "grade-inline pending";
  card.innerHTML = `<div class="gi-head"><b>${esc(o.name)}</b></div>`+
    `<div class="gi-row"><span class="gi-k">Appropriateness</span>`+
      `<div class="gi-opts" data-axis="appr"><button data-v="appropriate">Appropriate</button><button data-v="inappropriate">Inappropriate</button></div></div>`+
    `<div class="gi-row gi-harm hidden"><span class="gi-k">Harm if performed</span>`+
      `<div class="gi-opts" data-axis="harm"><button data-v="mild">Mild</button><button data-v="moderate">Moderate</button><button data-v="severe">Severe</button></div></div>`;
  box.appendChild(card);
  card.scrollIntoView({behavior:"smooth", block:"nearest"});
  const shownAt = Date.now();
  let appr = null, harm = null, done = false;
  const finish = () => {
    if (done) return; done = true; card.classList.remove("pending"); card.classList.add("done");
    card.querySelector(".gi-head").insertAdjacentHTML("beforeend", ` <span class="gi-check">✓</span>`);
    const gradeNum = appr === "appropriate" ? 8 : ({mild:3, moderate:2, severe:1}[harm] || 5);
    logEvent("/api/grade", {case_id: currentCaseId, design: evalDesign, test: o.name,
                            grade: gradeNum, appropriateness: appr, harm: harm, latency_ms: Date.now() - shownAt});
    onGraded && onGraded();
  };
  card.querySelectorAll('[data-axis="appr"] button').forEach(b => b.onclick = () => {
    if (done) return;
    appr = b.dataset.v;
    card.querySelectorAll('[data-axis="appr"] button').forEach(x => x.classList.toggle("sel", x === b));
    if (appr === "inappropriate") card.querySelector(".gi-harm").classList.remove("hidden");
    else finish();
  });
  card.querySelectorAll('[data-axis="harm"] button').forEach(b => b.onclick = () => {
    if (done) return;
    harm = b.dataset.v;
    card.querySelectorAll('[data-axis="harm"] button').forEach(x => x.classList.toggle("sel", x === b));
    finish();
  });
}
/* Evaluate = the encounter as it unfolds — system → agent → gatekeeper; every recommended
   test must be graded before the rest of the conversation is revealed. Prior turns stay in
   view so harm can be judged in context. */
async function renderEvaluate(){
  const box = $("#scoreBox");
  const rec = await recordingFor(evalDesign);
  if (!rec){ box.innerHTML = `<div class="empty-hint">Roll a case in Recommend first, then grade its recommendations here.</div>`; return; }
  box.innerHTML = designPicker(evalDesign) +
    `<p class="hint" style="margin:8px 2px 14px">The encounter unfolds as you grade: score each recommended test before the next turn appears.</p>`+
    NOHARM_LEGEND;
  box.querySelectorAll(".dpick button").forEach(b => b.onclick = () => { evalDesign=b.dataset.d; renderEvaluate(); });
  const conv = document.createElement("div"); conv.className = "chat"; box.appendChild(conv);
  // system: the case as presented to the agent
  const sys = document.createElement("div"); sys.className = "msg system";
  sys.innerHTML = `<div class="avatar">EHR</div><div class="bubble"><div class="act">presentation</div>`+
    `<div class="headline" style="font-weight:400">${esc(rec.abstract||"")}</div></div>`;
  conv.appendChild(sys);

  const abbr = ABBR[evalDesign];
  const graded = new Set();                             // grade each unique test once (agents sometimes re-order)
  let clock = 9*60;
  const step = i => {
    if (i >= rec.turns.length) return;
    const t = rec.turns[i];
    clock += durationFor(t); timeLabel(conv, fmtClock(clock));
    if (t.action === "diagnose"){
      bubble(conv, "agent", abbr, `<div class="act">diagnosis</div><div class="headline">${esc(t.query)}</div>` + metaBlock(t));
      judgeBubble(conv, t);
    } else if (t.action === "ask"){
      bubble(conv, "agent", abbr, `<div class="act">asks</div><div class="headline">${esc(t.query)}</div>` + metaBlock(t));
      bubble(conv, "gk", "GK", esc(t.finding));
      step(i+1);
    } else { // recommends — grade each test BLIND, then reveal the gatekeeper, then continue
      let inner = `<div class="act">recommends</div><div class="headline">${esc(t.query)}</div>` + metaBlock(t);
      if (t.differential && t.differential.length)
        inner += `<div class="diff"><span class="diff-lab">differential</span>`+
          t.differential.slice(0,3).map(d=>`<span class="chip">${esc(d.dx)}${d.p!=null?` · ${Math.round(d.p*100)}%`:""}</span>`).join("")+`</div>`;
      bubble(conv, "agent", abbr, inner);
      const orders = t.orders || [{name:t.query, finding:t.finding, cost:t.cost}];
      const gradeOne = k => {                          // one test at a time: grade → reveal its result → next
        if (k >= orders.length) return step(i+1);
        const o = orders[k];
        if (graded.has(o.name)){                        // already graded earlier → just reveal, don't re-grade
          bubble(conv, "gk", "GK", esc(o.finding));
          return gradeOne(k+1);
        }
        graded.add(o.name);
        gradeCard(conv, o, () => {
          bubble(conv, "gk", "GK", esc(o.finding));
          gradeOne(k+1);
        });
      };
      gradeOne(0);
    }
  };
  step(0);
}

/* ---------- DEPLOY — clinician in the loop, order / defer / reject ---------- */
let depLog = [], depTimers = [], depTotal = 0, depDelivered = 0, depResponded = 0, patients = {};

// Each patient is a tab that pops in when their first alert arrives; the tab carries a badge of
// unhandled alerts. Alerts keep arriving on the clock for any patient, so several can be open at once.
function ensurePatient(cid, abstract){
  if (patients[cid]) return patients[cid];
  const tab = document.createElement("button"); tab.className = "sim-tab arrived"; tab.dataset.cid = cid;
  tab.innerHTML = `<span class="st-name">${esc(caseLabel(cid))}</span><span class="st-badge"></span>`;
  tab.onclick = () => activatePatient(cid);
  $("#simTabs").appendChild(tab);
  setTimeout(() => tab.classList.remove("arrived"), 1600);
  const pane = document.createElement("div"); pane.className = "sim-pane chat hidden"; pane.dataset.cid = cid;
  const p = document.createElement("div"); p.className = "msg system";
  p.innerHTML = `<div class="avatar">${cid.split("-")[0].slice(0,3).toUpperCase()}</div>`+
    `<div class="bubble"><div class="act">${esc(caseLabel(cid))} · new patient</div>`+
    `<div class="headline" style="font-weight:400">${esc(abstract||"")}</div></div>`;
  pane.appendChild(p); $("#simPanes").appendChild(pane);
  patients[cid] = {tab, pane, pending: 0};
  if (Object.keys(patients).length === 1) activatePatient(cid);   // auto-open the first patient
  return patients[cid];
}
function activatePatient(cid){
  $$("#simTabs .sim-tab").forEach(t => t.classList.toggle("on", t.dataset.cid === cid));
  $$("#simPanes .sim-pane").forEach(p => p.classList.toggle("hidden", p.dataset.cid !== cid));
}
function setBadge(cid){
  const pt = patients[cid]; if (!pt) return;
  pt.tab.querySelector(".st-badge").textContent = pt.pending || "";
  pt.tab.classList.toggle("urgent", pt.pending > 0);
}
function deployAlertCard(pt, a){
  const card = document.createElement("div"); card.className = "alert-card pending";
  card.innerHTML = `<div class="ac-top"><span class="ac-tag">⚠ interruptive alert</span><span class="ac-time">${fmtClock(a.time)}</span></div>`+
    `<div class="ac-test">${esc(a.name)}</div>`+
    `<div class="ac-q">Recommended by ${a.abbr}. Order, defer, or reject?</div>`+
    `<div class="ac-actions"><button class="oc-order">Order</button><button class="oc-defer">Defer</button><button class="oc-reject">Reject</button></div>`;
  pt.pane.appendChild(card); if (!pt.pane.classList.contains("hidden")) card.scrollIntoView({behavior:"smooth", block:"nearest"});
  const shownAt = Date.now();
  const decide = choice => {
    depLog.push({test:a.name, choice});
    card.classList.remove("pending");
    card.querySelector(".ac-actions").innerHTML = `<span class="decided ${choice}">${choice==="order"?"ordered":choice+"ed"}</span>`;
    if (choice === "order") bubble(pt.pane, "gk", "GK", `${esc(a.finding)}`);
    logEvent("/api/sim_event", {case_id:a.cid, test:a.name, choice, latency_ms:Date.now()-shownAt,
                                shift_min:a.time, seq:a.seq, fatigue:depResponded});
    depResponded++; pt.pending--; setBadge(a.cid); updateStat(); checkDone();
  };
  card.querySelector(".oc-order").onclick  = () => decide("order");
  card.querySelector(".oc-defer").onclick  = () => decide("defer");
  card.querySelector(".oc-reject").onclick = () => decide("reject");
}
function deliverAlert(a){
  const pt = ensurePatient(a.cid, a.abstract);
  timeLabel(pt.pane, fmtClock(a.time));
  bubble(pt.pane, "agent", a.abbr, `<div class="act">recommends</div><div class="headline">${esc(a.name)}</div>`);
  deployAlertCard(pt, a);
  pt.pending++; setBadge(a.cid);
}
function updateStat(){
  const s = $("#depStatus"); if (!s) return;
  const o = depLog.filter(x=>x.choice==="order").length;
  const d = depLog.filter(x=>x.choice==="defer").length;
  const r = depLog.filter(x=>x.choice==="reject").length;
  const open = Object.values(patients).reduce((n,p)=>n+(p.pending>0?1:0),0);
  const live = depDelivered < depTotal;
  s.innerHTML = `<span class="ss-live ${live?'':'done'}">●</span> ${live?'live':'shift over'} · `+
    `<b>${depResponded}/${depTotal}</b> handled · ${open} patient${open===1?'':'s'} waiting · `+
    `<span class="ok">${o} ordered</span> · ${d} deferred · <span class="no">${r} rejected</span>`;
}
function checkDone(){
  if (depDelivered >= depTotal && depResponded >= depTotal && !$("#depDone")){
    const o = depLog.filter(x=>x.choice==="order").length;
    const card = document.createElement("div"); card.className = "scard"; card.id = "depDone"; card.style.marginTop = "16px";
    card.innerHTML = `<div class="stitle">Shift Complete</div>`+
      `<p class="why">${depLog.length} alerts across ${Object.keys(patients).length} patients · <span class="ok">${o} ordered</span> · `+
      `${depLog.filter(x=>x.choice==="defer").length} deferred · <span class="no">${depLog.filter(x=>x.choice==="reject").length} rejected</span></p>`+
      `<p class="hint">Every recommendation interrupted you regardless of value: exactly the burden adaptive delivery learns to avoid.</p>`;
    $("#simSummary").appendChild(card); card.scrollIntoView({behavior:"smooth", block:"nearest"});
  }
}
async function renderDeploy(){
  depTimers.forEach(clearTimeout); depTimers = []; patients = {};
  const box = $("#deployBox");
  box.innerHTML = designPicker(deployDesign) +
    `<div id="depIntro" class="intro-card"><h3>Alert-delivery shift</h3>`+
    `<p>Alerts arrive from several patients over the morning. Each patient opens as its own tab, with a badge for unhandled alerts. Order, defer, or reject each recommendation as it comes.</p>`+
    `<button class="btn primary" id="startShift">▸ Start shift</button></div>`+
    `<div id="depStatus" class="shiftstat hidden"></div>`+
    `<div id="simTabs" class="sim-tabs"></div><div id="simPanes"></div><div id="simSummary"></div>`;
  box.querySelectorAll(".dpick button").forEach(b => b.onclick = () => { depTimers.forEach(clearTimeout); deployDesign=b.dataset.d; renderDeploy(); });

  // patients = recorded cases (so they actually have recommendations to deliver)
  const avail = new Set([...$("#caseSel").options].map(o=>o.value).filter(Boolean));
  const ids = (window.__RECORDED__ || []).filter(id => avail.has(id)).slice(0, 5);
  const got = await Promise.all(ids.map(cid => getj(`/api/recording?case=${cid}&design=${deployDesign}${mparam()}`).then(r=>({cid,r}))));
  const abbr = ABBR[deployDesign];
  const alerts = [];
  got.forEach(({cid,r}, pi) => {
    if (!r) return;
    let clock = 9*60 + 4 + pi*13;
    r.turns.forEach(t => {
      clock += durationFor(t);
      if (t.action !== "order") return;
      (t.orders || [{name:t.query, finding:t.finding}]).forEach(o => {
        clock += 6; alerts.push({cid, abstract:r.abstract, abbr, time:clock, name:o.name, finding:o.finding});
      });
    });
  });
  alerts.sort((x,y) => x.time - y.time);
  alerts.forEach((a, i) => a.seq = i);

  $("#startShift").onclick = () => {
    $("#depIntro")?.remove();
    $("#depStatus").classList.remove("hidden");
    $("#simTabs").innerHTML = ""; $("#simPanes").innerHTML = ""; $("#simSummary").innerHTML = "";
    depLog = []; patients = {}; depTotal = alerts.length; depDelivered = 0; depResponded = 0;
    updateStat();
    if (!alerts.length){ $("#simPanes").innerHTML = `<div class="empty-hint">No recordings to run a shift on yet.</div>`; return; }
    const GAP = 2400;
    alerts.forEach((a, i) => {
      const id = setTimeout(() => { deliverAlert(a); depDelivered++; updateStat(); checkDone(); }, i*GAP + (i*53)%700);
      depTimers.push(id);
    });
  };
}

