(() => {
  const visEl  = document.querySelector(".visualization");
  const descEl = document.querySelector(".node_desc");
  const normalizeEdgeList = (edgeLike) => {
    if (!Array.isArray(edgeLike)) return [];
    return edgeLike
      .map((e) => {
        if (Array.isArray(e) && e.length >= 2) {
          return { from: String(e[0]), to: String(e[1]) };
        }
        if (e && typeof e === "object") {
          return { from: String(e.from ?? e.source), to: String(e.to ?? e.target) };
        }
        return null;
      })
      .filter((e) => e && e.from && e.to);
  };
  const toObjectMap = (value) => {
    if (!value) return {};
    if (typeof value === "string") {
      try {
        return toObjectMap(JSON.parse(value));
      } catch (_) {
        return {};
      }
    }
    if (Array.isArray(value)) {
      const out = {};
      for (const item of value) {
        if (!item || typeof item !== "object") continue;
        const key = item.node ?? item.id ?? item.name;
        const params = item.params ?? item.value ?? item.beta ?? item;
        if (key !== undefined) out[String(key)] = params;
      }
      return out;
    }
    if (typeof value === "object") return value;
    return {};
  };
  const resolveBetaParams = (obj) => {
    const directCandidates = [
      obj?.beta_parameters,
      obj?.global_betas,
      obj?.betas,
      obj?.parameters,
      obj?.params,
      obj?.model_params,
      obj?.final_global_params,
      obj?.final_local_params,
    ];
    for (const c of directCandidates) {
      const mapped = toObjectMap(c);
      if (Object.keys(mapped).length) return mapped;
    }
    const nestedCandidates = [obj?.network, obj?.model, obj?.result, obj?.payload];
    for (const n of nestedCandidates) {
      if (!n || typeof n !== "object") continue;
      const mapped = resolveBetaParams(n);
      if (Object.keys(mapped).length) return mapped;
    }
    return {};
  };
  const raw = sessionStorage.getItem("bayestitch_network");
  if (!raw) {
    descEl.innerHTML = `<div style="opacity:.6;font-size:14px;">No network found. Go back and upload a JSON/CSV first.</div>`;
    return;
  }
  const bn = JSON.parse(raw);
  const betaParams = resolveBetaParams(bn);
  const edges = normalizeEdgeList(
    bn?.dag?.edges ?? bn.dag_edges ?? bn.global_dag_edges ?? bn.edges ?? []
  );
  const baseNodes = Array.isArray(bn.nodes) ? bn.nodes : [];
  const nodeIds = new Set(baseNodes.map((n) => String(n.id ?? n.name ?? n.label)));
  edges.forEach((e) => {
    nodeIds.add(e.from);
    nodeIds.add(e.to);
  });
  Object.keys(betaParams || {}).forEach((nid) => nodeIds.add(String(nid)));
  const nodes = [...nodeIds].map((id) => {
    const existing = baseNodes.find((n) => String(n.id ?? n.name ?? n.label) === id) || {};
    const beta = betaParams?.[id];
    return {
      id: String(id),
      label: existing.label ?? existing.id ?? id,
      states: Array.isArray(existing.states)
        ? existing.states.map(String)
        : (Array.isArray(beta?.classes) ? beta.classes.map(String) : undefined),
      description: existing.description ?? "",
    };
  });
  const byId       = new Map(nodes.map((n) => [n.id, n]));
  const parentsOf  = new Map(nodes.map((n) => [n.id, []]));
  const childrenOf = new Map(nodes.map((n) => [n.id, []]));
  for (const e of edges) {
    if (parentsOf.has(e.to))    parentsOf.get(e.to).push(e.from);
    if (childrenOf.has(e.from)) childrenOf.get(e.from).push(e.to);
  }
  function escapeHtml(s) {
    return String(s)
      .replaceAll("&",  "&amp;")
      .replaceAll("<",  "&lt;")
      .replaceAll(">",  "&gt;")
      .replaceAll('"',  "&quot;")
      .replaceAll("'",  "&#039;");
  }
  function parseCombo(combo) {
    if (combo.includes("|") || (combo.includes("=") && !combo.includes(","))) {
      return combo.split("|").map((part) => {
        const eqIdx = part.indexOf("=");
        return eqIdx !== -1 ? part.slice(eqIdx + 1).trim() : part.trim();
      });
    }
    return combo.split(",").map((v) => v.trim());
  }
  function asArray(v) { return Array.isArray(v) ? v : (typeof v === "number" ? [v] : []); }
  function getNodeStates(nodeId) {
    const n = byId.get(nodeId);
    if (Array.isArray(n?.states) && n.states.length) return n.states.map(String);
    const cpt = bn.cpts?.[nodeId];
    if (Array.isArray(cpt?.states) && cpt.states.length) return cpt.states.map(String);
    const b = betaParams?.[nodeId];
    if (Array.isArray(b?.classes) && b.classes.length) return b.classes.map(String);
    const inter = asArray(b?.intercept);
    if (inter.length > 1) return inter.map((_, i) => String(i));
    return ["0", "1"];
  }
  function defaultFill(nodeId) {
    return bn.cpts?.[nodeId] ? "#fbc5bb" : "#e0e0e0";
  }
  function cptLookup(nodeId, stateVal, assignment) {
    const cpt = bn.cpts?.[nodeId];
    if (!cpt || !cpt.cpt) return null;
    const parents = Array.isArray(cpt.parents) ? cpt.parents : [];
    const states = Array.isArray(cpt.states) ? cpt.states : getNodeStates(nodeId);
    let prob = null;
    for (const [combo, probs] of Object.entries(cpt.cpt)) {
      const vals = parseCombo(combo);
      const matches = parents.every((pid, i) => {
        const av = assignment[pid];
        return av === undefined || String(vals[i]) === String(av);
      });
      if (matches || parents.length === 0) {
        const stateIdx = states.indexOf(String(stateVal));
        if (Array.isArray(probs)) {
          prob = stateIdx >= 0 ? +probs[stateIdx] : null;
        } else if (typeof probs === "object" && probs !== null) {
          prob = probs[stateVal] !== undefined ? +probs[stateVal] : null;
        } else {
          prob = stateIdx === 0 ? +probs : (1 - +probs);
        }
        if (parents.length === 0 || matches) break;
      }
    }
    return prob;
  }
  function normalizeProbabilities(probs, expectedLen) {
    const arr = asArray(probs).map((v) => +v);
    const safe = arr.filter((v) => Number.isFinite(v) && v >= 0);
    if (!safe.length) return Array.from({ length: expectedLen }, () => 1 / expectedLen);
    let out = safe;
    if (safe.length !== expectedLen) {
      if (safe.length === 1 && expectedLen === 2) out = [1 - safe[0], safe[0]];
      else if (safe.length < expectedLen) out = [...safe, ...Array.from({ length: expectedLen - safe.length }, () => 0)];
      else out = safe.slice(0, expectedLen);
    }
    const sum = out.reduce((a, b) => a + b, 0);
    if (sum <= 0) return Array.from({ length: expectedLen }, () => 1 / expectedLen);
    return out.map((v) => v / sum);
  }
  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((v) => v / sum);
  }
  function encodeValueForParent(parentId, value) {
    if (value === undefined || value === null) return 0;
    if (typeof value === "number") return value;
    const n = Number(value);
    if (Number.isFinite(n)) return n;
    const states = getNodeStates(parentId);
    const idx = states.indexOf(String(value));
    return idx >= 0 ? idx : 0;
  }
  function predictNodeDistribution(nodeId, assignment) {
    const states = getNodeStates(nodeId);
    const b = betaParams?.[nodeId];
    if (!b) return normalizeProbabilities([], states.length);
    if (!Array.isArray(b.coefficients)) {
      return normalizeProbabilities(b.intercept, states.length);
    }
    const parents = Array.isArray(b.parents) ? b.parents : [];
    const coeff = b.coefficients.map((row) => asArray(row).map((v) => +v));
    const inter = asArray(b.intercept).map((v) => +v);
    const x = parents.map((pid) => encodeValueForParent(pid, assignment[pid]));
    if (coeff.length === 1) {
      const row = coeff[0];
      const score = row.reduce((sum, c, i) => sum + c * (x[i] ?? 0), 0) + (inter[0] ?? 0);
      const p1 = 1 / (1 + Math.exp(-score));
      return normalizeProbabilities([1 - p1, p1], states.length);
    }
    const logits = coeff.map((row, r) =>
      row.reduce((sum, c, i) => sum + c * (x[i] ?? 0), 0) + (inter[r] ?? 0)
    );
    return normalizeProbabilities(softmax(logits), states.length);
  }
  function getStateProbability(nodeId, stateVal, assignment) {
    const cptProb = cptLookup(nodeId, stateVal, assignment);
    if (cptProb !== null && Number.isFinite(cptProb)) return cptProb;
    const states = getNodeStates(nodeId);
    const probs = predictNodeDistribution(nodeId, assignment);
    const idx = states.indexOf(String(stateVal));
    if (idx < 0) return 0;
    return probs[idx] ?? 0;
  }
  function topologicalNodeIds() {
    const ids = nodes.map((n) => n.id);
    const indeg = new Map(ids.map((id) => [id, 0]));
    for (const e of edges) indeg.set(e.to, (indeg.get(e.to) ?? 0) + 1);
    const q = ids.filter((id) => (indeg.get(id) ?? 0) === 0);
    const out = [];
    while (q.length) {
      const u = q.shift();
      out.push(u);
      for (const c of childrenOf.get(u) ?? []) {
        indeg.set(c, (indeg.get(c) ?? 0) - 1);
        if ((indeg.get(c) ?? 0) === 0) q.push(c);
      }
    }
    return out.length === ids.length ? out : ids;
  }
  function jointProb(assignment) {
    let p = 1;
    for (const nodeId of topologicalNodeIds()) {
      const prob = getStateProbability(nodeId, assignment[nodeId], assignment);
      if (!Number.isFinite(prob)) return null;
      p *= prob;
    }
    return p;
  }
  function enumerateAll(hiddenIds, fixedEvidence) {
    const results = [];
    function recurse(idx, current) {
      if (idx === hiddenIds.length) { results.push({ ...current }); return; }
      for (const s of getNodeStates(hiddenIds[idx])) {
        current[hiddenIds[idx]] = s;
        recurse(idx + 1, current);
      }
    }
    recurse(0, { ...fixedEvidence });
    return results;
  }
  function inferByEnumeration(queryId, evidence) {
    const queryStates = getNodeStates(queryId);
    const hiddenIds   = nodes.map((n) => n.id).filter((id) => id !== queryId && evidence[id] === undefined);
    const result = {};
    for (const qs of queryStates) {
      const fullEvidence = { ...evidence, [queryId]: qs };
      let total = 0;
      for (const assignment of enumerateAll(hiddenIds, fullEvidence)) {
        const jp = jointProb(assignment);
        if (jp === null) return null;
        total += jp;
      }
      result[qs] = total;
    }
    const Z = Object.values(result).reduce((a, b) => a + b, 0);
    if (Z === 0) return null;
    for (const s of queryStates) result[s] = result[s] / Z;
    return result;
  }
  visEl.innerHTML = "";
  visEl.style.position = "relative";
  const W = Math.max(520, visEl.clientWidth  || 520);
  const H = Math.max(420, visEl.clientHeight || 420);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width",  String(W));
  svg.setAttribute("height", String(H));
  svg.style.cssText = "width:100%;height:100%;display:block;cursor:grab;";
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  function makeMarker(id, color) {
    const m = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    m.setAttribute("id", id); m.setAttribute("markerWidth", "10");
    m.setAttribute("markerHeight", "10"); m.setAttribute("refX", "10");
    m.setAttribute("refY", "3"); m.setAttribute("orient", "auto");
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", "M0,0 L10,3 L0,6 Z"); p.setAttribute("fill", color);
    m.appendChild(p); return m;
  }
  defs.appendChild(makeMarker("arrow-gray",   "gray"));
  defs.appendChild(makeMarker("arrow-black",  "black"));
  defs.appendChild(makeMarker("arrow-black", "black"));
  svg.appendChild(defs);
  const mainGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  svg.appendChild(mainGroup);
  const cx = W / 2, cy = H / 2;
  const R  = Math.min(W, H) * 0.35;
  const NODE_R = 25;
  const positions = new Map();
  nodes.forEach((n, i) => {
    const a = (i / Math.max(1, nodes.length)) * Math.PI * 2;
    positions.set(n.id, { x: cx + Math.cos(a) * R, y: cy + Math.sin(a) * R });
  });
  const edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  mainGroup.appendChild(edgeGroup);
  function edgeCoords(fromId, toId) {
    const p1 = positions.get(fromId), p2 = positions.get(toId);
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const dist = Math.sqrt(dx*dx + dy*dy) || 1;
    const ux = dx/dist, uy = dy/dist;
    return { x1: p1.x+ux*NODE_R, y1: p1.y+uy*NODE_R, x2: p2.x-ux*NODE_R, y2: p2.y-uy*NODE_R };
  }
  for (const e of edges) {
    if (!positions.has(e.from) || !positions.has(e.to)) continue;
    const { x1, y1, x2, y2 } = edgeCoords(e.from, e.to);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(x1)); line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(x2)); line.setAttribute("y2", String(y2));
    line.setAttribute("stroke", "gray"); line.setAttribute("stroke-width", "1");
    line.setAttribute("marker-end", "url(#arrow-gray)");
    line.dataset.from = e.from; line.dataset.to = e.to;
    edgeGroup.appendChild(line);
  }
  const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  mainGroup.appendChild(nodeGroup);
  const nodeEls = new Map();
  for (const n of nodes) {
    const p = positions.get(n.id);
    if (!p) continue;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.style.cursor = "pointer"; g.dataset.node = n.id;
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", String(p.x)); circle.setAttribute("cy", String(p.y));
    circle.setAttribute("r",  String(NODE_R));
    circle.setAttribute("fill", defaultFill(n.id));
    circle.setAttribute("stroke", "gray"); circle.setAttribute("stroke-width", "1");
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", String(p.x)); text.setAttribute("y", String(p.y + 5));
    text.setAttribute("text-anchor", "middle"); text.setAttribute("font-size", "14");
    text.setAttribute("font-family", "Ubuntu Condensed, sans-serif");
    text.setAttribute("pointer-events", "none");
    text.textContent = (n.label ?? n.id).slice(0, 12);
    g.appendChild(circle); g.appendChild(text);
    g.addEventListener("click", () => { if (!didDrag) setQueryNode(n.id); });
    nodeGroup.appendChild(g);
    nodeEls.set(n.id, { g, circle, text });
  }
  visEl.appendChild(svg);
  let viewX = 0, viewY = 0, viewScale = 1;
  const applyTransform = () => {
    mainGroup.setAttribute("transform", `translate(${viewX},${viewY}) scale(${viewScale})`);
  };
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = svg.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    viewX = mx - (mx - viewX) * delta;
    viewY = my - (my - viewY) * delta;
    viewScale = Math.min(Math.max(viewScale * delta, 0.2), 5);
    applyTransform();
  }, { passive: false });
  let dragId = null, dragOffX = 0, dragOffY = 0;
  let isPanning = false, panStartX = 0, panStartY = 0;
  let didDrag = false;
  const svgPoint = (cx, cy) => {
    const rect = svg.getBoundingClientRect();
    return { x: (cx - rect.left - viewX) / viewScale, y: (cy - rect.top - viewY) / viewScale };
  };
  svg.addEventListener("mousedown", (e) => {
    didDrag = false;
    const nt = e.target.closest("[data-node]");
    if (nt) {
      dragId = nt.dataset.node;
      const pos = positions.get(dragId), sp = svgPoint(e.clientX, e.clientY);
      dragOffX = sp.x - pos.x; dragOffY = sp.y - pos.y;
      nt.style.cursor = "grabbing"; e.stopPropagation();
    } else {
      isPanning = true; panStartX = e.clientX - viewX; panStartY = e.clientY - viewY;
      svg.style.cursor = "grabbing";
    }
  });
  window.addEventListener("mousemove", (e) => {
    if (dragId) {
      didDrag = true;
      const sp = svgPoint(e.clientX, e.clientY);
      const x = sp.x - dragOffX, y = sp.y - dragOffY;
      positions.set(dragId, { x, y });
      const el = nodeEls.get(dragId);
      el.circle.setAttribute("cx", String(x)); el.circle.setAttribute("cy", String(y));
      el.text.setAttribute("x", String(x));    el.text.setAttribute("y", String(y + 5));
      for (const line of edgeGroup.querySelectorAll("line")) {
        if (line.dataset.from === dragId || line.dataset.to === dragId) {
          const { x1, y1, x2, y2 } = edgeCoords(line.dataset.from, line.dataset.to);
          line.setAttribute("x1", String(x1)); line.setAttribute("y1", String(y1));
          line.setAttribute("x2", String(x2)); line.setAttribute("y2", String(y2));
        }
      }
    } else if (isPanning) {
      didDrag = true;
      viewX = e.clientX - panStartX; viewY = e.clientY - panStartY;
      applyTransform();
    }
  });
  window.addEventListener("mouseup", () => {
    if (dragId) { const el = nodeEls.get(dragId); if (el) el.g.style.cursor = "pointer"; dragId = null; }
    if (isPanning) { isPanning = false; svg.style.cursor = "grab"; }
  });
  const resetBtn = document.createElement("button");
  resetBtn.textContent = "⟳ Reset View";
  resetBtn.style.cssText = `
    position:absolute;bottom:12px;right:12px;z-index:10;
    padding:5px 12px;border-radius:16px;border:1.5px solid rgba(0,0,0,0.3);
    background:rgba(244,241,232,0.95);font-size:12px;cursor:pointer;
    font-family:'Ubuntu Condensed',sans-serif;box-shadow:0 2px 8px rgba(0,0,0,0.12);
  `;
  resetBtn.addEventListener("click", () => { viewX=0; viewY=0; viewScale=1; applyTransform(); });
  visEl.appendChild(resetBtn);
  const hint = document.createElement("div");
  hint.textContent = "Click a node to set as query";
  hint.style.cssText = `
    position:absolute;bottom:12px;left:12px;z-index:10;
    font-size:11px;opacity:.5;font-family:'Ubuntu Condensed',sans-serif;
    pointer-events:none;
  `;
  visEl.appendChild(hint);
  const tooltip = document.createElement("div");
  tooltip.style.cssText = `
    position:fixed;pointer-events:none;z-index:9999;
    background:#1a1a1a;color:#f5f5f5;border-radius:8px;
    padding:10px 14px;font-family:'Ubuntu Condensed',sans-serif;
    font-size:13px;line-height:1.5;max-width:240px;
    box-shadow:0 4px 16px rgba(0,0,0,0.35);
    opacity:0;transition:opacity 0.15s ease;
  `;
  document.body.appendChild(tooltip);
  const positionTooltip = (mx, my) => {
    const pad=14, tw=tooltip.offsetWidth||200, th=tooltip.offsetHeight||70;
    let left=mx+pad, top=my+pad;
    if (left+tw > window.innerWidth-pad)  left = mx-tw-pad;
    if (top+th  > window.innerHeight-pad) top  = my-th-pad;
    tooltip.style.left = left+"px"; tooltip.style.top = top+"px";
  };
  for (const n of nodes) {
    const el = nodeEls.get(n.id);
    if (!el) continue;
    el.g.addEventListener("mouseenter", (e) => {
      const parents  = parentsOf.get(n.id)  ?? [];
      const children = childrenOf.get(n.id) ?? [];
      tooltip.innerHTML = `
        <div style="font-weight:600;margin-bottom:4px;">${escapeHtml(n.label ?? n.id)}</div>
        <div style="opacity:.8;margin-bottom:5px;font-size:12px;">${escapeHtml(n.description || "No description.")}</div>
        <div style="border-top:1px solid rgba(255,255,255,0.15);padding-top:5px;font-size:11px;opacity:.7;">
          <div><strong>Parents:</strong> ${parents.length ? parents.map(escapeHtml).join(", ") : "None"}</div>
          <div><strong>Children:</strong> ${children.length ? children.map(escapeHtml).join(", ") : "None"}</div>
        </div>
      `;
      positionTooltip(e.clientX, e.clientY);
      tooltip.style.opacity = "1";
    });
    el.g.addEventListener("mousemove",  (e) => positionTooltip(e.clientX, e.clientY));
    el.g.addEventListener("mouseleave", ()  => { tooltip.style.opacity = "0"; });
  }
  const clearHighlights = () => {
    for (const [id, { circle }] of nodeEls) {
      circle.setAttribute("fill",         defaultFill(id));
      circle.setAttribute("stroke",       "gray");
      circle.setAttribute("stroke-width", "1");
    }
    for (const line of edgeGroup.querySelectorAll("line")) {
      line.setAttribute("stroke",       "gray");
      line.setAttribute("stroke-width", "1");
      line.setAttribute("marker-end",   "url(#arrow-gray)");
    }
  };
  function highlightFromEvidence(queryId, evidence) {
    clearHighlights();
    const qel = nodeEls.get(queryId);
    if (qel) {
      qel.circle.setAttribute("fill",         "#ffa7a6");
      qel.circle.setAttribute("stroke",       "black");
      qel.circle.setAttribute("stroke-width", "3");
    }
    for (const [obsId] of Object.entries(evidence)) {
      const oel = nodeEls.get(obsId);
      if (oel) {
        oel.circle.setAttribute("stroke",       "black");
        oel.circle.setAttribute("stroke-width", "2.5");
      }
    }
    for (const line of edgeGroup.querySelectorAll("line")) {
      if (line.dataset.to === queryId) {
        line.setAttribute("stroke",       "black");
        line.setAttribute("stroke-width", "2");
        line.setAttribute("marker-end",   "url(#arrow-black)");
      }
    }
  }
  let currentQueryId = nodes.length ? nodes[0].id : null;
  function buildInferencePanel(queryId) {
    currentQueryId = queryId;
    const qs = document.getElementById("query_select");
    if (qs && qs.value !== queryId) qs.value = queryId;
    rebuildEvidenceRows(queryId);
    const resultEl = document.getElementById("inf_result");
    if (resultEl) resultEl.innerHTML = "";
  }
  function rebuildEvidenceRows(queryId) {
    const evidenceRows = document.getElementById("evidence_rows");
    if (!evidenceRows) return;
    evidenceRows.innerHTML = "";
    for (const n of nodes) {
      if (n.id === queryId) continue;
      const states = getNodeStates(n.id);
      const opts   = states.map((s) =>
        `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`
      ).join("");
      const row = document.createElement("div");
      row.className = "ev-row";
      row.innerHTML = `
        <label title="${escapeHtml(n.label ?? n.id)}">${escapeHtml(n.label ?? n.id)}</label>
        <select data-nodeid="${escapeHtml(n.id)}">
          <option value="">— unobserved —</option>
          ${opts}
        </select>
      `;
      row.querySelector("select").addEventListener("change", syncHighlights);
      evidenceRows.appendChild(row);
    }
  }
  function getEvidenceFromPanel() {
    const evidence = {};
    const evidenceRows = document.getElementById("evidence_rows");
    if (!evidenceRows) return evidence;
    for (const sel of evidenceRows.querySelectorAll("select[data-nodeid]")) {
      if (sel.value) evidence[sel.dataset.nodeid] = sel.value;
    }
    return evidence;
  }
  function syncHighlights() {
    if (!currentQueryId) return;
    highlightFromEvidence(currentQueryId, getEvidenceFromPanel());
  }
  function setQueryNode(nodeId) {
    currentQueryId = nodeId;
    buildInferencePanel(nodeId);
    syncHighlights();
  }
  const nodeOptions = nodes.map((n) =>
    `<option value="${escapeHtml(n.id)}">${escapeHtml(n.label ?? n.id)}</option>`
  ).join("");
  descEl.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:14px;height:100%;">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <label style="font-weight:600;font-size:14px;white-space:nowrap;">Query node:</label>
        <select id="query_select" style="
          flex:1;min-width:140px;padding:6px 10px;border-radius:8px;
          border:2px solid black;font-size:14px;
          font-family:'Ubuntu Condensed',sans-serif;
          background:#f4f1e8;cursor:pointer;
        ">${nodeOptions}</select>
      </div>
      <hr style="border:none;border-top:2px solid rgba(0,0,0,0.1);">
      <div style="font-weight:600;font-size:12px;opacity:.6;letter-spacing:.04em;">
        SET OBSERVED STATES &nbsp;<span style="font-weight:400;opacity:.7;"></span>
      </div>
      <div id="evidence_rows" style="
        display:flex;flex-direction:column;gap:5px;
        overflow-y:auto;flex:1;padding-right:4px;
      "></div>
      <hr style="border:none;border-top:2px solid rgba(0,0,0,0.1);">
      <button id="run_btn" style="
        padding:10px 0;border-radius:8px;border:3px solid black;
        background:#1a1a1a;color:white;font-size:15px;
        font-family:'Ubuntu Condensed',sans-serif;cursor:pointer;
        letter-spacing:.05em;transition:opacity 0.15s;flex-shrink:0;
      ">RUN INFERENCE</button>
      <div id="inf_result" style="flex-shrink:0;min-height:60px;"></div>
    </div>
    <style>
      #run_btn:hover { opacity: 0.75; }
      .ev-row {
        display:flex;align-items:center;justify-content:space-between;
        gap:8px;padding:5px 0;border-bottom:1px solid rgba(0,0,0,0.07);
      }
      .ev-row label {
        font-size:13px;flex:1;overflow:hidden;
        text-overflow:ellipsis;white-space:nowrap;
      }
      .ev-row select {
        padding:3px 7px;border-radius:6px;border:1.5px solid rgba(0,0,0,0.22);
        font-size:13px;font-family:'Ubuntu Condensed',sans-serif;
        background:white;cursor:pointer;max-width:130px;
      }
      .res-bar-wrap {
        background:rgba(0,0,0,0.08);border-radius:6px;
        overflow:hidden;height:20px;flex:1;
      }
      .res-bar {
        height:100%;background:#ffa7a6;border-radius:6px;
        transition:width 0.45s ease;
      }
      .res-row { display:flex;align-items:center;gap:8px;margin-bottom:6px; }
      .res-label { width:40px;font-size:13px;font-weight:600;text-align:right; }
      .res-pct   { font-size:13px;width:46px; }
    </style>
  `;
  document.getElementById("query_select").addEventListener("change", (e) => {
    setQueryNode(e.target.value);
  });
  document.getElementById("run_btn").addEventListener("click", () => {
    const queryId  = currentQueryId;
    const evidence = getEvidenceFromPanel();
    const resultEl = document.getElementById("inf_result");
    resultEl.innerHTML = `<div style="opacity:.5;font-size:13px;">Computing…</div>`;
    
    const hiddenCount = nodes.filter(n => n.id !== queryId && !evidence[n.id]).length;
    const avgStates = nodes.reduce((sum, n) => sum + (getNodeStates(n.id).length || 2), 0) / nodes.length;
    const complexity = Math.pow(avgStates, hiddenCount);
    const COMPLEXITY_THRESHOLD = 1e6;
    
    setTimeout(async () => {
      try {
        let result;
        let method = "Exact Enumeration";
        
        if (complexity >= COMPLEXITY_THRESHOLD && typeof window.performVariableElimination === 'function') {
          method = "Algorithm: Variable Elimination";
          console.log(`Using Variable Elimination - complexity: ${complexity.toExponential(2)}`);
          result = window.performVariableElimination(bn, queryId, evidence);
        } else {
          console.log(`Using exact enumeration - complexity: ${complexity.toExponential(2)}`);
          result = inferByEnumeration(queryId, evidence);
        }
        
        const queryNode = byId.get(queryId);
        if (!result) {
          resultEl.innerHTML = `
            <div style="color:#c0392b;font-size:12px;padding:6px 0;">
              Could not compute inference for this selection.
            </div>`;
          return;
        }
        const evidenceCount = Object.keys(evidence).length;
        const condStr = evidenceCount
          ? ` | ${evidenceCount} node${evidenceCount > 1 ? "s" : ""} observed`
          : "";
        const bars = Object.entries(result).map(([state, prob]) => {
          const pct = (prob * 100).toFixed(1);
          return `
            <div class="res-row">
              <div class="res-label">${escapeHtml(state)}</div>
              <div class="res-bar-wrap"><div class="res-bar" style="width:${pct}%"></div></div>
              <div class="res-pct">${pct}%</div>
            </div>
          `;
        }).join("");
        resultEl.innerHTML = `
          <div style="font-size:12px;opacity:.6;margin-bottom:8px;">
            P(<strong>${escapeHtml(queryNode.label ?? queryId)}</strong>${escapeHtml(condStr)})
          </div>
          ${bars}
          <div style="font-size:10px;opacity:.35;margin-top:6px;">
            ${method}
          </div>
        `;
      } catch (error) {
        console.error("Inference error:", error);
        resultEl.innerHTML = `
          <div style="color:#c0392b;font-size:12px;padding:6px 0;">
            Error: ${escapeHtml(error.message || String(error))}
          </div>
        `;
      }
    }, 20);
  });
  if (currentQueryId) {
    buildInferencePanel(currentQueryId);
    syncHighlights();
  }
})();