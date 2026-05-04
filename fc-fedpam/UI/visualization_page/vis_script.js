(() => {
  const visEl  = document.querySelector(".visualization");
  const descEl = document.querySelector(".node_desc");
  const infEl  = document.querySelector(".node_inference");
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
    descEl.textContent = "No network found. Go back and upload a JSON/CSV first.";
    infEl.textContent  = "";
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
  function asArray(v) {
    if (Array.isArray(v)) return v;
    if (typeof v === "number") return [v];
    return [];
  }
  function normalizeProbabilities(probs, expectedLen) {
    const arr = asArray(probs).map((v) => +v);
    const safe = arr.filter((v) => Number.isFinite(v) && v >= 0);
    if (!safe.length) return Array.from({ length: expectedLen }, () => 1 / expectedLen);
    let out = safe;
    if (safe.length !== expectedLen) {
      if (safe.length === 1 && expectedLen === 2) out = [1 - safe[0], safe[0]];
      else if (safe.length < expectedLen) {
        out = [...safe, ...Array.from({ length: expectedLen - safe.length }, () => 0)];
      } else {
        out = safe.slice(0, expectedLen);
      }
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
    const states  = Array.isArray(cpt.states)  ? cpt.states  : getNodeStates(nodeId);
    let prob = null;
    for (const [combo, probs] of Object.entries(cpt.cpt)) {
      const vals = parseCombo(combo);
      const matches = parents.every((pid, i) => {
        const assignedVal = assignment[pid];
        return assignedVal === undefined || String(vals[i]) === String(assignedVal);
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
        if (parents.length === 0) break;
        if (matches) break;
      }
    }
    return prob;
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
  function buildBetaTable(nodeId) {
    const b = betaParams?.[nodeId];
    if (!b) {
      return `<div style="opacity:.65;font-size:13px;">No beta parameters available for this node.</div>`;
    }
    const parents = Array.isArray(b.parents) ? b.parents : [];
    const classes = getNodeStates(nodeId);
    const intercept = asArray(b.intercept);
    const coefficients = Array.isArray(b.coefficients)
      ? b.coefficients.map((r) => asArray(r))
      : [];
    const interRows = intercept.length
      ? intercept.map((v, i) => `<tr><td>${escapeHtml(classes[i] ?? String(i))}</td><td>${(+v).toFixed(4)}</td></tr>`).join("")
      : `<tr><td colspan="2" style="opacity:.7;">No intercept values</td></tr>`;
    const coefHeader = parents.length
      ? parents.map((p) => `<th>${escapeHtml(p)}</th>`).join("")
      : "<th>—</th>";
    const coefRows = coefficients.length
      ? coefficients.map((row, i) => {
          const cells = parents.length
            ? parents.map((_, j) => `<td>${((+row[j]) || 0).toFixed(4)}</td>`).join("")
            : "<td>—</td>";
          return `<tr><td>${escapeHtml(classes[i] ?? String(i))}</td>${cells}</tr>`;
        }).join("")
      : `<tr><td colspan="${parents.length + 1}" style="opacity:.7;">No coefficient matrix (root/marginal node)</td></tr>`;
    return `
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">
          <thead>
            <tr style="background:rgba(0,0,0,0.08);">
              <th>Class</th><th>Intercept</th>
            </tr>
          </thead>
          <tbody>${interRows}</tbody>
        </table>
      </div>
      <div style="height:8px;"></div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">
          <thead>
            <tr style="background:rgba(0,0,0,0.08);">
              <th>Class</th>${coefHeader}
            </tr>
          </thead>
          <tbody>${coefRows}</tbody>
        </table>
      </div>
      <style>
        .node_desc table th, .node_desc table td {
          padding: 5px 10px;
          border: 1px solid rgba(0,0,0,0.15);
          text-align: center;
          white-space: nowrap;
        }
        .node_desc table tbody tr:nth-child(even) { background: rgba(0,0,0,0.04); }
        .node_desc table tbody tr:hover            { background: rgba(0,0,0,0.08); }
      </style>
    `;
  }
  function buildCptTable(cpt) {
    const states     = Array.isArray(cpt.states) ? cpt.states : [];
    const cptEntries = cpt.cpt ? Object.entries(cpt.cpt) : [];
    const tableRows = cptEntries.map(([combo, probs]) => {
      const parentCells = cpt.parents && cpt.parents.length
        ? parseCombo(combo).map((v) => `<td>${escapeHtml(v)}</td>`).join("")
        : "";
      let probCells;
      if (Array.isArray(probs)) {
        probCells = probs.map((p) => `<td>${(+p).toFixed(4)}</td>`).join("");
      } else if (typeof probs === "object" && probs !== null) {
        probCells = states.map((s) =>
          `<td>${probs[s] !== undefined ? (+probs[s]).toFixed(4) : "—"}</td>`
        ).join("");
      } else {
        const pTrue = +probs;
        probCells = states.length === 2
          ? `<td>${(1 - pTrue).toFixed(4)}</td><td>${pTrue.toFixed(4)}</td>`
          : `<td>${pTrue.toFixed(4)}</td>`;
      }
      return `<tr>${parentCells}${probCells}</tr>`;
    });
    const parentHeaders = cpt.parents && cpt.parents.length
      ? cpt.parents.map((p) => `<th>${escapeHtml(p)}</th>`).join("")
      : "";
    const stateHeaders = states.length
      ? states.map((s) => `<th>P(${escapeHtml(s)})</th>`).join("")
      : `<th>Probability</th>`;
    return `
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">
          <thead>
            <tr style="background:rgba(0,0,0,0.08);">
              ${parentHeaders}${stateHeaders}
            </tr>
          </thead>
          <tbody>${tableRows.join("")}</tbody>
        </table>
      </div>
      <style>
        .node_desc table th, .node_desc table td {
          padding: 5px 10px;
          border: 1px solid rgba(0,0,0,0.15);
          text-align: center;
          white-space: nowrap;
        }
        .node_desc table tbody tr:nth-child(even) { background: rgba(0,0,0,0.04); }
        .node_desc table tbody tr:hover            { background: rgba(0,0,0,0.08); }
      </style>
    `;
  }
  function enumerateAll(nodeIds, fixedEvidence) {
    const statesList = nodeIds.map((id) => getNodeStates(id));
    const results = [];
    function recurse(idx, current) {
      if (idx === nodeIds.length) {
        results.push({ assignment: { ...current }, prob: null });
        return;
      }
      for (const s of statesList[idx]) {
        current[nodeIds[idx]] = s;
        recurse(idx + 1, current);
      }
    }
    recurse(0, { ...fixedEvidence });
    return results;
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
  function inferByEnumeration(queryId, evidence) {
    const queryStates = getNodeStates(queryId);
    const hiddenIds   = nodes
      .map((n) => n.id)
      .filter((id) => id !== queryId && evidence[id] === undefined);
    const result = {};
    for (const qs of queryStates) {
      const fullEvidence = { ...evidence, [queryId]: qs };
      const rows = enumerateAll(hiddenIds, fullEvidence);
      let total = 0;
      for (const row of rows) {
        const jp = jointProb(row.assignment);
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
  let currentQueryId = null;
  function buildInferencePanel(queryId) {
    currentQueryId = queryId;
    const queryNode = byId.get(queryId);
    if (!queryNode) return;
    const otherNodes = nodes.filter((n) => n.id !== queryId);
    const rowsHtml = otherNodes.map((n) => {
      const states = getNodeStates(n.id);
      const opts   = states.map((s) =>
        `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`
      ).join("");
      return `
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;padding:4px 0;border-bottom:1px solid rgba(0,0,0,0.06);">
          <label style="font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(n.label ?? n.id)}">${escapeHtml(n.label ?? n.id)}</label>
          <select data-nodeid="${escapeHtml(n.id)}" style="font-size:12px;padding:2px 6px;border-radius:6px;border:1px solid rgba(0,0,0,0.2);background:white;max-width:110px;">
            <option value="">— unobserved —</option>
            ${opts}
          </select>
        </div>
      `;
    }).join("");
    infEl.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div style="font-size:12px;opacity:.7;">
          Set observed states for other nodes, then run inference to predict
          <strong>${escapeHtml(queryNode.label ?? queryId)}</strong>.
        </div>
        <div style="display:flex;flex-direction:column;gap:2px;max-height:220px;overflow-y:auto;padding-right:4px;">
          ${rowsHtml}
        </div>
        <button id="run_inference_btn" style="
          padding:7px 0;border-radius:8px;border:none;
          background:#1a1a1a;color:white;font-size:13px;
          font-family:Inter,system-ui,sans-serif;cursor:pointer;
          transition:opacity 0.15s;
        ">Run Inference</button>
        <div id="inference_result" style="font-size:13px;"></div>
      </div>
      <style>
        #run_inference_btn:hover { opacity: 0.8; }
        #inference_result .res-bar-wrap {
          background: rgba(0,0,0,0.07);
          border-radius: 6px;
          overflow: hidden;
          height: 18px;
          flex: 1;
        }
        #inference_result .res-bar {
          height: 100%;
          background: #ffa7a6;
          border-radius: 6px;
          transition: width 0.4s ease;
        }
      </style>
    `;
    document.getElementById("run_inference_btn").addEventListener("click", () => {
      runInference(queryId);
    });
  }
  function runInference(queryId) {
    const resultEl = document.getElementById("inference_result");
    resultEl.innerHTML = `<div style="opacity:.5;">Computing...</div>`;
    const evidence = {};
    for (const sel of infEl.querySelectorAll("select[data-nodeid]")) {
      if (sel.value) evidence[sel.dataset.nodeid] = sel.value;
    }
    setTimeout(() => {
      const result = inferByEnumeration(queryId, evidence);
      const queryNode = byId.get(queryId);
      if (!result) {
        resultEl.innerHTML = `
          <div style="color:#c0392b;font-size:12px;">
            Could not compute from beta parameters for this selection.
          </div>`;
        return;
      }
      const evidenceCount = Object.keys(evidence).length;
      const rows = Object.entries(result).map(([state, prob]) => {
        const pct = (prob * 100).toFixed(1);
        return `
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="width:36px;font-size:12px;font-weight:600;text-align:right;">${escapeHtml(state)}</div>
            <div class="res-bar-wrap">
              <div class="res-bar" style="width:${pct}%"></div>
            </div>
            <div style="font-size:12px;width:44px;">${pct}%</div>
          </div>
        `;
      }).join("");
      resultEl.innerHTML = `
        <div style="margin-bottom:8px;font-size:12px;opacity:.65;">
          P(<strong>${escapeHtml(queryNode.label ?? queryId)}</strong>${evidenceCount ? " | " + evidenceCount + " observed" : ""})
        </div>
        ${rows}
      `;
    }, 20);
  }
  visEl.innerHTML = "";
  visEl.style.position = "relative";
  const W = Math.max(520, visEl.clientWidth  || 520);
  const H = Math.max(420, visEl.clientHeight || 420);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width",  String(W));
  svg.setAttribute("height", String(H));
  svg.style.cssText = "width:100%;height:100%;display:block;border-radius:16px;cursor:grab;";
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  function makeMarker(id, color) {
    const m = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    m.setAttribute("id",           id);
    m.setAttribute("markerWidth",  "10");
    m.setAttribute("markerHeight", "10");
    m.setAttribute("refX",         "10");
    m.setAttribute("refY",         "3");
    m.setAttribute("orient",       "auto");
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d",    "M0,0 L10,3 L0,6 Z");
    p.setAttribute("fill", color);
    m.appendChild(p);
    return m;
  }
  defs.appendChild(makeMarker("arrow-gray",  "gray"));
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
    const p1   = positions.get(fromId);
    const p2   = positions.get(toId);
    const dx   = p2.x - p1.x;
    const dy   = p2.y - p1.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux   = dx / dist;
    const uy   = dy / dist;
    return {
      x1: p1.x + ux * NODE_R,
      y1: p1.y + uy * NODE_R,
      x2: p2.x - ux * NODE_R,
      y2: p2.y - uy * NODE_R,
    };
  }
  for (const e of edges) {
    if (!positions.has(e.from) || !positions.has(e.to)) continue;
    const { x1, y1, x2, y2 } = edgeCoords(e.from, e.to);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1",           String(x1));
    line.setAttribute("y1",           String(y1));
    line.setAttribute("x2",           String(x2));
    line.setAttribute("y2",           String(y2));
    line.setAttribute("stroke",       "gray");
    line.setAttribute("stroke-width", "1");
    line.setAttribute("marker-end",   "url(#arrow-gray)");
    line.dataset.from = e.from;
    line.dataset.to   = e.to;
    edgeGroup.appendChild(line);
  }
  const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  mainGroup.appendChild(nodeGroup);
  const nodeEls = new Map();
  for (const n of nodes) {
    const p = positions.get(n.id);
    if (!p) continue;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.style.cursor = "pointer";
    g.dataset.node = n.id;
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx",           String(p.x));
    circle.setAttribute("cy",           String(p.y));
    circle.setAttribute("r",            String(NODE_R));
    circle.setAttribute("fill",         defaultFill(n.id));
    circle.setAttribute("stroke",       "gray");
    circle.setAttribute("stroke-width", "1");
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x",              String(p.x));
    text.setAttribute("y",              String(p.y + 5));
    text.setAttribute("text-anchor",    "middle");
    text.setAttribute("font-size",      "14");
    text.setAttribute("font-family",    "Ubuntu Condensed, sans-serif");
    text.setAttribute("pointer-events", "none");
    text.textContent = (n.label ?? n.id).slice(0, 12);
    g.appendChild(circle);
    g.appendChild(text);
    g.addEventListener("click", () => { if (!didDrag) selectNode(n.id); });
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
    const rect  = svg.getBoundingClientRect();
    const mx    = e.clientX - rect.left;
    const my    = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    viewX     = mx - (mx - viewX) * delta;
    viewY     = my - (my - viewY) * delta;
    viewScale = Math.min(Math.max(viewScale * delta, 0.2), 5);
    applyTransform();
  }, { passive: false });
  let dragId = null, dragOffX = 0, dragOffY = 0;
  let isPanning = false, panStartX = 0, panStartY = 0;
  let didDrag = false;
  const svgPoint = (clientX, clientY) => {
    const rect = svg.getBoundingClientRect();
    return {
      x: (clientX - rect.left - viewX) / viewScale,
      y: (clientY - rect.top  - viewY) / viewScale,
    };
  };
  svg.addEventListener("mousedown", (e) => {
    didDrag = false;
    const nodeTarget = e.target.closest("[data-node]");
    if (nodeTarget) {
      const nid  = nodeTarget.dataset.node;
      dragId     = nid;
      const pos  = positions.get(nid);
      const sp   = svgPoint(e.clientX, e.clientY);
      dragOffX   = sp.x - pos.x;
      dragOffY   = sp.y - pos.y;
      nodeTarget.style.cursor = "grabbing";
      e.stopPropagation();
    } else {
      isPanning = true;
      panStartX = e.clientX - viewX;
      panStartY = e.clientY - viewY;
      svg.style.cursor = "grabbing";
    }
  });
  window.addEventListener("mousemove", (e) => {
    if (dragId) {
      didDrag = true;
      const sp = svgPoint(e.clientX, e.clientY);
      const x  = sp.x - dragOffX;
      const y  = sp.y - dragOffY;
      positions.set(dragId, { x, y });
      const el = nodeEls.get(dragId);
      el.circle.setAttribute("cx", String(x));
      el.circle.setAttribute("cy", String(y));
      el.text.setAttribute("x",    String(x));
      el.text.setAttribute("y",    String(y + 5));
      for (const line of edgeGroup.querySelectorAll("line")) {
        if (line.dataset.from === dragId || line.dataset.to === dragId) {
          const { x1, y1, x2, y2 } = edgeCoords(line.dataset.from, line.dataset.to);
          line.setAttribute("x1", String(x1));
          line.setAttribute("y1", String(y1));
          line.setAttribute("x2", String(x2));
          line.setAttribute("y2", String(y2));
        }
      }
    } else if (isPanning) {
      didDrag = true;
      viewX = e.clientX - panStartX;
      viewY = e.clientY - panStartY;
      applyTransform();
    }
  });
  window.addEventListener("mouseup", () => {
    if (dragId) {
      const el = nodeEls.get(dragId);
      if (el) el.g.style.cursor = "pointer";
      dragId = null;
    }
    if (isPanning) {
      isPanning = false;
      svg.style.cursor = "grab";
    }
  });
  const searchWrapper = document.createElement("div");
  searchWrapper.style.cssText = `
    position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:10;
  `;
  const searchInput = document.createElement("input");
  searchInput.type        = "text";
  searchInput.placeholder = "Search node...";
  searchInput.style.cssText = `
    padding:6px 14px;border-radius:20px;border:1.5px solid black;
    font-size:13px;font-family:Inter,system-ui,sans-serif;outline:none;
    background:#efe9db;width:200px;color:black;
  `;
  searchWrapper.appendChild(searchInput);
  visEl.appendChild(searchWrapper);
  searchInput.addEventListener("input", () => {
    const query = searchInput.value.trim().toLowerCase();
    clearHighlights();
    if (!query) return;
    const match = nodes.find((n) =>
      (n.label ?? n.id).toLowerCase().includes(query)
    );
    if (match) {
      selectNode(match.id);
      const el = nodeEls.get(match.id);
      if (el) {
        el.circle.setAttribute("stroke",       "orange");
        el.circle.setAttribute("stroke-width", "3");
      }
    }
  });
  const resetBtn = document.createElement("button");
  resetBtn.textContent = "⟳ Reset View";
  resetBtn.style.cssText = `
    position:absolute;bottom:12px;right:12px;z-index:10;
    padding:5px 12px;border-radius:16px;border:1.5px solid black;
    background:#efe9db;font-size:12px;cursor:pointer;
    font-family:Inter,system-ui,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,0.12);
  `;
  resetBtn.addEventListener("click", () => {
    viewX = 0; viewY = 0; viewScale = 1;
    applyTransform();
  });
  visEl.appendChild(resetBtn);
  const tooltip = document.createElement("div");
  tooltip.style.cssText = `
    position:fixed;pointer-events:none;z-index:9999;
    background:#1a1a1a;color:#f5f5f5;border-radius:8px;
    padding:10px 14px;font-family:Inter,system-ui,sans-serif;
    font-size:13px;line-height:1.5;max-width:260px;
    box-shadow:0 4px 16px rgba(0,0,0,0.35);
    opacity:0;transition:opacity 0.15s ease;white-space:normal;
  `;
  document.body.appendChild(tooltip);
  const positionTooltip = (mx, my) => {
    const pad = 14;
    const tw  = tooltip.offsetWidth  || 220;
    const th  = tooltip.offsetHeight || 80;
    let left  = mx + pad;
    let top   = my + pad;
    if (left + tw > window.innerWidth  - pad) left = mx - tw - pad;
    if (top  + th > window.innerHeight - pad) top  = my - th - pad;
    tooltip.style.left = left + "px";
    tooltip.style.top  = top  + "px";
  };
  const showTooltip = (n, mx, my) => {
    const parents  = parentsOf.get(n.id)  ?? [];
    const children = childrenOf.get(n.id) ?? [];
    tooltip.innerHTML = `
      <div style="font-weight:600;margin-bottom:5px;">${escapeHtml(n.label ?? n.id)}</div>
      <div style="opacity:0.85;margin-bottom:6px;">${escapeHtml(n.description || "No description provided.")}</div>
      <div style="border-top:1px solid rgba(255,255,255,0.15);padding-top:6px;font-size:11px;opacity:0.7;">
        <div><strong>Parents:</strong> ${parents.length  ? parents.map(escapeHtml).join(", ")  : "None"}</div>
        <div><strong>Children:</strong> ${children.length ? children.map(escapeHtml).join(", ") : "None"}</div>
      </div>
    `;
    positionTooltip(mx, my);
    tooltip.style.opacity = "1";
  };
  const hideTooltip = () => { tooltip.style.opacity = "0"; };
  for (const n of nodes) {
    const el = nodeEls.get(n.id);
    if (!el) continue;
    el.g.addEventListener("mouseenter", (e) => showTooltip(n, e.clientX, e.clientY));
    el.g.addEventListener("mousemove",  (e) => positionTooltip(e.clientX, e.clientY));
    el.g.addEventListener("mouseleave", hideTooltip);
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
  const selectNode = (nodeId) => {
    const n = byId.get(nodeId);
    if (!n) return;
    clearHighlights();
    const el = nodeEls.get(nodeId);
    if (el) {
      el.circle.setAttribute("fill",         "#ffa7a6");
      el.circle.setAttribute("stroke",       "black");
      el.circle.setAttribute("stroke-width", "3");
    }
    for (const parentId of parentsOf.get(nodeId) ?? []) {
      const parentEl = nodeEls.get(parentId);
      if (parentEl) {
        parentEl.circle.setAttribute("stroke",       "black");
        parentEl.circle.setAttribute("stroke-width", "1");
      }
    }
    for (const line of edgeGroup.querySelectorAll("line")) {
      if (line.dataset.to === nodeId) {
        line.setAttribute("stroke",       "black");
        line.setAttribute("stroke-width", "2");
        line.setAttribute("marker-end",   "url(#arrow-black)");
      }
    }
    const cpt = bn.cpts?.[nodeId];
    if (cpt) {
      const parentsLine = Array.isArray(cpt.parents) ? cpt.parents.join(", ") : "";
      const states = Array.isArray(cpt.states) ? cpt.states : getNodeStates(nodeId);
      descEl.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:600;">${escapeHtml(n.label ?? n.id)}</div>
          <div style="font-size:12px;opacity:.75;">
            <strong>Parents:</strong> ${escapeHtml(parentsLine || "None")}
            &nbsp;·&nbsp;
            <strong>States:</strong> ${escapeHtml(states.join(", ") || "—")}
          </div>
          ${buildCptTable(cpt)}
        </div>
      `;
    } else if (!betaParams?.[nodeId]) {
      descEl.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:8px;">
          <div style="font-weight:600;">${escapeHtml(n.label ?? n.id)}</div>
          <div style="opacity:.65;font-size:13px;">No CPT provided for this node.</div>
          <div style="opacity:.5;font-size:12px;">Include a cpts object in your JSON to display it here.</div>
        </div>
      `;
    } else {
      const beta = betaParams[nodeId];
      const parentsLine = Array.isArray(beta?.parents) ? beta.parents.join(", ") : "";
      const states = getNodeStates(nodeId);
      descEl.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:600;">${escapeHtml(n.label ?? n.id)}</div>
          <div style="font-size:12px;opacity:.75;">
            <strong>Parents:</strong> ${escapeHtml(parentsLine || "None")}
            &nbsp;·&nbsp;
            <strong>States:</strong> ${escapeHtml(states.join(", ") || "—")}
          </div>
          ${buildBetaTable(nodeId)}
        </div>
      `;
    }
    buildInferencePanel(nodeId);
  };
  if (nodes.length) selectNode(nodes[0].id);
})();
