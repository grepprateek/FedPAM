(() => {
  const visEl  = document.querySelector(".visualization");
  const descEl = document.querySelector(".node_desc");
  const raw = sessionStorage.getItem("bayestitch_network");
  if (!raw) {
    descEl.innerHTML = `<div style="opacity:.6;font-size:14px;">No network found. Go back and upload a JSON/CSV first.</div>`;
    return;
  }
  const bn = JSON.parse(raw);
  let nodes = bn.nodes.map((n) => ({
    id:          String(n.id),
    label:       n.label ?? n.id,
    states:      Array.isArray(n.states) ? n.states : undefined,
    description: n.description ?? "",
  }));
  let edges = bn.edges.map((e) => ({ from: String(e.from), to: String(e.to) }));
  const modifications = {
    blacklist: [],
    whitelist: []
  };
  const originalEdges = new Set(edges.map(e => `${e.from}->${e.to}`));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const parentsOf  = new Map(nodes.map((n) => [n.id, []]));
  const childrenOf = new Map(nodes.map((n) => [n.id, []]));
  function updateParentChildMaps() {
    parentsOf.clear();
    childrenOf.clear();
    nodes.forEach(n => {
      parentsOf.set(n.id, []);
      childrenOf.set(n.id, []);
    });
    for (const e of edges) {
      if (parentsOf.has(e.to))    parentsOf.get(e.to).push(e.from);
      if (childrenOf.has(e.from)) childrenOf.get(e.from).push(e.to);
    }
  }
  updateParentChildMaps();
  function escapeHtml(s) {
    return String(s)
      .replaceAll("&",  "&amp;")
      .replaceAll("<",  "&lt;")
      .replaceAll(">",  "&gt;")
      .replaceAll('"',  "&quot;")
      .replaceAll("'",  "&#039;");
  }
  function defaultFill(nodeId) {
    return bn.cpts?.[nodeId] ? "#fbc5bb" : "#e0e0e0";
  }
  let interactionMode = 'select';
  let selectedNode = null;
  let edgeStartNode = null;
  const history = {
    states: [],
    currentIndex: -1,
    maxStates: 50
  };
  let isRestoring = false;
  function saveState() {
    if (isRestoring) return;
    history.states = history.states.slice(0, history.currentIndex + 1);
    const state = {
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
      modifications: JSON.parse(JSON.stringify(modifications))
    };
    history.states.push(state);
    if (history.states.length > history.maxStates) {
      history.states.shift();
    } else {
      history.currentIndex++;
    }
    console.log('State saved. Total states:', history.states.length, 'Current index:', history.currentIndex);
    updateUndoRedoButtons();
  }
  function undo() {
    if (history.currentIndex > 0) {
      history.currentIndex--;
      console.log('Undo to index:', history.currentIndex, 'of', history.states.length);
      restoreState(history.states[history.currentIndex]);
    } else {
      console.log('Cannot undo - already at first state');
    }
  }
  function redo() {
    if (history.currentIndex < history.states.length - 1) {
      history.currentIndex++;
      console.log('Redo to index:', history.currentIndex, 'of', history.states.length);
      restoreState(history.states[history.currentIndex]);
    } else {
      console.log('Cannot redo - already at last state');
    }
  }
  function restoreState(state) {
    isRestoring = true;
    nodes = JSON.parse(JSON.stringify(state.nodes));
    edges = JSON.parse(JSON.stringify(state.edges));
    modifications.blacklist = JSON.parse(JSON.stringify(state.modifications.blacklist));
    modifications.whitelist = JSON.parse(JSON.stringify(state.modifications.whitelist));
    byId.clear();
    nodes.forEach(n => byId.set(n.id, n));
    updateParentChildMaps();
    renderNetwork();
    updateModificationPanel();
    updateUndoRedoButtons();
    isRestoring = false;
  }
  function updateUndoRedoButtons() {
    const undoBtn = document.getElementById('undo_btn');
    const redoBtn = document.getElementById('redo_btn');
    if (undoBtn) {
      undoBtn.disabled = history.currentIndex <= 0;
      undoBtn.style.opacity = undoBtn.disabled ? '0.3' : '1';
      undoBtn.style.cursor = undoBtn.disabled ? 'not-allowed' : 'pointer';
    }
    if (redoBtn) {
      redoBtn.disabled = history.currentIndex >= history.states.length - 1;
      redoBtn.style.opacity = redoBtn.disabled ? '0.3' : '1';
      redoBtn.style.cursor = redoBtn.disabled ? 'not-allowed' : 'pointer';
    }
  }
  let svg, mainGroup, edgeGroup, nodeGroup, positions, nodeEls;
  const W = Math.max(520, visEl.clientWidth  || 520);
  const H = Math.max(420, visEl.clientHeight || 420);
  const NODE_R = 25;
  let viewX = 0, viewY = 0, viewScale = 1;
  let dragId = null, dragOffX = 0, dragOffY = 0;
  let isPanning = false, panStartX = 0, panStartY = 0;
  let didDrag = false;
  function renderNetwork() {
    visEl.innerHTML = "";
    visEl.style.position = "relative";
    const controlsDiv = document.createElement('div');
    controlsDiv.style.cssText = `
      position: absolute;
      top: 10px;
      left: 10px;
      display: flex;
      gap: 6px;
      z-index: 10;
      flex-wrap: wrap;
      max-width: calc(100% - 20px);
    `;
    const buttonStyle = `
      padding: 6px 10px;
      border-radius: 6px;
      border: 2px solid black;
      background: white;
      font-size: 11px;
      font-family: 'Ubuntu Condensed', sans-serif;
      cursor: pointer;
      transition: all 0.15s;
      font-weight: 600;
      white-space: nowrap;
    `;
    const buttons = [
      { id: 'mode_select', text: 'Select Node', mode: 'select' },
      { id: 'mode_add_edge', text: 'Add Edge', mode: 'add-edge' },
      { id: 'mode_delete_edge', text: 'Delete Edge', mode: 'delete-edge' },
      { id: 'mode_delete_node', text: 'Delete Node', mode: 'delete-node' }
    ];
    buttons.forEach(({ id, text, mode }) => {
      const btn = document.createElement('button');
      btn.id = id;
      btn.textContent = text;
      btn.style.cssText = buttonStyle;
      if (mode === interactionMode) {
        btn.style.background = '#1a1a1a';
        btn.style.color = 'white';
      }
      btn.addEventListener('click', () => {
        interactionMode = mode;
        edgeStartNode = null;
        clearHighlights();
        const hintTexts = {
          'select': 'Click a node to select it',
          'add-edge': 'Click two nodes to add an edge',
          'delete-edge': 'Click an edge to delete it',
          'delete-node': 'Click a node to delete it'
        };
        const hint = document.getElementById('mode_hint');
        if (hint) hint.textContent = hintTexts[mode];
        controlsDiv.querySelectorAll('button').forEach(b => {
          b.style.background = 'white';
          b.style.color = 'black';
        });
        btn.style.background = '#1a1a1a';
        btn.style.color = 'white';
      });
      controlsDiv.appendChild(btn);
    });
    visEl.appendChild(controlsDiv);
    svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
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
    defs.appendChild(makeMarker("arrow-red",    "#e74c3c"));
    defs.appendChild(makeMarker("arrow-green",  "#27ae60"));
    defs.appendChild(makeMarker("arrow-blue",   "#3498db"));
    svg.appendChild(defs);
    mainGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    svg.appendChild(mainGroup);
    const cx = W / 2, cy = H / 2;
    const R  = Math.min(W, H) * 0.35;
    positions = new Map();
    nodes.forEach((n, i) => {
      const a = (i / Math.max(1, nodes.length)) * Math.PI * 2;
      positions.set(n.id, { x: cx + Math.cos(a) * R, y: cy + Math.sin(a) * R });
    });
    edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    mainGroup.appendChild(edgeGroup);
    function edgeCoords(fromId, toId) {
      const p1 = positions.get(fromId), p2 = positions.get(toId);
      if (!p1 || !p2) return null;
      const dx = p2.x - p1.x, dy = p2.y - p1.y;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      const ux = dx/dist, uy = dy/dist;
      return { x1: p1.x+ux*NODE_R, y1: p1.y+uy*NODE_R, x2: p2.x-ux*NODE_R, y2: p2.y-uy*NODE_R };
    }
    for (const e of edges) {
      const coords = edgeCoords(e.from, e.to);
      if (!coords) continue;
      const { x1, y1, x2, y2 } = coords;
      const edgeKey = `${e.from}->${e.to}`;
      const isNew = !originalEdges.has(edgeKey);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(x1)); line.setAttribute("y1", String(y1));
      line.setAttribute("x2", String(x2)); line.setAttribute("y2", String(y2));
      line.setAttribute("stroke", isNew ? "#27ae60" : "gray");
      line.setAttribute("stroke-width", isNew ? "2" : "1");
      line.setAttribute("marker-end", isNew ? "url(#arrow-green)" : "url(#arrow-gray)");
      line.dataset.from = e.from;
      line.dataset.to = e.to;
      line.style.cursor = "pointer";
      line.addEventListener("click", (evt) => {
        evt.stopPropagation();
        if (interactionMode === 'delete-edge' && !didDrag) {
          deleteEdge(e.from, e.to);
        }
      });
      edgeGroup.appendChild(line);
    }
    nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    mainGroup.appendChild(nodeGroup);
    nodeEls = new Map();
    for (const n of nodes) {
      const p = positions.get(n.id);
      if (!p) continue;
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.style.cursor = "pointer";
      g.dataset.node = n.id;
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", String(p.x));
      circle.setAttribute("cy", String(p.y));
      circle.setAttribute("r",  String(NODE_R));
      circle.setAttribute("fill", defaultFill(n.id));
      circle.setAttribute("stroke", "gray");
      circle.setAttribute("stroke-width", "1");
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", String(p.x));
      text.setAttribute("y", String(p.y + 5));
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("font-size", "14");
      text.setAttribute("font-family", "Ubuntu Condensed, sans-serif");
      text.setAttribute("pointer-events", "none");
      text.textContent = (n.label ?? n.id).slice(0, 12);
      g.appendChild(circle);
      g.appendChild(text);
      g.addEventListener("click", () => { if (!didDrag) handleNodeClick(n.id); });
      nodeGroup.appendChild(g);
      nodeEls.set(n.id, { g, circle, text });
    }
    visEl.appendChild(svg);
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
            const coords = edgeCoords(line.dataset.from, line.dataset.to);
            if (coords) {
              const { x1, y1, x2, y2 } = coords;
              line.setAttribute("x1", String(x1)); line.setAttribute("y1", String(y1));
              line.setAttribute("x2", String(x2)); line.setAttribute("y2", String(y2));
            }
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
    hint.id = "mode_hint";
    hint.textContent = "Click a node to select it";
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
  }
  function handleNodeClick(nodeId) {
    if (interactionMode === 'select') {
      selectedNode = nodeId;
      highlightNode(nodeId);
    } else if (interactionMode === 'add-edge') {
      if (!edgeStartNode) {
        edgeStartNode = nodeId;
        highlightNode(nodeId, '#ffa7a6');
      } else if (edgeStartNode !== nodeId) {
        addEdge(edgeStartNode, nodeId);
        edgeStartNode = null;
        clearHighlights();
      }
    } else if (interactionMode === 'delete-node') {
      deleteNode(nodeId);
    }
  }
  function addEdge(fromId, toId) {
    const exists = edges.some(e => e.from === fromId && e.to === toId);
    if (exists) {
      alert('Edge already exists!');
      return;
    }
    edges.push({ from: fromId, to: toId });
    const edgeKey = `${fromId}->${toId}`;
    if (originalEdges.has(edgeKey)) {
      const idx = modifications.blacklist.findIndex(e => e[0] === fromId && e[1] === toId);
      if (idx !== -1) modifications.blacklist.splice(idx, 1);
    } else {
      modifications.whitelist.push([fromId, toId]);
    }
    updateParentChildMaps();
    saveState();
    renderNetwork();
    updateModificationPanel();
  }
  function deleteEdge(fromId, toId) {
    const idx = edges.findIndex(e => e.from === fromId && e.to === toId);
    if (idx === -1) return;
    edges.splice(idx, 1);
    const edgeKey = `${fromId}->${toId}`;
    if (originalEdges.has(edgeKey)) {
      modifications.blacklist.push([fromId, toId]);
    } else {
      const wIdx = modifications.whitelist.findIndex(e => e[0] === fromId && e[1] === toId);
      if (wIdx !== -1) modifications.whitelist.splice(wIdx, 1);
    }
    updateParentChildMaps();
    saveState();
    renderNetwork();
    updateModificationPanel();
  }
  function deleteNode(nodeId) {
    if (!confirm(`Delete node "${nodeId}" and all connected edges?`)) return;
    const connectedEdges = edges.filter(e => e.from === nodeId || e.to === nodeId);
    edges = edges.filter(e => e.from !== nodeId && e.to !== nodeId);
    for (const e of connectedEdges) {
      const edgeKey = `${e.from}->${e.to}`;
      if (originalEdges.has(edgeKey)) {
        modifications.blacklist.push([e.from, e.to]);
      } else {
        const wIdx = modifications.whitelist.findIndex(w => w[0] === e.from && w[1] === e.to);
        if (wIdx !== -1) modifications.whitelist.splice(wIdx, 1);
      }
    }
    nodes = nodes.filter(n => n.id !== nodeId);
    byId.delete(nodeId);
    parentsOf.delete(nodeId);
    childrenOf.delete(nodeId);
    updateParentChildMaps();
    saveState();
    renderNetwork();
    updateModificationPanel();
  }
  function highlightNode(nodeId, color = '#ffa7a6') {
    clearHighlights();
    const nel = nodeEls.get(nodeId);
    if (nel) {
      nel.circle.setAttribute("fill", color);
      nel.circle.setAttribute("stroke", "black");
      nel.circle.setAttribute("stroke-width", "3");
    }
  }
  function clearHighlights() {
    for (const [id, nel] of nodeEls) {
      nel.circle.setAttribute("fill", defaultFill(id));
      nel.circle.setAttribute("stroke", "gray");
      nel.circle.setAttribute("stroke-width", "1");
    }
  }
  function updateModificationPanel() {
    const jsonDisplay = document.getElementById("json_display");
    const statsDiv = document.getElementById("modification_stats");
    if (jsonDisplay) {
      const jsonText = JSON.stringify(modifications, null, 2);
      jsonDisplay.textContent = jsonText;
    }
    if (statsDiv) {
      const blacklistCount = modifications.blacklist.length;
      const whitelistCount = modifications.whitelist.length;
      const totalChanges = blacklistCount + whitelistCount;
      statsDiv.innerHTML = `
        <div style="font-size:13px;line-height:1.6;">
          <strong>Total Changes:</strong> ${totalChanges}<br>
          <strong>Edges Removed:</strong> ${blacklistCount}<br>
          <strong>Edges Added:</strong> ${whitelistCount}
        </div>
      `;
    }
  }
  function buildModificationPanel() {
    descEl.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:14px;height:100%;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div style="font-weight:600;font-size:14px;">Network Modifications</div>
          <div style="display:flex;gap:6px;">
            <button id="undo_btn" style="
              padding:6px 12px;
              border-radius:6px;
              border:2px solid black;
              background:white;
              font-size:12px;
              font-family:'Ubuntu Condensed',sans-serif;
              cursor:pointer;
              font-weight:600;
              transition:opacity 0.15s;
            ">↶ Undo</button>
            <button id="redo_btn" style="
              padding:6px 12px;
              border-radius:6px;
              border:2px solid black;
              background:white;
              font-size:12px;
              font-family:'Ubuntu Condensed',sans-serif;
              cursor:pointer;
              font-weight:600;
              transition:opacity 0.15s;
            ">↷ Redo</button>
          </div>
        </div>
        <div id="modification_stats" style="
          padding:10px;
          background:rgba(0,0,0,0.05);
          border-radius:6px;
        ">
          <div style="font-size:13px;line-height:1.6;">
            <strong>Total Changes:</strong> 0<br>
            <strong>Edges Removed:</strong> 0<br>
            <strong>Edges Added:</strong> 0
          </div>
        </div>
        <hr style="border:none;border-top:2px solid rgba(0,0,0,0.1);">
        <div style="font-weight:600;font-size:14px;">Modifications JSON</div>
        <div style="flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:0;">
          <pre id="json_display" style="
            flex:1;
            overflow:auto;
            background:#f4f1e8;
            border:2px solid black;
            border-radius:8px;
            padding:12px;
            font-size:12px;
            font-family:monospace;
            margin:0;
          ">${JSON.stringify(modifications, null, 2)}</pre>
        </div>
        <button id="copy_json_btn" style="
          padding:10px 0;
          border-radius:8px;
          border:3px solid black;
          background:#1a1a1a;
          color:white;
          font-size:15px;
          font-family:'Ubuntu Condensed',sans-serif;
          cursor:pointer;
          letter-spacing:.05em;
          transition:opacity 0.15s;
          flex-shrink:0;
        ">COPY JSON</button>
        <div style="font-size:11px;opacity:0.6;line-height:1.5;">
          <strong>Legend:</strong><br>
          <span style="color:#27ae60;">●</span> Added edges (whitelist)<br>
          <span style="color:gray;">●</span> Original edges
        </div>
      </div>
      <style>
        #copy_json_btn:hover {
          opacity: 0.75;
        }
      </style>
    `;
    const undoBtn = document.getElementById("undo_btn");
    const redoBtn = document.getElementById("redo_btn");
    undoBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!undoBtn.disabled) {
        undo();
      }
    });
    redoBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!redoBtn.disabled) {
        redo();
      }
    });
    document.getElementById("copy_json_btn").addEventListener("click", () => {
      const jsonText = JSON.stringify(modifications, null, 2);
      navigator.clipboard.writeText(jsonText).then(() => {
        const btn = document.getElementById("copy_json_btn");
        const originalText = btn.textContent;
        btn.textContent = "COPIED!";
        setTimeout(() => {
          btn.textContent = originalText;
        }, 1500);
      });
    });
    updateUndoRedoButtons();
  }
  saveState();
  renderNetwork();
  buildModificationPanel();
})();
