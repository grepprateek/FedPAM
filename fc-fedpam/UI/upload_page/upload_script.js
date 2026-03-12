(() => {
  const fileInput = document.getElementById("fileInput");
  const fileNameEl = document.getElementById("fileName");
  const visualizeButton = document.getElementById("visualizeButton");
  const visualizeLink = visualizeButton?.querySelector("a");
  const getSelectedValue = (groupName) => {
    const checked = document.querySelector(`input[name="${groupName}"]:checked`);
    return checked ? checked.value : null;
  };
  const getExtension = (filename) => {
    const parts = filename.toLowerCase().split(".");
    return parts.length > 1 ? parts.pop() : "";
  };
  const setStatus = (text, isError = false) => {
    fileNameEl.textContent = text;
    fileNameEl.style.fontWeight = "600";
    fileNameEl.style.color = isError ? "crimson" : "black";
  };
  const readFileAsText = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Could not read the file."));
      reader.onload = () => resolve(String(reader.result || ""));
      reader.readAsText(file);
    });
  const parseCsvSimple = (csvText) => {
    const lines = csvText
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length === 0) return { headers: [], rows: [] };
    const headers = lines[0].split(",").map((h) => h.trim());
    const rows = lines.slice(1).map((line) => {
      const cols = line.split(",").map((c) => c.trim());
      const obj = {};
      headers.forEach((h, i) => (obj[h] = cols[i] ?? ""));
      return obj;
    });
    return { headers, rows };
  };
  const normalizeBnFromJson = (obj) => {
    if (obj && Array.isArray(obj.nodes) && Array.isArray(obj.edges)) {
      return {
        nodes: obj.nodes.map((n) => ({
          id: String(n.id ?? n.name ?? n.label),
          label: n.label ?? n.name ?? n.id,
          states: Array.isArray(n.states) ? n.states.map(String) : undefined,
          description: n.description ?? "",
        })),
        edges: obj.edges.map((e) => ({
          from: String(e.from ?? e.source),
          to: String(e.to ?? e.target),
        })),
        cpts: obj.cpts ?? obj.CPTs ?? undefined,
      };
    }
    if (Array.isArray(obj) && obj.length && (obj[0].from || obj[0].source)) {
      const edges = obj.map((e) => ({
        from: String(e.from ?? e.source),
        to: String(e.to ?? e.target),
      }));
      const nodeIds = new Set();
      edges.forEach((e) => (nodeIds.add(e.from), nodeIds.add(e.to)));
      const nodes = [...nodeIds].map((id) => ({ id, label: id }));
      return { nodes, edges };
    }
    if (obj && typeof obj === "object") {
      const edges = [];
      const nodeIds = new Set();
      for (const [from, tos] of Object.entries(obj)) {
        nodeIds.add(String(from));
        if (Array.isArray(tos)) {
          tos.forEach((to) => {
            nodeIds.add(String(to));
            edges.push({ from: String(from), to: String(to) });
          });
        }
      }
      if (edges.length) {
        const nodes = [...nodeIds].map((id) => ({ id, label: id }));
        return { nodes, edges };
      }
    }
    throw new Error(
      "JSON format not recognized. Provide {nodes:[...], edges:[...]} or an edge list / adjacency."
    );
  };
  const normalizeBnFromCsv = ({ headers, rows }) => {
    const lowerHeaders = headers.map((h) => h.toLowerCase());
    const fromKey =
      headers[lowerHeaders.indexOf("from")] ??
      headers[lowerHeaders.indexOf("source")] ??
      null;
    const toKey =
      headers[lowerHeaders.indexOf("to")] ??
      headers[lowerHeaders.indexOf("target")] ??
      null;
    if (!fromKey || !toKey) {
      throw new Error(
        "CSV must have headers 'from,to' (or 'source,target') for the edge list."
      );
    }
    const edges = rows
      .map((r) => ({
        from: String(r[fromKey] ?? "").trim(),
        to: String(r[toKey] ?? "").trim(),
      }))
      .filter((e) => e.from && e.to);
    if (!edges.length) throw new Error("CSV had no valid edges.");
    const nodeIds = new Set();
    edges.forEach((e) => (nodeIds.add(e.from), nodeIds.add(e.to)));
    const nodes = [...nodeIds].map((id) => ({ id, label: id }));
    return { nodes, edges };
  };
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (!file) return setStatus("No file selected");
    setStatus(file.name);
  });
  (visualizeLink ?? visualizeButton).addEventListener("click", async (evt) => {
    evt.preventDefault();
    const networkType = getSelectedValue("networkType");
    const fileType = getSelectedValue("fileType");
    const file = fileInput.files?.[0];
    if (!networkType || !fileType) return setStatus("Please select network type and file type.", true);
    if (!file) return setStatus("Please choose a file first.", true);
    const ext = getExtension(file.name);
    if (ext !== fileType) {
      return setStatus(`Selected ${fileType.toUpperCase()} but file is .${ext || "?"}`, true);
    }
    try {
      setStatus("Reading file...");
      const text = await readFileAsText(file);
      setStatus("Parsing...");
      let bn;
      if (fileType === "json") {
        const obj = JSON.parse(text);
        const norm = normalizeBnFromJson(obj);
        bn = { networkType, ...norm };
      } else {
        const parsed = parseCsvSimple(text);
        const norm = normalizeBnFromCsv(parsed);
        bn = { networkType, ...norm };
      }
      sessionStorage.setItem("bayestitch_network", JSON.stringify(bn));
      sessionStorage.setItem("bayestitch_filename", file.name);
      sessionStorage.setItem("bayestitch_filetype", fileType);
      setStatus("Loaded! Redirecting…");
      const target = visualizeLink?.getAttribute("href") || "../visualization_page/vis_index.html";
      window.location.href = target;
    } catch (err) {
      console.error(err);
      setStatus(err?.message || "Something went wrong.", true);
    }
  });
})();
