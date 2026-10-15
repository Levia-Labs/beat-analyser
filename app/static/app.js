"use strict";

// Every server-supplied string goes into the page through textContent, never innerHTML:
// folder and file names come from uploaded filenames.

const modal = new bootstrap.Modal(document.getElementById("loadingModal"));

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function sizeLabel(bytes) {
  return `${(bytes / 1024).toFixed(2)} KB`;
}

function folderItem(folder, files, index) {
  const collapseId = `collapse-${index}`;
  const item = el("div", "accordion-item");

  const header = el("h2", "accordion-header");
  const button = el("button", "accordion-button collapsed", folder);
  button.type = "button";
  button.dataset.bsToggle = "collapse";
  button.dataset.bsTarget = `#${collapseId}`;
  header.appendChild(button);

  const collapse = el("div", "accordion-collapse collapse");
  collapse.id = collapseId;
  collapse.dataset.bsParent = "#foldersAccordion";
  const body = el("div", "accordion-body p-0");

  const list = el("ul", "list-group list-group-flush");
  for (const [name, info] of Object.entries(files)) {
    const row = el("li", "list-group-item");
    row.appendChild(el("strong", null, name));
    row.appendChild(document.createTextNode(` (${sizeLabel(info.size)})`));
    if (info.content) {
      row.appendChild(el("pre", null, JSON.stringify(info.content, null, 2)));
    }
    list.appendChild(row);
  }
  body.appendChild(list);

  const footer = el("div", "text-center p-2 download-row");
  const link = el("a", "btn btn-accent btn-sm", "Download ZIP");
  link.href = `/download/${encodeURIComponent(folder)}`;
  footer.appendChild(link);
  body.appendChild(footer);

  collapse.appendChild(body);
  item.append(header, collapse);
  return item;
}

async function loadFolders() {
  const response = await fetch("/folders");
  const data = await response.json();
  const accordion = document.getElementById("foldersAccordion");
  accordion.replaceChildren(
    ...Object.entries(data).map(([folder, files], i) => folderItem(folder, files, i)),
  );
}

function filenameFrom(response, fallback) {
  const header = response.headers.get("Content-Disposition") || "";
  const match = header.match(/filename="?([^";]+)"?/);
  return match ? match[1] : fallback;
}

// POST, then hand the ZIP in the response to the browser as a download.
async function processAndDownload(url, options) {
  modal.show();
  try {
    const response = await fetch(url, { method: "POST", ...options });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      alert(body.error || `Request failed (${response.status})`);
      return;
    }
    const blob = await response.blob();
    const href = URL.createObjectURL(blob);
    const a = el("a");
    a.href = href;
    a.download = filenameFrom(response, "tracks.zip");
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(href);
    await loadFolders();
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    modal.hide();
  }
}

document.getElementById("uploadForm").addEventListener("submit", (e) => {
  e.preventDefault();
  processAndDownload("/upload", { body: new FormData(e.target) });
});

document.getElementById("ytForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const url = document.getElementById("ytUrl").value.trim();
  processAndDownload("/download_yt", {
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
});

window.addEventListener("DOMContentLoaded", loadFolders);
