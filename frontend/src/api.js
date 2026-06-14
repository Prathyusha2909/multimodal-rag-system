const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export const api = {
  getDocuments: () => request("/api/v1/documents"),
  getStats: () => request("/api/v1/stats"),
  query: (question) =>
    request("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 4 }),
    }),
  upload: (file) => {
    const data = new FormData();
    data.append("file", file);
    return request("/api/v1/documents/upload", { method: "POST", body: data });
  },
};

