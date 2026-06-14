import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { demoDocuments, demoStats, initialResult } from "./demoData";

const sampleQuestions = [
  "What is the revenue trend shown in Figure 3?",
  "Compare Figure 3 and Table 2.",
  "Summarize the findings from pages 12-18.",
  "How are sales and profit related?",
];

const Icon = ({ name, size = 18 }) => {
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8M8 17h6"/></>,
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
    upload: <><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/></>,
    spark: <><path d="m12 3-1.4 3.6L7 8l3.6 1.4L12 13l1.4-3.6L17 8l-3.6-1.4L12 3z"/><path d="m5 14-.8 2.2L2 17l2.2.8L5 20l.8-2.2L8 17l-2.2-.8L5 14z"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    chart: <><path d="M4 19V9M10 19V5M16 19v-7M22 19V3"/><path d="M2 19h22"/></>,
    database: <><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v6c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 11v6c0 1.7 4 3 9 3s9-1.3 9-3v-6"/></>,
  };
  return <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
};

function App() {
  const [documents, setDocuments] = useState(demoDocuments);
  const [stats, setStats] = useState(demoStats);
  const [question, setQuestion] = useState("What is the revenue trend shown in Figure 3?");
  const [result, setResult] = useState(initialResult);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState("");
  const [apiOnline, setApiOnline] = useState(false);
  const fileInput = useRef(null);

  useEffect(() => {
    Promise.all([api.getDocuments(), api.getStats()])
      .then(([nextDocuments, nextStats]) => {
        setDocuments(nextDocuments);
        setStats(nextStats);
        setApiOnline(true);
      })
      .catch(() => {
        setApiOnline(false);
        setNotice("Demo mode: start the API to enable live uploads and retrieval.");
      });
  }, []);

  const modalityCount = useMemo(
    () => Object.values(stats.modalities || {}).reduce((sum, count) => sum + count, 0),
    [stats],
  );

  const ask = async (nextQuestion = question) => {
    const cleaned = nextQuestion.trim();
    if (!cleaned || loading) return;
    setQuestion(cleaned);
    setLoading(true);
    setNotice("");
    try {
      setResult(await api.query(cleaned));
      setApiOnline(true);
    } catch (error) {
      setApiOnline(false);
      setNotice(`${error.message}. Showing the built-in grounded demo response.`);
      setResult(initialResult);
    } finally {
      setLoading(false);
    }
  };

  const upload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setNotice("");
    try {
      const document = await api.upload(file);
      setApiOnline(true);
      setDocuments((current) => [document, ...current.filter((item) => item.id !== document.id)]);
      setStats(await api.getStats());
      setNotice(`${file.name} indexed successfully.`);
    } catch (error) {
      setApiOnline(false);
      setNotice(error.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><span></span><span></span><span></span></div>
          <div><strong>Prism</strong><small>Multimodal RAG</small></div>
        </div>

        <nav aria-label="Main navigation">
          <button className="nav-item active"><Icon name="grid" />Workspace</button>
          <button className="nav-item"><Icon name="file" />Documents<span>{stats.documents}</span></button>
          <button className="nav-item"><Icon name="search" />Search history</button>
        </nav>

        <div className="sidebar-section">
          <p>INDEX HEALTH</p>
          <div className="health-row"><span className="health-dot"></span><div><strong>Demo index ready</strong><small>{stats.chunks} chunks indexed</small></div></div>
        </div>

        <div className="pipeline-card">
          <div className="pipeline-label"><Icon name="database" /><span>Retrieval pipeline</span></div>
          <strong>Hybrid + reranking</strong>
          <small>Semantic and lexical evidence fused before generation.</small>
          <div className="pipeline-tags"><span>SentenceTransformers</span><span>FAISS</span><span>BM25</span><span>CrossEncoder</span></div>
        </div>

        <div className="profile">
          <div className="avatar">P</div>
          <div><strong>Prathyusha</strong><small>AI Engineer</small></div>
          <button aria-label="Open profile">...</button>
        </div>
      </aside>

      <main>
        <header>
          <div><p className="eyebrow">DOCUMENT INTELLIGENCE</p><h1>Ask across document evidence.</h1></div>
          <div className="header-actions">
            <span className={`live-badge ${apiOnline ? "" : "offline"}`}><i></i>{apiOnline ? "API live" : "Demo mode"}</span>
            <input ref={fileInput} type="file" hidden accept=".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff,.txt,.md,.csv" onChange={upload} />
            <button className="upload-button" onClick={() => fileInput.current?.click()} disabled={uploading}><Icon name="upload" />{uploading ? "Indexing..." : "Upload document"}</button>
          </div>
        </header>

        {notice && <div className="notice">{notice}</div>}

        <section className="hero-grid">
          <div className="query-panel">
            <div className="panel-heading"><span className="step">01</span><div><p>QUERY WORKSPACE</p><h2>What do you want to understand?</h2></div></div>
            <div className={`question-box ${loading ? "loading" : ""}`}>
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ask(); } }} aria-label="Ask a question about your documents" />
              <div className="question-footer"><span><Icon name="spark" size={16} />Grounded in {stats.documents} documents</span><button onClick={() => ask()} disabled={loading} aria-label="Send query"><Icon name="send" size={17} /></button></div>
            </div>
            <div className="suggestions">
              <span>TRY AN EXAMPLE</span>
              {sampleQuestions.map((sample) => <button key={sample} onClick={() => ask(sample)}>{sample}<Icon name="chevron" size={14} /></button>)}
            </div>
          </div>

          <div className="metrics-panel">
            <div className="metric-top"><p>INDEX OVERVIEW</p><span>{apiOnline ? "API" : "Sample"}</span></div>
            <div className="big-metric"><strong>{String(stats.pages).padStart(2, "0")}</strong><span>pages<br />indexed</span></div>
            <div className="metric-grid">
              <div><span>DOCUMENTS</span><strong>{stats.documents}</strong><small>ready to query</small></div>
              <div><span>MODAL SIGNALS</span><strong>{modalityCount}</strong><small>across the index</small></div>
            </div>
            <div className="modality-list">
              {Object.entries(stats.modalities || {}).map(([name, count]) => <div key={name}><span className={`modality-icon ${name}`}>{name.slice(0, 1).toUpperCase()}</span><span>{name}</span><i><b style={{ width: `${Math.max(16, (count / Math.max(...Object.values(stats.modalities))) * 100)}%` }}></b></i><strong>{count}</strong></div>)}
            </div>
          </div>
        </section>

        <section className="answer-section">
          <div className="section-title"><span className="step">02</span><div><p>GROUNDED RESPONSE</p><h2>Evidence, synthesized.</h2></div><div className="timing"><span>{result.retrieval_ms} ms retrieval</span><span>{result.generation_ms} ms generation</span></div></div>
          <div className="answer-grid">
            <article className="answer-card">
              <div className="answer-label"><Icon name="spark" /><span>ANSWER</span><small>{result.model}</small></div>
              <p>{result.answer}</p>
              <div className="confidence"><Icon name="check" size={15} /><span>Answer includes {result.citations.length} retrieved sources for inspection</span></div>
            </article>
            <aside className="sources-card">
              <div className="sources-heading"><span>RETRIEVED EVIDENCE</span><strong>{result.citations.length} sources</strong></div>
              {result.citations.map((citation) => <button className="source" key={`${citation.document_name}-${citation.page}-${citation.index}`}>
                <span className={`source-number ${citation.modality}`}>{citation.index}</span>
                <div><p>{citation.document_name}</p><span>Page {citation.page} <i></i> {citation.modality}</span><small>{citation.excerpt}</small></div>
                <strong title="Internal reranking score">{citation.score.toFixed(2)}</strong>
              </button>)}
            </aside>
          </div>
        </section>

        <section className="documents-section">
          <div className="documents-heading"><div><p>KNOWLEDGE BASE</p><h2>Indexed documents</h2></div><button onClick={() => fileInput.current?.click()}>Add source</button></div>
          <div className="document-list">
            {documents.map((document) => <div className="document-row" key={document.id}>
              <div className="file-mark">PDF</div><div className="document-info"><strong>{document.name}</strong><span>{document.pages} pages <i></i> {document.chunks} chunks</span></div>
              <div className="document-modalities">{document.modalities.slice(0, 4).map((modality) => <span key={modality}>{modality}</span>)}</div>
              <div className="ready"><i></i>{document.status}</div>
            </div>)}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
