"use client";

import {
  ChangeEvent,
  FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";

type IconName =
  | "activity"
  | "arrow-up"
  | "book"
  | "check"
  | "chevron-down"
  | "chevron-right"
  | "clock"
  | "cloud-upload"
  | "file"
  | "file-text"
  | "folder"
  | "grid"
  | "info"
  | "message"
  | "more"
  | "plus"
  | "search"
  | "shield"
  | "sparkles"
  | "trash"
  | "x";

type DocumentStatus = "Ready" | "Processing" | "Needs review";

type KnowledgeDocument = {
  id: string;
  name: string;
  type: string;
  size: string;
  updated: string;
  status: DocumentStatus;
};

type Source = {
  id: string;
  title: string;
  detail: string;
  score: number;
  snippet: string;
  citation: string;
};

type AssistantMessage = {
  question: string;
  answer: string;
  domain?: string;
  citations: string[];
  sources: Source[];
  abstained: boolean;
  generationMode: string;
  retrievalMode: string;
  groundedness?: number;
  citationValidity?: number;
};

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

type ApiDocument = {
  document_id: string;
  filename: string;
  extension: string;
  size_bytes: number;
  chunk_count: number;
};

function formatFileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function mapDocument(document: ApiDocument): KnowledgeDocument {
  return {
    id: document.document_id,
    name: document.filename,
    type: document.extension.replace(".", "").toUpperCase(),
    size: formatFileSize(document.size_bytes),
    updated: `${document.chunk_count} chunk${document.chunk_count === 1 ? "" : "s"}`,
    status: "Ready",
  };
}

const suggestedQuestions = [
  "What is our remote work policy?",
  "Summarize our hiring process",
  "What made Project X successful?",
];

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  const paths: Record<IconName, React.ReactNode> = {
    activity: <><path d="M3 12h4l2.3-7 4.4 14 2.3-7H21" /></>,
    "arrow-up": <><path d="m5 12 7-7 7 7" /><path d="M12 19V5" /></>,
    book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21.5z" /><path d="M4 5.5v16" /><path d="M8 7h8M8 11h7" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    "chevron-down": <path d="m6 9 6 6 6-6" />,
    "chevron-right": <path d="m9 18 6-6-6-6" />,
    clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7v5l3 2" /></>,
    "cloud-upload": <><path d="M16 16l-4-4-4 4" /><path d="M12 12v9" /><path d="M20.4 17.5A4.5 4.5 0 0 0 19 9h-1.2A6 6 0 0 0 6 10.5 3.5 3.5 0 0 0 6.5 17" /></>,
    file: <><path d="M14 2.5H6.5A1.5 1.5 0 0 0 5 4v16a1.5 1.5 0 0 0 1.5 1.5h11A1.5 1.5 0 0 0 19 20V7.5z" /><path d="M14 2.5V8h5" /></>,
    "file-text": <><path d="M14 2.5H6.5A1.5 1.5 0 0 0 5 4v16a1.5 1.5 0 0 0 1.5 1.5h11A1.5 1.5 0 0 0 19 20V7.5z" /><path d="M14 2.5V8h5M8 12h8M8 16h6" /></>,
    folder: <><path d="M3.5 6.5A1.5 1.5 0 0 1 5 5h5l2 2h7.5A1.5 1.5 0 0 1 21 8.5v9A1.5 1.5 0 0 1 19.5 19h-14A1.5 1.5 0 0 1 4 17.5z" /></>,
    grid: <><rect x="4" y="4" width="6" height="6" rx="1" /><rect x="14" y="4" width="6" height="6" rx="1" /><rect x="4" y="14" width="6" height="6" rx="1" /><rect x="14" y="14" width="6" height="6" rx="1" /></>,
    info: <><circle cx="12" cy="12" r="8.5" /><path d="M12 11v5M12 8h.01" /></>,
    message: <><path d="M20 11.5a7.5 7.5 0 0 1-8 7.5 8.3 8.3 0 0 1-3.7-.9L4 20l1.8-3.7A7.2 7.2 0 0 1 4 11.5 7.5 7.5 0 0 1 12 4a7.5 7.5 0 0 1 8 7.5Z" /></>,
    more: <><circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    search: <><circle cx="10.8" cy="10.8" r="6.5" /><path d="m16 16 4.5 4.5" /></>,
    shield: <><path d="M12 3 19 6v5.7c0 4.3-2.8 7.5-7 9.3-4.2-1.8-7-5-7-9.3V6z" /><path d="m9 12 2 2 4-4" /></>,
    sparkles: <><path d="m12 3-1.2 4.1L7 8.5l3.8 1.4L12 14l1.2-4.1L17 8.5l-3.8-1.4z" /><path d="m19 14-.6 2.4L16 17l2.4.6L19 20l.6-2.4L22 17l-2.4-.6z" /></>,
    trash: <><path d="M4 7h16M10 11v5M14 11v5" /><path d="m6 7 .7 13h10.6L18 7M9 7l.7-3h4.6l.7 3" /></>,
    x: <><path d="m6 6 12 12M18 6 6 18" /></>,
  };

  return <svg {...common}>{paths[name]}</svg>;
}

function StatusBadge({ status }: { status: DocumentStatus }) {
  const statusClass = status === "Ready" ? "ready" : status === "Processing" ? "processing" : "review";
  return (
    <span className={`status-badge ${statusClass}`}>
      {status === "Ready" ? <Icon name="check" size={12} /> : status === "Processing" ? <Icon name="clock" size={12} /> : <Icon name="info" size={12} />}
      {status}
    </span>
  );
}

function SourceCard({ source }: { source: Source }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <article className="source-card">
      <div className="source-icon"><Icon name="file-text" size={16} /></div>
      <div className="source-copy">
        <div className="source-title-row">
          <h4>{source.title}</h4>
          <span>{Math.round(source.score * 100)}% match</span>
        </div>
        <p>{source.detail} · {source.citation}</p>
        {expanded && <blockquote className="source-snippet">“{source.snippet}”</blockquote>}
        <button className="source-link" type="button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Hide passage" : "View passage"} <Icon name="chevron-right" size={13} />
        </button>
      </div>
    </article>
  );
}

export default function Home() {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [query, setQuery] = useState("");
  const [activeQuery, setActiveQuery] = useState("");
  const [message, setMessage] = useState<AssistantMessage | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);
  const [queryError, setQueryError] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [documentError, setDocumentError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [filter, setFilter] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const readyCount = documents.filter((document) => document.status === "Ready").length;
  const filteredDocuments = documents.filter((document) => document.name.toLowerCase().includes(filter.toLowerCase()));

  async function refreshDocuments() {
    const response = await fetch(`${API_URL}/documents`);
    if (!response.ok) throw new Error(`Document request failed with status ${response.status}`);
    const data = (await response.json()) as { documents?: ApiDocument[] };
    setDocuments((data.documents ?? []).map(mapDocument));
    setDocumentError("");
  }

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_URL}/documents`)
      .then((response) => {
        if (!response.ok) throw new Error(`Document request failed with status ${response.status}`);
        return response.json() as Promise<{ documents?: ApiDocument[] }>;
      })
      .then((data) => {
        if (!cancelled) setDocuments((data.documents ?? []).map(mapDocument));
      })
      .catch(() => {
        if (!cancelled) setDocumentError("The knowledge base is unavailable. Start the API to load indexed documents.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleQuery(event?: FormEvent<HTMLFormElement>, suggestedQuery?: string) {
    event?.preventDefault();
    const nextQuery = (suggestedQuery ?? query).trim();
    if (!nextQuery || isQuerying) return;

    setQuery(nextQuery);
    setActiveQuery(nextQuery);
    setIsQuerying(true);
    setQueryError("");
    setMessage(null);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: nextQuery }),
      });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      const data = (await response.json()) as {
        response?: string;
        domain?: string;
        citations?: string[];
        evidence?: Array<{ citation: string; filename: string; text: string; score: number; chunk_index: number }>;
        abstained?: boolean;
        generation_mode?: string;
        retrieval?: { mode?: string };
        evaluation?: { groundedness?: number; citation_validity?: number };
      };
      const citations = Array.isArray(data.citations) ? data.citations : [];
      setMessage({
        question: nextQuery,
        answer: data.response || "I found no answer in the connected knowledge base.",
        domain: data.domain,
        citations,
        abstained: data.abstained ?? citations.length === 0,
        generationMode: data.generation_mode ?? "unknown",
        retrievalMode: data.retrieval?.mode ?? "none",
        groundedness: data.evaluation?.groundedness,
        citationValidity: data.evaluation?.citation_validity,
        sources: (data.evidence ?? []).map((item) => ({
          id: item.citation,
          title: item.filename,
          detail: `Chunk ${item.chunk_index + 1}`,
          score: item.score,
          snippet: item.text,
          citation: item.citation,
        })),
      });
    } catch {
      setQueryError(`Couldn’t reach the knowledge service at ${API_URL}. Check that the API is running and try again.`);
    } finally {
      setIsQuerying(false);
    }
  }

  async function handleFiles(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setUploading(true);
    setUploadError("");
    const temporaryId = `${file.name}-${file.lastModified}`;
    setDocuments((current) => [
      { id: temporaryId, name: file.name, type: file.name.split(".").pop()?.toUpperCase() || "FILE", size: formatFileSize(file.size), updated: "Uploading now", status: "Processing" },
      ...current,
    ]);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_URL}/documents`, { method: "POST", body: formData });
      if (!response.ok) throw new Error("Upload failed");
      await refreshDocuments();
    } catch {
      setDocuments((current) => current.filter((document) => document.id !== temporaryId));
      setUploadError(`We couldn’t upload ${file.name}. Make sure the document service is available and try again.`);
    } finally {
      setUploading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Icon name="sparkles" size={18} /></div>
          <span>know<span className="brand-accent">wise</span></span>
        </div>
        <div className="workspace-switcher">
          <div className="workspace-avatar">AC</div>
          <div><strong>Acme Corp</strong><span>Enterprise workspace</span></div>
          <Icon name="chevron-down" size={15} />
        </div>
        <nav className="primary-nav" aria-label="Primary navigation">
          <span className="nav-label">Workspace</span>
          <a className="nav-item active" href="#ask"><Icon name="message" size={18} />Ask knowledge <span className="nav-shortcut">⌘ K</span></a>
          <a className="nav-item" href="#documents"><Icon name="folder" size={18} />Documents <span className="nav-count">{documents.length}</span></a>
          <a className="nav-item" href="#activity"><Icon name="activity" size={18} />Activity</a>
          <span className="nav-label nav-label-spaced">Manage</span>
          <a className="nav-item" href="#team"><Icon name="grid" size={18} />Team & access</a>
          <a className="nav-item" href="#settings"><Icon name="shield" size={18} />Workspace settings</a>
        </nav>
        <div className="sidebar-footer">
          <div className="usage-card"><div className="usage-heading"><span>Knowledge usage</span><Icon name="more" size={16} /></div><strong>{documents.length * 2.1 + 1.6 > 10 ? "10.0" : (documents.length * 2.1 + 1.6).toFixed(1)} GB</strong><span className="usage-muted">of 25 GB used</span><div className="usage-bar"><span style={{ width: `${Math.min(100, documents.length * 8 + 12)}%` }} /></div><span className="usage-muted">Resets on Jan 01, 2025</span></div>
          <div className="profile"><div className="profile-avatar">JD</div><div><strong>Jordan Davis</strong><span>Admin</span></div><Icon name="more" size={17} /></div>
        </div>
      </aside>

      <section className="main-content">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><Icon name="chevron-right" size={14} /><strong>Ask knowledge</strong></div>
          <div className="topbar-actions"><button className="icon-button" type="button" aria-label="Search"><Icon name="search" size={18} /></button><button className="help-button" type="button"><Icon name="info" size={16} />Help center</button><div className="topbar-avatar">JD</div></div>
        </header>

        <div className="content-grid">
          <section className="knowledge-column" id="ask">
             <div className="page-heading"><div><p className="eyebrow"><span className="live-dot" />Knowledge assistant</p><h1>Find the signal in your knowledge.</h1><p className="heading-copy">Ask questions about your company’s documents and inspect the evidence behind each result.</p></div><div className="secure-note"><Icon name="shield" size={16} /><span>Private to your workspace</span></div></div>

            <div className={`answer-panel ${message ? "has-answer" : ""}`}>
              {!message && !isQuerying && !queryError && <div className="welcome-state"><div className="welcome-icon"><Icon name="sparkles" size={22} /></div><h2>What would you like to know?</h2><p>Search across your indexed documents to get a grounded answer with clear sources.</p><div className="suggestion-list">{suggestedQuestions.map((suggestion) => <button key={suggestion} type="button" onClick={() => void handleQuery(undefined, suggestion)}><span>{suggestion}</span><Icon name="arrow-up" size={15} /></button>)}</div></div>}
              {isQuerying && <div className="loading-state"><div className="loading-orb"><span /><span /><span /></div><h2>Searching your knowledge...</h2><p>Retrieving the most relevant passages and checking their sources.</p><div className="loading-lines"><span /><span /><span /></div></div>}
              {queryError && !isQuerying && <div className="error-state"><div className="error-icon"><Icon name="info" size={21} /></div><h2>We couldn’t complete that search.</h2><p>{queryError}</p><button className="secondary-button" type="button" onClick={() => void handleQuery(undefined, activeQuery)}>Try again <Icon name="arrow-up" size={15} /></button></div>}
               {message && !isQuerying && <div className="answer-content"><div className="question-row"><div className="question-avatar">JD</div><div><span className="answer-label">Your question</span><p>{message.question}</p></div></div><div className="assistant-row"><div className="assistant-avatar"><Icon name="sparkles" size={16} /></div><div className="assistant-copy"><div className="answer-meta"><span>Answer</span>{message.domain && <span className="domain-chip">{message.domain}</span>}<span className="signal-chip">{message.generationMode}</span><span className="signal-chip">{message.retrievalMode}</span><span className="answer-time">Just now</span></div><p className="answer-text">{message.answer}</p>{message.abstained && <div className="no-sources"><Icon name="info" size={15} /> No indexed passage matched this question, so the assistant abstained.</div>}{message.sources.length > 0 && <div className="sources-section"><div className="sources-heading"><span>Evidence passages</span><span>{message.sources.length} result{message.sources.length === 1 ? "" : "s"}{message.groundedness !== undefined ? ` · ${Math.round(message.groundedness * 100)}% grounded` : ""}{message.citationValidity !== undefined ? ` · ${Math.round(message.citationValidity * 100)}% cited` : ""}</span></div><div className="source-list">{message.sources.map((source) => <SourceCard key={source.id} source={source} />)}</div></div>}</div></div></div>}
            </div>

             <form className="composer" onSubmit={handleQuery}><div className="composer-top"><textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask a question about your knowledge base..." rows={2} aria-label="Ask a question" /><button className="send-button" type="submit" disabled={!query.trim() || isQuerying} aria-label="Send question"><Icon name="arrow-up" size={18} /></button></div><div className="composer-bottom"><div className="composer-hint"><Icon name="sparkles" size={14} /> Answers are assembled from indexed evidence</div><span className="keyboard-hint">Press <kbd>⌘</kbd><kbd>↵</kbd> to send</span></div></form>
            <p className="disclaimer"><Icon name="info" size={13} /> AI-generated answers may be incomplete. Always verify important information in the source document.</p>
          </section>

          <aside className="documents-column" id="documents">
            <div className="panel-heading"><div><h2>Knowledge base</h2><p>{readyCount} ready to answer questions</p></div><button className="add-button" type="button" onClick={() => fileInputRef.current?.click()}><Icon name="plus" size={16} />Add files</button></div>
             <div className="upload-panel" onClick={() => fileInputRef.current?.click()} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInputRef.current?.click(); }}><div className="upload-icon"><Icon name="cloud-upload" size={20} /></div><div><strong>{uploading ? "Uploading document..." : "Add a source document"}</strong><p>TXT, MD, or Markdown up to 5 MB</p></div><Icon name="chevron-right" size={17} /></div>
             <input ref={fileInputRef} type="file" accept=".txt,.md,.markdown" onChange={(event) => void handleFiles(event)} hidden />
             {uploadError && <div className="inline-error"><Icon name="info" size={15} /><span>{uploadError}</span><button type="button" onClick={() => setUploadError("")} aria-label="Dismiss upload error"><Icon name="x" size={14} /></button></div>}
             {documentError && <div className="inline-error"><Icon name="info" size={15} /><span>{documentError}</span><button type="button" onClick={() => { setDocumentError(""); void refreshDocuments().catch(() => setDocumentError("The knowledge base is still unavailable.")); }} aria-label="Retry loading documents"><Icon name="clock" size={14} /></button></div>}
            <div className="document-toolbar"><div className="doc-title"><span>Documents</span><span className="doc-count">{documents.length}</span></div><div className="doc-tools"><label className="search-field"><Icon name="search" size={15} /><input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter" aria-label="Filter documents" /></label><button className="icon-button small" type="button" aria-label="Document options"><Icon name="more" size={16} /></button></div></div>
             <div className="document-list">{filteredDocuments.length === 0 ? <div className="empty-documents"><div className="empty-icon"><Icon name="folder" size={20} /></div><strong>No documents found</strong><p>{documents.length === 0 ? "Upload a file to start building your knowledge base." : "Try a different search term."}</p>{documents.length === 0 && <button type="button" onClick={() => fileInputRef.current?.click()}>Upload your first file <Icon name="arrow-up" size={14} /></button>}</div> : filteredDocuments.map((document) => <div className="document-row" key={document.id}><div className={`document-icon ${document.type.toLowerCase()}`}><Icon name="file-text" size={17} /></div><div className="document-details"><strong title={document.name}>{document.name}</strong><span>{document.type} · {document.size} · {document.updated}</span></div><StatusBadge status={document.status} /><button type="button" className="row-more" aria-label={`Options for ${document.name}`}><Icon name="more" size={16} /></button></div>)}</div>
            <button className="view-all" type="button">View all documents <Icon name="chevron-right" size={15} /></button>
            <div className="index-status"><div className="index-status-icon"><Icon name="check" size={16} /></div><div><strong>Index is up to date</strong><p>Last synced just now</p></div><button type="button" aria-label="Index status information"><Icon name="info" size={15} /></button></div>
          </aside>
        </div>
      </section>
    </main>
  );
}
