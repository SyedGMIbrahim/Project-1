"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type SourceDocument = {
  content: string;
  metadata: Record<string, unknown>;
};

type QueryResponse = {
  answer: string;
  source_documents: SourceDocument[];
};

type UploadResponse = {
  message: string;
  original_file_name: string;
  stored_file_name: string;
  file_type: string;
  total_chunks_indexed: number;
  source_path: string;
};

type DocumentRecord = {
  file_name: string;
  source_path: string | null;
  file_type: string | null;
  upload_date: string | null;
  file_size: number | null;
  status: string;
  chunk_count: number | null;
};

const API_URL = "http://localhost:8000/api/query";
const UPLOAD_URL = "http://localhost:8000/api/upload";
const DOCUMENTS_URL = "http://localhost:8000/api/documents";

function Spinner() {
  return (
    <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" aria-hidden="true" />
  );
}

export default function Page() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SourceDocument[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [notification, setNotification] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [availableSymptoms, setAvailableSymptoms] = useState<string[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [diagnosisLoading, setDiagnosisLoading] = useState(false);
  const [diagnosisResult, setDiagnosisResult] = useState<{ predicted: string; probability: number } | null>(null);
  const DIAGNOSE_URL = "http://localhost:8000/api/diagnose";
  const EXTRACT_URL = "http://localhost:8000/api/extract-symptoms";
  const EXTRACT_FROM_DOCS_URL = "http://localhost:8000/api/extract-from-docs";
  const SYMPTOMS_URL = "http://localhost:8000/api/symptoms";
  const [nlText, setNlText] = useState("");
  const [extractingFromText, setExtractingFromText] = useState(false);
  const [extractingFromDocs, setExtractingFromDocs] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  const canSubmit = useMemo(() => query.trim().length > 0 && !loading, [query, loading]);
  const symptomSuggestions = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    if (!term) return [];
    const selectedSet = new Set(selectedSymptoms.map((symptom) => symptom.toLowerCase()));
    return availableSymptoms.filter(
      (symptom) => symptom.toLowerCase().includes(term) && !selectedSet.has(symptom.toLowerCase())
    );
  }, [availableSymptoms, selectedSymptoms, searchTerm]);

  async function loadDocuments(options?: { notifyOnError?: boolean }) {
    setDocumentsLoading(true);

    try {
      const response = await fetch(DOCUMENTS_URL, { cache: "no-store" });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Failed to load indexed documents");
      }

      const data = (await response.json()) as DocumentRecord[];
      setDocuments(Array.isArray(data) ? data : []);
    } catch (err) {
      if (options?.notifyOnError !== false) {
        showNotification("error", err instanceof Error ? err.message : "Failed to load indexed documents");
      }
    } finally {
      setDocumentsLoading(false);
    }
  }

  useEffect(() => {
    void loadDocuments({ notifyOnError: false });
    void fetch(SYMPTOMS_URL, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return (await res.json()) as string[];
      })
      .then((data) => {
        setAvailableSymptoms(Array.isArray(data) ? data : []);
      })
      .catch(() => {
        // Silently catch initial load errors if backend is starting up
      });
  }, []);

  function addSymptom(symptom: string) {
    setSelectedSymptoms((prev) => {
      if (prev.some((item) => item.toLowerCase() === symptom.toLowerCase())) {
        return prev;
      }
      return [...prev, symptom];
    });
    setSearchTerm("");
  }

  function removeSymptom(symptom: string) {
    setSelectedSymptoms((prev) => prev.filter((item) => item.toLowerCase() !== symptom.toLowerCase()));
  }

  function applyExtractedSymptoms(extracted: string[]) {
    const allowedMap = new Map(availableSymptoms.map((symptom) => [symptom.toLowerCase(), symptom]));
    const selected: string[] = [];
    const seen = new Set<string>();

    for (const item of Array.isArray(extracted) ? extracted : []) {
      if (typeof item !== "string") continue;
      const mapped = allowedMap.get(item.toLowerCase());
      if (mapped && !seen.has(mapped.toLowerCase())) {
        selected.push(mapped);
        seen.add(mapped.toLowerCase());
      }
    }

    setSelectedSymptoms(selected);
    showNotification("success", `Extracted ${selected.length} symptom${selected.length === 1 ? "" : "s"}`);
  }

  async function handleExtractFromText() {
    if (!nlText.trim()) {
      showNotification("error", "Please enter a description to extract symptoms.");
      return;
    }

    setExtractingFromText(true);
    try {
      const res = await fetch(EXTRACT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: nlText, description: nlText }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const symptomsList = Array.isArray(data) ? data : Array.isArray(data?.symptoms) ? data.symptoms : [];
      applyExtractedSymptoms(symptomsList);
    } catch (err) {
      showNotification("error", err instanceof Error ? err.message : "Extraction failed");
    } finally {
      setExtractingFromText(false);
    }
  }

  async function handleExtractFromDocs() {
    setExtractingFromDocs(true);
    try {
      const res = await fetch(EXTRACT_FROM_DOCS_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const symptomsList = Array.isArray(data) ? data : Array.isArray(data?.symptoms) ? data.symptoms : [];
      applyExtractedSymptoms(symptomsList);
    } catch (err) {
      showNotification("error", err instanceof Error ? err.message : "Document extraction failed");
    } finally {
      setExtractingFromDocs(false);
    }
  }

  async function handleDiagnose() {
    if (selectedSymptoms.length === 0) {
      showNotification("error", "Please select one or more symptoms to run diagnosis.");
      return;
    }

    setDiagnosisLoading(true);
    setDiagnosisResult(null);

    try {
      const payload = { symptoms: selectedSymptoms };
      const res = await fetch(DIAGNOSE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Diagnosis failed");
      }

      const data = await res.json();
      setDiagnosisResult({ predicted: data.predicted, probability: data.probability });
      showNotification("success", `Predicted: ${data.predicted} (${Math.round(data.probability * 100)}%)`);
    } catch (err) {
      showNotification("error", err instanceof Error ? err.message : "Diagnosis failed");
    } finally {
      setDiagnosisLoading(false);
    }
  }

  async function searchGuidelines(disease: string) {
    // reuse the query endpoint to search local guidelines for the disease name
    setLoading(true);
    setError(null);
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: disease }),
      });
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as QueryResponse;
      setAnswer(data.answer);
      setSources(Array.isArray(data.source_documents) ? data.source_documents : []);
    } catch (err) {
      showNotification("error", err instanceof Error ? err.message : "Failed to search guidelines");
    } finally {
      setLoading(false);
    }
  }

  function showNotification(type: "success" | "error", text: string) {
    setNotification({ type, text });
    window.setTimeout(() => {
      setNotification((current) => (current?.text === text ? null : current));
    }, 4000);
  }

  async function uploadClinicalFile(file: File) {
    const validExtension = /\.(pdf|txt)$/i.test(file.name);
    if (!validExtension) {
      showNotification("error", "Only PDF and TXT files can be uploaded.");
      return;
    }

    setUploading(true);
    setNotification(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(UPLOAD_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Failed to upload and index the clinical file");
      }

      const data = (await response.json()) as UploadResponse;
      showNotification(
        "success",
        `${data.original_file_name} indexed successfully with ${data.total_chunks_indexed} chunk${data.total_chunks_indexed === 1 ? "" : "s"}.`
      );
      void loadDocuments({ notifyOnError: false });
    } catch (err) {
      showNotification("error", err instanceof Error ? err.message : "An unexpected upload error occurred");
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);

    const droppedFile = event.dataTransfer.files?.[0];
    if (droppedFile) {
      void uploadClinicalFile(droppedFile);
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery || loading) {
      return;
    }

    setLoading(true);
    setError(null);
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: trimmedQuery }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Failed to fetch response from the local API");
      }

      const data = (await response.json()) as QueryResponse;
      setAnswer(data.answer);
      setSources(Array.isArray(data.source_documents) ? data.source_documents : []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-clinical-gradient text-ink-950">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8 flex items-center justify-between rounded-3xl border border-plum-100 bg-white/80 px-5 py-4 shadow-soft backdrop-blur">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.35em] text-plum-700">Local Clinical RAG</p>
            <h1 className="mt-1 text-2xl font-semibold text-ink-950 sm:text-3xl">Clinical Document Intelligence</h1>
          </div>
          <div className="hidden rounded-full border border-plum-100 bg-plum-50 px-4 py-2 text-sm text-plum-800 sm:block">
            100% local analysis
          </div>
        </header>

        {notification ? (
          <div
            className={`mb-6 rounded-2xl border px-4 py-3 text-sm shadow-sm ${notification.type === "success" ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-red-200 bg-red-50 text-red-700"}`}
          >
            {notification.text}
          </div>
        ) : null}

        <section className="grid flex-1 gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="flex flex-col rounded-3xl border border-plum-100 bg-white/90 p-5 shadow-soft backdrop-blur sm:p-8">
            <div className="mb-6">
              <p className="text-sm font-medium text-plum-700">Ask a question</p>
              <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-600">
                Query the local medical record index and review the generated answer alongside the retrieved evidence.
              </p>
            </div>

            <div className="mb-6 rounded-3xl border border-dashed border-plum-200 bg-plum-50/50 p-4">
              <div
                onDragEnter={() => setDragActive(true)}
                onDragLeave={() => setDragActive(false)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={handleDrop}
                className={`rounded-2xl border-2 border-dashed p-5 text-center transition ${dragActive ? "border-plum-500 bg-white" : "border-plum-200 bg-white/80"}`}
              >
                <p className="text-sm font-semibold text-ink-950">Upload Clinical File</p>
                <p className="mt-1 text-sm text-slate-600">Drag and drop a .pdf or .txt document to index it locally.</p>
                <div className="mt-4 flex flex-col items-center justify-center gap-3 sm:flex-row">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                    className="inline-flex items-center justify-center gap-2 rounded-2xl bg-plum-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-plum-800 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {uploading ? <Spinner /> : null}
                    {uploading ? "Processing..." : "Choose file"}
                  </button>
                  <span className="text-xs text-slate-500">Supported: PDF, TXT</span>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt"
                  className="hidden"
                  onChange={(event) => {
                    const selectedFile = event.target.files?.[0];
                    if (selectedFile) {
                      void uploadClinicalFile(selectedFile);
                    }
                  }}
                />
              </div>
              {uploading ? (
                <div className="mt-3 rounded-2xl bg-plum-50 px-4 py-3 text-sm text-plum-800">Processing and indexing document...</div>
              ) : null}
            </div>

            <section className="mb-8 rounded-3xl border border-plum-100 bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold text-ink-950">Currently Indexed Datasets</h2>
                  <p className="mt-1 text-sm text-slate-600">Documents available in the local Chroma index.</p>
                </div>
                {documentsLoading ? <span className="text-xs font-medium text-plum-700">Refreshing...</span> : null}
              </div>

              {documents.length > 0 ? (
                <div className="overflow-hidden rounded-2xl border border-plum-100">
                  <div className="grid grid-cols-[1.2fr_1.4fr_0.8fr_0.9fr_0.9fr] gap-3 border-b border-plum-100 bg-plum-50 px-4 py-3 text-xs font-semibold uppercase tracking-[0.18em] text-plum-700">
                    <span>File</span>
                    <span>Source</span>
                    <span>Size</span>
                    <span>Status</span>
                    <span>Indexed</span>
                  </div>
                  <div className="divide-y divide-plum-100 bg-white">
                    {documents.map((document) => {
                      const fileType = (document.file_type || "doc").toUpperCase();
                      const uploadDate = document.upload_date ? new Date(document.upload_date).toLocaleString() : "N/A";
                      const fileSize = typeof document.file_size === "number" ? `${(document.file_size / 1024).toFixed(1)} KB` : "N/A";

                      return (
                        <div key={document.source_path ?? document.file_name} className="grid grid-cols-[1.2fr_1.4fr_0.8fr_0.9fr_0.9fr] gap-3 px-4 py-3 text-sm">
                          <div className="flex items-start gap-3">
                            <div className="mt-0.5 flex h-9 w-9 items-center justify-center rounded-xl bg-plum-50 text-xs font-bold text-plum-700 ring-1 ring-plum-100">
                              {fileType}
                            </div>
                            <div>
                              <p className="font-medium text-ink-950">{document.file_name}</p>
                              <p className="text-xs text-slate-500">{uploadDate}</p>
                            </div>
                          </div>
                          <div className="truncate text-slate-600" title={document.source_path ?? undefined}>
                            {document.source_path || "Local index"}
                          </div>
                          <div className="text-slate-600">{fileSize}</div>
                          <div>
                            <span className="inline-flex rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
                              {document.status}
                            </span>
                          </div>
                          <div className="text-slate-600">{document.chunk_count ?? 0} chunks</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-plum-200 bg-plum-50/40 px-5 py-10 text-sm text-slate-500">
                  No clinical datasets are currently indexed.
                </div>
              )}
            </section>

            <form onSubmit={handleSubmit} className="mb-8">
              <label htmlFor="query" className="sr-only">
                Medical document query
              </label>
              <div className="flex flex-col gap-3 rounded-3xl border border-plum-100 bg-plum-50/70 p-3 shadow-sm sm:flex-row">
                <textarea
                  id="query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  rows={3}
                  placeholder="e.g. What follow-up treatment is recommended for the patient's hypertension?"
                  className="min-h-[88px] flex-1 resize-none rounded-2xl border border-transparent bg-white px-4 py-3 text-base outline-none ring-0 placeholder:text-slate-400 focus:border-plum-300 focus:shadow-[0_0_0_4px_rgba(140,73,223,0.10)]"
                />
                <button
                  type="submit"
                  disabled={!canSubmit}
                  className="inline-flex items-center justify-center gap-2 rounded-2xl bg-plum-700 px-5 py-3 text-sm font-semibold text-white transition hover:bg-plum-800 disabled:cursor-not-allowed disabled:opacity-50 sm:min-w-[148px]"
                >
                  {loading ? <Spinner /> : null}
                  {loading ? "Searching..." : "Query records"}
                </button>
              </div>
            </form>

            <div className="flex-1 rounded-3xl border border-plum-100 bg-white p-5">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-ink-950">Generated answer</h2>
                {loading ? <span className="text-sm text-plum-700">Analyzing local sources...</span> : null}
              </div>

              {error ? (
                <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              ) : answer ? (
                <p className="whitespace-pre-wrap text-[15px] leading-7 text-slate-700">{answer}</p>
              ) : (
                <div className="rounded-2xl border border-dashed border-plum-200 bg-plum-50/40 px-5 py-10 text-sm text-slate-500">
                  The answer will appear here after a query is submitted.
                </div>
              )}
            </div>
          </div>

          <div className="mt-8 rounded-3xl border border-plum-100 bg-white/90 p-5 shadow-soft backdrop-blur sm:p-6">
            <div className="mb-4">
              <h3 className="text-xl font-semibold text-ink-950">Symptom Checker</h3>
              <p className="mt-1 text-sm text-slate-600">Select symptoms manually or use AI extraction from text and indexed clinical documents.</p>
            </div>

            <div className="rounded-2xl border border-plum-100 bg-plum-50/40 p-4">
              <label htmlFor="nl-symptoms" className="mb-2 block text-sm font-medium text-plum-800">Describe symptoms naturally</label>
              <textarea
                id="nl-symptoms"
                value={nlText}
                onChange={(e) => setNlText(e.target.value)}
                rows={3}
                placeholder="Describe symptoms naturally..."
                className="min-h-[88px] w-full resize-none rounded-2xl border border-plum-100 bg-white px-4 py-3 text-sm text-slate-700 placeholder:text-slate-400 focus:border-plum-300 focus:outline-none focus:shadow-[0_0_0_4px_rgba(140,73,223,0.10)]"
              />

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handleExtractFromText()}
                  className="inline-flex items-center justify-center rounded-xl bg-plum-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-plum-800 disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={extractingFromText}
                >
                  {extractingFromText ? "Extracting..." : "🪄 Auto-Select from Text"}
                </button>
                <button
                  type="button"
                  onClick={() => void handleExtractFromDocs()}
                  className="inline-flex items-center justify-center rounded-xl bg-white px-4 py-2 text-sm font-semibold text-plum-800 ring-1 ring-plum-200 transition hover:bg-plum-50 disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={extractingFromDocs}
                >
                  {extractingFromDocs ? "Extracting..." : "📄 Extract from Uploaded Docs"}
                </button>
                <button
                  type="button"
                  onClick={() => setNlText("")}
                  className="text-sm font-medium text-slate-500 underline underline-offset-2"
                >
                  Clear
                </button>
              </div>
            </div>

            <div className="mt-4">
              <label htmlFor="symptom-search" className="mb-2 block text-sm font-medium text-plum-800">Search and add symptoms</label>
              <input
                id="symptom-search"
                value={searchTerm}
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="Type symptom name..."
                className="w-full rounded-2xl border border-plum-100 bg-white px-4 py-2.5 text-sm text-slate-700 placeholder:text-slate-400 focus:border-plum-300 focus:outline-none focus:shadow-[0_0_0_4px_rgba(140,73,223,0.10)]"
              />
            </div>

            {searchTerm.trim() ? (
              <div className="mt-3 max-h-60 overflow-y-auto rounded-2xl border border-plum-100 bg-white p-2">
                {symptomSuggestions.length === 0 ? (
                  <div className="px-3 py-2 text-sm text-slate-500">No matching symptoms found.</div>
                ) : (
                  <ul className="space-y-1">
                    {symptomSuggestions.map((symptom) => (
                      <li key={symptom}>
                        <button
                          type="button"
                          onClick={() => addSymptom(symptom)}
                          className="w-full rounded-xl px-3 py-2 text-left text-sm text-slate-700 transition hover:bg-plum-50"
                        >
                          {symptom}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ) : null}

            <div className="mt-3 rounded-2xl border border-plum-100 bg-white p-4">
              {availableSymptoms.length === 0 ? (
                <div className="rounded-xl bg-plum-50 px-3 py-4 text-sm text-slate-500">No symptom list available.</div>
              ) : selectedSymptoms.length === 0 ? (
                <div className="rounded-xl bg-plum-50 px-3 py-4 text-sm text-slate-500">No symptoms selected yet.</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {selectedSymptoms.map((symptom) => (
                    <span
                      key={symptom}
                      className="inline-flex items-center gap-2 rounded-full bg-plum-100 px-3 py-1.5 text-sm font-medium text-plum-800"
                    >
                      <span>{symptom}</span>
                      <button
                        type="button"
                        onClick={() => removeSymptom(symptom)}
                        className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-white text-plum-700 ring-1 ring-plum-200 hover:bg-plum-50"
                        aria-label={`Remove ${symptom}`}
                      >
                        X
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => void handleDiagnose()}
                className="inline-flex items-center justify-center rounded-xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={diagnosisLoading}
              >
                {diagnosisLoading ? "Diagnosing..." : "Run Diagnosis"}
              </button>
              {diagnosisResult ? (
                <div className="rounded-xl bg-emerald-50 px-3 py-2 text-sm text-emerald-800 ring-1 ring-emerald-100">
                  <strong>{diagnosisResult.predicted}</strong> - {Math.round(diagnosisResult.probability * 100)}% confidence
                  <button
                    type="button"
                    className="ml-3 font-semibold text-plum-700 underline underline-offset-2"
                    onClick={() => void searchGuidelines(diagnosisResult.predicted)}
                  >
                    Search Guidelines
                  </button>
                </div>
              ) : null}
            </div>
          </div>

          <aside className="rounded-3xl border border-plum-100 bg-white/90 p-5 shadow-soft backdrop-blur sm:p-8">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-ink-950">Evidence panel</h2>
                <p className="mt-1 text-sm text-slate-600">Retrieved source chunks from the local vector store.</p>
              </div>
              <span className="rounded-full bg-plum-50 px-3 py-1 text-xs font-semibold text-plum-700">
                Top {sources.length || 3}
              </span>
            </div>

            <div className="space-y-3">
              {sources.length > 0 ? (
                sources.map((source, index) => {
                  const fileName = typeof source.metadata.file_name === "string" ? source.metadata.file_name : "Unknown source";
                  const page = source.metadata.page;
                  const pageLabel = typeof page === "number" ? `Page ${page}` : "";

                  return (
                    <details key={`${fileName}-${index}`} className="group rounded-2xl border border-plum-100 bg-plum-50/40 p-4 open:bg-white">
                      <summary className="cursor-pointer list-none text-sm font-semibold text-ink-950">
                        <div className="flex items-start justify-between gap-3">
                          <span>{fileName}</span>
                          <span className="rounded-full bg-white px-2.5 py-1 text-xs font-medium text-plum-700 ring-1 ring-plum-100">
                            Chunk {index + 1}
                          </span>
                        </div>
                        <div className="mt-1 text-xs font-normal text-slate-500">{pageLabel || "Document evidence"}</div>
                      </summary>
                      <div className="mt-3 rounded-2xl border border-plum-100 bg-white p-4">
                        <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">{source.content}</p>
                        {Object.keys(source.metadata).length > 0 ? (
                          <dl className="mt-4 grid gap-2 rounded-xl bg-plum-50 p-3 text-xs text-slate-600 sm:grid-cols-2">
                            {Object.entries(source.metadata).map(([key, value]) => (
                              <div key={key}>
                                <dt className="font-semibold text-plum-800">{key}</dt>
                                <dd className="break-words">{String(value)}</dd>
                              </div>
                            ))}
                          </dl>
                        ) : null}
                      </div>
                    </details>
                  );
                })
              ) : (
                <div className="rounded-2xl border border-dashed border-plum-200 bg-plum-50/40 px-5 py-10 text-sm text-slate-500">
                  Evidence chunks will appear here after the backend returns the top 3 matches.
                </div>
              )}
            </div>
          </aside>
        </section>
      </div>
    </main>
  );
}
