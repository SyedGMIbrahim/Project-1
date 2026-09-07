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

type DiagnoseResponseData = {
  predicted: string;
  probability: number;
  probabilities?: Record<string, number>;
  decision: "DIAGNOSE" | "CLARIFY" | "ABSTAIN";
  sparsity_score: number;
  active_symptom_count: number;
  evidence_coverage: number;
  abstention_reason?: string | null;
  clarification_question?: string | null;
  symptom_states?: {
    supported: string[];
    uncertain: string[];
    unsupported: string[];
  };
  evidence_details?: {
    candidate_disease: string;
    model_probability: number;
    evidence_coverage: number;
    adaptive_threshold: number;
    present_hallmarks: Array<{ symptom: string; weight: number }>;
    missing_hallmarks: Array<{ symptom: string; weight: number }>;
    sparsity_metrics: {
      sparsity_score: number;
      active_symptom_count: number;
      z_score: number;
      is_sparse: boolean;
    };
  };
};

const API_URL = "http://localhost:8000/api/query";
const UPLOAD_URL = "http://localhost:8000/api/upload";
const DOCUMENTS_URL = "http://localhost:8000/api/documents";
const SUMMARIZE_URL = "http://localhost:8000/api/summarize";

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
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [notification, setNotification] = useState<{ type: "success" | "error" | "info"; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [availableSymptoms, setAvailableSymptoms] = useState<string[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [uncertainSymptoms, setUncertainSymptoms] = useState<string[]>([]);
  const [unsupportedSymptoms, setUnsupportedSymptoms] = useState<string[]>([]);
  const [diagnosisLoading, setDiagnosisLoading] = useState(false);
  const [diagnosisResult, setDiagnosisResult] = useState<DiagnoseResponseData | null>(null);
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
      const docs = Array.isArray(data) ? data : [];
      setDocuments(docs);
      if (docs.length > 0) {
        setSelectedDocument((prev) => (prev && docs.some((d) => d.file_name === prev) ? prev : docs[0].file_name));
      }
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

  function applyExtractedSymptoms(extracted: string[], uncertain: string[] = [], unsupported: string[] = []) {
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
    setUncertainSymptoms(Array.isArray(uncertain) ? uncertain : []);
    setUnsupportedSymptoms(Array.isArray(unsupported) ? unsupported : []);
    const uncertMsg = uncertain.length ? `, ${uncertain.length} uncertain` : "";
    const unsuppMsg = unsupported.length ? `, ${unsupported.length} unmapped` : "";
    showNotification("success", `Extracted ${selected.length} supported symptom${selected.length === 1 ? "" : "s"}${uncertMsg}${unsuppMsg}`);
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
      applyExtractedSymptoms(symptomsList, data?.uncertain || [], data?.unsupported || []);
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
        body: JSON.stringify({ file_name: selectedDocument || undefined }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const symptomsList = Array.isArray(data) ? data : Array.isArray(data?.symptoms) ? data.symptoms : [];
      applyExtractedSymptoms(symptomsList, data?.uncertain || [], data?.unsupported || []);
    } catch (err) {
      showNotification("error", err instanceof Error ? err.message : "Document extraction failed");
    } finally {
      setExtractingFromDocs(false);
    }
  }

  async function handleDiagnose(overrideSymptoms?: string[]) {
    const symptomsToUse = overrideSymptoms || selectedSymptoms;
    if (symptomsToUse.length === 0) {
      showNotification("error", "Please select one or more symptoms to run diagnosis.");
      return;
    }

    setDiagnosisLoading(true);
    setDiagnosisResult(null);

    try {
      const payload = {
        symptoms: symptomsToUse,
        uncertain_symptoms: uncertainSymptoms,
        unsupported_symptoms: unsupportedSymptoms,
      };
      const res = await fetch(DIAGNOSE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Diagnosis failed");
      }

      const data = (await res.json()) as DiagnoseResponseData;
      setDiagnosisResult(data);
      const probPct = Math.round(data.probability * 100);
      const covPct = Math.round((data.evidence_coverage || 0) * 100);
      showNotification("success", `[${data.decision}] ${data.predicted} (${probPct}% prob, ${covPct}% coverage)`);
    } catch (err) {
      showNotification("error", err instanceof Error ? err.message : "Diagnosis failed");
    } finally {
      setDiagnosisLoading(false);
    }
  }

  function handleClarificationResponse(answerYes: boolean) {
    if (!diagnosisResult || !diagnosisResult.clarification_question) return;

    if (answerYes) {
      const candidateSymptom =
        diagnosisResult.evidence_details?.missing_hallmarks?.[0]?.symptom ||
        uncertainSymptoms[0];

      if (candidateSymptom) {
        const updated = [...new Set([...selectedSymptoms, candidateSymptom])];
        setSelectedSymptoms(updated);
        setUncertainSymptoms((prev) => prev.filter((s) => s.toLowerCase() !== candidateSymptom.toLowerCase()));
        showNotification("success", `Added '${candidateSymptom}'. Re-evaluating decision gate...`);
        void handleDiagnose(updated);
        return;
      }
    }

    showNotification("info", "Clarification response recorded: symptom absent.");
  }

  function promoteUncertainSymptom(symptom: string) {
    addSymptom(symptom);
    setUncertainSymptoms((prev) => prev.filter((s) => s.toLowerCase() !== symptom.toLowerCase()));
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
        body: JSON.stringify({ query: disease, file_name: selectedDocument || undefined }),
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

  function showNotification(type: "success" | "error" | "info", text: string) {
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
      setSelectedDocument(data.stored_file_name);
      showNotification(
        "success",
        `${data.original_file_name} indexed successfully with ${data.total_chunks_indexed} chunk${data.total_chunks_indexed === 1 ? "" : "s"}. Set as active report.`
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

  async function handleSummarize() {
    if (loading || summarizing) return;
    setSummarizing(true);
    setLoading(true);
    setError(null);
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(SUMMARIZE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_name: selectedDocument || undefined }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Failed to generate clinical summary");
      }

      const data = (await response.json()) as QueryResponse;
      setAnswer(data.answer);
      setSources(Array.isArray(data.source_documents) ? data.source_documents : []);
      showNotification("success", "Clinical summary generated strictly from report.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate clinical summary");
      showNotification("error", "Summarization failed");
    } finally {
      setLoading(false);
      setSummarizing(false);
    }
  }

  async function handleSubmit(event?: React.FormEvent<HTMLFormElement>, overrideQuery?: string) {
    if (event) {
      event.preventDefault();
    }
    const targetQuery = (overrideQuery ?? query).trim();
    if (!targetQuery || loading) {
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
        body: JSON.stringify({
          query: targetQuery,
          file_name: selectedDocument || undefined,
        }),
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
            className={`mb-6 rounded-2xl border px-4 py-3 text-sm shadow-sm ${
              notification.type === "success"
                ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                : notification.type === "info"
                ? "border-plum-200 bg-plum-50 text-plum-800"
                : "border-red-200 bg-red-50 text-red-700"
            }`}
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
                  <p className="mt-1 text-sm text-slate-600">Documents available in the local Chroma index. Select an active report to scope your Q&A and summaries.</p>
                </div>
                {documentsLoading ? <span className="text-xs font-medium text-plum-700">Refreshing...</span> : null}
              </div>

              {documents.length > 0 ? (
                <div className="overflow-hidden rounded-2xl border border-plum-100">
                  <div className="grid grid-cols-[1.2fr_1.2fr_0.7fr_0.8fr_0.8fr_0.9fr] gap-3 border-b border-plum-100 bg-plum-50 px-4 py-3 text-xs font-semibold uppercase tracking-[0.18em] text-plum-700">
                    <span>File</span>
                    <span>Source</span>
                    <span>Size</span>
                    <span>Status</span>
                    <span>Indexed</span>
                    <span>Target Scope</span>
                  </div>
                  <div className="divide-y divide-plum-100 bg-white">
                    {documents.map((document) => {
                      const fileType = (document.file_type || "doc").toUpperCase();
                      const uploadDate = document.upload_date ? new Date(document.upload_date).toLocaleString() : "N/A";
                      const fileSize = typeof document.file_size === "number" ? `${(document.file_size / 1024).toFixed(1)} KB` : "N/A";
                      const isActive = selectedDocument === document.file_name;

                      return (
                        <div key={document.source_path ?? document.file_name} className={`grid grid-cols-[1.2fr_1.2fr_0.7fr_0.8fr_0.8fr_0.9fr] gap-3 px-4 py-3 text-sm items-center transition ${isActive ? "bg-plum-50/40" : ""}`}>
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
                          <div>
                            {isActive ? (
                              <span className="inline-flex items-center gap-1 rounded-full bg-plum-100 px-2.5 py-1 text-xs font-semibold text-plum-800 ring-1 ring-plum-300">
                                ✓ Active
                              </span>
                            ) : (
                              <button
                                type="button"
                                onClick={() => setSelectedDocument(document.file_name)}
                                className="rounded-lg bg-white px-2.5 py-1 text-xs font-medium text-slate-700 ring-1 ring-slate-200 hover:bg-plum-50 hover:text-plum-700 transition"
                              >
                                Set Active
                              </button>
                            )}
                          </div>
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

            {selectedDocument ? (
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-plum-200 bg-plum-50/70 px-4 py-2.5 text-xs text-plum-900">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-plum-700">Target Report:</span>
                  <span className="rounded-md bg-white px-2 py-0.5 font-mono text-[11px] font-semibold text-ink-950 ring-1 ring-plum-100">{selectedDocument}</span>
                </div>
                <span className="text-slate-500">Strictly grounded • Uses data only in report • No hallucinations</span>
              </div>
            ) : null}

            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
              <span className="font-medium text-slate-500">Quick Analysis:</span>
              <button
                type="button"
                onClick={() => void handleSummarize()}
                disabled={loading || summarizing}
                className="inline-flex items-center gap-1 rounded-xl bg-plum-100 px-3 py-1.5 font-semibold text-plum-800 transition hover:bg-plum-200 disabled:opacity-50"
              >
                📋 Full Clinical Summary
              </button>
              <button
                type="button"
                onClick={() => {
                  const q = "What symptoms are documented for the patient and what is their recorded severity, measurements, or lab values in this report?";
                  setQuery(q);
                  void handleSubmit(undefined, q);
                }}
                disabled={loading}
                className="inline-flex items-center gap-1 rounded-xl bg-white px-3 py-1.5 font-medium text-slate-700 ring-1 ring-plum-200 transition hover:bg-plum-50 disabled:opacity-50"
              >
                🩸 Symptoms & Severity
              </button>
              <button
                type="button"
                onClick={() => {
                  const q = "What are the patient's documented laboratory results, vitals, and diagnostic findings in this report?";
                  setQuery(q);
                  void handleSubmit(undefined, q);
                }}
                disabled={loading}
                className="inline-flex items-center gap-1 rounded-xl bg-white px-3 py-1.5 font-medium text-slate-700 ring-1 ring-plum-200 transition hover:bg-plum-50 disabled:opacity-50"
              >
                🧪 Lab Results & Vitals
              </button>
              <button
                type="button"
                onClick={() => {
                  const q = "What treatments, prescriptions, or follow-ups are explicitly ordered by the physician in this report?";
                  setQuery(q);
                  void handleSubmit(undefined, q);
                }}
                disabled={loading}
                className="inline-flex items-center gap-1 rounded-xl bg-white px-3 py-1.5 font-medium text-slate-700 ring-1 ring-plum-200 transition hover:bg-plum-50 disabled:opacity-50"
              >
                💊 Documented Treatment Plan
              </button>
            </div>

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
                  placeholder="e.g. In the report, if sugar is high and leg pain, what does the report state about severity and treatment?"
                  className="min-h-[88px] flex-1 resize-none rounded-2xl border border-transparent bg-white px-4 py-3 text-base outline-none ring-0 placeholder:text-slate-400 focus:border-plum-300 focus:shadow-[0_0_0_4px_rgba(140,73,223,0.10)]"
                />
                <div className="flex flex-col gap-2 sm:min-w-[160px]">
                  <button
                    type="submit"
                    disabled={!canSubmit}
                    className="inline-flex items-center justify-center gap-2 rounded-2xl bg-plum-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-plum-800 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {loading && !summarizing ? <Spinner /> : null}
                    {loading && !summarizing ? "Searching..." : "Query report"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleSummarize()}
                    disabled={loading || summarizing}
                    className="inline-flex items-center justify-center gap-2 rounded-2xl bg-white px-3 py-2 text-xs font-semibold text-plum-800 ring-1 ring-plum-300 transition hover:bg-plum-50 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {summarizing ? <Spinner /> : null}
                    {summarizing ? "Summarizing..." : "📋 Summarize"}
                  </button>
                </div>
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

            {/* ── 3-State Symptom Epistemic Categorization ── */}
            <div className="mt-3 space-y-3">
              {/* State 1: SUPPORTED Symptoms (Enters Diagnostic Vector) */}
              <div className="rounded-2xl border border-plum-100 bg-white p-4">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-emerald-700">
                    Supported Symptoms (Active in Vector: {selectedSymptoms.length})
                  </span>
                  {selectedSymptoms.length > 0 ? (
                    <button
                      type="button"
                      onClick={() => setSelectedSymptoms([])}
                      className="text-xs text-slate-400 hover:text-slate-600"
                    >
                      Clear all
                    </button>
                  ) : null}
                </div>
                {availableSymptoms.length === 0 ? (
                  <div className="rounded-xl bg-plum-50 px-3 py-3 text-sm text-slate-500">Loading clinical schema...</div>
                ) : selectedSymptoms.length === 0 ? (
                  <div className="rounded-xl bg-plum-50 px-3 py-3 text-sm text-slate-500">No supported symptoms active yet.</div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {selectedSymptoms.map((symptom) => (
                      <span
                        key={symptom}
                        className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1.5 text-sm font-medium text-emerald-900 ring-1 ring-emerald-200"
                      >
                        <span>{symptom}</span>
                        <button
                          type="button"
                          onClick={() => removeSymptom(symptom)}
                          className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-white text-emerald-700 ring-1 ring-emerald-200 hover:bg-emerald-100"
                          aria-label={`Remove ${symptom}`}
                        >
                          X
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* State 2: UNCERTAIN Symptoms (Near-Tie / Moderate Similarity) */}
              {uncertainSymptoms.length > 0 ? (
                <div className="rounded-2xl border border-amber-200 bg-amber-50/50 p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-amber-800">
                      Borderline / Uncertain Symptoms ({uncertainSymptoms.length})
                    </span>
                    <span className="text-[11px] text-amber-600">Near-tie or moderate confidence</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {uncertainSymptoms.map((sym) => (
                      <span
                        key={sym}
                        className="inline-flex items-center gap-2 rounded-full bg-white px-3 py-1.5 text-sm font-medium text-amber-900 ring-1 ring-amber-300"
                      >
                        <span>{sym}</span>
                        <button
                          type="button"
                          onClick={() => promoteUncertainSymptom(sym)}
                          className="rounded-full bg-amber-600 px-2 py-0.5 text-xs font-semibold text-white hover:bg-amber-700"
                        >
                          + Confirm
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              {/* State 3: UNSUPPORTED Symptoms (Described in text but unrepresented in schema) */}
              {unsupportedSymptoms.length > 0 ? (
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-700">
                      Described but Unmapped Findings ({unsupportedSymptoms.length})
                    </span>
                    <span className="rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                      Honest Abstention · Schema Gap
                    </span>
                  </div>
                  <p className="mb-2 text-xs text-slate-500">
                    Documented by patient/doctor but not in the 230-feature dataset schema. Preserved rather than force-mapped.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {unsupportedSymptoms.map((sym) => (
                      <span
                        key={sym}
                        className="inline-flex items-center rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-300"
                      >
                        {sym}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>

            {/* ── Run Diagnosis Trigger & Evidence-Adaptive Result Panel ── */}
            <div className="mt-5 space-y-4">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => void handleDiagnose()}
                  className="inline-flex items-center justify-center rounded-xl bg-plum-700 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-plum-800 disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={diagnosisLoading}
                >
                  {diagnosisLoading ? "Evaluating Evidence..." : "Run Evidence-Adaptive Diagnosis"}
                </button>
              </div>

              {diagnosisResult ? (
                <div
                  className={`rounded-3xl border p-5 shadow-sm transition-all ${
                    diagnosisResult.decision === "DIAGNOSE"
                      ? "border-emerald-200 bg-emerald-50/40"
                      : diagnosisResult.decision === "CLARIFY"
                      ? "border-amber-200 bg-amber-50/40"
                      : "border-slate-200 bg-slate-50/60"
                  }`}
                >
                  {/* Decision Header */}
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200/60 pb-3">
                    <div className="flex items-center gap-2">
                      <span
                        className={`rounded-full px-3 py-1 text-xs font-bold tracking-wide uppercase ${
                          diagnosisResult.decision === "DIAGNOSE"
                            ? "bg-emerald-600 text-white"
                            : diagnosisResult.decision === "CLARIFY"
                            ? "bg-amber-600 text-white"
                            : "bg-slate-700 text-white"
                        }`}
                      >
                        Decision: {diagnosisResult.decision}
                      </span>
                      <span className="text-xs font-medium text-slate-600">
                        {diagnosisResult.decision === "DIAGNOSE"
                          ? "Sufficient Evidence Grounding"
                          : diagnosisResult.decision === "CLARIFY"
                          ? "Targeted Information-Gain Opportunity"
                          : "Honest Clinical Abstention"}
                      </span>
                    </div>

                    <button
                      type="button"
                      className="text-xs font-semibold text-plum-700 underline underline-offset-2 hover:text-plum-900"
                      onClick={() => void searchGuidelines(diagnosisResult.predicted)}
                    >
                      Search Guidelines for &ldquo;{diagnosisResult.predicted}&rdquo;
                    </button>
                  </div>

                  {/* Primary Prediction Label */}
                  <div className="mt-3">
                    <div className="text-lg font-bold text-ink-950">
                      {diagnosisResult.predicted}
                    </div>
                  </div>

                  {/* Dual Comparison Gauges: Model Probability vs Evidence Coverage */}
                  <div className="mt-4 grid gap-4 sm:grid-cols-2">
                    {/* Gauge 1: Model Softmax Probability */}
                    <div className="rounded-2xl border border-plum-100 bg-white p-3.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-700">Model Probability</span>
                        <span className="font-bold text-plum-700">{Math.round(diagnosisResult.probability * 100)}%</span>
                      </div>
                      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full bg-plum-600 transition-all duration-500"
                          style={{ width: `${Math.min(100, Math.round(diagnosisResult.probability * 100))}%` }}
                        />
                      </div>
                      <p className="mt-1.5 text-[11px] text-slate-400">Classifier ensemble softmax score</p>
                    </div>

                    {/* Gauge 2: Diagnostic Evidence Coverage */}
                    <div className="rounded-2xl border border-plum-100 bg-white p-3.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-700">Diagnostic Evidence Coverage</span>
                        <span className="font-bold text-emerald-700">
                          {Math.round((diagnosisResult.evidence_coverage || 0) * 100)}%
                        </span>
                      </div>
                      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={`h-full transition-all duration-500 ${
                            (diagnosisResult.evidence_coverage || 0) >= 0.35 ? "bg-emerald-600" : "bg-amber-500"
                          }`}
                          style={{ width: `${Math.min(100, Math.round((diagnosisResult.evidence_coverage || 0) * 100))}%` }}
                        />
                      </div>
                      <p className="mt-1.5 text-[11px] text-slate-400">
                        Weighted hallmark overlap from dataset ground truth
                      </p>
                    </div>
                  </div>

                  {/* Vector Sparsity Information */}
                  <div className="mt-3 flex items-center justify-between rounded-xl bg-white/70 px-3.5 py-2 text-xs text-slate-600">
                    <span>
                      Active Presentation: <strong>{diagnosisResult.active_symptom_count}</strong> symptoms (Sparsity:{" "}
                      <strong>{(diagnosisResult.sparsity_score * 100).toFixed(1)}%</strong>)
                    </span>
                    <span className="text-[11px] text-slate-500">
                      Training domain shift: z ={" "}
                      {diagnosisResult.evidence_details?.sparsity_metrics?.z_score ?? "0.0"}
                    </span>
                  </div>

                  {/* Clarification Box (when CLARIFY is triggered) */}
                  {diagnosisResult.decision === "CLARIFY" && diagnosisResult.clarification_question ? (
                    <div className="mt-4 rounded-2xl border border-amber-300 bg-white p-4 shadow-sm">
                      <div className="flex items-center gap-2 text-xs font-bold text-amber-800">
                        <span>🎯 Targeted Clarification Question (Information Gain):</span>
                      </div>
                      <p className="mt-2 text-sm font-semibold text-slate-800">
                        {diagnosisResult.clarification_question}
                      </p>
                      <div className="mt-3 flex gap-2">
                        <button
                          type="button"
                          onClick={() => handleClarificationResponse(true)}
                          className="rounded-xl bg-emerald-600 px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-emerald-700"
                        >
                          ✓ Yes, I have this symptom
                        </button>
                        <button
                          type="button"
                          onClick={() => handleClarificationResponse(false)}
                          className="rounded-xl bg-white px-3.5 py-1.5 text-xs font-semibold text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50"
                        >
                          ✕ No, symptom is absent
                        </button>
                      </div>
                    </div>
                  ) : null}

                  {/* Abstention Rationale Box (when ABSTAIN is triggered) */}
                  {diagnosisResult.decision === "ABSTAIN" && diagnosisResult.abstention_reason ? (
                    <div className="mt-4 rounded-2xl border border-rose-200 bg-white p-4 shadow-sm">
                      <div className="text-xs font-bold text-rose-800">
                        🛡️ Clinical Decision Rationale:
                      </div>
                      <p className="mt-1.5 text-xs leading-5 text-slate-700">
                        {diagnosisResult.abstention_reason}
                      </p>
                    </div>
                  ) : null}

                  {/* Hallmark Symptom Verification Details */}
                  {diagnosisResult.evidence_details ? (
                    <details className="mt-4 rounded-2xl border border-slate-200 bg-white p-3.5 open:bg-white text-xs">
                      <summary className="cursor-pointer font-semibold text-slate-700">
                        View Disease Hallmark Evidence Breakdown
                      </summary>
                      <div className="mt-3 space-y-3 pt-2 border-t border-slate-100">
                        <div>
                          <span className="font-semibold text-emerald-800">Present Hallmarks:</span>
                          <div className="mt-1 flex flex-wrap gap-1.5">
                            {diagnosisResult.evidence_details.present_hallmarks.length > 0 ? (
                              diagnosisResult.evidence_details.present_hallmarks.map((h) => (
                                <span
                                  key={h.symptom}
                                  className="rounded-md bg-emerald-50 px-2 py-0.5 text-emerald-800 ring-1 ring-emerald-200"
                                >
                                  {h.symptom} (freq {Math.round(h.weight * 100)}%)
                                </span>
                              ))
                            ) : (
                              <span className="text-slate-400">None present in current vector</span>
                            )}
                          </div>
                        </div>

                        <div>
                          <span className="font-semibold text-amber-800">Key Hallmarks Missing:</span>
                          <div className="mt-1 flex flex-wrap gap-1.5">
                            {diagnosisResult.evidence_details.missing_hallmarks.slice(0, 4).map((h) => (
                              <span
                                key={h.symptom}
                                className="rounded-md bg-amber-50 px-2 py-0.5 text-amber-800 ring-1 ring-amber-200"
                              >
                                {h.symptom} (freq {Math.round(h.weight * 100)}%)
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </details>
                  ) : null}
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
