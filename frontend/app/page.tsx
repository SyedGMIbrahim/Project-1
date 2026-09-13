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
  clarification_target?: string | null;
  symptom_states?: {
    supported: string[];
    uncertain: string[];
    unsupported: string[];
    denied?: string[];
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
    epistemic_states?: {
      supported: string[];
      uncertain: string[];
      unsupported: string[];
      denied?: string[];
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
  const [deniedSymptoms, setDeniedSymptoms] = useState<string[]>([]);
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
    setDeniedSymptoms([]); // Reset denied symptoms on fresh extraction
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

  async function handleDiagnose(
    overrideSymptoms?: string[],
    overrideDenied?: string[],
    overrideUncertain?: string[],
    overrideUnsupported?: string[]
  ) {
    const symptomsToUse = overrideSymptoms || selectedSymptoms;
    const deniedToUse = overrideDenied !== undefined ? overrideDenied : deniedSymptoms;
    const uncertainToUse = overrideUncertain !== undefined ? overrideUncertain : uncertainSymptoms;
    const unsupportedToUse = overrideUnsupported !== undefined ? overrideUnsupported : unsupportedSymptoms;

    if (symptomsToUse.length === 0) {
      showNotification("error", "Please select one or more symptoms to run diagnosis.");
      return;
    }

    setDiagnosisLoading(true);

    try {
      const payload = {
        symptoms: symptomsToUse,
        uncertain_symptoms: uncertainToUse,
        unsupported_symptoms: unsupportedToUse,
        denied_symptoms: deniedToUse,
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

    // Determine target symptom precisely from backend response
    const targetSymptom =
      diagnosisResult.clarification_target ||
      diagnosisResult.evidence_details?.missing_hallmarks?.[0]?.symptom ||
      uncertainSymptoms[0];

    if (!targetSymptom) {
      showNotification("error", "No target symptom identified for clarification.");
      return;
    }

    if (answerYes) {
      // User confirms symptom is present
      const updatedSymptoms = [...new Set([...selectedSymptoms, targetSymptom])];
      const updatedUncertain = uncertainSymptoms.filter(
        (s) => s.toLowerCase() !== targetSymptom.toLowerCase()
      );
      const updatedDenied = deniedSymptoms.filter(
        (s) => s.toLowerCase() !== targetSymptom.toLowerCase()
      );

      setSelectedSymptoms(updatedSymptoms);
      setUncertainSymptoms(updatedUncertain);
      setDeniedSymptoms(updatedDenied);
      showNotification("success", `Added '${targetSymptom}'. Re-evaluating decision gate...`);
      void handleDiagnose(updatedSymptoms, updatedDenied, updatedUncertain, unsupportedSymptoms);
    } else {
      // User clarifies symptom is ABSENT
      const updatedDenied = [...new Set([...deniedSymptoms, targetSymptom])];
      const updatedUncertain = uncertainSymptoms.filter(
        (s) => s.toLowerCase() !== targetSymptom.toLowerCase()
      );

      setDeniedSymptoms(updatedDenied);
      setUncertainSymptoms(updatedUncertain);
      showNotification("info", `Recorded '${targetSymptom}' as absent. Moving to next clarification...`);
      void handleDiagnose(selectedSymptoms, updatedDenied, updatedUncertain, unsupportedSymptoms);
    }
  }

  function removeDeniedSymptom(symptom: string) {
    const updatedDenied = deniedSymptoms.filter((s) => s.toLowerCase() !== symptom.toLowerCase());
    setDeniedSymptoms(updatedDenied);
    showNotification("info", `Unmarked '${symptom}' from absent list.`);
    void handleDiagnose(selectedSymptoms, updatedDenied, uncertainSymptoms, unsupportedSymptoms);
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
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8 flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500">Local Clinical RAG</p>
            <h1 className="mt-1 text-2xl font-black tracking-tight text-slate-900 sm:text-3xl">Clinical Document Intelligence</h1>
          </div>
          <div className="hidden rounded-sm border border-slate-200 bg-slate-50 px-4 py-1.5 text-[11px] font-bold uppercase tracking-widest text-slate-500 sm:block">
            100% local analysis
          </div>
        </header>

        {notification ? (
          <div
            className={`mb-6 rounded-sm border px-4 py-3 text-sm shadow-sm ${notification.type === "success"
                ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                : notification.type === "info"
                  ? "border-slate-300 bg-slate-50 text-sky-800"
                  : "border-red-200 bg-red-50 text-red-700"
              }`}
          >
            {notification.text}
          </div>
        ) : null}

        <section className="grid flex-1 gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="flex flex-col border border-slate-200 bg-white p-5 shadow-sm sm:p-8">
            <div className="mb-6">
              <p className="text-[11px] font-bold uppercase tracking-widest text-sky-700">Ask a question</p>
              <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-600">
                Query the local medical record index and review the generated answer alongside the retrieved evidence.
              </p>
            </div>

            <div className="mb-6 border border-dashed border-slate-300 bg-slate-50 p-6">
              <div
                onDragEnter={() => setDragActive(true)}
                onDragLeave={() => setDragActive(false)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={handleDrop}
                className={`rounded-sm border-2 border-dashed p-5 text-center transition ${dragActive ? "border-slate-500 bg-white" : "border-slate-300 bg-white/80"}`}
              >
                <p className="text-sm font-semibold text-slate-900">Upload Clinical File</p>
                <p className="mt-1 text-sm text-slate-600">Drag and drop a .pdf or .txt document to index it locally.</p>
                <div className="mt-4 flex flex-col items-center justify-center gap-3 sm:flex-row">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                    className="inline-flex items-center justify-center gap-2 rounded-sm bg-sky-700 px-5 py-2.5 text-[13px] font-bold tracking-wide text-white transition hover:bg-slate-800 disabled:opacity-50"
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
                <div className="mt-3 rounded-sm bg-slate-50 px-4 py-3 text-sm text-sky-800">Processing and indexing document...</div>
              ) : null}
            </div>

            <section className="mb-8 border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">Currently Indexed Datasets</h2>
                  <p className="mt-1 text-sm text-slate-600">Documents available in the local Chroma index. Select an active report to scope your Q&A and summaries.</p>
                </div>
                {documentsLoading ? <span className="text-[11px] font-bold uppercase text-sky-700">Refreshing...</span> : null}
              </div>

              {documents.length > 0 ? (
                <div className="border border-slate-200">
                  <div className="grid grid-cols-[1.2fr_1.2fr_0.7fr_0.8fr_0.8fr_0.9fr] gap-3 border-b border-slate-200 bg-slate-50 px-4 py-3 text-[11px] font-bold uppercase tracking-widest text-slate-500">
                    <span>File</span>
                    <span>Source</span>
                    <span>Size</span>
                    <span>Status</span>
                    <span>Indexed</span>
                    <span>Target Scope</span>
                  </div>
                  <div className="divide-y divide-slate-200 bg-white">
                    {documents.map((document) => {
                      const fileType = (document.file_type || "doc").toUpperCase();
                      const uploadDate = document.upload_date ? new Date(document.upload_date).toLocaleString() : "N/A";
                      const fileSize = typeof document.file_size === "number" ? `${(document.file_size / 1024).toFixed(1)} KB` : "N/A";
                      const isActive = selectedDocument === document.file_name;

                      return (
                        <div key={document.source_path ?? document.file_name} className={`grid grid-cols-[1.2fr_1.2fr_0.7fr_0.8fr_0.8fr_0.9fr] gap-3 px-4 py-3 text-sm items-center transition ${isActive ? "bg-slate-50/40" : ""}`}>
                          <div className="flex items-start gap-3">
                            <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-sm bg-slate-100 text-[10px] font-bold text-slate-700 ring-1 ring-slate-200">
                              {fileType}
                            </div>
                            <div>
                              <p className="font-medium text-slate-900">{document.file_name}</p>
                              <p className="text-xs text-slate-500">{uploadDate}</p>
                            </div>
                          </div>
                          <div className="truncate text-slate-600" title={document.source_path ?? undefined}>
                            {document.source_path || "Local index"}
                          </div>
                          <div className="text-slate-600">{fileSize}</div>
                          <div>
                            <span className="inline-flex rounded-sm bg-emerald-50 px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-emerald-700">
                              {document.status}
                            </span>
                          </div>
                          <div className="text-slate-600">{document.chunk_count ?? 0} chunks</div>
                          <div>
                            {isActive ? (
                              <span className="inline-flex items-center gap-1 rounded-sm bg-sky-50 px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-sky-700 ring-1 ring-sky-200">
                                ✓ Active
                              </span>
                            ) : (
                              <button
                                type="button"
                                onClick={() => setSelectedDocument(document.file_name)}
                                className="rounded-sm bg-white px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-slate-500 ring-1 ring-slate-200 hover:bg-slate-50 hover:text-slate-900 transition"
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
                <div className="rounded-sm border border-dashed border-slate-300 bg-slate-50/40 px-5 py-10 text-sm text-slate-500">
                  No clinical datasets are currently indexed.
                </div>
              )}
            </section>

            {selectedDocument ? (
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border border-sky-200 bg-sky-50 px-4 py-3 text-xs text-sky-900">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold uppercase tracking-widest text-sky-700">Target Report:</span>
                  <span className="rounded-sm bg-white px-2 py-1 font-mono text-[11px] font-bold text-sky-900 ring-1 ring-sky-200">{selectedDocument}</span>
                </div>
                <span className="text-slate-500">Strictly grounded • Uses data only in report • No hallucinations</span>
              </div>
            ) : null}

            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-[11px] font-bold uppercase tracking-widest text-slate-500">Quick Analysis:</span>
              <button
                type="button"
                onClick={() => void handleSummarize()}
                disabled={loading || summarizing}
                className="inline-flex items-center gap-1 rounded-sm bg-sky-700 px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide text-white transition hover:bg-sky-800 disabled:opacity-50"
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
                className="inline-flex items-center gap-1 rounded-sm bg-white px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 transition hover:bg-slate-50 disabled:opacity-50"
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
                className="inline-flex items-center gap-1 rounded-sm bg-white px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 transition hover:bg-slate-50 disabled:opacity-50"
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
                className="inline-flex items-center gap-1 rounded-sm bg-white px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 transition hover:bg-slate-50 disabled:opacity-50"
              >
                💊 Documented Treatment Plan
              </button>
            </div>

            <form onSubmit={handleSubmit} className="mb-8">
              <label htmlFor="query" className="sr-only">
                Medical document query
              </label>
              <div className="flex flex-col gap-3 border border-slate-200 bg-slate-50 p-4 shadow-sm sm:flex-row">
                <textarea
                  id="query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  rows={3}
                  placeholder="e.g. In the report, if sugar is high and leg pain, what does the report state about severity and treatment?"
                  className="min-h-[88px] flex-1 resize-none rounded-sm border border-transparent bg-white px-4 py-3 text-base outline-none ring-0 placeholder:text-slate-400 focus:border-slate-400 focus:ring-2 focus:ring-sky-600 focus:border-sky-600"
                />
                <div className="flex flex-col gap-2 sm:min-w-[160px]">
                  <button
                    type="submit"
                    disabled={!canSubmit}
                    className="inline-flex items-center justify-center gap-2 rounded-sm bg-slate-900 px-5 py-2.5 text-[13px] font-bold tracking-wide text-white transition hover:bg-slate-800 disabled:opacity-50"
                  >
                    {loading && !summarizing ? <Spinner /> : null}
                    {loading && !summarizing ? "Searching..." : "Query report"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleSummarize()}
                    disabled={loading || summarizing}
                    className="inline-flex items-center justify-center gap-2 rounded-sm bg-white px-3 py-2 text-[11px] font-bold uppercase tracking-wide text-slate-700 ring-1 ring-slate-300 transition hover:bg-slate-50 disabled:opacity-50"
                  >
                    {summarizing ? <Spinner /> : null}
                    {summarizing ? "Summarizing..." : "📋 Summarize"}
                  </button>
                </div>
              </div>
            </form>

            <div className="flex-1 border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-[11px] font-bold uppercase tracking-widest text-slate-500">GENERATED ANSWER</h2>
                {loading ? <span className="text-[11px] font-bold uppercase tracking-widest text-sky-700">Analyzing sources...</span> : null}
              </div>

              {error ? (
                <div className="rounded-sm border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              ) : answer ? (
                <p className="whitespace-pre-wrap font-serif text-[15px] leading-7 text-slate-900">{answer}</p>
              ) : (
                <div className="rounded-sm border border-dashed border-slate-300 bg-slate-50/40 px-5 py-10 text-sm text-slate-500">
                  The answer will appear here after a query is submitted.
                </div>
              )}
            </div>
          </div>

          <div className="mt-8 border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-4">
              <h3 className="text-[11px] font-bold uppercase tracking-widest text-slate-500">SYMPTOM CHECKER & DIAGNOSTIC GATE</h3>
              <p className="mt-1 text-sm text-slate-600">Select symptoms manually or use AI extraction from text and indexed clinical documents.</p>
            </div>

            <div className="rounded-sm border border-slate-200 bg-slate-50/40 p-4">
              <label htmlFor="nl-symptoms" className="mb-2 block text-sm font-medium text-sky-800">Describe symptoms naturally</label>
              <textarea
                id="nl-symptoms"
                value={nlText}
                onChange={(e) => setNlText(e.target.value)}
                rows={3}
                placeholder="Describe symptoms naturally..."
                className="min-h-[88px] w-full resize-none rounded-sm border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 placeholder:text-slate-400 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-600 focus:border-sky-600"
              />

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handleExtractFromText()}
                  className="inline-flex items-center justify-center rounded bg-sky-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={extractingFromText}
                >
                  {extractingFromText ? "Extracting..." : "🪄 Auto-Select from Text"}
                </button>
                <button
                  type="button"
                  onClick={() => void handleExtractFromDocs()}
                  className="inline-flex items-center justify-center rounded bg-white px-4 py-2 text-sm font-semibold text-sky-800 ring-1 ring-slate-300 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
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
              <label htmlFor="symptom-search" className="mb-2 block text-sm font-medium text-sky-800">Search and add symptoms</label>
              <input
                id="symptom-search"
                value={searchTerm}
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="Type symptom name..."
                className="w-full rounded-sm border border-slate-200 bg-white px-4 py-2.5 text-sm text-slate-700 placeholder:text-slate-400 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-600 focus:border-sky-600"
              />
            </div>

            {searchTerm.trim() ? (
              <div className="mt-3 max-h-60 overflow-y-auto rounded-sm border border-slate-200 bg-white p-2">
                {symptomSuggestions.length === 0 ? (
                  <div className="px-3 py-2 text-sm text-slate-500">No matching symptoms found.</div>
                ) : (
                  <ul className="space-y-1">
                    {symptomSuggestions.map((symptom) => (
                      <li key={symptom}>
                        <button
                          type="button"
                          onClick={() => addSymptom(symptom)}
                          className="w-full rounded px-3 py-2 text-left text-sm text-slate-700 transition hover:bg-slate-50"
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
            <div className="mt-4 space-y-4 font-sans">
              {/* State 1: SUPPORTED Symptoms (Enters Diagnostic Vector) */}
              <div className="rounded-sm border border-slate-300 bg-slate-50 p-4">
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-widest text-slate-900">
                    Supported Evidence ({selectedSymptoms.length})
                  </span>
                  {selectedSymptoms.length > 0 ? (
                    <button
                      type="button"
                      onClick={() => setSelectedSymptoms([])}
                      className="text-[11px] font-semibold text-slate-500 hover:text-slate-900"
                    >
                      CLEAR ALL
                    </button>
                  ) : null}
                </div>
                {availableSymptoms.length === 0 ? (
                  <div className="text-sm text-slate-500">Loading clinical schema...</div>
                ) : selectedSymptoms.length === 0 ? (
                  <div className="text-sm text-slate-500">No supported symptoms active.</div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {selectedSymptoms.map((symptom) => (
                      <span
                        key={symptom}
                        className="inline-flex items-center gap-2 rounded-sm border border-slate-300 bg-white px-2.5 py-1 text-[13px] font-medium text-slate-900 shadow-sm"
                      >
                        <span>{symptom}</span>
                        <button
                          type="button"
                          onClick={() => removeSymptom(symptom)}
                          className="font-bold text-slate-400 hover:text-slate-800"
                          aria-label={`Remove ${symptom}`}
                        >
                          ✕
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* State 2: UNCERTAIN Symptoms (Near-Tie / Moderate Similarity) */}
              {uncertainSymptoms.length > 0 ? (
                <div className="rounded-sm border border-dashed border-amber-400 bg-amber-50 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <span className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-amber-900">
                      ⚠️ Uncertain Findings ({uncertainSymptoms.length})
                    </span>
                    <span className="text-[11px] font-medium text-amber-700">Near-tie or moderate confidence</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {uncertainSymptoms.map((sym) => (
                      <span
                        key={sym}
                        className="inline-flex items-center gap-2 rounded-sm border border-amber-300 bg-white px-2.5 py-1 text-[13px] font-medium text-amber-900 shadow-sm"
                      >
                        <span>{sym}</span>
                        <button
                          type="button"
                          onClick={() => promoteUncertainSymptom(sym)}
                          className="rounded-sm bg-amber-100 px-1.5 py-0.5 text-[11px] font-bold text-amber-800 hover:bg-amber-200"
                        >
                          CONFIRM
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              {/* State 3: UNSUPPORTED Symptoms (Described in text but unrepresented in schema) */}
              {unsupportedSymptoms.length > 0 ? (
                <div className="pt-2">
                  <div className="mb-2 flex items-center gap-2">
                    <span className="text-[11px] font-bold uppercase tracking-widest text-slate-500">
                      Dropped / Unmapped ({unsupportedSymptoms.length})
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {unsupportedSymptoms.map((sym) => (
                      <span
                        key={sym}
                        className="text-[13px] font-medium text-slate-400 line-through"
                      >
                        {sym}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              {/* State 4: DENIED / ABSENT Symptoms (Explicitly confirmed absent by user) */}
              {deniedSymptoms.length > 0 ? (
                <div className="pt-2">
                  <div className="mb-2 flex items-center gap-2">
                    <span className="text-[11px] font-bold uppercase tracking-widest text-slate-500">
                      Confirmed Absent ({deniedSymptoms.length})
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {deniedSymptoms.map((sym) => (
                      <span
                        key={sym}
                        className="inline-flex items-center gap-1.5 text-[13px] font-medium text-slate-400 line-through"
                      >
                        <span>{sym}</span>
                        <button
                          type="button"
                          onClick={() => removeDeniedSymptom(sym)}
                          className="font-bold text-slate-300 hover:text-slate-600"
                          title="Unmark absent"
                        >
                          ✕
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>

            {/* ── Run Diagnosis Trigger & Evidence-Adaptive Result Panel ── */}
            <div className="mt-8 border-t border-slate-200 pt-6 font-sans">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => void handleDiagnose()}
                  className="rounded-sm bg-sky-700 px-6 py-2.5 text-[13px] font-bold tracking-wide text-white hover:bg-sky-800 disabled:opacity-50"
                  disabled={diagnosisLoading}
                >
                  {diagnosisLoading ? "COMPUTING..." : "COMPUTE DIAGNOSIS"}
                </button>
              </div>

              {diagnosisResult ? (
                <div
                  className={`mt-6 rounded-sm border p-6 transition-all ${diagnosisResult.decision === "DIAGNOSE"
                      ? "border-emerald-300 bg-white shadow-[0_4px_0_0_#6ee7b7]"
                      : diagnosisResult.decision === "CLARIFY"
                        ? "border-amber-300 bg-white shadow-[0_4px_0_0_#fcd34d]"
                        : "border-slate-800 bg-slate-900 shadow-[0_4px_0_0_#0f172a]"
                    }`}
                >
                  {/* Decision Header */}
                  <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200/20 pb-4">
                    <div>
                      <div className={`text-[11px] font-bold tracking-widest ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-400" : "text-slate-500"}`}>
                        DECISION GATE
                      </div>
                      <div className={`mt-1 text-2xl font-black tracking-tight ${diagnosisResult.decision === "DIAGNOSE"
                          ? "text-emerald-700"
                          : diagnosisResult.decision === "CLARIFY"
                            ? "text-amber-700"
                            : "text-white"
                        }`}>
                        {diagnosisResult.decision}
                      </div>
                    </div>

                    <button
                      type="button"
                      className={`text-[12px] font-semibold underline underline-offset-4 ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-300 hover:text-white" : "text-sky-700 hover:text-sky-900"}`}
                      onClick={() => void searchGuidelines(diagnosisResult.predicted)}
                    >
                      Search Guidelines
                    </button>
                  </div>

                  {/* Primary Prediction Label */}
                  <div className="mt-5">
                    <div className={`text-[11px] font-bold tracking-widest ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-400" : "text-slate-500"}`}>
                      TOP PREDICTION
                    </div>
                    <div className={`mt-1 text-xl font-bold ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-200" : "text-slate-900"}`}>
                      {diagnosisResult.predicted}
                    </div>
                  </div>

                  {/* Dual Comparison Gauges: Model Probability vs Evidence Coverage */}
                  <div className="mt-6 grid gap-6 sm:grid-cols-2">
                    {/* Gauge 1: Model Softmax Probability */}
                    <div>
                      <div className="flex items-end justify-between">
                        <span className={`text-[11px] font-bold tracking-widest ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-400" : "text-slate-500"}`}>PROBABILITY</span>
                        <span className={`font-mono text-xl font-bold ${diagnosisResult.decision === "ABSTAIN" ? "text-white" : "text-slate-900"}`}>{Math.round(diagnosisResult.probability * 100)}%</span>
                      </div>
                      <div className={`mt-2 h-1.5 w-full bg-slate-200 ${diagnosisResult.decision === "ABSTAIN" ? "bg-slate-700" : ""}`}>
                        <div
                          className={`h-full ${diagnosisResult.decision === "ABSTAIN" ? "bg-white" : "bg-sky-600"}`}
                          style={{ width: `${Math.min(100, Math.round(diagnosisResult.probability * 100))}%` }}
                        />
                      </div>
                    </div>

                    {/* Gauge 2: Diagnostic Evidence Coverage */}
                    <div>
                      <div className="flex items-end justify-between">
                        <span className={`text-[11px] font-bold tracking-widest ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-400" : "text-slate-500"}`}>EVIDENCE COVERAGE</span>
                        <span className={`font-mono text-xl font-bold ${diagnosisResult.decision === "ABSTAIN" ? "text-white" : "text-slate-900"}`}>
                          {Math.round((diagnosisResult.evidence_coverage || 0) * 100)}%
                        </span>
                      </div>
                      <div className={`mt-2 h-1.5 w-full bg-slate-200 ${diagnosisResult.decision === "ABSTAIN" ? "bg-slate-700" : ""}`}>
                        <div
                          className={`h-full ${(diagnosisResult.evidence_coverage || 0) >= 0.35 ? (diagnosisResult.decision === "ABSTAIN" ? "bg-emerald-400" : "bg-emerald-600") : (diagnosisResult.decision === "ABSTAIN" ? "bg-amber-400" : "bg-amber-500")
                            }`}
                          style={{ width: `${Math.min(100, Math.round((diagnosisResult.evidence_coverage || 0) * 100))}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Vector Sparsity Information */}
                  <div className={`mt-6 flex items-center justify-between border-t py-3 text-[12px] ${diagnosisResult.decision === "ABSTAIN" ? "border-slate-700 text-slate-400" : "border-slate-200 text-slate-600"}`}>
                    <span>
                      ACTIVE SYMPTOMS: <span className={`font-mono font-bold ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-300" : "text-slate-900"}`}>{diagnosisResult.active_symptom_count}</span>
                      <span className="mx-3 opacity-50">|</span>
                      SPARSITY: <span className={`font-mono font-bold ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-300" : "text-slate-900"}`}>{(diagnosisResult.sparsity_score * 100).toFixed(1)}%</span>
                    </span>
                  </div>

                  {/* Clarification Box (when CLARIFY is triggered) */}
                  {diagnosisResult.decision === "CLARIFY" && diagnosisResult.clarification_question ? (
                    <div className="mt-4 rounded-sm border border-amber-300 bg-amber-50 p-5">
                      <div className="text-[11px] font-bold tracking-widest text-amber-900">
                        TARGETED CLARIFICATION REQUIRED
                      </div>
                      <p className="mt-2 text-[14px] font-medium text-slate-900">
                        {diagnosisResult.clarification_question}
                      </p>
                      <div className="mt-4 flex gap-3">
                        <button
                          type="button"
                          onClick={() => handleClarificationResponse(true)}
                          className="rounded-sm bg-slate-900 px-4 py-2 text-[12px] font-bold text-white hover:bg-slate-800"
                        >
                          YES, PRESENT
                        </button>
                        <button
                          type="button"
                          onClick={() => handleClarificationResponse(false)}
                          className="rounded-sm border border-slate-300 bg-white px-4 py-2 text-[12px] font-bold text-slate-700 hover:bg-slate-50"
                        >
                          NO, ABSENT
                        </button>
                      </div>
                    </div>
                  ) : null}

                  {/* Abstention Rationale Box (when ABSTAIN is triggered) */}
                  {diagnosisResult.decision === "ABSTAIN" && diagnosisResult.abstention_reason ? (
                    <div className="mt-4 rounded-sm border border-slate-700 bg-slate-800 p-5">
                      <div className="text-[11px] font-bold tracking-widest text-rose-400">
                        CLINICAL DECISION RATIONALE
                      </div>
                      <p className="mt-2 text-[13px] leading-relaxed text-slate-300">
                        {diagnosisResult.abstention_reason}
                      </p>
                    </div>
                  ) : null}

                  {/* Hallmark Symptom Verification Details */}
                  {diagnosisResult.evidence_details ? (
                    <details className={`mt-5 text-[13px] ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-300" : "text-slate-700"}`}>
                      <summary className={`cursor-pointer font-semibold outline-none hover:opacity-80 ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-200" : "text-slate-900"}`}>
                        [+] SHOW EVIDENCE BREAKDOWN
                      </summary>
                      <div className={`mt-4 space-y-4 border-l-2 pl-4 ${diagnosisResult.decision === "ABSTAIN" ? "border-slate-700" : "border-slate-200"}`}>
                        <div>
                          <span className={`text-[11px] font-bold tracking-widest uppercase ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-400" : "text-slate-500"}`}>Present Hallmarks</span>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {diagnosisResult.evidence_details.present_hallmarks.length > 0 ? (
                              diagnosisResult.evidence_details.present_hallmarks.map((h) => (
                                <span
                                  key={h.symptom}
                                  className={`rounded-sm border px-2 py-0.5 ${diagnosisResult.decision === "ABSTAIN" ? "border-slate-600 bg-slate-700 text-slate-200" : "border-slate-200 bg-white text-slate-800"}`}
                                >
                                  {h.symptom} <span className="font-mono text-[11px] opacity-70">({Math.round(h.weight * 100)}%)</span>
                                </span>
                              ))
                            ) : (
                              <span className="text-slate-500">None present in current vector</span>
                            )}
                          </div>
                        </div>

                        <div>
                          <span className={`text-[11px] font-bold tracking-widest uppercase ${diagnosisResult.decision === "ABSTAIN" ? "text-slate-400" : "text-slate-500"}`}>Missing Hallmarks</span>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {diagnosisResult.evidence_details.missing_hallmarks.slice(0, 4).map((h) => (
                              <span
                                key={h.symptom}
                                className={`rounded-sm border px-2 py-0.5 opacity-80 ${diagnosisResult.decision === "ABSTAIN" ? "border-slate-700 bg-slate-800 text-slate-400" : "border-slate-200 bg-slate-50 text-slate-500"}`}
                              >
                                {h.symptom} <span className="font-mono text-[11px] opacity-70">({Math.round(h.weight * 100)}%)</span>
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

          <aside className="border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-[11px] font-bold uppercase tracking-widest text-slate-500">EVIDENCE PANEL</h2>
                <p className="mt-1 text-sm text-slate-600">Retrieved source chunks from the local vector store.</p>
              </div>
              <span className="rounded-sm bg-slate-100 px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-widest text-slate-600">
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
                    <details key={`${fileName}-${index}`} className="group rounded-sm border border-slate-200 bg-slate-50 p-4 open:bg-white shadow-sm">
                      <summary className="cursor-pointer list-none text-sm font-semibold text-slate-900">
                        <div className="flex items-start justify-between gap-3">
                          <span>{fileName}</span>
                          <span className="rounded-sm bg-white px-2 py-1 font-mono text-[10px] font-bold text-slate-600 ring-1 ring-slate-200">
                            Chunk {index + 1}
                          </span>
                        </div>
                        <div className="mt-1 text-xs font-normal text-slate-500">{pageLabel || "Document evidence"}</div>
                      </summary>
                      <div className="mt-3 rounded-sm border border-slate-200 bg-white p-4">
                        <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">{source.content}</p>
                        {Object.keys(source.metadata).length > 0 ? (
                          <dl className="mt-4 grid gap-2 rounded bg-slate-50 p-3 text-xs text-slate-600 sm:grid-cols-2">
                            {Object.entries(source.metadata).map(([key, value]) => (
                              <div key={key}>
                                <dt className="font-semibold text-sky-800">{key}</dt>
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
                <div className="rounded-sm border border-dashed border-slate-300 bg-slate-50/40 px-5 py-10 text-sm text-slate-500">
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
