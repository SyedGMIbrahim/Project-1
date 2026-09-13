import re

with open('frontend/app/page.tsx', 'r') as f:
    content = f.read()

replacement = """            {/* ── 3-State Symptom Epistemic Categorization ── */}
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
                  className={`mt-6 rounded-sm border p-6 transition-all ${
                    diagnosisResult.decision === "DIAGNOSE"
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
                      <div className={`mt-1 text-2xl font-black tracking-tight ${
                        diagnosisResult.decision === "DIAGNOSE"
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
                          className={`h-full ${
                            (diagnosisResult.evidence_coverage || 0) >= 0.35 ? (diagnosisResult.decision === "ABSTAIN" ? "bg-emerald-400" : "bg-emerald-600") : (diagnosisResult.decision === "ABSTAIN" ? "bg-amber-400" : "bg-amber-500")
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
"""

start_marker = "{/* ── 3-State Symptom Epistemic Categorization ── */}"
end_marker = "</div>\n          </div>\n\n          <aside className=\"rounded-md border border-slate-200"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    new_content = content[:start_idx] + replacement + "            " + content[end_idx:]
    with open('frontend/app/page.tsx', 'w') as f:
        f.write(new_content)
    print("Success")
else:
    print("Markers not found", start_idx, end_idx)
