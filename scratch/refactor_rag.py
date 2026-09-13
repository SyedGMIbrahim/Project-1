import re

with open('frontend/app/page.tsx', 'r') as f:
    content = f.read()

# Replace focus rings on all textareas and inputs
content = content.replace(
    'focus:shadow-[0_0_0_4px_rgba(140,73,223,0.10)]',
    'focus:ring-2 focus:ring-sky-600 focus:border-sky-600'
)

# Header update: remove rounded-3xl, make it crisp
content = content.replace(
    '<header className="mb-8 flex items-center justify-between rounded-md border border-slate-200 bg-white/80 px-5 py-4 shadow-sm ">',
    '<header className="mb-8 flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4">'
)
content = content.replace(
    '<p className="text-xs font-semibold uppercase tracking-[0.35em] text-sky-700">',
    '<p className="text-[11px] font-bold uppercase tracking-widest text-slate-500">'
)
content = content.replace(
    '<h1 className="mt-1 text-2xl font-semibold text-slate-900 sm:text-3xl">',
    '<h1 className="mt-1 text-2xl font-black tracking-tight text-slate-900 sm:text-3xl">'
)
content = content.replace(
    '<div className="hidden rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-sm text-sky-800 sm:block">',
    '<div className="hidden rounded-sm border border-slate-200 bg-slate-50 px-4 py-1.5 text-[11px] font-bold uppercase tracking-widest text-slate-500 sm:block">'
)

# Main RAG box
content = content.replace(
    '<div className="flex flex-col rounded-md border border-slate-200 bg-white/90 p-5 shadow-sm  sm:p-8">',
    '<div className="flex flex-col border border-slate-200 bg-white p-5 shadow-sm sm:p-8">'
)

content = content.replace(
    '<p className="text-sm font-medium text-sky-700">',
    '<p className="text-[11px] font-bold uppercase tracking-widest text-sky-700">'
)

# Drag and drop upload box
content = content.replace(
    '<div className="mb-6 rounded-md border border-dashed border-slate-300 bg-slate-50/50 p-4">',
    '<div className="mb-6 border border-dashed border-slate-300 bg-slate-50 p-6">'
)
content = content.replace(
    'className={`rounded-sm border-2 border-dashed p-5 text-center transition ${dragActive ? "border-sky-500 bg-white" : "border-slate-300 bg-white/80"}`}',
    'className={`rounded-sm border-2 border-dashed p-6 text-center transition ${dragActive ? "border-sky-600 bg-sky-50" : "border-slate-300 bg-white"}`}'
)
content = content.replace(
    'className="inline-flex items-center justify-center gap-2 rounded-sm bg-sky-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-60"',
    'className="inline-flex items-center justify-center gap-2 rounded-sm bg-slate-900 px-5 py-2.5 text-[13px] font-bold tracking-wide text-white transition hover:bg-slate-800 disabled:opacity-50"'
)

# Indexed Datasets table header
content = content.replace(
    '<section className="mb-8 rounded-md border border-slate-200 bg-white p-5 shadow-sm">',
    '<section className="mb-8 border border-slate-200 bg-white p-5 shadow-sm">'
)
content = content.replace(
    '<span className="text-xs font-medium text-sky-700">Refreshing...</span>',
    '<span className="text-[11px] font-bold uppercase text-sky-700">Refreshing...</span>'
)
content = content.replace(
    '<div className="overflow-hidden rounded-sm border border-slate-200">',
    '<div className="border border-slate-200">'
)
content = content.replace(
    '<div className="grid grid-cols-[1.2fr_1.2fr_0.7fr_0.8fr_0.8fr_0.9fr] gap-3 border-b border-slate-200 bg-slate-50 px-4 py-3 text-xs font-semibold uppercase tracking-[0.18em] text-sky-700">',
    '<div className="grid grid-cols-[1.2fr_1.2fr_0.7fr_0.8fr_0.8fr_0.9fr] gap-3 border-b border-slate-200 bg-slate-50 px-4 py-3 text-[11px] font-bold uppercase tracking-widest text-slate-500">'
)
content = content.replace(
    '<div className="mt-0.5 flex h-9 w-9 items-center justify-center rounded bg-slate-50 text-xs font-bold text-sky-700 ring-1 ring-slate-200">',
    '<div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-sm bg-slate-100 text-[10px] font-bold text-slate-700 ring-1 ring-slate-200">'
)
content = content.replace(
    '<span className="inline-flex items-center gap-1 rounded-full bg-slate-200 px-2.5 py-1 text-xs font-semibold text-sky-800 ring-1 ring-slate-400">',
    '<span className="inline-flex items-center gap-1 rounded-sm bg-sky-50 px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-sky-700 ring-1 ring-sky-200">'
)
content = content.replace(
    '<span className="inline-flex rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">',
    '<span className="inline-flex rounded-sm bg-emerald-50 px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-emerald-700">'
)
content = content.replace(
    '<button\n                                type="button"\n                                onClick={() => setSelectedDocument(document.file_name)}\n                                className="rounded-lg bg-white px-2.5 py-1 text-xs font-medium text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50 hover:text-sky-700 transition"\n                              >',
    '<button\n                                type="button"\n                                onClick={() => setSelectedDocument(document.file_name)}\n                                className="rounded-sm bg-white px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-slate-500 ring-1 ring-slate-200 hover:bg-slate-50 hover:text-slate-900 transition"\n                              >'
)

# Active Target Banner
content = content.replace(
    '<div className="mb-3 flex flex-wrap items-center justify-between gap-2 rounded-sm border border-slate-300 bg-slate-50/70 px-4 py-2.5 text-xs text-sky-900">',
    '<div className="mb-4 flex flex-wrap items-center justify-between gap-2 border border-sky-200 bg-sky-50 px-4 py-3 text-xs text-sky-900">'
)
content = content.replace(
    '<span className="font-semibold text-sky-700">Target Report:</span>',
    '<span className="text-[11px] font-bold uppercase tracking-widest text-sky-700">Target Report:</span>'
)
content = content.replace(
    '<span className="rounded-md bg-white px-2 py-0.5 font-mono text-[11px] font-semibold text-slate-900 ring-1 ring-slate-200">{selectedDocument}</span>',
    '<span className="rounded-sm bg-white px-2 py-1 font-mono text-[11px] font-bold text-sky-900 ring-1 ring-sky-200">{selectedDocument}</span>'
)

# Quick Analysis Buttons
content = content.replace(
    '<span className="font-medium text-slate-500">Quick Analysis:</span>',
    '<span className="text-[11px] font-bold uppercase tracking-widest text-slate-500">Quick Analysis:</span>'
)
content = content.replace(
    'className="inline-flex items-center gap-1 rounded bg-slate-200 px-3 py-1.5 font-semibold text-sky-800 transition hover:bg-slate-300 disabled:opacity-50"',
    'className="inline-flex items-center gap-1 rounded-sm bg-sky-700 px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide text-white transition hover:bg-sky-800 disabled:opacity-50"'
)
content = content.replace(
    'className="inline-flex items-center gap-1 rounded bg-white px-3 py-1.5 font-medium text-slate-700 ring-1 ring-slate-300 transition hover:bg-slate-50 disabled:opacity-50"',
    'className="inline-flex items-center gap-1 rounded-sm bg-white px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 transition hover:bg-slate-50 disabled:opacity-50"'
)

# RAG Query Input
content = content.replace(
    '<div className="flex flex-col gap-3 rounded-md border border-slate-200 bg-slate-50/70 p-3 shadow-sm sm:flex-row">',
    '<div className="flex flex-col gap-3 border border-slate-200 bg-slate-50 p-4 shadow-sm sm:flex-row">'
)

content = content.replace(
    'className="inline-flex items-center justify-center gap-2 rounded-sm bg-sky-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-50"',
    'className="inline-flex items-center justify-center gap-2 rounded-sm bg-slate-900 px-5 py-2.5 text-[13px] font-bold tracking-wide text-white transition hover:bg-slate-800 disabled:opacity-50"'
)
content = content.replace(
    'className="inline-flex items-center justify-center gap-2 rounded-sm bg-white px-3 py-2 text-xs font-semibold text-sky-800 ring-1 ring-slate-400 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"',
    'className="inline-flex items-center justify-center gap-2 rounded-sm bg-white px-3 py-2 text-[11px] font-bold uppercase tracking-wide text-slate-700 ring-1 ring-slate-300 transition hover:bg-slate-50 disabled:opacity-50"'
)


# RAG Response Box
content = content.replace(
    '<div className="flex-1 rounded-md border border-slate-200 bg-white p-5">',
    '<div className="flex-1 border border-slate-200 bg-white p-6 shadow-sm">'
)
content = content.replace(
    '<h2 className="text-lg font-semibold text-slate-900">Generated answer</h2>',
    '<h2 className="text-[11px] font-bold uppercase tracking-widest text-slate-500">GENERATED ANSWER</h2>'
)
content = content.replace(
    '<span className="text-sm text-sky-700">Analyzing local sources...</span>',
    '<span className="text-[11px] font-bold uppercase tracking-widest text-sky-700">Analyzing sources...</span>'
)
content = content.replace(
    '<p className="whitespace-pre-wrap text-[15px] leading-7 text-slate-700">{answer}</p>',
    '<p className="whitespace-pre-wrap font-serif text-[15px] leading-7 text-slate-900">{answer}</p>'
)

# Symptom Checker Header
content = content.replace(
    '<div className="mt-8 rounded-md border border-slate-200 bg-white/90 p-5 shadow-sm  sm:p-6">',
    '<div className="mt-8 border border-slate-200 bg-white p-6 shadow-sm">'
)
content = content.replace(
    '<h3 className="text-xl font-semibold text-slate-900">Symptom Checker</h3>',
    '<h3 className="text-[11px] font-bold uppercase tracking-widest text-slate-500">SYMPTOM CHECKER & DIAGNOSTIC GATE</h3>'
)

# Evidence Panel (Aside)
content = content.replace(
    '<aside className="rounded-md border border-slate-200 bg-white/90 p-5 shadow-sm  sm:p-8">',
    '<aside className="border border-slate-200 bg-white p-6 shadow-sm">'
)
content = content.replace(
    '<h2 className="text-lg font-semibold text-slate-900">Evidence panel</h2>',
    '<h2 className="text-[11px] font-bold uppercase tracking-widest text-slate-500">EVIDENCE PANEL</h2>'
)
content = content.replace(
    '<span className="rounded-full bg-slate-50 px-3 py-1 text-xs font-semibold text-sky-700">',
    '<span className="rounded-sm bg-slate-100 px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-widest text-slate-600">'
)
content = content.replace(
    '<details key={`${fileName}-${index}`} className="group rounded-sm border border-slate-200 bg-slate-50/40 p-4 open:bg-white">',
    '<details key={`${fileName}-${index}`} className="group rounded-sm border border-slate-200 bg-slate-50 p-4 open:bg-white shadow-sm">'
)
content = content.replace(
    '<span className="rounded-full bg-white px-2.5 py-1 text-xs font-medium text-sky-700 ring-1 ring-slate-200">',
    '<span className="rounded-sm bg-white px-2 py-1 font-mono text-[10px] font-bold text-slate-600 ring-1 ring-slate-200">'
)


with open('frontend/app/page.tsx', 'w') as f:
    f.write(content)

