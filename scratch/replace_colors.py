import re

with open('frontend/app/page.tsx', 'r') as f:
    content = f.read()

# Replace general colors
content = content.replace('bg-clinical-gradient', 'bg-slate-50')
content = content.replace('text-ink-950', 'text-slate-900')
content = content.replace('plum-50', 'slate-50')
content = content.replace('plum-100', 'slate-200')
content = content.replace('plum-200', 'slate-300')
content = content.replace('plum-300', 'slate-400')
content = content.replace('plum-400', 'sky-400')
content = content.replace('plum-500', 'sky-500')
content = content.replace('plum-600', 'sky-600')
content = content.replace('plum-700', 'sky-700')
content = content.replace('plum-800', 'sky-800')
content = content.replace('plum-900', 'sky-900')

# Replace shapes
content = content.replace('rounded-3xl', 'rounded-md')
content = content.replace('rounded-2xl', 'rounded-sm')
content = content.replace('rounded-xl', 'rounded')
content = content.replace('shadow-soft', 'shadow-sm')
content = content.replace('backdrop-blur', '')

with open('frontend/app/page.tsx', 'w') as f:
    f.write(content)

