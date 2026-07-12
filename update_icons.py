import re

with open('web_dashboard/index.html', 'r') as f:
    content = f.read()

# Add Lucide script to head
if 'lucide' not in content:
    content = content.replace('</head>', '    <script src="https://unpkg.com/lucide@latest"></script>\n</head>')

# Replace specific Emojis with Lucide Icons
replacements = {
    '<span class="icon">💬</span>': '<i data-lucide="message-square" class="icon"></i>',
    '🎙️': '<i data-lucide="mic"></i>',
    '🧠 Plan Mission': '<i data-lucide="brain-circuit"></i> Plan Mission',
    '✏️': '<i data-lucide="edit-3" style="width: 14px; height: 14px;"></i>',
    '📍': '<i data-lucide="map-pin" style="width: 14px; height: 14px;"></i>',
    '🔁': '<i data-lucide="repeat" style="width: 14px; height: 14px;"></i>',
    '∞': '<i data-lucide="infinity" style="width: 14px; height: 14px;"></i>',
    '🔤': '<i data-lucide="type" style="width: 14px; height: 14px;"></i>',
    '⏱': '<i data-lucide="clock" style="width: 14px; height: 14px;"></i>',
    '<span class="icon">📋</span>': '<i data-lucide="file-json" class="icon"></i>',
    '<div class="empty-state-icon">📋</div>': '<div class="empty-state-icon"><i data-lucide="file-json" style="width: 48px; height: 48px;"></i></div>',
    '<span class="icon">🛡️</span>': '<i data-lucide="shield-check" class="icon"></i>',
    '<div class="empty-state-icon">🛡️</div>': '<div class="empty-state-icon"><i data-lucide="shield-check" style="width: 48px; height: 48px;"></i></div>',
    '⚙️ Edit Safety Limits': '<i data-lucide="settings" style="width: 14px; height: 14px; margin-right: 4px;"></i> Edit Safety Limits',
    '<span class="icon">🗺️</span>': '<i data-lucide="map" class="icon"></i>',
    '<span class="icon">🚀</span>': '<i data-lucide="terminal" class="icon"></i>',
    '<div class="empty-state-icon">🚀</div>': '<div class="empty-state-icon"><i data-lucide="terminal" style="width: 48px; height: 48px;"></i></div>',
    '🚀 Execute Mission': '<i data-lucide="rocket"></i> Execute Mission',
    '⏸️ Hold': '<i data-lucide="pause-circle"></i> Hold',
    '🏠 RTH': '<i data-lucide="home"></i> RTH',
    '🛑 Terminate': '<i data-lucide="power"></i> Terminate',
    '<span class="icon">⚙️</span>': '<i data-lucide="settings" class="icon"></i>',
    '💾 Save & Apply': '<i data-lucide="save"></i> Save & Apply'
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Initialize Lucide at end of script
if 'lucide.createIcons()' not in content:
    content = content.replace('loadSafetyLimits();', 'loadSafetyLimits();\n        lucide.createIcons();')
    content = content.replace('lucide.createIcons();\n        lucide.createIcons();', 'lucide.createIcons();')

# Convert inline styles on action buttons to classes or clear them up
content = content.replace('background-color: var(--accent-amber); color: #1e293b; border-radius: 8px; border: none; padding: 12px 24px; font-weight: 600; cursor: pointer; transition: background 0.2s;', 'background: #f59e0b; color: white;')
content = content.replace('background-color: var(--accent-blue); color: white; border-radius: 8px; border: none; padding: 12px 24px; font-weight: 600; cursor: pointer; transition: background 0.2s;', 'background: #0ea5e9; color: white;')

with open('web_dashboard/index.html', 'w') as f:
    f.write(content)
