/* Pipeline Observability Dashboard — vanilla JS */

let autoRefreshInterval = null;

// ── Utilities ──────────────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '...' : str;
}

function formatTime(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ── Fetchers ───────────────────────────────────────────────────────
async function fetchPipelineLatest() {
    try {
        const res = await fetch('/api/pipeline-latest');
        return await res.json();
    } catch { return null; }
}

async function fetchPipelineRuns() {
    try {
        const res = await fetch('/api/pipeline-runs');
        return await res.json();
    } catch { return []; }
}

async function fetchMemoryHealth() {
    try {
        const res = await fetch('/semantic-memory-stats');
        return await res.json();
    } catch { return null; }
}

// ── Refresh all sections ───────────────────────────────────────────
async function refreshAll() {
    const btn = document.getElementById('refresh-btn');
    btn.disabled = true;
    btn.textContent = 'Refreshing...';

    try {
        const [latest, runs, health] = await Promise.all([
            fetchPipelineLatest(),
            fetchPipelineRuns(),
            fetchMemoryHealth(),
        ]);
        renderStageCards(latest);
        renderPipelineRuns(runs);
        renderMemoryHealth(health);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Refresh';
    }
}

// ── Stage Cards ────────────────────────────────────────────────────
const STAGE_NAMES = ['classify', 'retrieve', 'score', 'build_context', 'generate', 'post_process'];

function renderStageCards(run) {
    const subtitle = document.getElementById('pipeline-subtitle');

    if (!run) {
        subtitle.textContent = 'No pipeline runs yet';
        STAGE_NAMES.forEach(name => {
            const card = document.getElementById('stage-' + name);
            if (card) {
                card.className = 'stage-card stage-empty';
                card.querySelector('.stage-timing').textContent = '—';
                card.querySelector('.stage-detail').textContent = 'Waiting for data';
            }
        });
        return;
    }

    subtitle.textContent = formatTime(run.timestamp) + '  ·  ' + run.total_time_ms + 'ms total';

    STAGE_NAMES.forEach(name => {
        const card = document.getElementById('stage-' + name);
        if (!card) return;

        const stage = (run.stages && run.stages[name]) || {};
        const timing = (run.timings && run.timings[name]) != null ? run.timings[name] : null;
        const status = stage.status || 'empty';

        card.className = 'stage-card stage-' + status;
        card.querySelector('.stage-timing').textContent = timing != null ? timing.toFixed(1) + ' ms' : '—';
        card.querySelector('.stage-detail').textContent = stage.detail || '';
    });
}

// ── Memory Health ──────────────────────────────────────────────────
function renderMemoryHealth(stats) {
    if (!stats) return;

    document.getElementById('mem-total').textContent = stats.total_facts || 0;

    const tiers = stats.strength_tiers || {};
    document.getElementById('mem-strong').textContent = tiers.strong || 0;
    document.getElementById('mem-medium').textContent = tiers.medium || 0;
    document.getElementById('mem-weak').textContent = tiers.weak || 0;
    document.getElementById('mem-consolidated').textContent = stats.consolidated_count || 0;

    const lastRun = document.getElementById('lifecycle-last-run');
    if (stats.last_lifecycle_run) {
        lastRun.textContent = 'Last lifecycle run: ' + formatTime(stats.last_lifecycle_run);
    } else {
        lastRun.textContent = '';
    }
}

// ── Lifecycle Actions ──────────────────────────────────────────────
async function runLifecycleAction(action, btn) {
    const statusEl = document.getElementById('lifecycle-status');
    btn.disabled = true;
    statusEl.textContent = 'Running ' + action + '...';
    statusEl.className = 'lifecycle-status';

    try {
        const res = await fetch('/memory-lifecycle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action }),
        });
        const data = await res.json();
        if (res.ok) {
            statusEl.textContent = action + ' completed';
            statusEl.className = 'lifecycle-status status-ok';
        } else {
            statusEl.textContent = data.error || 'Error';
            statusEl.className = 'lifecycle-status status-err';
        }
        // Refresh memory health after lifecycle action
        const health = await fetchMemoryHealth();
        renderMemoryHealth(health);
    } catch (e) {
        statusEl.textContent = 'Network error';
        statusEl.className = 'lifecycle-status status-err';
    } finally {
        btn.disabled = false;
    }
}

// ── Pipeline Runs Table ────────────────────────────────────────────
const CLASS_COLORS = {
    NONE: 'badge-none',
    RECENT: 'badge-recent',
    SEMANTIC: 'badge-semantic',
    PROFILE: 'badge-profile',
    MULTI: 'badge-multi',
};

function renderPipelineRuns(runs) {
    const tbody = document.getElementById('runs-tbody');

    if (!runs || runs.length === 0) {
        tbody.innerHTML = '<tr class="empty-row"><td colspan="6">No pipeline runs recorded yet. Send a chat message to generate data.</td></tr>';
        return;
    }

    tbody.innerHTML = runs.map(run => {
        const cls = run.classification || {};
        const need = cls.memory_need || 'SEMANTIC';
        const badgeClass = CLASS_COLORS[need] || 'badge-semantic';
        const errCount = (run.errors && run.errors.length) || 0;
        const errCell = errCount > 0
            ? '<span class="error-count">' + errCount + '</span>'
            : '<span class="no-errors">0</span>';

        return '<tr>' +
            '<td>' + escapeHtml(formatTime(run.timestamp)) + '</td>' +
            '<td class="msg-cell" title="' + escapeHtml(run.user_message || '') + '">' + escapeHtml(truncate(run.user_message, 50)) + '</td>' +
            '<td><span class="class-badge ' + badgeClass + '">' + escapeHtml(need) + '</span></td>' +
            '<td>' + (run.candidate_count || 0) + '</td>' +
            '<td>' + (run.total_time_ms || 0) + '</td>' +
            '<td>' + errCell + '</td>' +
            '</tr>';
    }).join('');
}

// ── Auto-refresh ───────────────────────────────────────────────────
function startAutoRefresh() {
    if (autoRefreshInterval) return;
    autoRefreshInterval = setInterval(refreshAll, 5000);
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
}

document.getElementById('auto-refresh').addEventListener('change', function () {
    if (this.checked) {
        startAutoRefresh();
    } else {
        stopAutoRefresh();
    }
});

// ── Init ───────────────────────────────────────────────────────────
refreshAll();
startAutoRefresh();
