// State management
let currentConversationId = null;  // null = unsaved, number = saved
let conversationHistory = [];      // in-memory history for context
let currentMode = 'normal';

// Store raw text for streaming messages to enable proper formatting
const streamingRawText = {};

// Memory system state (simplified - auto-summarization)

// Token speed tracking
let tokenCount = 0;
let tokenStartTime = null;
let tokenSpeedInterval = null;
let sessionTokens = 0;  // Tokens used in current browser session

// Note: Event listeners and initialization are at the bottom of the file

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(`btn-${mode}`).classList.add('active');
}

// Token speed tracking functions
function startTokenSpeedTracking() {
    tokenCount = 0;
    tokenStartTime = Date.now();
    const speedEl = document.getElementById('token-speed');
    if (speedEl) speedEl.classList.add('active');

    // Update speed display every 200ms
    tokenSpeedInterval = setInterval(updateTokenSpeedDisplay, 200);
}

function stopTokenSpeedTracking() {
    if (tokenSpeedInterval) {
        clearInterval(tokenSpeedInterval);
        tokenSpeedInterval = null;
    }
    const speedEl = document.getElementById('token-speed');
    if (speedEl) speedEl.classList.remove('active');

    // Calculate final stats
    const elapsed = tokenStartTime ? Date.now() - tokenStartTime : 0;
    const tokensPerSecond = elapsed > 0 ? tokenCount / (elapsed / 1000) : 0;

    // Add to session tokens
    sessionTokens += tokenCount;

    // Save to database (async, don't block)
    if (tokenCount > 0) {
        logTokenUsage(tokenCount, tokensPerSecond, elapsed);
    }

    // Final display update
    updateTokenSpeedDisplay();
}

async function logTokenUsage(tokens, tokensPerSecond, durationMs) {
    try {
        await fetch('/token-usage', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_id: currentConversationId,
                tokens_output: tokens,
                tokens_per_second: Math.round(tokensPerSecond * 10) / 10,
                duration_ms: Math.round(durationMs)
            })
        });
    } catch (error) {
        console.error('Error logging token usage:', error);
    }
}

function updateTokenSpeedDisplay() {
    const valueEl = document.getElementById('token-speed-value');
    if (!valueEl) return;

    if (tokenStartTime && tokenCount > 0) {
        const elapsed = (Date.now() - tokenStartTime) / 1000;
        const tokensPerSecond = elapsed > 0 ? Math.round(tokenCount / elapsed) : 0;
        valueEl.textContent = tokensPerSecond;
    } else {
        valueEl.textContent = '0';
    }
}

function incrementTokenCount() {
    tokenCount++;
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();

    if (!message) return;

    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    appendMessage('user', message);
    input.value = '';
    input.style.height = 'auto';

    input.disabled = true;
    document.getElementById('send-btn').disabled = true;

    const aiMessageId = 'ai-msg-' + Date.now();
    appendStreamingMessage(aiMessageId);

    try {
        console.log('Sending request...');
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: message,
                mode: currentMode,
                conversation_id: currentConversationId,
                history: conversationHistory
            })
        });

        console.log('Response received, status:', response.status);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                console.log('Stream finished');
                break;
            }

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));

                    if (data.error) {
                        updateStreamingMessage(aiMessageId, data.error, true);
                    } else if (data.token) {
                        appendToStreamingMessage(aiMessageId, data.token);
                    } else if (data.done) {
                        break;
                    }
                }
            }
        }

        // After streaming completes, save to history
        const assistantResponse = streamingRawText[aiMessageId] || '';

        // Add to local history
        conversationHistory.push({ role: 'user', content: message });
        conversationHistory.push({ role: 'assistant', content: assistantResponse });

        // If this is a saved conversation, persist messages to DB
        if (currentConversationId) {
            await saveMessageToDb(currentConversationId, 'user', message);
            await saveMessageToDb(currentConversationId, 'assistant', assistantResponse);
            // Refresh list to update last_updated
            loadConversationsList();
            // Update memory stats
            updateMemoryStats();
        }

        // Update current chat section visibility
        updateCurrentChatUI();

    } catch (error) {
        console.error('Error:', error);
        updateStreamingMessage(aiMessageId, 'Error: ' + error.message, true);
    } finally {
        // Stop token speed tracking
        stopTokenSpeedTracking();

        input.disabled = false;
        document.getElementById('send-btn').disabled = false;
        input.focus();
    }
}

async function saveMessageToDb(conversationId, role, content) {
    try {
        await fetch('/message', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                conversation_id: conversationId,
                role: role,
                content: content
            })
        });
    } catch (error) {
        console.error('Error saving message:', error);
    }
}

// Chat management functions
async function saveChat() {
    if (conversationHistory.length === 0) {
        alert('No messages to save');
        return;
    }

    const saveBtn = document.querySelector('.save-btn');
    if (saveBtn) {
        saveBtn.textContent = 'Saving...';
        saveBtn.disabled = true;
    }

    try {
        const response = await fetch('/save-chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ messages: conversationHistory })
        });

        const result = await response.json();

        if (result.error) {
            alert('Error saving chat: ' + result.error);
            return;
        }

        currentConversationId = result.id;
        loadConversationsList();
        updateCurrentChatUI();
        updateMemoryStats();  // Show memory stats for newly saved conversation

    } catch (error) {
        console.error('Error saving chat:', error);
        alert('Error saving chat');
    } finally {
        if (saveBtn) {
            saveBtn.textContent = 'Save';
            saveBtn.disabled = false;
        }
    }
}

async function loadConversation(id) {
    // Prompt to save if current chat is unsaved and has messages
    if (!currentConversationId && conversationHistory.length > 0) {
        if (confirm('You have unsaved messages. Save before switching?')) {
            await saveChat();
        }
    }

    try {
        const response = await fetch(`/conversation/${id}`);
        const convo = await response.json();

        currentConversationId = convo.id;
        conversationHistory = convo.messages;
        renderMessages(convo.messages);
        updateCurrentChatUI();
        loadConversationsList();  // Update active state in list
        updateMemoryStats();  // Load memory stats for this conversation

    } catch (error) {
        console.error('Error loading conversation:', error);
        alert('Error loading conversation');
    }
}

function newChat() {
    // Prompt to save if current chat is unsaved and has messages
    if (!currentConversationId && conversationHistory.length > 0) {
        if (confirm('You have unsaved messages. Save before starting new chat?')) {
            saveChat();
        }
    }

    currentConversationId = null;
    conversationHistory = [];
    clearChatBox();
    updateCurrentChatUI();
    loadConversationsList();  // Update active state
    updateMemoryStats();  // Hide memory section for unsaved chats
}

async function deleteConversation(id, event) {
    if (event) {
        event.stopPropagation();
    }

    if (!confirm('Delete this conversation?')) return;

    try {
        await fetch(`/conversation/${id}`, { method: 'DELETE' });

        // If we deleted the current conversation, start a new chat
        if (currentConversationId === id) {
            currentConversationId = null;
            conversationHistory = [];
            clearChatBox();
            updateCurrentChatUI();
        }

        loadConversationsList();

    } catch (error) {
        console.error('Error deleting conversation:', error);
        alert('Error deleting conversation');
    }
}

// UI functions
async function loadConversationsList() {
    try {
        const response = await fetch('/conversations');
        const conversations = await response.json();
        renderConversationsList(conversations);
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

function renderConversationsList(conversations) {
    const list = document.getElementById('saved-chats-list');
    list.innerHTML = conversations.map(c => `
        <div class="chat-item ${currentConversationId === c.id ? 'active' : ''}"
             onclick="loadConversation(${c.id})">
            <span class="title">${escapeHtmlAttr(c.title)}</span>
            <button class="delete-btn" onclick="deleteConversation(${c.id}, event)">×</button>
        </div>
    `).join('');
}

function updateCurrentChatUI() {
    const section = document.getElementById('current-chat-section');
    const currentItem = document.getElementById('current-chat-item');

    if (currentConversationId) {
        // We're in a saved conversation, hide current chat section
        section.style.display = 'none';
    } else {
        // We're in an unsaved chat
        section.style.display = 'block';
        if (conversationHistory.length > 0) {
            currentItem.querySelector('.title').textContent = 'Unsaved Chat';
            currentItem.querySelector('.save-btn').style.display = 'inline-block';
        } else {
            currentItem.querySelector('.title').textContent = 'New Chat';
            currentItem.querySelector('.save-btn').style.display = 'none';
        }
    }
}

function renderMessages(messages) {
    clearChatBox();
    messages.forEach(m => {
        appendMessage(m.role === 'user' ? 'user' : 'ai', m.content);
    });
}

function clearChatBox() {
    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML = `
        <div class="welcome-message">
            <h3>Welcome to Joey Bot</h3>
            <p>Start a conversation below</p>
        </div>
    `;
}

function appendMessage(type, text) {
    const chatBox = document.getElementById('chat-box');

    // Remove welcome message if present
    const welcome = chatBox.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    messageDiv.innerHTML = `
        <div class="message-content">${formatText(text)}</div>
    `;

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendStreamingMessage(id) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.id = id;
    messageDiv.className = 'message ai';

    messageDiv.innerHTML = `
        <div class="message-content"></div>
    `;

    streamingRawText[id] = '';
    chatBox.appendChild(messageDiv);

    // Start token speed tracking
    startTokenSpeedTracking();
}

function appendToStreamingMessage(id, token) {
    const messageDiv = document.getElementById(id);
    if (messageDiv) {
        // Increment token count for speed tracking
        incrementTokenCount();

        // Accumulate raw text
        streamingRawText[id] = (streamingRawText[id] || '') + token;

        // Re-render with formatting
        const content = messageDiv.querySelector('.message-content');
        content.innerHTML = formatText(streamingRawText[id]);

        const chatBox = document.getElementById('chat-box');
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

function updateStreamingMessage(id, text, isError = false) {
    const messageDiv = document.getElementById(id);
    if (messageDiv) {
        if (isError) {
            messageDiv.classList.add('error');
        }
        const content = messageDiv.querySelector('.message-content');
        content.innerHTML = formatText(text);
        // Clean up raw text storage
        delete streamingRawText[id];
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeHtmlAttr(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function formatText(text) {
    // First escape HTML
    let formatted = escapeHtml(text);

    // Split into lines to handle bullet points
    const lines = formatted.split('\n');
    const formattedLines = lines.map(line => {
        // Check for bullet point: line starting with "* " (asterisk + space)
        if (line.match(/^\* /)) {
            line = '<li>' + line.slice(2) + '</li>';
        }
        return line;
    });

    // Join lines back, wrapping consecutive <li> in <ul>
    let result = '';
    let inList = false;
    for (const line of formattedLines) {
        if (line.startsWith('<li>')) {
            if (!inList) {
                result += '<ul>';
                inList = true;
            }
            result += line;
        } else {
            if (inList) {
                result += '</ul>';
                inList = false;
            }
            result += line + '\n';
        }
    }
    if (inList) {
        result += '</ul>';
    }

    // Remove trailing newline
    result = result.replace(/\n$/, '');

    // Bold: **text** (do this before italic to avoid conflicts)
    result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic: *text* (but not at line start which is bullet, and not part of **)
    // Use a simpler approach that doesn't require lookbehind
    result = result.replace(/([^*]|^)\*([^*\n]+)\*([^*]|$)/g, '$1<em>$2</em>$3');

    // Convert remaining newlines to <br>
    result = result.replace(/\n/g, '<br>');

    return result;
}

// Memory management functions (simplified - auto-summarization)
async function updateMemoryStats() {
    const section = document.getElementById('memory-section');

    if (!currentConversationId) {
        section.style.display = 'none';
        return;
    }

    try {
        const response = await fetch(`/memory-stats/${currentConversationId}`);
        const stats = await response.json();

        section.style.display = 'block';
        document.getElementById('total-messages').textContent = stats.total_messages;
        document.getElementById('messages-summarized').textContent = stats.messages_summarized;

        // Update note to show summary status
        const note = document.getElementById('memory-note');
        if (stats.has_summary && stats.last_summary_at) {
            note.textContent = `Last summarized: ${new Date(stats.last_summary_at).toLocaleString()}`;
        } else {
            note.textContent = 'Auto-summarizes every 10 messages';
        }
    } catch (error) {
        console.error('Error fetching memory stats:', error);
        section.style.display = 'none';
    }
}

// Initialize event listeners when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');

    if (userInput) {
        // Auto-resize textarea
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });

        // Enter to send, Shift+Enter for new line
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // Load conversations list
    loadConversationsList();

    // Initialize memory stats (will hide section if no conversation loaded)
    updateMemoryStats();
});

// ====================
// Settings View Functions
// ====================

function showSettings() {
    document.getElementById('chat-sidebar').style.display = 'none';
    document.getElementById('chat-main').style.display = 'none';
    document.getElementById('settings-view').style.display = 'flex';
}

function hideSettings() {
    document.getElementById('chat-sidebar').style.display = 'flex';
    document.getElementById('chat-main').style.display = 'flex';
    document.getElementById('settings-view').style.display = 'none';
}

function showSettingsSection(section) {
    // Update nav active state
    document.querySelectorAll('.settings-nav-item').forEach(item => {
        item.classList.remove('active');
    });
    document.getElementById('nav-' + section).classList.add('active');

    // Hide all sections, show selected
    document.querySelectorAll('.settings-section').forEach(sec => {
        sec.style.display = 'none';
    });
    document.getElementById('section-' + section).style.display = 'block';

    // Load data when entering specific sections
    if (section === 'userdetails') {
        loadUserProfile();
    } else if (section === 'usage') {
        loadUsageStats();
    }
}

// ====================
// User Profile Functions
// ====================

async function loadUserProfile() {
    try {
        const response = await fetch('/user-profile');
        const data = await response.json();
        document.getElementById('user-name').value = data.name || '';
        document.getElementById('user-details').value = data.details || '';
        updateWordCount();
    } catch (error) {
        console.error('Error loading user profile:', error);
    }
}

async function saveUserProfile() {
    const name = document.getElementById('user-name').value.trim();
    const details = document.getElementById('user-details').value.trim();
    const statusEl = document.getElementById('save-status');

    // Check word count
    const wordCount = details.split(/\s+/).filter(w => w.length > 0).length;
    if (wordCount > 125) {
        statusEl.textContent = 'Details exceed 125 word limit';
        statusEl.className = 'save-status error';
        return;
    }

    try {
        const response = await fetch('/user-profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, details })
        });

        if (response.ok) {
            statusEl.textContent = 'Profile saved!';
            statusEl.className = 'save-status';
            setTimeout(() => { statusEl.textContent = ''; }, 3000);
        }
    } catch (error) {
        statusEl.textContent = 'Error saving profile';
        statusEl.className = 'save-status error';
    }
}

function updateWordCount() {
    const textarea = document.getElementById('user-details');
    const countEl = document.getElementById('word-count');
    if (!textarea || !countEl) return;

    const words = textarea.value.trim().split(/\s+/).filter(w => w.length > 0);
    const count = textarea.value.trim() === '' ? 0 : words.length;

    countEl.textContent = `${count}/125 words`;
    countEl.className = count > 125 ? 'word-count limit-reached' : 'word-count';
}

// Add event listener for word count
document.addEventListener('DOMContentLoaded', () => {
    const userDetailsTextarea = document.getElementById('user-details');
    if (userDetailsTextarea) {
        userDetailsTextarea.addEventListener('input', updateWordCount);
    }
});

// ====================
// Usage Stats Functions
// ====================

async function loadUsageStats() {
    try {
        const response = await fetch('/usage-stats');
        const data = await response.json();

        document.getElementById('total-tokens-used').textContent = data.total_tokens.toLocaleString();
        document.getElementById('session-tokens').textContent = sessionTokens.toLocaleString();
        document.getElementById('avg-tokens-per-chat').textContent = data.avg_per_chat.toLocaleString();
        document.getElementById('total-conversations').textContent = data.total_conversations;
    } catch (error) {
        console.error('Error loading usage stats:', error);
    }
}
