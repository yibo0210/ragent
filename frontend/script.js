const { createApp } = Vue;

createApp({
    data() {
        return {
            messages: [],
            userInput: '',
            isLoading: false,
            activeNav: 'newChat',
            abortController: null,
            sessionId: 'session_' + Date.now(),
            sessions: [],
            sidebarCollapsed: false,
            isComposing: false,
            liveAgents: [],
            traceSteps: [],
            hitlState: null,
            hitlAction: null,
            hitlModifiedInput: '',
            hitlPending: false,
            documents: [],
            documentsLoading: false,
            selectedFile: null,
            isUploading: false,
            uploadProgress: '',
            uploadPercent: 0,
            theme: localStorage.getItem('ragent-theme') || 'light',
            toasts: [],
            toastId: 0,
            confirmState: null,
            isDragOver: false,
            sessionsLoading: false,
            skipAnimation: false,
            showTrace: false
        };
    },
    mounted() {
        document.documentElement.setAttribute('data-theme', this.theme);
        this.configureMarked();
        this.setupCodeCopyListener();
        this.loadSessions();
    },
    methods: {
        configureMarked() {
            const renderer = new marked.Renderer();
            renderer.code = function(code, language) {
                const lang = (language || '').split(' ')[0];
                const highlighted = hljs.getLanguage(lang)
                    ? hljs.highlight(code, { language: lang }).value
                    : hljs.highlight(code, { language: 'plaintext' }).value;
                return `<div class="code-block"><div class="code-header"><span class="code-lang">${lang || 'text'}</span><button class="code-copy-btn"><i class="fas fa-copy"></i></button></div><pre><code class="hljs language-${lang}">${highlighted}</code></pre></div>`;
            };
            marked.setOptions({ renderer, breaks: true, gfm: true });
        },

        setupCodeCopyListener() {
            const container = this.$refs.chatContainer;
            if (!container) return;
            container.addEventListener('click', (e) => {
                const btn = e.target.closest('.code-copy-btn');
                if (!btn) return;
                const code = btn.closest('.code-block').querySelector('code').textContent;
                navigator.clipboard.writeText(code).then(() => {
                    btn.innerHTML = '<i class="fas fa-check"></i>';
                    setTimeout(() => { btn.innerHTML = '<i class="fas fa-copy"></i>'; }, 1500);
                });
            });
        },

        parseMarkdown(text) {
            return marked.parse(text);
        },

        renderMessage(msg) {
            if (!msg.text) return '';
            if (msg._streaming) {
                return this.escapeHtml(msg.text).replace(/\n/g, '<br>');
            }
            return marked.parse(msg.text);
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        handleCompositionStart() { this.isComposing = true; },
        handleCompositionEnd() { this.isComposing = false; },

        handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },

        handleStop() {
            if (this.abortController) this.abortController.abort();
        },

        sendSuggestion(text) {
            this.userInput = text;
            this.handleSend();
        },

        async handleSend() {
            const text = this.userInput.trim();
            if (!text || this.isLoading || this.isComposing) return;
            if (this.hitlState) { this.showToast('请先完成人工审核', 'warning'); return; }

            this.messages.push({ text, isUser: true });
            this.userInput = '';
            this.$nextTick(() => { this.resetTextareaHeight(); this.scrollToBottom(); });

            this.showTrace = true;
            this.isLoading = true;
            this.messages.push({ text: '', isUser: false, isThinking: true, _streaming: true, ragTrace: null, ragSteps: [], agentRoutes: [], webSearchResults: [], agentTrace: null });
            const botIdx = this.messages.length - 1;
            this.abortController = new AbortController();

            try {
                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text, session_id: this.sessionId }),
                    signal: this.abortController.signal,
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let idx;
                    while ((idx = buffer.indexOf('\n\n')) !== -1) {
                        const ev = buffer.slice(0, idx);
                        buffer = buffer.slice(idx + 2);
                        if (!ev.startsWith('data: ')) continue;
                        const raw = ev.slice(6);
                        if (raw === '[DONE]') continue;
                        try {
                            const data = JSON.parse(raw);
                            if (data.type === 'content') {
                                if (this.messages[botIdx].isThinking) this.messages[botIdx].isThinking = false;
                                this.messages[botIdx].text += data.content;
                            } else if (data.type === 'trace') {
                                const trace = data.rag_trace;
                                trace._pipelineOpen = false;
                                trace._retrievalOpen = false;
                                trace._rewriteOpen = false;
                                trace._resultsOpen = true;
                                this.messages[botIdx].ragTrace = trace;
                            } else if (data.type === 'rag_step') {
                                if (!this.messages[botIdx].ragSteps) this.messages[botIdx].ragSteps = [];
                                this.messages[botIdx].ragSteps.push(data.step);
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: (data.step.agent) || 'rag',
                                    message: (data.step.icon || '') + ' ' + (data.step.message || '')
                                });
                            } else if (data.type === 'routing') {
                                if (!this.messages[botIdx].agentRoutes) this.messages[botIdx].agentRoutes = [];
                                this.messages[botIdx].agentRoutes.push({
                                    agent: data.agent,
                                    reason: data.reason,
                                    timestamp: Date.now()
                                });
                            } else if (data.type === 'web_search_results') {
                                this.messages[botIdx].webSearchResults = data.results || [];
                            } else if (data.type === 'agent_trace') {
                                this.messages[botIdx].agentTrace = data.agent_trace;
                            } else if (data.type === 'agent_start') {
                                const existing = this.liveAgents.find(a => a.name === data.agent);
                                if (existing) {
                                    existing.status = 'active';
                                } else {
                                    this.liveAgents.push({ name: data.agent, status: 'active' });
                                }
                                this.traceSteps.push({ timestamp: Date.now(), agent: data.agent, message: '开始回答' });
                            } else if (data.type === 'agent_done') {
                                const agent = this.liveAgents.find(a => a.name === data.agent);
                                if (agent) agent.status = 'done';
                            } else if (data.type === 'worker_content') {
                                // 不推送完整回答到追踪面板
                            } else if (data.type === 'graph_expand') {
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: data.agent,
                                    message: '🔗 ' + data.message
                                });
                            } else if (data.type === 'community_match') {
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: data.agent,
                                    message: '📊 ' + data.message
                                });
                            } else if (data.type === 'hitl_interrupt') {
                                this.hitlState = data.data;
                                this.hitlAction = null;
                                this.hitlModifiedInput = '';
                                this.isLoading = false;
                                this.traceSteps.push({ timestamp: Date.now(), agent: 'system', message: 'HITL 中断，等待人工干预' });
                            } else if (data.type === 'error') {
                                this.messages[botIdx].isThinking = false;
                                this.messages[botIdx].text += `\n[Error: ${data.content}]`;
                            }
                        } catch (e) { console.warn('SSE parse error:', e); }
                    }
                    this.$nextTick(() => this.scrollToBottom());
                }
            } catch (error) {
                this.messages[botIdx].isThinking = false;
                if (error.name === 'AbortError') {
                    this.messages[botIdx].text = this.messages[botIdx].text || '(已终止回答)';
                    if (this.messages[botIdx].text && !this.messages[botIdx].text.includes('已终止')) {
                        this.messages[botIdx].text += '\n\n_(回答已被终止)_';
                    }
                } else {
                    console.error('Error:', error);
                    this.messages[botIdx].text = `抱歉，出了点问题：${error.message}`;
                }
            } finally {
                this.isLoading = false;
                this.abortController = null;
                if (this.messages[botIdx]) this.messages[botIdx]._streaming = false;
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        autoResize(event) {
            const ta = event.target;
            ta.style.height = 'auto';
            ta.style.height = ta.scrollHeight + 'px';
        },

        resetTextareaHeight() {
            if (this.$refs.textarea) this.$refs.textarea.style.height = 'auto';
        },

        scrollToBottom() {
            if (this.$refs.chatContainer) this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
        },

        handleNewChat() {
            this.messages = [];
            this.sessionId = 'session_' + Date.now();
            this.activeNav = 'newChat';
            this.liveAgents = [];
            this.traceSteps = [];
            this.showTrace = false;
            this.refreshSessionList();
        },

        async refreshSessionList() {
            // 只刷新侧边栏会话列表，不加载最新会话的消息
            try {
                const res = await fetch('/sessions');
                if (!res.ok) throw new Error('Failed');
                const data = await res.json();
                this.sessions = data.sessions;
            } catch (e) {
                console.error('Error loading sessions:', e);
            }
        },

        async loadSessions() {
            this.sessionsLoading = true;
            try {
                const res = await fetch('/sessions');
                if (!res.ok) throw new Error('Failed');
                const data = await res.json();
                this.sessions = data.sessions;
            } catch (e) {
                console.error('Error loading sessions:', e);
            } finally {
                this.sessionsLoading = false;
            }
        },

        async loadSessionMessages(sessionId) {
            try {
                this.skipAnimation = true;
                const res = await fetch(`/sessions/${sessionId}`);
                if (!res.ok) throw new Error('Failed');
                const data = await res.json();
                this.messages = data.messages.map(msg => ({
                    text: msg.content,
                    isUser: msg.type === 'human',
                    ragTrace: msg.rag_trace || null
                }));
                this.$nextTick(() => { this.skipAnimation = false; this.scrollToBottom(); });
            } catch (e) {
                console.error('Error loading session messages:', e);
                this.skipAnimation = false;
            }
        },

        async loadSession(sessionId) {
            this.sessionId = sessionId;
            this.activeNav = 'newChat';
            this.liveAgents = [];
            this.traceSteps = [];
            await this.loadSessionMessages(sessionId);
        },

        async deleteSession(sessionId) {
            const ok = await this.showConfirm('确定要删除该会话吗？');
            if (!ok) return;
            try {
                const res = await fetch(`/sessions/${sessionId}`, { method: 'DELETE' });
                const payload = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(payload.detail || 'Delete failed');
                this.sessions = this.sessions.filter(s => s.session_id !== sessionId);
                if (this.sessionId === sessionId) {
                    this.messages = [];
                    this.sessionId = 'session_' + Date.now();
                }
                this.showToast('会话已删除', 'success');
            } catch (e) {
                console.error('Error deleting session:', e);
                this.showToast('删除会话失败：' + e.message, 'error');
            }
        },

        handleSettings() {
            this.activeNav = 'settings';
            this.loadDocuments();
        },

        async loadDocuments() {
            this.documentsLoading = true;
            try {
                const res = await fetch('/documents');
                if (!res.ok) throw new Error('Failed');
                const data = await res.json();
                this.documents = data.documents;
            } catch (e) {
                console.error('Error loading documents:', e);
            } finally {
                this.documentsLoading = false;
            }
        },

        handleFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.selectedFile = files[0];
                this.uploadProgress = '';
            }
        },

        async uploadDocument() {
            if (!this.selectedFile) return;
            this.isUploading = true;
            this.uploadProgress = '正在上传...';
            this.uploadPercent = 0;
            try {
                const fd = new FormData();
                fd.append('file', this.selectedFile);
                const response = await fetch('/documents/upload', { method: 'POST', body: fd });
                if (!response.ok) {
                    const e = await response.json().catch(() => ({}));
                    throw new Error(e.detail || 'Upload failed');
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let idx;
                    while ((idx = buffer.indexOf('\n\n')) !== -1) {
                        const ev = buffer.slice(0, idx);
                        buffer = buffer.slice(idx + 2);
                        if (!ev.startsWith('data: ')) continue;
                        try {
                            const data = JSON.parse(ev.slice(6));
                            if (data.type === 'progress') {
                                this.uploadProgress = data.status;
                                if (data.total > 0) {
                                    this.uploadPercent = Math.round((data.current / data.total) * 100);
                                }
                            } else if (data.type === 'complete') {
                                this.uploadProgress = data.message;
                                this.uploadPercent = 100;
                                this.selectedFile = null;
                                if (this.$refs.fileInput) this.$refs.fileInput.value = '';
                                await this.loadDocuments();
                                setTimeout(() => { this.uploadProgress = ''; this.uploadPercent = 0; }, 3000);
                            } else if (data.type === 'error') {
                                throw new Error(data.message);
                            }
                        } catch (e) {
                            if (e.message && !e.message.includes('JSON')) throw e;
                        }
                    }
                }
            } catch (e) {
                console.error('Error uploading:', e);
                this.uploadProgress = '上传失败：' + e.message;
            } finally {
                this.isUploading = false;
            }
        },

        async deleteDocument(filename) {
            const ok = await this.showConfirm(`确定要删除文档 "${filename}" 吗？`);
            if (!ok) return;
            try {
                const res = await fetch(`/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
                if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Delete failed'); }
                await this.loadDocuments();
                this.showToast('文档已删除', 'success');
            } catch (e) {
                console.error('Error deleting document:', e);
                this.showToast('删除文档失败：' + e.message, 'error');
            }
        },

        getFileIcon(fileType) {
            const map = {
                'PDF': 'fas fa-file-pdf',
                'Word': 'fas fa-file-word',
                'Excel': 'fas fa-file-excel',
                'Markdown': 'fas fa-file-code',
                'Image': 'fas fa-file-image'
            };
            return map[fileType] || 'fas fa-file';
        },

        // Theme
        toggleTheme() {
            this.theme = this.theme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', this.theme);
            localStorage.setItem('ragent-theme', this.theme);
            const meta = document.querySelector('meta[name="theme-color"]');
            if (meta) meta.content = this.theme === 'dark' ? '#1a1b1e' : '#3b82f6';
        },

        // Toast
        showToast(message, type = 'info', duration = 3000) {
            const id = ++this.toastId;
            this.toasts.push({ id, message, type });
            if (duration > 0) setTimeout(() => this.removeToast(id), duration);
        },
        removeToast(id) {
            this.toasts = this.toasts.filter(t => t.id !== id);
        },
        toastIcon(type) {
            const map = { success: 'fas fa-circle-check', error: 'fas fa-circle-xmark', info: 'fas fa-circle-info', warning: 'fas fa-triangle-exclamation' };
            return map[type] || map.info;
        },

        // Confirm
        showConfirm(message) {
            return new Promise(resolve => {
                this.confirmState = { message, resolve };
            });
        },

        // Message Actions
        async copyMessage(text) {
            try {
                await navigator.clipboard.writeText(text);
                this.showToast('已复制到剪贴板', 'success', 2000);
            } catch (e) {
                this.showToast('复制失败', 'error');
            }
        },
        regenerateResponse(botIndex) {
            let userText = '';
            for (let i = botIndex - 1; i >= 0; i--) {
                if (this.messages[i].isUser) { userText = this.messages[i].text; break; }
            }
            if (!userText) return;
            this.messages.splice(botIndex, 1);
            this.userInput = userText;
            this.$nextTick(() => this.handleSend());
        },

        // Drag & Drop
        handleDrop(event) {
            this.isDragOver = false;
            const files = event.dataTransfer.files;
            if (files && files.length > 0) {
                const file = files[0];
                const validTypes = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.md', '.markdown', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'];
                const ext = '.' + file.name.split('.').pop().toLowerCase();
                if (validTypes.includes(ext)) {
                    this.selectedFile = file;
                    this.uploadProgress = '';
                } else {
                    this.showToast('不支持的文件格式，请上传 PDF、Word 或 Excel 文件', 'warning');
                }
            }
        },

        // HITL resolution
        async resolveHitl(action) {
            this.hitlPending = true;
            try {
                const body = { session_id: this.sessionId, action: action };
                if (action === 'modify') body.modified_input = this.hitlModifiedInput;

                const resp = await fetch('/chat/hitl/resume', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                if (!resp.ok) {
                    const err = await resp.json();
                    this.showToast('HITL 操作失败: ' + (err.detail || '未知错误'), 'error');
                    this.hitlPending = false;
                    return;
                }

                this.hitlState = null;
                this.hitlPending = false;
                this.messages.push({
                    type: 'bot',
                    text: '',
                    isThinking: true,
                    _streaming: true,
                    ragTrace: null,
                    ragSteps: [],
                    agentRoutes: []
                });
                const botIdx = this.messages.length - 1;
                this.messages[botIdx].isThinking = false;

                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let idx;
                    while ((idx = buffer.indexOf('\n\n')) !== -1) {
                        const ev = buffer.slice(0, idx);
                        buffer = buffer.slice(idx + 2);
                        if (!ev.startsWith('data: ')) continue;
                        const raw = ev.slice(6);
                        if (raw === '[DONE]') continue;
                        try {
                            const data = JSON.parse(raw);
                            if (data.type === 'content') {
                                this.messages[botIdx].text += data.content;
                            }
                        } catch (e) {}
                    }
                }
                this.messages[botIdx]._streaming = false;
                this.$nextTick(() => this.scrollToBottom());
            } catch (e) {
                this.showToast('HITL 操作异常: ' + e.message, 'error');
                this.hitlPending = false;
            }
        },

        // HITL scenario label
        hitlScenarioLabel(state) {
            const labels = {
                'low_confidence_rag': 'RAG 检索置信度过低',
                'non_select_sql': 'SQL 语句安全审核',
            };
            return labels[state.scenario] || '人工干预';
        },

        // Format timestamp for trace
        formatTime(ts) {
            const d = new Date(ts);
            return d.getHours().toString().padStart(2,'0') + ':' +
                   d.getMinutes().toString().padStart(2,'0') + ':' +
                   d.getSeconds().toString().padStart(2,'0');
        },

        // Agent Label Helper
        agentLabel(agent) {
            const labels = {
                'rag_specialist': { icon: 'fas fa-database', text: '知识库专家', color: '#3b82f6' },
                'web_searcher': { icon: 'fas fa-globe', text: '联网搜索', color: '#10b981' },
                'data_analyst': { icon: 'fas fa-chart-bar', text: '数据分析', color: '#f59e0b' },
                'direct_answer': { icon: 'fas fa-robot', text: '直接回答', color: '#8b5cf6' },
                'local_graph_search': { icon: 'fas fa-project-diagram', text: '图谱检索', color: '#ec4899' },
                'global_graph_search': { icon: 'fas fa-globe', text: '全局图谱', color: '#14b8a6' }
            };
            return labels[agent] || { icon: 'fas fa-cog', text: agent || '未知', color: '#6b7280' };
        },

        // RAG Trace Helper
        totalChunkCount(trace) {
            let count = 0;
            for (const key of ['initial_retrieved_chunks', 'expanded_retrieved_chunks', 'retrieved_chunks']) {
                if (trace[key]) count += trace[key].length;
            }
            return count;
        }
    },
    watch: {
        messages: {
            handler() { this.$nextTick(() => this.scrollToBottom()); },
            deep: true
        }
    }
}).mount('#app');
