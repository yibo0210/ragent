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
            showTrace: false,
            // Auth
            authToken: localStorage.getItem('ragent-token') || '',
            authUsername: localStorage.getItem('ragent-username') || '',
            authMode: 'login',
            authLoading: false,
            authError: '',
            authTenantName: '',
            // Workflow
            wfGoal: '',
            wfPlanning: false,
            wfPlanData: null,
            wfPlanError: '',
            wfExecuting: false,
            wfExecutionId: '',
            wfStatus: '',
            wfProgress: 0,
            wfStatusText: '',
            wfErrorMessage: '',
            wfCurrentStep: '',
            wfCompletedSteps: [],
            wfStepResults: {},
            wfStepErrors: {},
            wfArtifacts: [],
            wfPollTimer: null,
            wfArtifactModal: null,
            wfHistory: [],
            // Research state
            researchGoal: '',
            researchRunning: false,
            researchState: null,
            researchEvidence: [],
            researchReportContent: '',
            reportFormat: 'markdown',
            researchHistory: [],
            researchTimer: null,
            researchStartTime: null,
            researchElapsed: '0:00',
            // Translations for research enums
            sourceLabels: {
                web_search: '网络搜索', graph_rag: '知识图谱', data_analyst: '数据分析',
                internal_kb: '内部知识库', mcp: 'MCP', user_upload: '用户上传',
            },
            confidenceLabels: { high: '高', medium: '中', low: '低' },
            statusLabels: { pending: '等待中', running: '运行中', completed: '已完成', failed: '失败', cancelled: '已取消' },
            // History filter
            historyFilter: 'all',
            // v21: Dynamic Research
            researchHypotheses: [],
            researchConflicts: [],
            evidenceGraphData: { nodes: [], edges: [] },
            _evidenceChartInstance: null,
        };
    },
    mounted() {
        document.documentElement.setAttribute('data-theme', this.theme);
        this.configureMarked();
        this.setupCodeCopyListener();
        if (this.authToken) { this.loadSessions(); this._loadAllHistory(); }
    },

    computed: {
        // Merge sessions, workflows, research into unified sidebar history
        combinedHistory() {
            const items = [];
            (this.sessions || []).forEach(s => {
                items.push({
                    type: 'chat', id: s.session_id,
                    title: s.first_message || s.session_id || '新对话',
                    timestamp: s.updated_at || s.created_at || '',
                    action: () => this.loadSession(s.session_id),
                    del: () => this.deleteSession(s.session_id),
                });
            });
            (this.wfHistory || []).forEach(w => {
                items.push({
                    type: 'workflow', id: w.execution_id,
                    title: w.goal || '工作流',
                    timestamp: w.created_at || '',
                    action: () => { this.activeNav = 'workflow'; this.wfLoadHistory(); },
                    del: () => this._deleteWorkflow(w.execution_id),
                });
            });
            (this.researchHistory || []).forEach(r => {
                items.push({
                    type: 'research', id: r.execution_id,
                    title: r.goal || '研究任务',
                    timestamp: r.created_at || '',
                    action: () => { this.activeNav = 'research'; this.loadResearch(r.execution_id); },
                    del: () => this._deleteResearch(r.execution_id),
                });
            });
            items.sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''));
            return items;
        },
    },

    methods: {
        // Load workflow + research history for sidebar
        async _loadAllHistory() {
            try {
                const [wfR, rsR] = await Promise.all([
                    this._authFetch('/workflows'),
                    this._authFetch('/research/list'),
                ]);
                const wfData = await wfR.json();
                const rsData = await rsR.json();
                this.wfHistory = wfData.executions || [];
                this.researchHistory = rsData.executions || [];
            } catch (e) { /* ignore */ }
        },
        // Delete helpers for sidebar history
        async _deleteWorkflow(id) {
            if (!confirm('确定删除此工作流记录？')) return;
            try {
                await this._authFetch('/workflows/' + id, { method: 'DELETE' });
                this.showToast('已删除');
                this._loadAllHistory();
            } catch (e) { /* ignore */ }
        },
        async _deleteResearch(id) {
            if (!confirm('确定删除此研究记录？')) return;
            try {
                await this._authFetch('/research/' + id, { method: 'DELETE' });
                this.showToast('已删除');
                this._loadAllHistory();
            } catch (e) { /* ignore */ }
        },
        // Translate research source/confidence/status
        tSource(val) { return this.sourceLabels[val] || val; },
        tConfidence(val) { return this.confidenceLabels[val] || val; },
        tStatus(val) { return this.statusLabels[val] || val; },
        // Auth helper: common fetch with Authorization header
        _authFetch(url, options = {}) {
            const headers = options.headers || {};
            if (this.authToken) {
                headers['Authorization'] = 'Bearer ' + this.authToken;
            }
            return fetch(url, { ...options, headers });
        },
        configureMarked() {
            const renderer = new marked.Renderer();
            renderer.code = function(code, language) {
                const lang = (language || '').split(' ')[0];
                // v9: Echarts 图表渲染
                if (lang === 'echarts') {
                    const chartId = 'echarts-' + Math.random().toString(36).substr(2, 9);
                    // 延迟渲染，等 DOM 挂载后初始化
                    setTimeout(() => {
                        try {
                            const el = document.getElementById(chartId);
                            if (el && typeof echarts !== 'undefined') {
                                const chart = echarts.init(el);
                                const config = JSON.parse(code);
                                chart.setOption(config);
                                window.addEventListener('resize', () => chart.resize());
                            }
                        } catch (e) { console.warn('Echarts render error:', e); }
                    }, 100);
                    return `<div id="${chartId}" class="echarts-chart" style="width:100%;height:400px;margin:16px 0;border-radius:8px;background:var(--bg-card)"></div>`;
                }
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
                const response = await this._authFetch('/chat/stream', {
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
                                const agentNames = {
                                    rag_specialist: '知识库专家',
                                    local_graph_search: '局部图谱搜索',
                                    global_graph_search: '全局图谱搜索',
                                    web_searcher: '联网搜索',
                                    data_analyst: '数据分析师',
                                    direct_answer: '直接回答',
                                    multimodal_specialist: '多模态专家',
                                    planner: '任务规划',
                                    critique: '事实核查',
                                };
                                const name = agentNames[data.agent] || data.agent;
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: 'supervisor',
                                    message: '路由 → ' + name + (data.reason ? '（' + data.reason + '）' : '')
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
                                const labels = {
                                    supervisor: '分析意图并路由任务',
                                    rag_specialist: '私有知识库混合检索',
                                    local_graph_search: '向量+图谱多跳推理',
                                    global_graph_search: '社区摘要全局匹配',
                                    web_searcher: 'Tavily实时联网搜索',
                                    data_analyst: 'Text-to-SQL数据分析',
                                    direct_answer: '轻量直接回答',
                                    multimodal_specialist: '多模态图表解读',
                                    planner: '分解复杂查询为多步计划',
                                    critique: '交叉验证回答与检索依据',
                                };
                                const desc = labels[data.agent] || '开始执行任务';
                                this.traceSteps.push({ timestamp: Date.now(), agent: data.agent, message: desc });
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
                            } else if (data.type === 'cache_hit') {
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: 'system',
                                    message: '语义缓存命中 — 相似度' + (data.similarity || 0) + '，直接返回缓存答案（0 Token，<200ms）'
                                });
                            } else if (data.type === 'plan_generated') {
                                // v8: 显示 Planner 生成的查询计划
                                const stepDescs = (data.steps || []).map(s => s.query || s.agent).join(' → ');
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: 'planner',
                                    message: '查询计划: ' + (stepDescs || '简单查询，无需拆解')
                                });
                            } else if (data.type === 'critique_feedback') {
                                // v8: 显示 Critique 事实核查结果
                                const icon = data.is_valid ? '通过' : '驳回';
                                const msg = data.is_valid ? '事实核查通过' : '事实核查驳回: ' + (data.feedback || '');
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: 'critique',
                                    message: msg
                                });
                            } else if (data.type === 'self_correction') {
                                // v8: 自纠错循环
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: 'critique',
                                    message: '自纠错 — ' + (data.message || '正在补充信息...')
                                });
                            } else if (data.type === 'mcp_tool_call') {
                                // v9: MCP 工具调用
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: data.agent || 'data_analyst',
                                    message: '调用外部系统: ' + (data.server_name || '') + '/' + (data.tool_name || '')
                                });
                            } else if (data.type === 'mcp_tool_result') {
                                // v9: MCP 工具结果
                                const summary = (data.result_summary || '').substring(0, 100);
                                this.traceSteps.push({
                                    timestamp: Date.now(),
                                    agent: data.agent || 'data_analyst',
                                    message: '外部系统返回: ' + summary + (data.is_error ? ' (错误)' : '')
                                });
                            } else if (data.type === 'hitl_interrupt') {
                                this.hitlState = data.data;
                                this.hitlAction = null;
                                this.hitlModifiedInput = '';
                                this.isLoading = false;
                                this.traceSteps.push({ timestamp: Date.now(), agent: 'system', message: 'HITL 中断 — 工作流挂起，等待人工决策' });
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
                } else if (error.message && error.message.includes('401')) {
                    this.messages[botIdx].text = '登录已过期，请重新登录';
                    this.handleLogout();
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
                const res = await this._authFetch('/sessions');
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
                const res = await this._authFetch('/sessions');
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
                const res = await this._authFetch(`/sessions/${sessionId}`);
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
                const res = await this._authFetch(`/sessions/${sessionId}`, { method: 'DELETE' });
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
                const res = await this._authFetch('/documents');
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
                const response = await this._authFetch('/documents/upload', { method: 'POST', body: fd });
                if (response.status === 401) {
                    this.showToast('登录已过期，请重新登录', 'error', 3000);
                    this.handleLogout();
                    return;
                }
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
                const res = await this._authFetch(`/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
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

        // Auth
        async handleAuth() {
            this.authLoading = true;
            this.authError = '';
            const endpoint = this.authMode === 'login' ? '/auth/token' : '/auth/register';
            const isLogin = this.authMode === 'login';
            try {
                let body;
                if (isLogin) {
                    body = new URLSearchParams();
                    body.append('username', this.authUsername);
                    body.append('password', this.authPassword);
                } else {
                    body = JSON.stringify({
                        username: this.authUsername,
                        password: this.authPassword,
                        tenant_name: this.authTenantName || this.authUsername + '_org',
                        role: 'admin',
                    });
                }
                const headers = { 'Content-Type': isLogin ? 'application/x-www-form-urlencoded' : 'application/json' };
                const resp = await fetch(endpoint, { method: 'POST', headers, body: body.toString() });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || 'Request failed');
                if (isLogin) {
                    this.authToken = data.access_token;
                    this.authUsername = this.authUsername;
                    localStorage.setItem('ragent-token', data.access_token);
                    localStorage.setItem('ragent-username', this.authUsername);
                    this.loadSessions();
                    this.showToast('登录成功', 'success');
                } else {
                    this.authUsername = data.username || this.authUsername;
                    this.showToast('注册成功，请登录', 'success');
                    this.authMode = 'login';
                }
            } catch (e) {
                this.authError = e.message;
            } finally {
                this.authLoading = false;
            }
        },
        handleLogout() {
            this.authToken = '';
            this.authUsername = '';
            localStorage.removeItem('ragent-token');
            localStorage.removeItem('ragent-username');
            this.messages = [];
            this.sessions = [];
            this.showToast('已退出登录', 'info');
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

                const resp = await this._authFetch('/chat/hitl/resume', {
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
                'global_graph_search': { icon: 'fas fa-globe', text: '全局图谱', color: '#14b8a6' },
                'planner': { icon: 'fas fa-sitemap', text: '任务规划', color: '#6366f1' },
                'critique': { icon: 'fas fa-check-double', text: '事实核查', color: '#f97316' }
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
        },

        // === Workflow Methods ===

        async wfPlan() {
            if (this.wfPlanning || !this.wfGoal.trim()) return;
            this.wfPlanning = true;
            this.wfPlanError = '';
            this.wfPlanData = null;
            try {
                const res = await this._authFetch('/workflows/plan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ goal: this.wfGoal.trim() }),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                const data = await res.json();
                this.wfPlanData = data.plan;
                this.wfDefinitionId = data.definition_id;
                this.showToast('计划生成成功', 'success');
            } catch (e) {
                this.wfPlanError = '计划生成失败: ' + e.message;
                this.showToast(this.wfPlanError, 'error');
            } finally {
                this.wfPlanning = false;
            }
        },

        async wfExecute() {
            if (this.wfExecuting || !this.wfDefinitionId) return;
            this.wfExecuting = true;
            this.wfStatus = 'running';
            this.wfStatusText = '正在启动...';
            this.wfProgress = 0;
            this.wfCurrentStep = '';
            this.wfCompletedSteps = [];
            this.wfStepResults = {};
            this.wfStepErrors = {};
            this.wfErrorMessage = '';
            this.wfArtifacts = [];
            try {
                const res = await this._authFetch('/workflows/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ definition_id: this.wfDefinitionId }),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                const data = await res.json();
                this.wfExecutionId = data.execution_id;
                this.showToast('工作流已启动', 'info');
                this.wfPollStatus();
            } catch (e) {
                this.wfExecuting = false;
                this.wfStatus = 'failed';
                this.wfStatusText = '启动失败';
                this.wfErrorMessage = e.message;
                this.showToast('执行失败: ' + e.message, 'error');
            }
        },

        wfPollStatus() {
            if (this.wfPollTimer) clearInterval(this.wfPollTimer);
            this.wfPollTimer = setInterval(async () => {
                try {
                    const res = await this._authFetch(`/workflows/${this.wfExecutionId}/status`);
                    if (!res.ok) return;
                    const data = await res.json();
                    this.wfStatus = data.status;
                    this.wfProgress = data.progress;
                    this.wfCurrentStep = data.current_step_id;
                    this.wfErrorMessage = data.error_message || '';
                    if (data.step_results) {
                        for (const [sid, r] of Object.entries(data.step_results)) {
                            if (r.success) {
                                if (!this.wfCompletedSteps.includes(sid)) this.wfCompletedSteps.push(sid);
                                this.wfStepResults[sid] = r.data?.response || JSON.stringify(r.data);
                            } else {
                                this.wfStepErrors[sid] = r.error;
                            }
                        }
                    }

                    if (data.status === 'completed') {
                        this.wfStatusText = '执行完成';
                        this.wfExecuting = false;
                        clearInterval(this.wfPollTimer);
                        this.wfPollTimer = null;
                        this.showToast('工作流执行完成', 'success');
                        this.wfLoadArtifacts();
                    } else if (data.status === 'failed') {
                        this.wfStatusText = '执行失败';
                        this.wfExecuting = false;
                        clearInterval(this.wfPollTimer);
                        this.wfPollTimer = null;
                        this.showToast('工作流执行失败', 'error');
                    } else if (data.status === 'cancelled') {
                        this.wfStatusText = '已取消';
                        this.wfExecuting = false;
                        clearInterval(this.wfPollTimer);
                        this.wfPollTimer = null;
                    } else if (data.status === 'running') {
                        const done = this.wfCompletedSteps.length;
                        const total = this.wfPlanData?.steps?.length || 1;
                        this.wfStatusText = `执行中 (${done}/${total})`;
                    }
                } catch (e) {
                    // polling error, continue
                }
            }, 1500);
        },

        async wfLoadArtifacts() {
            try {
                const res = await this._authFetch(`/workflows/${this.wfExecutionId}/artifacts`);
                if (res.ok) {
                    const data = await res.json();
                    this.wfArtifacts = data.artifacts || [];
                }
            } catch (e) { /* ignore */ }
        },

        wfReset() {
            if (this.wfPollTimer) { clearInterval(this.wfPollTimer); this.wfPollTimer = null; }
            this.wfGoal = '';
            this.wfPlanning = false;
            this.wfPlanData = null;
            this.wfPlanError = '';
            this.wfExecuting = false;
            this.wfExecutionId = '';
            this.wfDefinitionId = null;
            this.wfStatus = '';
            this.wfProgress = 0;
            this.wfStatusText = '';
            this.wfErrorMessage = '';
            this.wfCurrentStep = '';
            this.wfCompletedSteps = [];
            this.wfStepResults = {};
            this.wfStepErrors = {};
            this.wfArtifacts = [];
            this.wfArtifactModal = null;
        },

        artifactIcon(type) {
            const map = { report: 'fas fa-file-lines', excel: 'fas fa-file-excel', csv: 'fas fa-file-csv', chart: 'fas fa-chart-bar', dashboard: 'fas fa-grip', pdf: 'fas fa-file-pdf' };
            return map[type] || 'fas fa-file';
        },

        showArtifactContent(art) {
            this.wfArtifactModal = art;
        },

        renderMarkdown(text) {
            if (!text) return '';
            try {
                return marked.parse(text);
            } catch (e) {
                return this.escapeHtml(text);
            }
        },

        truncate(text, maxLen) {
            if (!text) return '';
            text = String(text);
            return text.length > maxLen ? text.substring(0, maxLen) + '...' : text;
        },

        async wfLoadHistory() {
            try {
                const res = await this._authFetch('/workflows');
                if (res.ok) {
                    const data = await res.json();
                    this.wfHistory = data.executions || [];
                }
            } catch (e) { /* ignore */ }
        },

        wfViewHistory(h) {
            this.wfReset();
            this.wfExecutionId = h.execution_id;
            this.wfStatus = h.status;
            this.wfProgress = h.progress;
            this.wfExecuting = h.status === 'running';
            if (h.status === 'completed') {
                this.wfStatusText = '执行完成';
                this.wfLoadArtifacts();
            } else if (h.status === 'running') {
                this.wfStatusText = '执行中...';
                this.wfPollStatus();
            } else if (h.status === 'failed') {
                this.wfStatusText = '执行失败';
                this.wfErrorMessage = h.error_message || '';
            }
            this.$nextTick(() => {
                document.querySelector('.wf-goal-card')?.scrollIntoView({ behavior: 'smooth' });
            });
        },

        // === Research Methods ===

        async startResearch() {
            if (!this.researchGoal.trim() || this.researchRunning) return;
            this.researchRunning = true;
            this.researchState = { status: 'running', progress: 0, review_count: 0 };
            this.researchStartTime = Date.now();
            this.researchElapsed = '0:00';
            this.researchTimer = setInterval(() => {
                const elapsed = Math.floor((Date.now() - this.researchStartTime) / 1000);
                this.researchElapsed = Math.floor(elapsed / 60) + ':' + String(elapsed % 60).padStart(2, '0');
            }, 1000);

            try {
                const resp = await this._authFetch('/research/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ goal: this.researchGoal, session_id: this.currentSessionId }),
                });
                const data = await resp.json();
                this._pollResearchStatus(data.execution_id);
            } catch (e) {
                this.researchRunning = false;
                clearInterval(this.researchTimer);
            }
        },

        async _pollResearchStatus(executionId) {
            const poll = async () => {
                try {
                    const resp = await this._authFetch('/research/' + executionId);
                    const data = await resp.json();
                    this.researchState = data;

                    if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
                        this.researchRunning = false;
                        clearInterval(this.researchTimer);
                        if (data.status === 'completed') {
                            await this._loadResearchEvidence(executionId);
                            await this._loadResearchReport(executionId);
                            // v21: load hypothesis and conflict data
                            this.researchHypotheses = data.hypotheses || [];
                            this.researchConflicts = data.conflicts || [];
                            this.evidenceGraphData = data.evidence_graph_data || { nodes: [], edges: [] };
                            if (this.evidenceGraphData.nodes && this.evidenceGraphData.nodes.length) {
                                this._renderEvidenceGraph();
                            }
                        }
                        await this._loadResearchHistory();
                        return;
                    }
                    setTimeout(poll, 3000);
                } catch (e) {
                    this.researchRunning = false;
                    clearInterval(this.researchTimer);
                }
            };
            poll();
        },

        async _loadResearchEvidence(executionId) {
            try {
                const resp = await this._authFetch('/research/' + executionId + '/evidence');
                const data = await resp.json();
                this.researchEvidence = data.evidence || [];
            } catch (e) { /* ignore */ }
        },

        async _loadResearchReport(executionId) {
            try {
                const resp = await this._authFetch('/research/' + executionId + '/report?format=markdown');
                const data = await resp.json();
                this.researchReportContent = data.content || '';
            } catch (e) { /* ignore */ }
        },

        async loadResearch(executionId) {
            // Load completed research: only show evidence + report, skip active flow
            this.researchRunning = false;
            this.researchHypotheses = [];
            this.researchConflicts = [];
            this.evidenceGraphData = { nodes: [], edges: [] };
            try {
                const resp = await this._authFetch('/research/' + executionId);
                const data = await resp.json();
                this.researchState = data;
                if (data.status === 'completed') {
                    await this._loadResearchEvidence(executionId);
                    await this._loadResearchReport(executionId);
                }
            } catch (e) { /* ignore */ }
        },

        async _loadResearchHistory() {
            try {
                const resp = await this._authFetch('/research/list');
                const data = await resp.json();
                this.researchHistory = data.executions || [];
            } catch (e) { /* ignore */ }
        },

        // v21: Evidence graph visualization
        _renderEvidenceGraph() {
            if (!this.evidenceGraphData || !this.evidenceGraphData.nodes || !this.evidenceGraphData.nodes.length) return;
            this.$nextTick(() => {
                const container = this.$refs.evidenceGraphChart;
                if (!container) return;
                if (this._evidenceChartInstance) this._evidenceChartInstance.dispose();
                const chart = echarts.init(container);
                this._evidenceChartInstance = chart;
                const nodes = this.evidenceGraphData.nodes.map(n => ({
                    id: n.id,
                    name: n.label,
                    symbolSize: 20 + (n.confidence || 0.5) * 30,
                    category: n.hypothesis_id ? (n.hypothesis_id.charCodeAt(1) || 49) - 49 : 0,
                    itemStyle: { color: (n.confidence || 0.5) > 0.7 ? '#22c55e' : (n.confidence || 0.5) > 0.4 ? '#f59e0b' : '#ef4444' },
                }));
                const edges = (this.evidenceGraphData.edges || []).map(e => ({
                    source: e.from, target: e.to,
                    lineStyle: { color: e.type === 'REFUTES' ? '#ef4444' : '#3b82f6', width: e.type === 'REFUTES' ? 2 : 1 },
                }));
                chart.setOption({
                    series: [{
                        type: 'graph', layout: 'force', data: nodes, links: edges,
                        roam: true, draggable: true,
                        force: { repulsion: 300, edgeLength: 150 },
                        label: { show: true, fontSize: 10, formatter: p => (p.name || '').substring(0, 20) },
                    }],
                });
                window.addEventListener('resize', () => chart.resize());
            });
        },

    },
    watch: {
        messages: {
            handler() { this.$nextTick(() => this.scrollToBottom()); },
            deep: true
        },
        activeNav(val) {
            if (val === 'workflow') this.wfLoadHistory();
            if (val === 'research') this._loadResearchHistory();
        }
    }
}).mount('#app');
