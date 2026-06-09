<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>The Jargon Buster</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['Roboto', 'monospace'],
                    },
                    colors: {
                        slate: {
                            850: '#151e2e',
                            950: '#020617',
                        }
                    },
                    animation: {
                        'fade-in': 'fadeIn 0.5s ease-out forwards',
                        'spin-slow': 'spin 3s linear infinite',
                    },
                    keyframes: {
                        fadeIn: {
                            '0%': { opacity: '0', transform: 'translateY(10px)' },
                            '100%': { opacity: '1', transform: 'translateY(0)' },
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #020617;
            color: #f8fafc;
        }
        /* Custom scrollbar */
        textarea::-webkit-scrollbar {
            width: 8px;
        }
        textarea::-webkit-scrollbar-track {
            background: #1e293b; 
            border-radius: 4px;
        }
        textarea::-webkit-scrollbar-thumb {
            background: #475569; 
            border-radius: 4px;
        }
        textarea::-webkit-scrollbar-thumb:hover {
            background: #64748b; 
        }
        .glass-panel {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(51, 65, 85, 0.5);
        }
    </style>
</head>
<body class="antialiased min-h-screen selection:bg-indigo-500 selection:text-white overflow-x-hidden relative">

    <!-- Background Gradient Orbs -->
    <div class="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div class="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-indigo-600/20 rounded-full blur-[120px]"></div>
        <div class="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-cyan-600/10 rounded-full blur-[120px]"></div>
    </div>

    <div class="min-h-screen w-full px-4 py-12 sm:px-6 lg:px-8 max-w-5xl mx-auto">
        
        <!-- Header -->
        <header class="flex flex-col items-center justify-center space-y-4 mb-8 text-center">
            <div class="relative group">
                <div class="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-full blur opacity-25 group-hover:opacity-75 transition duration-1000 group-hover:duration-200"></div>
                <div class="relative p-3 bg-slate-900 ring-1 ring-slate-900/5 rounded-full">
                    <!-- Icon: ShieldCheck -->
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-indigo-400 w-10 h-10"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/></svg>
                </div>
            </div>
            
            <div>
                <h1 class="text-4xl md:text-5xl font-extrabold tracking-tight">
                    <span class="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-cyan-400 to-emerald-400">
                        Jargon Buster
                    </span>
                </h1>
                <p class="mt-2 text-slate-400 text-lg max-w-lg mx-auto">
                    Transform complex jargon into clear, actionable insights.
                </p>
            </div>
        </header>

        <!-- Main Content -->
        <main class="relative z-10 space-y-8">
            
            <!-- API Key Input Section -->
            <div class="w-full max-w-3xl mx-auto">
                <div class="relative">
                    <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-slate-500"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/></svg>
                    </div>
                    <input 
                        type="password" 
                        id="apiKeyInput" 
                        placeholder="Paste your Google Gemini API Key here..." 
                        class="block w-full pl-10 pr-4 py-2 bg-slate-900/30 border border-slate-700/50 rounded-lg text-slate-300 placeholder-slate-600 focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 text-sm transition-all"
                    >
                </div>
            </div>

            <!-- Analysis Form -->
            <form id="analysisForm" class="w-full max-w-3xl mx-auto space-y-6">
                
                <!-- Controls Bar -->
                <div class="flex flex-col sm:flex-row gap-4">
                    <!-- Dropdown -->
                    <div class="relative flex-1">
                        <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <!-- Icon: FileText -->
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-indigo-400"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>
                        </div>
                        <select id="docTypeSelect" class="block w-full pl-10 pr-10 py-3 bg-slate-900/50 border border-slate-700 rounded-xl text-slate-200 focus:ring-2 focus:ring-indigo-500 focus:border-transparent appearance-none transition-all backdrop-blur-sm hover:bg-slate-800/50 cursor-pointer">
                            <option value="Legal Contract">Legal Contract</option>
                            <option value="Medical Report">Medical Report</option>
                            <option value="Technical Specification">Technical Spec</option>
                        </select>
                        <div class="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                            <!-- Icon: ChevronDown -->
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-slate-500"><path d="m6 9 6 6 6-6"/></svg>
                        </div>
                    </div>

                    <!-- Action Button -->
                    <button type="submit" id="analyzeBtn" class="relative px-8 py-3 rounded-xl font-bold text-white transition-all duration-300 flex items-center justify-center gap-2 shadow-lg bg-indigo-600 hover:bg-indigo-500 hover:shadow-[0_0_20px_rgba(99,102,241,0.5)] active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">
                        <span id="btnIcon">
                            <!-- Icon: Play -->
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                        </span>
                        <span id="btnSpinner" class="hidden">
                            <div class="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                        </span>
                        <span id="btnText">Analyze & Simplify</span>
                    </button>
                </div>

                <!-- Text Area -->
                <div class="relative group">
                    <div class="absolute -inset-0.5 bg-gradient-to-r from-indigo-500/20 to-cyan-500/20 rounded-2xl blur opacity-75 group-hover:opacity-100 transition duration-500"></div>
                    <textarea 
                        id="textInput"
                        placeholder="Paste your confusing legal agreement, medical test results, or engineering specifications here..." 
                        class="relative block w-full h-64 p-6 bg-slate-900/80 border border-slate-700/50 rounded-xl text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none transition-all backdrop-blur-md focus:bg-slate-900/90 font-mono text-sm leading-relaxed"
                    ></textarea>
                    <div class="absolute bottom-4 right-4 text-xs text-slate-500 font-mono">
                        <span id="charCount">0</span> characters
                    </div>
                </div>
            </form>

            <!-- Error Message -->
            <div id="errorMessage" class="hidden max-w-3xl mx-auto p-4 rounded-xl bg-rose-950/50 border border-rose-500/30 flex items-center gap-3 text-rose-200 animate-fade-in backdrop-blur-sm">
                <!-- Icon: AlertCircle -->
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="flex-shrink-0"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>
                <p id="errorText"></p>
            </div>

            <!-- Result Display -->
            <div id="resultDisplay" class="hidden w-full max-w-3xl mx-auto mt-12 animate-fade-in space-y-8 pb-12">
                
                <!-- Section A: Summary -->
                <div class="glass-panel relative overflow-hidden rounded-2xl shadow-2xl">
                    <div class="absolute top-0 left-0 w-1 h-full bg-emerald-500"></div>
                    <div class="p-8">
                        <div class="flex items-center gap-3 mb-4">
                            <!-- Icon: CheckCircle2 -->
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-emerald-400"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                            <h2 class="text-2xl font-bold text-slate-100">The Breakdown</h2>
                        </div>
                        <div id="summaryContent" class="prose prose-invert max-w-none text-slate-300 leading-relaxed">
                            <!-- Summary inserted here -->
                        </div>
                    </div>
                </div>

                <!-- Section B: Red Flags -->
                <div class="glass-panel relative overflow-hidden rounded-2xl shadow-2xl">
                    <div class="absolute top-0 left-0 w-1 h-full bg-rose-500"></div>
                    <div class="p-8">
                        <div class="flex items-center gap-3 mb-6">
                            <!-- Icon: AlertTriangle -->
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-rose-400"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
                            <h2 class="text-2xl font-bold text-slate-100">Red Flags & Critical Info</h2>
                        </div>
                        <ul id="redFlagsList" class="space-y-4">
                            <!-- Red flags inserted here -->
                        </ul>
                    </div>
                </div>

                <div class="flex justify-center items-center gap-2 text-slate-600 text-sm">
                    <!-- Icon: FileSearch -->
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h14a2 2 0 0 0 2-2V7.5L14.5 2H6a2 2 0 0 0-2 2v4"/><path d="M14 2v6h6"/><path d="M5 17a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/><path d="m9 18-1.5-1.5"/></svg>
                    <span>Generated by Gemini 2.5 Flash</span>
                </div>
            </div>

        </main>

        <footer class="mt-20 text-center text-slate-600 text-sm">
            <p>© 2025 Jargon Buster. Powered by Google Gemini.</p>
        </footer>
    </div>

    <script>
        // DOM Elements
        const form = document.getElementById('analysisForm');
        const textInput = document.getElementById('textInput');
        const apiKeyInput = document.getElementById('apiKeyInput');
        const docTypeSelect = document.getElementById('docTypeSelect');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const btnIcon = document.getElementById('btnIcon');
        const btnSpinner = document.getElementById('btnSpinner');
        const btnText = document.getElementById('btnText');
        const resultDisplay = document.getElementById('resultDisplay');
        const summaryContent = document.getElementById('summaryContent');
        const redFlagsList = document.getElementById('redFlagsList');
        const errorMessage = document.getElementById('errorMessage');
        const errorText = document.getElementById('errorText');
        const charCount = document.getElementById('charCount');

        // Character Count Updater
        textInput.addEventListener('input', () => {
            charCount.textContent = textInput.value.length;
            validateForm();
        });

        // Form Validation
        function validateForm() {
            const hasText = textInput.value.trim().length > 0;
            if (hasText) {
                analyzeBtn.classList.remove('bg-slate-700', 'text-slate-400', 'cursor-not-allowed');
                analyzeBtn.classList.add('bg-indigo-600', 'hover:bg-indigo-500');
                analyzeBtn.disabled = false;
            } else {
                analyzeBtn.classList.add('bg-slate-700', 'text-slate-400', 'cursor-not-allowed');
                analyzeBtn.classList.remove('bg-indigo-600', 'hover:bg-indigo-500');
                analyzeBtn.disabled = true;
            }
        }

        // Helper: Toggle Loading State
        function setLoading(isLoading) {
            if (isLoading) {
                analyzeBtn.disabled = true;
                btnIcon.classList.add('hidden');
                btnSpinner.classList.remove('hidden');
                btnText.textContent = 'Analyzing...';
                
                // Hide previous results/errors
                resultDisplay.classList.add('hidden');
                errorMessage.classList.add('hidden');
            } else {
                analyzeBtn.disabled = false;
                btnIcon.classList.remove('hidden');
                btnSpinner.classList.add('hidden');
                btnText.textContent = 'Analyze & Simplify';
            }
        }

        // Helper: Show Error
        function showError(message) {
            errorText.textContent = message;
            errorMessage.classList.remove('hidden');
            setLoading(false);
        }

        // Handle Form Submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const text = textInput.value.trim();
            const apiKey = apiKeyInput.value.trim();
            const docType = docTypeSelect.value;

            if (!text) return;

            if (!apiKey) {
                showError("Please enter your Google Gemini API Key.");
                return;
            }

            setLoading(true);

            // Construct Prompt
            const prompt = `
                You are an expert consultant specializing in ${docType}.
                The user has provided a ${docType} text.
                
                Your goal is to:
                1. Summarize the text in simple, plain English that a layperson can understand.
                2. Identify 3-5 critical "Red Flags", risks, or important clauses/details they must know about.
                
                Here is the text to analyze:
                """
                ${text}
                """
            `;

            // Call Gemini API (REST)
            try {
                const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        contents: [{
                            parts: [{ text: prompt }]
                        }],
                        generationConfig: {
                            responseMimeType: "application/json",
                            responseSchema: {
                                type: "OBJECT",
                                properties: {
                                    summary: { type: "STRING" },
                                    redFlags: { 
                                        type: "ARRAY",
                                        items: { type: "STRING" }
                                    }
                                },
                                required: ["summary", "redFlags"]
                            }
                        }
                    })
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error?.message || `API Error: ${response.status}`);
                }

                const data = await response.json();
                
                // Parse Content
                const candidate = data.candidates?.[0];
                if (!candidate) throw new Error("No response candidates returned.");

                const resultText = candidate.content.parts[0].text;
                const resultJson = JSON.parse(resultText);

                // Update UI
                summaryContent.textContent = resultJson.summary;
                
                redFlagsList.innerHTML = '';
                resultJson.redFlags.forEach(flag => {
                    const li = document.createElement('li');
                    li.className = "flex gap-4 p-4 rounded-xl bg-rose-500/5 border border-rose-500/10 hover:bg-rose-500/10 transition-colors";
                    li.innerHTML = `
                        <span class="flex-shrink-0 text-2xl select-none">⚠️</span>
                        <span class="text-slate-200">${flag}</span>
                    `;
                    redFlagsList.appendChild(li);
                });

                // Show Results
                setLoading(false);
                resultDisplay.classList.remove('hidden');
                
                // Smooth scroll to results
                resultDisplay.scrollIntoView({ behavior: 'smooth' });

            } catch (err) {
                console.error(err);
                showError(err.message || "Failed to analyze text. Please check your API key and try again.");
            }
        });

        // Initialize state
        validateForm();
    </script>
</body>
</html>

