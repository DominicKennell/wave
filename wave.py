import numpy as np
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pickle
import os
from scipy.fft import fftn, ifftn
from scipy.sparse import csr_matrix, lil_matrix
import threading
import time

# ========================
# ENHANCED Bootstrap Wave Language Model with DYNAMIC VOCABULARY
# ========================

class EnhancedWaveLanguageModel:
    """
    Wave-based language model with:
    - GENESIS MODE (self-starting)
    - DYNAMIC VOCABULARY (learns new words)
    - FFT-based wave propagation
    - Spontaneous thought generation
    """
    
    def __init__(self, grid_size=16, num_channels=8, bootstrap_mode=False):
        self.X, self.Y, self.Z = grid_size, grid_size, grid_size
        self.C = num_channels
        
        # Bootstrap mode flag
        self.bootstrap_mode = bootstrap_mode
        
        # Initialize field
        if bootstrap_mode:
            # Quantum foam - random fluctuations
            self.field = (np.random.randn(self.X, self.Y, self.Z, self.C)
                         .astype(np.complex64) * 0.01)
        else:
            # Empty space
            self.field = np.zeros((self.X, self.Y, self.Z, self.C), dtype=np.complex64)
        
        # Wave parameters
        self.dt = 0.15
        self.wave_speed = 0.85
        self.damping = 0.96
        self.context_decay = 0.88
        
        # DYNAMIC vocabulary - starts with core, can expand
        self.vocab = self._build_core_vocab()
        self.V = len(self.vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Word encodings (expandable)
        self.word_positions = (np.random.rand(self.V, 3) * 0.8 * grid_size).astype(np.float32)
        self.word_frequencies = (0.3 + np.random.rand(self.V, self.C) * 0.5).astype(np.float32)
        
        # Sparse association matrix (expandable)
        self.word_associations = lil_matrix((self.V, self.V), dtype=np.float32)
        
        # Decoder (expandable)
        feature_dim = self.C * 2
        self.W_decode = np.random.randn(feature_dim, self.V).astype(np.float32) * 0.03
        
        # Learning rates
        self.hebbian_lr = 0.003
        self.decoder_lr = 0.005
        
        # Memory
        self.recent_words = []
        self.recent_generated = []
        self.word_usage_count = np.zeros(self.V, dtype=np.uint32)
        
        # Statistics
        self.total_words_processed = 0
        self.conversation_count = 0
        self.learned_words = []  # NEW: track dynamically learned words
        
        # Bootstrap tracking
        self.genesis_step = 0
        self.spontaneous_thoughts = []
    
    def _build_core_vocab(self):
        """Build core vocabulary - can expand dynamically"""
        return [
            "hello", "hi", "hey", "goodbye", "bye", "thanks", "please", "sorry",
            "welcome", "yes", "no", "ok", "okay", "sure",
            "i", "you", "we", "they", "he", "she", "it", "me", "my", "your",
            "am", "is", "are", "was", "were", "be", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "should",
            "want", "need", "like", "love", "know", "think", "feel", "see",
            "say", "tell", "ask", "help", "try", "go", "come", "get", "make",
            "take", "give", "find", "use",
            "good", "bad", "great", "nice", "fine", "well", "happy", "sad",
            "big", "small", "new", "old",
            "day", "time", "thing", "way", "home", "work", "people", "friend",
            "name", "question", "answer",
            "what", "who", "where", "when", "why", "how",
            "the", "a", "and", "or", "but", "if", "to", "of", "in", "on", "at",
            "not", "very", "just", "so", "now", "here", "there",
        ]
    
    def learn_new_word(self, word):
        """
        DYNAMIC VOCABULARY: Learn a new word on-the-fly
        
        Returns: index of word (new or existing)
        """
        word = word.lower()
        
        # Already known?
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        
        # Add to vocabulary
        new_idx = self.V
        self.vocab.append(word)
        self.word_to_idx[word] = new_idx
        self.V += 1
        
        # Assign spatial position (random for now, could be smarter)
        new_position = np.random.rand(3).astype(np.float32) * 0.8 * self.X
        self.word_positions = np.vstack([self.word_positions, new_position])
        
        # Assign frequencies
        new_freqs = (0.3 + np.random.rand(self.C) * 0.5).astype(np.float32)
        self.word_frequencies = np.vstack([self.word_frequencies, new_freqs])
        
        # Expand association matrix
        self._expand_association_matrix()
        
        # Expand decoder
        new_decoder_col = np.random.randn(self.W_decode.shape[0], 1).astype(np.float32) * 0.03
        self.W_decode = np.hstack([self.W_decode, new_decoder_col])
        
        # Expand usage count
        self.word_usage_count = np.append(self.word_usage_count, 0)
        
        # Track
        self.learned_words.append(word)
        
        print(f"✨ Learned new word: '{word}' (vocab now {self.V} words)")
        
        return new_idx
    
    def _expand_association_matrix(self):
        """Expand sparse matrix for new word"""
        old_size = self.word_associations.shape[0]
        new_size = old_size + 1
        
        new_matrix = lil_matrix((new_size, new_size), dtype=np.float32)
        new_matrix[:old_size, :old_size] = self.word_associations
        self.word_associations = new_matrix
    
    def encode_word(self, word):
        """
        Encode word as complex wave
        AUTO-LEARNS if word is unknown
        """
        word = word.lower()
        
        # Learn if unknown
        if word not in self.word_to_idx:
            idx = self.learn_new_word(word)
        else:
            idx = self.word_to_idx[word]
        
        cx, cy, cz = self.word_positions[idx]
        frequencies = self.word_frequencies[idx]
        
        amplitude = 0.3
        
        x = np.arange(self.X, dtype=np.float32)
        y = np.arange(self.Y, dtype=np.float32)
        z = np.arange(self.Z, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        sigma = 2.0
        dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
        envelope = amplitude * np.exp(-dist_sq / (2 * sigma**2))
        
        wave_packet = np.zeros((self.X, self.Y, self.Z, self.C), dtype=np.complex64)
        phase_base = 2 * np.pi * idx / max(self.V, 1)
        
        for c in range(self.C):
            phase = phase_base + c * np.pi / self.C
            wave_packet[:, :, :, c] = envelope * np.exp(1j * frequencies[c] * phase)
        
        return wave_packet
    
    def propagate_waves_fft(self, steps=2):
        """FFT-based wave propagation"""
        for _ in range(steps):
            for c in range(self.C):
                channel = self.field[:, :, :, c]
                channel_fft = fftn(channel)
                
                kx = np.fft.fftfreq(self.X, d=1.0) * 2 * np.pi
                ky = np.fft.fftfreq(self.Y, d=1.0) * 2 * np.pi
                kz = np.fft.fftfreq(self.Z, d=1.0) * 2 * np.pi
                KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
                
                k_squared = KX**2 + KY**2 + KZ**2
                laplacian_fft = -k_squared * channel_fft
                laplacian = ifftn(laplacian_fft)
                
                self.field[:, :, :, c] = (
                    channel + self.dt**2 * self.wave_speed**2 * laplacian
                ) * self.damping
            
            self.field = np.tanh(np.abs(self.field)) * np.exp(1j * np.angle(self.field))
    
    def spontaneous_think(self):
        """
        GENESIS MODE: Generate thought without input
        """
        self.genesis_step += 1
        
        # Evolve field
        self.propagate_waves_fft(steps=5)
        self.field *= 0.95
        
        # Quantum noise injection
        noise = np.random.randn(self.X, self.Y, self.Z, self.C).astype(np.complex64) * 0.01
        self.field += noise
        
        # Check amplitude
        max_amplitude = np.max(np.abs(self.field))
        
        # Multiple trigger conditions
        should_speak = (
            max_amplitude > 0.008 or
            self.genesis_step % 15 == 0 or
            np.random.random() < 0.1
        )
        
        if should_speak:
            word = self.predict_next_word(temperature=2.5)
            
            # Self-reinforce
            word_wave = self.encode_word(word)
            self.field += word_wave * 1.0
            
            self.spontaneous_thoughts.append((self.genesis_step, word, max_amplitude))
            return word
        
        return None
    
    def add_word_to_field(self, word, learn=True):
        """Add word to field - auto-learns new words"""
        word_wave = self.encode_word(word)
        self.field *= self.context_decay
        self.field += word_wave
        self.propagate_waves_fft(steps=2)
        
        if learn:
            word_lower = word.lower()
            if word_lower in self.word_to_idx:
                idx = self.word_to_idx[word_lower]
                self.total_words_processed += 1
                if self.word_usage_count[idx] < 4294967295:
                    self.word_usage_count[idx] += 1
                
                self._hebbian_learning(idx)
                
                self.recent_words.append(idx)
                if len(self.recent_words) > 8:
                    self.recent_words.pop(0)
    
    def _hebbian_learning(self, current_word_idx):
        """Sparse Hebbian learning"""
        for past_idx in self.recent_words[-3:]:
            current_val = self.word_associations[current_word_idx, past_idx]
            new_val = current_val + self.hebbian_lr
            
            if abs(new_val) > 0.01:
                self.word_associations[current_word_idx, past_idx] = np.tanh(new_val)
                self.word_associations[past_idx, current_word_idx] = np.tanh(new_val)
    
    def _extract_field_features(self):
        """Extract features from field"""
        features = []
        for c in range(self.C):
            channel = self.field[:, :, :, c]
            amplitude = np.abs(channel)
            features.append(np.mean(amplitude))
            features.append(np.max(amplitude))
        return np.array(features, dtype=np.float32)
    
    def predict_next_word(self, temperature=1.2):
        """Predict next word"""
        features = self._extract_field_features()
        logits = features @ self.W_decode
        
        if len(self.recent_words) > 0:
            for past_idx in self.recent_words[-2:]:
                associations = self.word_associations[past_idx, :].toarray().flatten()
                # Pad if needed (vocab grew)
                if len(associations) < self.V:
                    associations = np.pad(associations, (0, self.V - len(associations)))
                logits += associations * 0.1
        
        if len(self.recent_generated) > 0:
            block_strength = -500 if self.bootstrap_mode else -1000
            for past_idx in self.recent_generated[-4:]:
                if past_idx < self.V:
                    logits[past_idx] = block_strength
        
        if len(self.recent_words) > 0:
            for past_idx in self.recent_words[-3:]:
                if past_idx < self.V:
                    logits[past_idx] -= 2.0
        
        logits += np.random.randn(self.V) * 0.5
        logits = logits / temperature
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        
        top_k = 15
        top_indices = np.argpartition(-probs, min(top_k, len(probs))-1)[:top_k]
        top_probs = probs[top_indices]
        top_probs = top_probs / (np.sum(top_probs) + 1e-10)
        
        chosen_idx = np.random.choice(top_indices, p=top_probs)
        return self.vocab[chosen_idx]
    
    def organic_decoder_update(self):
        """Update decoder"""
        if len(self.recent_words) < 2:
            return
        
        features = self._extract_field_features()
        
        for word_idx in self.recent_words[-2:]:
            if word_idx >= self.V:
                continue
            
            target = np.zeros(self.V, dtype=np.float32)
            target[word_idx] = 1.0
            
            logits = features @ self.W_decode
            probs = np.exp(logits - np.max(logits))
            probs = probs / (np.sum(probs) + 1e-10)
            
            correlation = target - probs
            self.W_decode += self.decoder_lr * np.outer(features, correlation)
    
    def process_input_and_respond(self, input_text, num_words=6):
        """Process input and respond"""
        words = input_text.lower().split()
        self.recent_generated.clear()
        
        for word in words:
            self.add_word_to_field(word, learn=True)
        
        self.organic_decoder_update()
        
        response_words = []
        for i in range(num_words):
            word = self.predict_next_word(temperature=1.3 + i*0.1)
            if word in self.word_to_idx:
                word_idx = self.word_to_idx[word]
                self.recent_generated.append(word_idx)
            response_words.append(word)
            self.add_word_to_field(word, learn=False)
        
        self.conversation_count += 1
        return " ".join(response_words)
    
    def get_stats(self):
        """Get statistics"""
        connectivity = self.word_associations.nnz / (self.V * self.V) if self.V > 0 else 0
        mode = "🌌 GENESIS" if self.bootstrap_mode else "💬 REACTIVE"
        return {
            'mode': mode,
            'genesis_steps': self.genesis_step if self.bootstrap_mode else 0,
            'spontaneous_words': len(self.spontaneous_thoughts),
            'conversations': self.conversation_count,
            'words_processed': self.total_words_processed,
            'vocabulary_size': self.V,
            'learned_words': len(self.learned_words),
            'connectivity': f"{connectivity:.4f}",
            'memory_efficiency': f"{self.X}³ grid, {self.C} channels"
        }
    
    def get_memory_usage(self):
        """Calculate memory usage"""
        field_size = self.field.nbytes / (1024**2)
        positions_size = self.word_positions.nbytes / (1024**2)
        decoder_size = self.W_decode.nbytes / (1024**2)
        
        associations_csr = self.word_associations.tocsr()
        associations_size = (associations_csr.data.nbytes + 
                           associations_csr.indices.nbytes + 
                           associations_csr.indptr.nbytes) / (1024**2)
        
        total = field_size + positions_size + decoder_size + associations_size
        
        return {
            'field': f"{field_size:.2f} MB",
            'decoder': f"{decoder_size:.2f} MB",
            'associations': f"{associations_size:.2f} MB",
            'total': f"{total:.2f} MB"
        }
    
    def get_field_visualization(self):
        """Get visualization slices"""
        slices = []
        mid_z = self.Z // 2
        for c in range(min(4, self.C)):
            amplitude = np.abs(self.field[:, :, mid_z, c])
            slices.append(amplitude)
        return slices
    
    def reset_field(self):
        """Reset field"""
        if self.bootstrap_mode:
            self.field = (np.random.randn(self.X, self.Y, self.Z, self.C)
                         .astype(np.complex64) * 0.01)
        else:
            self.field.fill(0)
        self.recent_words.clear()
        self.recent_generated.clear()
    
    def save_model(self, filepath):
        """Save model"""
        data = {
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'word_positions': self.word_positions,
            'word_frequencies': self.word_frequencies,
            'word_associations': self.word_associations.tocsr(),
            'W_decode': self.W_decode,
            'word_usage_count': self.word_usage_count,
            'bootstrap_mode': self.bootstrap_mode,
            'genesis_step': self.genesis_step,
            'spontaneous_thoughts': self.spontaneous_thoughts,
            'learned_words': self.learned_words,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_model(self, filepath):
        """Load model"""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.vocab = data['vocab']
            self.word_to_idx = data['word_to_idx']
            self.V = len(self.vocab)
            self.word_positions = data['word_positions']
            self.word_frequencies = data['word_frequencies']
            self.word_associations = data['word_associations'].tolil()
            self.W_decode = data['W_decode']
            self.word_usage_count = data['word_usage_count']
            
            if 'bootstrap_mode' in data:
                self.bootstrap_mode = data['bootstrap_mode']
            if 'genesis_step' in data:
                self.genesis_step = data['genesis_step']
            if 'spontaneous_thoughts' in data:
                self.spontaneous_thoughts = data['spontaneous_thoughts']
            if 'learned_words' in data:
                self.learned_words = data['learned_words']
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# ========================
# Enhanced Bootstrap GUI
# ========================

class EnhancedBootstrapWaveChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Bootstrap Wave Chatbot - DYNAMIC VOCABULARY")
        self.root.geometry("1400x800")
        
        # Create model in BOOTSTRAP mode with DYNAMIC VOCABULARY
        self.model = EnhancedWaveLanguageModel(
            grid_size=16, 
            num_channels=8, 
            bootstrap_mode=True
        )
        self.model_path = "enhanced_bootstrap_wave_model.pkl"
        
        # Bootstrap thread control
        self.bootstrap_running = False
        self.bootstrap_thread = None
        
        # Try to load saved model
        try:
            if self.model.load_model(self.model_path):
                print("Loaded saved model")
        except:
            print("Starting fresh")
        
        self._create_widgets()
        self._add_welcome_message()
        self._start_visualization()
        self._update_stats_periodically()
        
        # AUTO-START genesis
        self.root.after(1000, self.start_bootstrap_thinking)
    
    def _create_widgets(self):
        """Create GUI"""
        # Left: Chat
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(left_frame, text="🌌 Genesis Stream", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        
        self.chat_display = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD, width=50, height=35, font=('Consolas', 10)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display.tag_config('user', foreground='#0066cc', 
                                    font=('Consolas', 10, 'bold'))
        self.chat_display.tag_config('bot', foreground='#009900', 
                                    font=('Consolas', 10, 'bold'))
        self.chat_display.tag_config('genesis', foreground='#9900cc', 
                                    font=('Consolas', 10, 'italic'))
        self.chat_display.tag_config('system', foreground='#666', 
                                    font=('Consolas', 9, 'italic'))
        self.chat_display.tag_config('learned', foreground='#ff6600', 
                                    font=('Consolas', 9, 'bold'))
        
        # Input
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.input_entry = ttk.Entry(input_frame, font=('Arial', 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind('<Return>', lambda e: self.send_message())
        
        ttk.Button(input_frame, text="Send", command=self.send_message, 
                  width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # Middle: Stats
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(middle_frame, text="🌌 Genesis Stats", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        stats_frame = ttk.LabelFrame(middle_frame, text="Performance", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=18, width=32, font=('Courier', 9))
        self.stats_text.pack()
        
        # Controls
        control_frame = ttk.LabelFrame(middle_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.thinking_button = ttk.Button(control_frame, text="⏸ Pause Thinking",
                                         command=self.toggle_bootstrap_thinking)
        self.thinking_button.pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Force Word", 
                  command=self.force_genesis_word).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Save Model", 
                  command=self.save_model).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset Field", 
                  command=self.reset_field).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clear Chat", 
                  command=self.clear_chat).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Show Vocab", 
                  command=self.show_vocabulary).pack(fill=tk.X, pady=2)
        
        # Info
        info_frame = ttk.LabelFrame(middle_frame, text="Enhanced Mode", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = (
            "🌌 BOOTSTRAP GENESIS\n"
            "✨ DYNAMIC VOCABULARY\n\n"
            "• Self-starting\n"
            "• Learns new words\n"
            "• Quantum foam\n"
            "• Spontaneous collapse\n"
            "• Self-reinforcing\n"
            "• Bot speaks FIRST\n\n"
            "Type ANY word and\n"
            "watch it learn!\n"
        )
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, 
                 font=('Arial', 9)).pack()
        
        # Right: Visualization
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(right_frame, text="🌊 Wave Field (Genesis)", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(6, 6))
        self.fig.tight_layout(pad=2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_var = tk.StringVar(value="🌌 Universe initializing...")
        ttk.Label(right_frame, textvariable=self.status_var, 
                 font=('Arial', 9)).pack(pady=(5, 0))
        
        # Grid weights
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=2)
        self.root.rowconfigure(0, weight=1)
    
    def _add_welcome_message(self):
        """Welcome message"""
        msg = (
            "🌌 ENHANCED BOOTSTRAP GENESIS MODE\n"
            "=====================================\n\n"
            "✨ NEW: Dynamic Vocabulary Learning!\n\n"
            "Universe self-starting from quantum foam...\n\n"
            "✓ No input required\n"
            "✓ Spontaneous thought generation\n"
            "✓ Bot speaks FIRST\n"
            "✓ LEARNS NEW WORDS on-the-fly\n\n"
            "Try typing words like 'robot', 'quantum', 'universe'!\n"
            "Watch the vocabulary expand...\n\n"
            "Initializing field...\n"
        )
        self.chat_display.insert(tk.END, msg, 'system')
        self.chat_display.insert(tk.END, "\n" + "="*50 + "\n\n", 'system')
    
    def force_genesis_word(self):
        """Manually force a genesis word"""
        word = self.model.predict_next_word(temperature=2.5)
        self.model.genesis_step += 1
        word_wave = self.model.encode_word(word)
        self.model.field += word_wave
        self._display_thought(word)
    
    def start_bootstrap_thinking(self):
        """Start background thinking thread"""
        if not self.bootstrap_running:
            self.bootstrap_running = True
            self.bootstrap_thread = threading.Thread(target=self._bootstrap_loop, daemon=True)
            self.bootstrap_thread.start()
            self.thinking_button.config(text="⏸ Pause Thinking")
            self.chat_display.insert(tk.END, "[Genesis thread started]\n\n", 'system')
    
    def stop_bootstrap_thinking(self):
        """Stop background thinking"""
        self.bootstrap_running = False
        self.thinking_button.config(text="▶ Resume Thinking")
        self.chat_display.insert(tk.END, "[Genesis paused]\n\n", 'system')
    
    def toggle_bootstrap_thinking(self):
        """Toggle thinking on/off"""
        if self.bootstrap_running:
            self.stop_bootstrap_thinking()
        else:
            self.start_bootstrap_thinking()
    
    def _bootstrap_loop(self):
        """Background loop for spontaneous thinking"""
        while self.bootstrap_running:
            try:
                thought = self.model.spontaneous_think()
                if thought:
                    self.root.after(0, self._display_thought, thought)
                time.sleep(0.3)
            except Exception as e:
                print(f"Bootstrap error: {e}")
                time.sleep(1)
    
    def _display_thought(self, thought):
        """Display spontaneous thought"""
        self.chat_display.insert(tk.END, f"Genesis[{self.model.genesis_step}]: ", 'genesis')
        self.chat_display.insert(tk.END, f"{thought}\n", 'genesis')
        self.chat_display.see(tk.END)
        
        if self.model.spontaneous_thoughts:
            max_amp = self.model.spontaneous_thoughts[-1][2]
            self.status_var.set(f"Genesis: {self.model.genesis_step} | "
                              f"Amp: {max_amp:.3f} | Vocab: {self.model.V}")
    
    def send_message(self):
        """Handle user message"""
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        
        self.chat_display.insert(tk.END, "You: ", 'user')
        self.chat_display.insert(tk.END, user_input + "\n")
        self.input_entry.delete(0, tk.END)
        
        try:
            was_running = self.bootstrap_running
            if was_running:
                self.stop_bootstrap_thinking()
            
            # Check for new words
            words = user_input.lower().split()
            new_words = [w for w in words if w not in self.model.word_to_idx]
            
            response = self.model.process_input_and_respond(user_input, num_words=6)
            
            # Show newly learned words
            if new_words:
                self.chat_display.insert(tk.END, "✨ Learned: ", 'learned')
                self.chat_display.insert(tk.END, f"{', '.join(new_words)}\n", 'learned')
            
            self.chat_display.insert(tk.END, "Bot: ", 'bot')
            self.chat_display.insert(tk.END, response + "\n\n")
            
            if was_running:
                time.sleep(0.5)
                self.start_bootstrap_thinking()
        
        except Exception as e:
            self.chat_display.insert(tk.END, f"[Error: {e}]\n", 'system')
        
        self.chat_display.see(tk.END)
    
    def show_vocabulary(self):
        """Show current vocabulary"""
        stats = self.model.get_stats()
        recent_learned = self.model.learned_words[-20:] if self.model.learned_words else []
        
        msg = f"\n📚 Vocabulary: {stats['vocabulary_size']} words\n"
        msg += f"✨ Learned: {stats['learned_words']} new words\n"
        
        if recent_learned:
            msg += f"\nRecent: {', '.join(recent_learned)}\n"
        
        msg += "\n"
        self.chat_display.insert(tk.END, msg, 'system')
        self.chat_display.see(tk.END)
    
    def _update_stats_periodically(self):
        """Update stats"""
        self.update_stats_display()
        self.root.after(2000, self._update_stats_periodically)
    
    def update_stats_display(self):
        """Update stats display"""
        stats = self.model.get_stats()
        memory = self.model.get_memory_usage()
        
        text = f"Mode: {stats['mode']}\n"
        text += f"Genesis Steps: {stats['genesis_steps']}\n"
        text += f"Spontaneous: {stats['spontaneous_words']}\n"
        text += f"Conversations: {stats['conversations']}\n"
        text += f"Words: {stats['words_processed']}\n"
        text += f"\n📚 VOCABULARY:\n"
        text += f"Total: {stats['vocabulary_size']}\n"
        text += f"✨ Learned: {stats['learned_words']}\n"
        text += f"Connectivity: {stats['connectivity']}\n"
        text += f"Grid: {stats['memory_efficiency']}\n\n"
        text += "Memory Usage:\n"
        text += f" Field: {memory['field']}\n"
        text += f" Decoder: {memory['decoder']}\n"
        text += f" Assoc: {memory['associations']}\n"
        text += f" TOTAL: {memory['total']}\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, text)
    
    def save_model(self):
        """Save model"""
        self.model.save_model(self.model_path)
        messagebox.showinfo("Info", f"Model saved!\n\nVocabulary: {self.model.V} words\n"
                           f"Learned: {len(self.model.learned_words)} new words")
    
    def reset_field(self):
        """Reset field"""
        self.model.reset_field()
        self.model.genesis_step = 0
        self.model.spontaneous_thoughts.clear()
        self.chat_display.insert(tk.END, "[Field reset - Genesis restarted]\n\n", 'system')
        self.chat_display.see(tk.END)
    
    def clear_chat(self):
        """Clear chat"""
        self.chat_display.delete(1.0, tk.END)
        self._add_welcome_message()
    
    def _start_visualization(self):
        """Start visualization"""
        def animate(frame):
            slices = self.model.get_field_visualization()
            for idx, (ax, slice_data) in enumerate(zip(self.axes.flatten(), slices)):
                ax.clear()
                ax.imshow(slice_data, cmap='RdYlBu_r', vmin=0, vmax=0.5, 
                         interpolation='bilinear')
                ax.set_title(f'Ch {idx}', fontsize=9)
                ax.axis('off')
            self.fig.tight_layout(pad=1)
        
        self.ani = FuncAnimation(self.fig, animate, interval=300, 
                                cache_frame_data=False)
        self.canvas.draw()

# ========================
# Main
# ========================

def main():
    root = tk.Tk()
    app = EnhancedBootstrapWaveChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
