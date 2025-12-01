import numpy as np
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pickle
import os

# ========================
# Organic Wave Language Model
# ========================

class OrganicWaveLanguageModel:
    """
    Wave-based language model with ORGANIC learning:
    - No explicit supervision or feedback needed
    - Learns from conversation patterns via Hebbian plasticity
    - Self-organizing through wave interference
    - Emergent behavior from local rules
    """
    
    def __init__(self, grid_size=20, num_channels=12):
        self.X, self.Y, self.Z = grid_size, grid_size, grid_size
        self.C = num_channels
        
        # Wave field state
        self.field = np.zeros((self.X, self.Y, self.Z, self.C, 2), dtype=np.float32)
        self.velocity_field = np.zeros_like(self.field)
        
        # Wave parameters
        self.dt = 0.1
        self.wave_speed = 0.85
        self.damping = 0.97
        self.coupling = 0.1
        self.context_decay = 0.88  # Increased decay from 0.94 - faster forgetting
        
        # Vocabulary
        self.vocab = self._build_vocab()
        self.V = len(self.vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        
        # ORGANIC LEARNABLE COMPONENTS
        # Word encodings evolve through use
        self.word_positions = np.random.rand(self.V, 3) * np.array([self.X, self.Y, self.Z]) * 0.8
        self.word_frequencies = 0.3 + np.random.rand(self.V, self.C) * 0.5
        self.word_amplitudes = np.ones(self.V) * 0.3
        
        # Hebbian connection matrix: words that co-occur strengthen
        self.word_associations = np.zeros((self.V, self.V), dtype=np.float32)
        
        # Decoder evolves through correlation-based learning
        feature_dim = self.C * 4
        self.W_decode = np.random.randn(feature_dim, self.V) * 0.03
        
        # Organic learning rates
        self.hebbian_lr = 0.003  # Reduced from 0.01
        self.position_lr = 0.002  # Reduced from 0.005
        self.decoder_lr = 0.005   # Reduced from 0.008
        
        # Memory trace for temporal patterns
        self.recent_words = []  # Input words
        self.recent_generated = []  # Generated words (for anti-repetition)
        self.word_usage_count = np.zeros(self.V)
        
        # Statistics
        self.total_words_processed = 0
        self.conversation_count = 0
    
    def _build_vocab(self):
        """Build vocabulary"""
        return [
            # Core words
            "hello", "hi", "hey", "goodbye", "bye", "thanks", "thank", "please", 
            "sorry", "welcome", "greetings", "yes", "no", "ok", "okay", "sure",
            # Pronouns
            "i", "you", "we", "they", "he", "she", "it", "me", "my", "your",
            "our", "their", "this", "that", "these", "those",
            # Verbs
            "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "done",
            "can", "could", "will", "would", "should", "may", "might",
            "want", "need", "like", "love", "hate", "know", "think", "feel",
            "see", "hear", "say", "tell", "ask", "answer", "help", "try",
            "go", "come", "get", "make", "take", "give", "find", "use",
            "work", "play", "learn", "teach", "understand", "remember", "mean",
            # Adjectives
            "good", "bad", "great", "nice", "fine", "well",
            "happy", "sad", "angry", "excited", "tired", "bored",
            "big", "small", "large", "little", "new", "old", "young",
            "hot", "cold", "warm", "cool", "fast", "slow", "easy", "hard",
            "right", "wrong", "true", "false", "maybe",
            # Nouns
            "day", "night", "time", "week", "year", "today", "tomorrow",
            "thing", "things", "way", "place", "home", "work", "school",
            "people", "person", "friend", "family", "name", "life",
            "question", "answer", "problem", "idea", "information",
            # Question words
            "what", "who", "where", "when", "why", "how", "which",
            # Function words
            "the", "a", "an", "and", "or", "but", "if", "then",
            "to", "of", "in", "on", "at", "by", "for", "with",
            "from", "about", "as", "into", "through",
            "not", "very", "really", "just", "only",
            "also", "too", "so", "now", "here", "there",
        ]
    
    def encode_word(self, word):
        """Encode word as wave packet"""
        word = word.lower()
        if word not in self.word_to_idx:
            return np.zeros((self.X, self.Y, self.Z, self.C, 2))
        
        idx = self.word_to_idx[word]
        wave_packet = np.zeros((self.X, self.Y, self.Z, self.C, 2))
        
        # Get current parameters (evolving)
        cx, cy, cz = self.word_positions[idx]
        amplitude = self.word_amplitudes[idx]
        frequencies = self.word_frequencies[idx]
        
        # Create spatial grid
        sigma = 2.5
        x = np.arange(self.X)
        y = np.arange(self.Y)
        z = np.arange(self.Z)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Gaussian envelope
        dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
        envelope = amplitude * np.exp(-dist_sq / (2 * sigma**2))
        
        # Apply frequencies
        phase_base = 2 * np.pi * idx / self.V
        for c in range(self.C):
            phase = phase_base + c * np.pi / self.C
            wave_packet[:, :, :, c, 0] = envelope * np.cos(frequencies[c] * phase)
            wave_packet[:, :, :, c, 1] = envelope * np.sin(frequencies[c] * phase)
        
        return wave_packet
    
    def propagate_waves(self, steps=4):
        """Propagate waves"""
        for _ in range(steps):
            laplacian = self._compute_laplacian()
            acceleration = (self.wave_speed ** 2) * laplacian
            
            # Channel coupling
            for c in range(1, self.C):
                coupling_term = self.coupling * (
                    self.field[:, :, :, c-1, :] - self.field[:, :, :, c, :]
                )
                acceleration[:, :, :, c, :] += coupling_term
            
            self.velocity_field += acceleration * self.dt
            self.velocity_field *= self.damping
            self.field += self.velocity_field * self.dt
            self.field = np.tanh(self.field)
    
    def _compute_laplacian(self):
        """3D Laplacian"""
        lap = np.zeros_like(self.field)
        lap[1:-1, 1:-1, 1:-1] = (
            self.field[2:, 1:-1, 1:-1] + self.field[:-2, 1:-1, 1:-1] +
            self.field[1:-1, 2:, 1:-1] + self.field[1:-1, :-2, 1:-1] +
            self.field[1:-1, 1:-1, 2:] + self.field[1:-1, 1:-1, :-2] -
            6 * self.field[1:-1, 1:-1, 1:-1]
        )
        return lap
    
    def add_word_to_field(self, word, learn=True):
        """Add word and organically learn from it"""
        word = word.lower()
        word_wave = self.encode_word(word)
        
        # Apply context decay
        self.field *= self.context_decay
        self.field += word_wave
        self.propagate_waves(steps=4)
        
        if learn and word in self.word_to_idx:
            idx = self.word_to_idx[word]
            self.total_words_processed += 1
            self.word_usage_count[idx] += 1
            
            # ORGANIC LEARNING: Hebbian plasticity
            self._hebbian_learning(idx)
            
            # Update word encoding based on field state
            self._adapt_word_encoding(idx)
            
            # Track for temporal patterns
            self.recent_words.append(idx)
            if len(self.recent_words) > 10:
                self.recent_words.pop(0)
    
    def _hebbian_learning(self, current_word_idx):
        """
        Hebbian rule: neurons that fire together wire together.
        Strengthen associations between co-occurring words.
        """
        # Strengthen connections to recently seen words
        for past_idx in self.recent_words[-5:]:
            # Symmetric strengthening
            self.word_associations[current_word_idx, past_idx] += self.hebbian_lr
            self.word_associations[past_idx, current_word_idx] += self.hebbian_lr
            
            # Normalize to prevent unbounded growth
            self.word_associations[current_word_idx, past_idx] = np.tanh(
                self.word_associations[current_word_idx, past_idx]
            )
            self.word_associations[past_idx, current_word_idx] = np.tanh(
                self.word_associations[past_idx, current_word_idx]
            )
    
    def _adapt_word_encoding(self, word_idx):
        """
        Organically adapt word's spatial position and frequencies
        based on current field activity (like spike-timing dependent plasticity)
        """
        # Find center of field activity
        activity_center = self._compute_activity_center()
        
        # Move word position slightly toward activity center
        direction = activity_center - self.word_positions[word_idx]
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            # Adaptive step size based on usage
            step = self.position_lr * (1.0 + np.log1p(self.word_usage_count[word_idx]) * 0.1)
            self.word_positions[word_idx] += direction * step
            
            # Keep in bounds
            self.word_positions[word_idx] = np.clip(
                self.word_positions[word_idx],
                [1, 1, 1],
                [self.X-2, self.Y-2, self.Z-2]
            )
    
    def _compute_activity_center(self):
        """Compute center of mass of field activity"""
        total_activity = np.sum(np.abs(self.field), axis=(3, 4))
        total_mass = np.sum(total_activity) + 1e-10
        
        x_coords = np.arange(self.X)
        y_coords = np.arange(self.Y)
        z_coords = np.arange(self.Z)
        
        x_center = np.sum(np.sum(total_activity, axis=(1, 2)) * x_coords) / total_mass
        y_center = np.sum(np.sum(total_activity, axis=(0, 2)) * y_coords) / total_mass
        z_center = np.sum(np.sum(total_activity, axis=(0, 1)) * z_coords) / total_mass
        
        return np.array([x_center, y_center, z_center])
    
    def _extract_field_features(self):
        """Extract features from wave field"""
        features = []
        
        for c in range(self.C):
            channel = self.field[:, :, :, c, :]
            amplitude = np.sqrt(channel[:, :, :, 0]**2 + channel[:, :, :, 1]**2)
            
            features.append(np.mean(amplitude))
            features.append(np.max(amplitude))
            features.append(np.std(amplitude))
            features.append(np.sum(amplitude**2))
        
        return np.array(features)
    
    def predict_next_word(self, temperature=1.2):
        """
        Predict next word using:
        1. Current field state
        2. Hebbian associations from recent words
        3. Usage statistics
        4. STRONG ANTI-REPETITION: Block recently generated words
        """
        # Get field-based prediction
        features = self._extract_field_features()
        field_logits = features @ self.W_decode
        
        # Add Hebbian association bias (very weak)
        association_bias = np.zeros(self.V)
        if len(self.recent_words) > 0:
            # Only look at input words, not generated ones
            for i, past_idx in enumerate(self.recent_words[-2:]):
                weight = 0.1 / (i + 1)
                association_bias += self.word_associations[past_idx, :] * weight
        
        # Combine signals
        logits = field_logits + association_bias
        
        # CRITICAL: Block recently generated words completely
        if len(self.recent_generated) > 0:
            for past_idx in self.recent_generated[-4:]:
                logits[past_idx] = -1000  # Effectively zero probability
        
        # Also penalize the input words (don't just echo)
        if len(self.recent_words) > 0:
            for past_idx in self.recent_words[-3:]:
                logits[past_idx] -= 2.0
        
        # Add strong diversity noise
        diversity_noise = np.random.randn(self.V) * 0.5
        logits += diversity_noise
        
        # Temperature and softmax
        logits = logits / max(temperature, 0.01)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        
        # Ensure we have valid probabilities
        if np.sum(probs) < 1e-9 or np.isnan(probs).any():
            # Fallback: uniform random
            probs = np.ones(self.V) / self.V
            # Still block recent
            if len(self.recent_generated) > 0:
                for past_idx in self.recent_generated[-4:]:
                    probs[past_idx] = 0
            probs = probs / (np.sum(probs) + 1e-10)
        
        # Sample from top-k with more diversity
        top_k = 20
        top_indices = np.argpartition(-probs, min(top_k, len(probs))-1)[:top_k]
        top_probs = probs[top_indices]
        top_probs = top_probs / (np.sum(top_probs) + 1e-10)
        
        chosen_idx = np.random.choice(top_indices, p=top_probs)
        return self.vocab[chosen_idx]
    
    def organic_decoder_update(self):
        """
        Update decoder through correlation-based learning (Oja's rule).
        No supervision needed - learns from statistical regularities.
        """
        if len(self.recent_words) < 2:
            return
        
        # Get current field state
        features = self._extract_field_features()
        
        # Update based on recent word co-occurrences
        for word_idx in self.recent_words[-3:]:
            # Target: one-hot for this word
            target = np.zeros(self.V)
            target[word_idx] = 1.0
            
            # Current prediction
            logits = features @ self.W_decode
            probs = np.exp(logits - np.max(logits))
            probs = probs / (np.sum(probs) + 1e-10)
            
            # Correlation-based update (Oja's rule variant)
            # Strengthen connections that correlate with actual usage
            correlation = target - probs
            self.W_decode += self.decoder_lr * np.outer(features, correlation)
            
            # Weight normalization to prevent explosion
            norms = np.linalg.norm(self.W_decode, axis=0) + 1e-10
            self.W_decode /= (norms * 0.01 + 1.0)
    
    def process_input_and_respond(self, input_text, num_words=6):
        """Process input organically and generate response"""
        words = input_text.lower().split()
        
        # Clear previous generated words for new response
        self.recent_generated.clear()
        
        # Process each input word (learning happens here)
        for word in words:
            self.add_word_to_field(word, learn=True)
        
        # Update decoder organically
        self.organic_decoder_update()
        
        # Generate response
        response_words = []
        for i in range(num_words):
            word = self.predict_next_word(temperature=1.3 + i*0.1)  # Increase temp each word
            
            # Get word index for tracking
            if word in self.word_to_idx:
                word_idx = self.word_to_idx[word]
                self.recent_generated.append(word_idx)
            
            response_words.append(word)
            
            # Add generated word to field (but don't learn from it)
            self.add_word_to_field(word, learn=False)
        
        self.conversation_count += 1
        return " ".join(response_words)
    
    def get_stats(self):
        """Get learning statistics"""
        # Measure network connectivity
        connectivity = np.mean(np.abs(self.word_associations))
        
        # Most associated word pairs
        top_pairs = []
        for i in range(min(5, self.V)):
            for j in range(i+1, min(5, self.V)):
                if self.word_associations[i, j] > 0.1:
                    top_pairs.append((self.vocab[i], self.vocab[j], self.word_associations[i, j]))
        
        return {
            'conversations': self.conversation_count,
            'words_processed': self.total_words_processed,
            'connectivity': f"{connectivity:.3f}",
            'top_pairs': sorted(top_pairs, key=lambda x: x[2], reverse=True)[:3]
        }
    
    def save_model(self, filepath):
        """Save learned parameters"""
        data = {
            'word_positions': self.word_positions,
            'word_frequencies': self.word_frequencies,
            'word_amplitudes': self.word_amplitudes,
            'word_associations': self.word_associations,
            'W_decode': self.W_decode,
            'word_usage_count': self.word_usage_count,
            'stats': self.get_stats()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_model(self, filepath):
        """Load learned parameters"""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.word_positions = data['word_positions']
        self.word_frequencies = data['word_frequencies']
        self.word_amplitudes = data['word_amplitudes']
        self.word_associations = data['word_associations']
        self.W_decode = data['W_decode']
        self.word_usage_count = data['word_usage_count']
        
        return True
    
    def get_field_visualization(self):
        """Get visualization slices"""
        slices = []
        mid_z = self.Z // 2
        
        for c in range(min(4, self.C)):
            amplitude = np.sqrt(
                self.field[:, :, mid_z, c, 0]**2 + 
                self.field[:, :, mid_z, c, 1]**2
            )
            slices.append(amplitude)
        
        return slices
    
    def reset_field(self):
        """Reset wave field"""
        self.field.fill(0)
        self.velocity_field.fill(0)
        self.recent_words.clear()
        self.recent_generated.clear()


# ========================
# Organic Learning GUI
# ========================

class OrganicWaveChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Organic Wave Chatbot - Self-Learning AI")
        self.root.geometry("1400x800")
        
        self.model = OrganicWaveLanguageModel(grid_size=20, num_channels=12)
        self.model_path = "organic_wave_model.pkl"
        
        # Try to load existing model
        if self.model.load_model(self.model_path):
            self.show_info("Loaded previously learned model!")
        
        self._create_widgets()
        self._add_welcome_message()
        self._start_visualization()
        self._update_stats_periodically()
    
    def _create_widgets(self):
        """Create GUI"""
        # Left: Chat
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(left_frame, text="💬 Conversation", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        
        self.chat_display = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD, width=50, height=35, font=('Consolas', 10)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display.tag_config('user', foreground='#0066cc', font=('Consolas', 10, 'bold'))
        self.chat_display.tag_config('bot', foreground='#009900', font=('Consolas', 10, 'bold'))
        self.chat_display.tag_config('system', foreground='#666', font=('Consolas', 9, 'italic'))
        
        # Input
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.input_entry = ttk.Entry(input_frame, font=('Arial', 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind('<Return>', lambda e: self.send_message())
        
        ttk.Button(input_frame, text="Send", command=self.send_message, 
                  width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # Middle: Stats & Info
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(middle_frame, text="🧠 Learning Stats", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        stats_frame = ttk.LabelFrame(middle_frame, text="Organic Learning", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=12, width=32, font=('Courier', 9))
        self.stats_text.pack()
        
        # Controls
        control_frame = ttk.LabelFrame(middle_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Save Model", command=self.save_model).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset Field", command=self.reset_field).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clear Chat", command=self.clear_chat).pack(fill=tk.X, pady=2)
        
        # Info
        info_frame = ttk.LabelFrame(middle_frame, text="How It Learns", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = (
            "✨ ORGANIC LEARNING\n\n"
            "No feedback needed!\n\n"
            "• Hebbian plasticity:\n"
            "  Co-occurring words\n"
            "  strengthen connections\n\n"
            "• Self-organizing:\n"
            "  Word positions adapt\n"
            "  based on usage\n\n"
            "• Emergent behavior:\n"
            "  Learns patterns from\n"
            "  conversation structure\n\n"
            "Just chat naturally!"
        )
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, 
                 font=('Arial', 9)).pack()
        
        # Right: Visualization
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(right_frame, text="🌊 Wave Field", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(6, 6))
        self.fig.tight_layout(pad=2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_var = tk.StringVar(value="Learning organically...")
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
            "🧠 ORGANIC WAVE CHATBOT\n"
            "=======================\n\n"
            "This bot learns ORGANICALLY!\n"
            "No feedback buttons needed.\n\n"
            "It learns from:\n"
            "• Word co-occurrence patterns (Hebbian)\n"
            "• Conversation structure\n"
            "• Wave interference dynamics\n"
            "• Statistical regularities\n\n"
            "Just chat naturally and watch it evolve!\n"
        )
        self.chat_display.insert(tk.END, msg, 'system')
        self.chat_display.insert(tk.END, "\n" + "="*50 + "\n\n")
    
    def send_message(self):
        """Handle message"""
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        
        self.chat_display.insert(tk.END, "You: ", 'user')
        self.chat_display.insert(tk.END, user_input + "\n")
        self.input_entry.delete(0, tk.END)
        
        try:
            # Generate response (learning happens automatically)
            response = self.model.process_input_and_respond(user_input, num_words=6)
            
            self.chat_display.insert(tk.END, "Bot: ", 'bot')
            self.chat_display.insert(tk.END, response + "\n\n")
            
            self.status_var.set(f"Words processed: {self.model.total_words_processed}")
            
        except Exception as e:
            self.chat_display.insert(tk.END, f"[Error: {e}]\n", 'system')
        
        self.chat_display.see(tk.END)
    
    def _update_stats_periodically(self):
        """Update stats display periodically"""
        self.update_stats_display()
        self.root.after(2000, self._update_stats_periodically)
    
    def update_stats_display(self):
        """Update stats"""
        stats = self.model.get_stats()
        
        text = f"Conversations: {stats['conversations']}\n"
        text += f"Words Processed: {stats['words_processed']}\n"
        text += f"Network Connectivity: {stats['connectivity']}\n\n"
        text += "Top Word Associations:\n"
        
        for word1, word2, strength in stats['top_pairs']:
            text += f"  {word1} ↔ {word2}: {strength:.2f}\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, text)
    
    def save_model(self):
        """Save model"""
        self.model.save_model(self.model_path)
        self.show_info("Model saved!")
    
    def reset_field(self):
        """Reset field"""
        self.model.reset_field()
        self.chat_display.insert(tk.END, "[Wave field reset]\n\n", 'system')
        self.chat_display.see(tk.END)
    
    def clear_chat(self):
        """Clear chat"""
        self.chat_display.delete(1.0, tk.END)
        self._add_welcome_message()
    
    def show_info(self, message):
        """Show info dialog"""
        messagebox.showinfo("Info", message)
    
    def _start_visualization(self):
        """Start visualization"""
        def animate(frame):
            slices = self.model.get_field_visualization()
            
            for idx, (ax, slice_data) in enumerate(zip(self.axes.flatten(), slices)):
                ax.clear()
                ax.imshow(slice_data, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='bilinear')
                ax.set_title(f'Channel {idx}', fontsize=9)
                ax.axis('off')
            
            self.fig.tight_layout(pad=1)
        
        self.ani = FuncAnimation(self.fig, animate, interval=250, cache_frame_data=False)
        self.canvas.draw()


# ========================
# Main
# ========================

def main():
    root = tk.Tk()
    app = OrganicWaveChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
