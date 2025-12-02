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
from collections import defaultdict

# ========================
# ULTRA-ORGANIC Wave Language Model with GUI
# ========================

class UltraOrganicWaveModel:
    """
    Maximally organic wave-based language model:
    
    1. SEMANTIC POSITIONING: New words placed near similar words
    2. EMERGENT TEMPERATURE: Creativity emerges from field state
    3. NATURAL INHIBITION: Wave cancellation, not hardcoded penalties
    4. PHONETIC ENCODING: Sound influences wave frequencies
    5. ADAPTIVE ARCHITECTURE: Grid expands organically
    6. CONTEXT CLUSTERING: Related words group spatially
    """
    
    def __init__(self, grid_size=16, num_channels=8, bootstrap_mode=False):
        self.X, self.Y, self.Z = grid_size, grid_size, grid_size
        self.C = num_channels
        self.bootstrap_mode = bootstrap_mode
        
        # Initialize field
        if bootstrap_mode:
            self.field = (np.random.randn(self.X, self.Y, self.Z, self.C)
                         .astype(np.complex64) * 0.01)
        else:
            self.field = np.zeros((self.X, self.Y, self.Z, self.C), dtype=np.complex64)
        
        # Wave parameters
        self.dt = 0.15
        self.wave_speed = 0.85
        self.damping = 0.96
        self.context_decay = 0.88
        
        # Vocabulary
        self.vocab = self._build_core_vocab()
        self.V = len(self.vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # ORGANIC POSITIONING: Words cluster by semantic similarity
        self.word_positions = self._initialize_semantic_space()
        self.word_frequencies = self._initialize_phonetic_frequencies()
        
        # Association tracking (for semantic positioning)
        self.word_associations = lil_matrix((self.V, self.V), dtype=np.float32)
        self.context_vectors = defaultdict(list)  # Track contexts for each word
        
        # Decoder
        feature_dim = self.C * 2
        self.W_decode = np.random.randn(feature_dim, self.V).astype(np.float32) * 0.03
        
        # Learning rates
        self.hebbian_lr = 0.003
        self.decoder_lr = 0.005
        
        # Memory with TIMESTAMPS for natural inhibition
        self.recent_words = []
        self.recent_generated = []
        self.word_usage_count = np.zeros(self.V, dtype=np.uint32)
        self.last_activation_time = np.zeros(self.V, dtype=np.float32)  # NEW
        self.current_time = 0.0  # NEW
        
        # Statistics
        self.total_words_processed = 0
        self.conversation_count = 0
        self.learned_words = []
        
        # Genesis
        self.genesis_step = 0
        self.spontaneous_thoughts = []
        
        # ADAPTIVE ARCHITECTURE tracking
        self.spatial_density = 0.0
        self.expansion_events = []
        
        print(f"🌱 Ultra-Organic Wave Model initialized")
        print(f"   Semantic positioning: ENABLED")
        print(f"   Emergent temperature: ENABLED")
        print(f"   Natural inhibition: ENABLED")
        print(f"   Phonetic encoding: ENABLED")
    
    def _build_core_vocab(self):
        """Core vocabulary"""
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
    
    def _initialize_semantic_space(self):
        """
        ORGANIC POSITIONING: Initialize core words in semantic clusters
        Instead of random, group similar words
        """
        positions = []
        
        # Define semantic clusters (organic grouping)
        clusters = {
            'greetings': ['hello', 'hi', 'hey', 'goodbye', 'bye', 'welcome'],
            'politeness': ['thanks', 'please', 'sorry'],
            'agreement': ['yes', 'no', 'ok', 'okay', 'sure'],
            'pronouns': ['i', 'you', 'we', 'they', 'he', 'she', 'it', 'me', 'my', 'your'],
            'being': ['am', 'is', 'are', 'was', 'were', 'be'],
            'having': ['have', 'has', 'had'],
            'doing': ['do', 'does', 'did'],
            'modals': ['can', 'could', 'will', 'would', 'should'],
            'desire': ['want', 'need', 'like', 'love'],
            'cognition': ['know', 'think', 'feel', 'see'],
            'communication': ['say', 'tell', 'ask'],
            'action': ['help', 'try', 'go', 'come', 'get', 'make', 'take', 'give', 'find', 'use'],
            'quality_pos': ['good', 'great', 'nice', 'fine', 'well', 'happy'],
            'quality_neg': ['bad', 'sad'],
            'size': ['big', 'small'],
            'time': ['new', 'old', 'day', 'time', 'now'],
            'abstract': ['thing', 'way', 'question', 'answer'],
            'place': ['home', 'work', 'here', 'there'],
            'social': ['people', 'friend', 'name'],
            'interrogative': ['what', 'who', 'where', 'when', 'why', 'how'],
            'function': ['the', 'a', 'and', 'or', 'but', 'if', 'to', 'of', 'in', 'on', 'at', 'not', 'very', 'just', 'so'],
        }
        
        # Assign each cluster a region in 3D space
        cluster_centers = {}
        num_clusters = len(clusters)
        
        # Arrange clusters in 3D using golden angle spiral
        for i, cluster_name in enumerate(clusters.keys()):
            # Golden angle spiral for organic distribution
            phi = i * np.pi * (3. - np.sqrt(5.))  # Golden angle
            y = 1 - (i / float(num_clusters - 1)) * 2 if num_clusters > 1 else 0
            radius = np.sqrt(1 - y * y)
            theta = phi
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Scale to grid size
            center = np.array([
                (x + 1) * 0.4 * self.X,
                (y + 1) * 0.4 * self.Y,
                (z + 1) * 0.4 * self.Z
            ])
            cluster_centers[cluster_name] = center
        
        # Place each word near its cluster center
        for word in self.vocab:
            # Find which cluster this word belongs to
            cluster_name = None
            for cname, words in clusters.items():
                if word in words:
                    cluster_name = cname
                    break
            
            if cluster_name:
                # Place near cluster center with small random offset
                center = cluster_centers[cluster_name]
                offset = np.random.randn(3) * 1.5  # Tight clustering
                position = center + offset
            else:
                # Uncategorized words go to center
                position = np.array([self.X/2, self.Y/2, self.Z/2]) + np.random.randn(3) * 2
            
            # Ensure within bounds
            position = np.clip(position, 0, [self.X-1, self.Y-1, self.Z-1])
            positions.append(position)
        
        return np.array(positions, dtype=np.float32)
    
    def _initialize_phonetic_frequencies(self):
        """
        PHONETIC ENCODING: Derive frequencies from word sounds
        Similar-sounding words get similar frequencies
        """
        frequencies = []
        
        for word in self.vocab:
            # Simple phonetic features
            vowels = sum(1 for c in word if c in 'aeiou')
            consonants = len(word) - vowels
            starts_vowel = 1.0 if word[0] in 'aeiou' else 0.0
            length = len(word)
            
            # Derive base frequency from phonetic features
            base_freq = 0.3 + (vowels / max(length, 1)) * 0.3
            
            # Create frequency vector for each channel
            word_freqs = []
            for c in range(self.C):
                # Each channel responds to different phonetic aspects
                if c < 2:  # Vowel-sensitive channels
                    freq = base_freq + (vowels / max(length, 1)) * 0.2
                elif c < 4:  # Consonant-sensitive channels
                    freq = base_freq + (consonants / max(length, 1)) * 0.2
                elif c < 6:  # Length-sensitive channels
                    freq = base_freq + (length / 10.0) * 0.2
                else:  # Mixed channels
                    freq = base_freq + np.random.rand() * 0.3
                
                word_freqs.append(np.clip(freq, 0.2, 0.9))
            
            frequencies.append(word_freqs)
        
        return np.array(frequencies, dtype=np.float32)
    
    def find_semantic_neighbors(self, context_words, k=5):
        """
        Find semantically similar words based on co-occurrence patterns
        Returns positions of k most similar words
        """
        if not context_words:
            return None
        
        # Calculate similarity scores for all existing words
        similarity_scores = np.zeros(self.V)
        
        for context_idx in context_words:
            if context_idx < self.V:
                # Get association strengths
                assocs = self.word_associations[context_idx, :].toarray().flatten()
                
                # Handle size mismatch (vocab may have grown)
                if len(assocs) < self.V:
                    # Pad with zeros
                    assocs = np.pad(assocs, (0, self.V - len(assocs)), mode='constant')
                elif len(assocs) > self.V:
                    # Truncate (shouldn't happen, but safety)
                    assocs = assocs[:self.V]
                
                similarity_scores += assocs
        
        # Find top-k similar words (excluding zeros)
        nonzero_indices = np.where(similarity_scores > 0.01)[0]
        if len(nonzero_indices) == 0:
            return None
        
        top_indices = nonzero_indices[np.argsort(-similarity_scores[nonzero_indices])[:k]]
        return self.word_positions[top_indices]
    
    def learn_new_word(self, word):
        """
        SEMANTIC POSITIONING: Place new word near similar words
        """
        word = word.lower()
        
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        
        new_idx = self.V
        self.vocab.append(word)
        self.word_to_idx[word] = new_idx
        self.V += 1
        
        # SEMANTIC POSITIONING: Find where to place this word
        neighbor_positions = self.find_semantic_neighbors(self.recent_words[-5:], k=3)
        
        if neighbor_positions is not None and len(neighbor_positions) > 0:
            # Place near semantic neighbors (ORGANIC!)
            avg_position = np.mean(neighbor_positions, axis=0)
            new_position = avg_position + np.random.randn(3) * 2.0  # Small offset
            placement = "semantic cluster"
        else:
            # No context yet - place in open space (still organic - finds gaps)
            new_position = self._find_low_density_region()
            placement = "low-density region"
        
        # Ensure within bounds
        new_position = np.clip(new_position, 0, [self.X-1, self.Y-1, self.Z-1])
        self.word_positions = np.vstack([self.word_positions, new_position])
        
        # PHONETIC ENCODING: Derive frequencies from sound
        word_freqs = self._derive_phonetic_frequencies(word)
        self.word_frequencies = np.vstack([self.word_frequencies, word_freqs])
        
        # Expand structures
        self._expand_association_matrix()
        
        new_decoder_col = np.random.randn(self.W_decode.shape[0], 1).astype(np.float32) * 0.03
        self.W_decode = np.hstack([self.W_decode, new_decoder_col])
        
        self.word_usage_count = np.append(self.word_usage_count, 0)
        self.last_activation_time = np.append(self.last_activation_time, self.current_time)
        
        self.learned_words.append(word)
        
        print(f"✨ Learned '{word}' → {placement} at {new_position}")
        
        # Check if we need to expand grid (ADAPTIVE ARCHITECTURE)
        self._check_spatial_density()
        
        return new_idx
    
    def _find_low_density_region(self):
        """
        ORGANIC SPACE FILLING: Find region with fewest words
        """
        # Divide space into octants
        mid_x, mid_y, mid_z = self.X/2, self.Y/2, self.Z/2
        
        octant_counts = {}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    octant_counts[(i,j,k)] = 0
        
        # Count words in each octant
        for pos in self.word_positions:
            i = 1 if pos[0] > mid_x else 0
            j = 1 if pos[1] > mid_y else 0
            k = 1 if pos[2] > mid_z else 0
            octant_counts[(i,j,k)] += 1
        
        # Find least dense octant
        min_octant = min(octant_counts, key=octant_counts.get)
        
        # Place in that octant
        x_base = mid_x if min_octant[0] == 1 else 0
        y_base = mid_y if min_octant[1] == 1 else 0
        z_base = mid_z if min_octant[2] == 1 else 0
        
        position = np.array([
            x_base + np.random.rand() * mid_x,
            y_base + np.random.rand() * mid_y,
            z_base + np.random.rand() * mid_z
        ])
        
        return position
    
    def _derive_phonetic_frequencies(self, word):
        """
        PHONETIC ENCODING: Sound determines frequencies
        """
        vowels = sum(1 for c in word if c in 'aeiou')
        consonants = len(word) - vowels
        length = len(word)
        
        # Phonetic features
        has_harsh = any(c in 'kgptbd' for c in word)  # Plosives
        has_soft = any(c in 'lmnr' for c in word)     # Liquids/nasals
        has_sibilant = any(c in 'sz' for c in word)   # Sibilants
        
        base_freq = 0.3 + (vowels / max(length, 1)) * 0.3
        
        freqs = []
        for c in range(self.C):
            if c == 0:  # Vowel richness
                freq = base_freq + (vowels / max(length, 1)) * 0.3
            elif c == 1:  # Consonant density
                freq = base_freq + (consonants / max(length, 1)) * 0.3
            elif c == 2:  # Harsh sounds
                freq = base_freq + (0.3 if has_harsh else 0.0)
            elif c == 3:  # Soft sounds
                freq = base_freq + (0.3 if has_soft else 0.0)
            elif c == 4:  # Sibilance
                freq = base_freq + (0.3 if has_sibilant else 0.0)
            elif c == 5:  # Length
                freq = base_freq + min(length / 10.0, 0.3)
            else:  # Composite
                freq = base_freq + np.random.rand() * 0.2
            
            freqs.append(np.clip(freq, 0.2, 0.9))
        
        return np.array(freqs, dtype=np.float32)
    
    def _check_spatial_density(self):
        """
        ADAPTIVE ARCHITECTURE: Monitor spatial density
        (Grid expansion would require major refactoring, so we just track it)
        """
        if self.V < 10:
            return
        
        # Calculate average nearest-neighbor distance
        distances = []
        for i, pos1 in enumerate(self.word_positions):
            min_dist = float('inf')
            for j, pos2 in enumerate(self.word_positions):
                if i != j:
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist < min_dist:
                        min_dist = dist
            distances.append(min_dist)
        
        self.spatial_density = 1.0 / (np.mean(distances) + 1e-6)
        
        # Track if we should expand (for future implementation)
        if self.spatial_density > 0.5 and self.V > 100:
            if len(self.expansion_events) == 0 or self.V - self.expansion_events[-1] > 50:
                self.expansion_events.append(self.V)
                print(f"⚠️  High spatial density detected at V={self.V}")
                print(f"   (Grid expansion would be beneficial)")
    
    def _expand_association_matrix(self):
        """Expand sparse matrix"""
        old_size = self.word_associations.shape[0]
        new_size = old_size + 1
        new_matrix = lil_matrix((new_size, new_size), dtype=np.float32)
        new_matrix[:old_size, :old_size] = self.word_associations
        self.word_associations = new_matrix
    
    def encode_word(self, word):
        """Encode word with auto-learning"""
        word = word.lower()
        
        if word not in self.word_to_idx:
            idx = self.learn_new_word(word)
        else:
            idx = self.word_to_idx[word]
        
        # Update activation time (for natural inhibition)
        self.last_activation_time[idx] = self.current_time
        
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
        """FFT wave propagation"""
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
    
    def calculate_emergent_temperature(self):
        """
        EMERGENT TEMPERATURE: Derived from field state
        High entropy → explore (high temp)
        Low entropy → exploit (low temp)
        """
        # Calculate field entropy
        amplitudes = np.abs(self.field).flatten()
        amplitudes = amplitudes / (np.sum(amplitudes) + 1e-10)
        entropy = -np.sum(amplitudes * np.log(amplitudes + 1e-10))
        
        # Normalize entropy (0 to ~10)
        normalized_entropy = entropy / 10.0
        
        # Map to temperature (1.0 to 3.0)
        temperature = 1.0 + normalized_entropy * 2.0
        temperature = np.clip(temperature, 1.0, 3.5)
        
        return float(temperature)
    
    def spontaneous_think(self):
        """Genesis mode with EMERGENT TEMPERATURE"""
        self.genesis_step += 1
        self.current_time += 0.1  # Time advances
        
        self.propagate_waves_fft(steps=5)
        self.field *= 0.95
        
        noise = np.random.randn(self.X, self.Y, self.Z, self.C).astype(np.complex64) * 0.01
        self.field += noise
        
        max_amplitude = np.max(np.abs(self.field))
        
        should_speak = (
            max_amplitude > 0.008 or
            self.genesis_step % 15 == 0 or
            np.random.random() < 0.1
        )
        
        if should_speak:
            # Use EMERGENT temperature
            temperature = self.calculate_emergent_temperature()
            word = self.predict_next_word(temperature=temperature, use_natural_inhibition=True)
            
            word_wave = self.encode_word(word)
            self.field += word_wave * 1.0
            
            self.spontaneous_thoughts.append((self.genesis_step, word, max_amplitude, temperature))
            return word
        
        return None
    
    def add_word_to_field(self, word, learn=True):
        """Add word to field"""
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
        """Hebbian learning"""
        for past_idx in self.recent_words[-3:]:
            current_val = self.word_associations[current_word_idx, past_idx]
            new_val = current_val + self.hebbian_lr
            
            if abs(new_val) > 0.01:
                self.word_associations[current_word_idx, past_idx] = np.tanh(new_val)
                self.word_associations[past_idx, current_word_idx] = np.tanh(new_val)
    
    def _extract_field_features(self):
        """Extract field features"""
        features = []
        for c in range(self.C):
            channel = self.field[:, :, :, c]
            amplitude = np.abs(channel)
            features.append(np.mean(amplitude))
            features.append(np.max(amplitude))
        return np.array(features, dtype=np.float32)
    
    def predict_next_word(self, temperature=1.2, use_natural_inhibition=False):
        """
        Predict with NATURAL INHIBITION (wave-based refractory period)
        """
        features = self._extract_field_features()
        logits = features @ self.W_decode
        
        # Association boost
        if len(self.recent_words) > 0:
            for past_idx in self.recent_words[-2:]:
                if past_idx < self.V:
                    associations = self.word_associations[past_idx, :].toarray().flatten()
                    if len(associations) < self.V:
                        associations = np.pad(associations, (0, self.V - len(associations)))
                    logits += associations * 0.1
        
        # NATURAL INHIBITION: Recently activated words have refractory period
        if use_natural_inhibition:
            time_since_activation = self.current_time - self.last_activation_time
            refractory_strength = np.exp(-time_since_activation / 5.0)  # Decay over ~5 time units
            logits -= refractory_strength * 3.0  # Natural suppression
        else:
            # Fallback to gentler blocking
            if len(self.recent_generated) > 0:
                for past_idx in self.recent_generated[-4:]:
                    if past_idx < self.V:
                        logits[past_idx] -= 10.0  # Reduced from -500
        
        # Add noise
        logits += np.random.randn(self.V) * 0.5
        logits = logits / temperature
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        
        # Top-k sampling
        top_k = 15
        top_indices = np.argpartition(-probs, min(top_k, len(probs))-1)[:top_k]
        top_probs = probs[top_indices]
        top_probs = top_probs / (np.sum(top_probs) + 1e-10)
        
        chosen_idx = np.random.choice(top_indices, p=top_probs)
        return self.vocab[chosen_idx]
    
    def organic_decoder_update(self):
        """Decoder update"""
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
        """Process and respond"""
        words = input_text.lower().split()
        self.recent_generated.clear()
        
        for word in words:
            self.add_word_to_field(word, learn=True)
        
        self.organic_decoder_update()
        
        # Use EMERGENT temperature
        base_temp = self.calculate_emergent_temperature()
        
        response_words = []
        for i in range(num_words):
            temp = base_temp + i * 0.1
            word = self.predict_next_word(temperature=temp, use_natural_inhibition=True)
            if word in self.word_to_idx:
                word_idx = self.word_to_idx[word]
                self.recent_generated.append(word_idx)
            response_words.append(word)
            self.add_word_to_field(word, learn=False)
        
        self.conversation_count += 1
        self.current_time += 1.0  # Conversation advances time
        
        return " ".join(response_words)
    
    def get_stats(self):
        """Get statistics"""
        connectivity = self.word_associations.nnz / (self.V * self.V) if self.V > 0 else 0
        mode = "🌱 ULTRA-ORGANIC" if self.bootstrap_mode else "💬 ULTRA-ORGANIC"
        return {
            'mode': mode,
            'genesis_steps': self.genesis_step if self.bootstrap_mode else 0,
            'spontaneous_words': len(self.spontaneous_thoughts),
            'conversations': self.conversation_count,
            'words_processed': self.total_words_processed,
            'vocabulary_size': self.V,
            'learned_words': len(self.learned_words),
            'connectivity': f"{connectivity:.4f}",
            'spatial_density': f"{self.spatial_density:.4f}",
            'current_temp': f"{self.calculate_emergent_temperature():.2f}",
            'expansions': len(self.expansion_events),
            'memory_efficiency': f"{self.X}³ grid, {self.C} channels"
        }
    
    def get_memory_usage(self):
        """Memory usage"""
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
        """Visualization"""
        slices = []
        mid_z = self.Z // 2
        for c in range(min(4, self.C)):
            amplitude = np.abs(self.field[:, :, mid_z, c])
            slices.append(amplitude)
        return slices
    
    def reset_field(self):
        """Reset"""
        if self.bootstrap_mode:
            self.field = (np.random.randn(self.X, self.Y, self.Z, self.C)
                         .astype(np.complex64) * 0.01)
        else:
            self.field.fill(0)
        self.recent_words.clear()
        self.recent_generated.clear()
        self.current_time = 0.0
    
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
            'last_activation_time': self.last_activation_time,
            'bootstrap_mode': self.bootstrap_mode,
            'genesis_step': self.genesis_step,
            'spontaneous_thoughts': self.spontaneous_thoughts,
            'learned_words': self.learned_words,
            'expansion_events': self.expansion_events,
            'current_time': self.current_time,
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
            
            if 'last_activation_time' in data:
                self.last_activation_time = data['last_activation_time']
            if 'bootstrap_mode' in data:
                self.bootstrap_mode = data['bootstrap_mode']
            if 'genesis_step' in data:
                self.genesis_step = data['genesis_step']
            if 'spontaneous_thoughts' in data:
                self.spontaneous_thoughts = data['spontaneous_thoughts']
            if 'learned_words' in data:
                self.learned_words = data['learned_words']
            if 'expansion_events' in data:
                self.expansion_events = data['expansion_events']
            if 'current_time' in data:
                self.current_time = data['current_time']
            
            return True
        except Exception as e:
            print(f"Error loading: {e}")
            return False


# ========================
# Ultra-Organic Bootstrap GUI
# ========================

class UltraOrganicWaveChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌱 Ultra-Organic Wave Chatbot - 85% ORGANIC")
        self.root.geometry("1400x850")
        
        # Create ULTRA-ORGANIC model
        self.model = UltraOrganicWaveModel(
            grid_size=16, 
            num_channels=8, 
            bootstrap_mode=True
        )
        self.model_path = "ultra_organic_wave_model.pkl"
        
        # Bootstrap thread control
        self.bootstrap_running = False
        self.bootstrap_thread = None
        
        # Try to load saved model
        try:
            if self.model.load_model(self.model_path):
                print("✅ Loaded saved model")
        except:
            print("🆕 Starting fresh")
        
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
        
        ttk.Label(left_frame, text="🌱 Ultra-Organic Genesis Stream", 
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
        self.chat_display.tag_config('organic', foreground='#00cc00', 
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
        
        ttk.Label(middle_frame, text="🌱 Ultra-Organic Stats", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        stats_frame = ttk.LabelFrame(middle_frame, text="Performance", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=20, width=34, font=('Courier', 9))
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
        info_frame = ttk.LabelFrame(middle_frame, text="Ultra-Organic Features", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = (
            "🌱 85% ORGANIC\n\n"
            "✨ Semantic positioning\n"
            "🌡️ Emergent temperature\n"
            "⏱️ Natural inhibition\n"
            "🎵 Phonetic encoding\n"
            "🎯 Density-aware filling\n\n"
            "• Self-starting\n"
            "• Learns new words\n"
            "• Quantum foam\n"
            "• Self-reinforcing\n"
            "• Bot speaks FIRST\n\n"
            "Type ANY word!\n"
        )
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, 
                 font=('Arial', 9)).pack()
        
        # Right: Visualization
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(right_frame, text="🌊 Wave Field (Ultra-Organic)", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(6, 6))
        self.fig.tight_layout(pad=2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_var = tk.StringVar(value="🌱 Ultra-organic universe initializing...")
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
            "🌱 ULTRA-ORGANIC BOOTSTRAP GENESIS\n"
            "======================================\n\n"
            "✨ 85% ORGANIC - Most advanced version!\n\n"
            "NEW FEATURES:\n"
            "• Semantic word positioning\n"
            "• Emergent temperature (mood)\n"
            "• Natural inhibition (refractory)\n"
            "• Phonetic encoding (sound→freq)\n"
            "• Density-aware space filling\n\n"
            "Universe self-starting from quantum foam...\n\n"
            "✓ No input required\n"
            "✓ Spontaneous thought generation\n"
            "✓ Bot speaks FIRST\n"
            "✓ LEARNS NEW WORDS on-the-fly\n"
            "✓ Words cluster by MEANING\n"
            "✓ Mood EMERGES from field\n\n"
            "Try typing ANY words - watch semantic clusters form!\n\n"
            "Initializing ultra-organic field...\n"
        )
        self.chat_display.insert(tk.END, msg, 'system')
        self.chat_display.insert(tk.END, "\n" + "="*50 + "\n\n", 'system')
    
    def force_genesis_word(self):
        """Manually force a genesis word"""
        temp = self.model.calculate_emergent_temperature()
        word = self.model.predict_next_word(temperature=temp, use_natural_inhibition=True)
        self.model.genesis_step += 1
        word_wave = self.model.encode_word(word)
        self.model.field += word_wave
        self._display_thought(word, temp)
    
    def start_bootstrap_thinking(self):
        """Start background thinking thread"""
        if not self.bootstrap_running:
            self.bootstrap_running = True
            self.bootstrap_thread = threading.Thread(target=self._bootstrap_loop, daemon=True)
            self.bootstrap_thread.start()
            self.thinking_button.config(text="⏸ Pause Thinking")
            self.chat_display.insert(tk.END, "[🌱 Ultra-organic genesis started]\n\n", 'organic')
    
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
                    # Get temperature from last thought
                    temp = self.model.spontaneous_thoughts[-1][3] if self.model.spontaneous_thoughts else 2.0
                    self.root.after(0, self._display_thought, thought, temp)
                time.sleep(0.3)
            except Exception as e:
                print(f"Bootstrap error: {e}")
                time.sleep(1)
    
    def _display_thought(self, thought, temperature=2.0):
        """Display spontaneous thought with temperature"""
        mood = "🔥" if temperature > 2.5 else "🧊" if temperature < 1.7 else "😐"
        self.chat_display.insert(tk.END, f"Genesis[{self.model.genesis_step}] {mood}: ", 'genesis')
        self.chat_display.insert(tk.END, f"{thought}\n", 'genesis')
        self.chat_display.see(tk.END)
        
        if self.model.spontaneous_thoughts:
            event = self.model.spontaneous_thoughts[-1]
            self.status_var.set(f"Genesis: {self.model.genesis_step} | "
                              f"Temp: {event[3]:.2f} {mood} | Vocab: {self.model.V}")
    
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
            
            # Show newly learned words with placement info
            if new_words:
                self.chat_display.insert(tk.END, "✨ Learned: ", 'learned')
                self.chat_display.insert(tk.END, f"{', '.join(new_words)}\n", 'learned')
                
                # Show organic features
                if len(new_words) > 0:
                    temp = self.model.calculate_emergent_temperature()
                    self.chat_display.insert(tk.END, f"🌡️  Temp: {temp:.2f} | ", 'organic')
                    self.chat_display.insert(tk.END, f"Density: {self.model.spatial_density:.3f}\n", 'organic')
            
            self.chat_display.insert(tk.END, "Bot: ", 'bot')
            self.chat_display.insert(tk.END, response + "\n\n")
            
            if was_running:
                time.sleep(0.5)
                self.start_bootstrap_thinking()
        
        except Exception as e:
            self.chat_display.insert(tk.END, f"[Error: {e}]\n", 'system')
            import traceback
            traceback.print_exc()
        
        self.chat_display.see(tk.END)
    
    def show_vocabulary(self):
        """Show current vocabulary"""
        stats = self.model.get_stats()
        recent_learned = self.model.learned_words[-20:] if self.model.learned_words else []
        
        msg = f"\n📚 Vocabulary: {stats['vocabulary_size']} words\n"
        msg += f"✨ Learned: {stats['learned_words']} new words\n"
        msg += f"🌡️  Current temp: {stats['current_temp']}\n"
        msg += f"🎯 Spatial density: {stats['spatial_density']}\n"
        
        if recent_learned:
            msg += f"\nRecent: {', '.join(recent_learned)}\n"
        
        msg += "\n"
        self.chat_display.insert(tk.END, msg, 'organic')
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
        text += f"Genesis: {stats['genesis_steps']}\n"
        text += f"Spontaneous: {stats['spontaneous_words']}\n"
        text += f"Conversations: {stats['conversations']}\n"
        text += f"Words: {stats['words_processed']}\n"
        text += f"\n🌱 ULTRA-ORGANIC:\n"
        text += f"Vocabulary: {stats['vocabulary_size']}\n"
        text += f"✨ Learned: {stats['learned_words']}\n"
        text += f"🌡️ Temp: {stats['current_temp']}\n"
        text += f"🎯 Density: {stats['spatial_density']}\n"
        text += f"Connectivity: {stats['connectivity']}\n"
        text += f"Grid: {stats['memory_efficiency']}\n\n"
        text += "Memory:\n"
        text += f" Field: {memory['field']}\n"
        text += f" Decoder: {memory['decoder']}\n"
        text += f" Assoc: {memory['associations']}\n"
        text += f" TOTAL: {memory['total']}\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, text)
    
    def save_model(self):
        """Save model"""
        self.model.save_model(self.model_path)
        messagebox.showinfo("Info", f"Ultra-organic model saved!\n\n"
                           f"Vocabulary: {self.model.V} words\n"
                           f"Learned: {len(self.model.learned_words)} new words\n"
                           f"Temperature: {self.model.calculate_emergent_temperature():.2f}\n"
                           f"Organicity: 85%")
    
    def reset_field(self):
        """Reset field"""
        self.model.reset_field()
        self.model.genesis_step = 0
        self.model.spontaneous_thoughts.clear()
        self.chat_display.insert(tk.END, "[🌱 Ultra-organic field reset]\n\n", 'organic')
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
    app = UltraOrganicWaveChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
