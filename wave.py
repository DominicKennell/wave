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
# COMPLETE MULTIMODAL SPINNING WAVE LANGUAGE MODEL
# WITH ROBUST NaN PROTECTION
# ========================

class MultimodalOutputSystem:
    """
    Allows genesis to EXPRESS its internal state externally
    through light and sound emissions
    """
    
    def __init__(self, model):
        self.model = model
        
        # Output devices
        self.light_output_enabled = True
        self.sound_output_enabled = True
        
        # Output history
        self.light_emissions = []
        self.sound_emissions = []
        
        # Thresholds for spontaneous output
        self.light_emission_threshold = 0.8
        self.sound_emission_threshold = 0.7
        
    def check_for_spontaneous_output(self):
        """Check if genesis wants to OUTPUT light/sound"""
        outputs = {}
        
        # LIGHT EMISSION
        visual_energy = np.sum(np.abs(self.model.visual_field)**2)
        
        # NaN protection
        if np.isnan(visual_energy) or np.isinf(visual_energy):
            visual_energy = 0.0
        
        if visual_energy > self.light_emission_threshold:
            # Extract dominant color
            rgb_totals = np.sum(np.abs(self.model.visual_field[:,:,:,:3]), axis=(0,1,2))
            
            # NaN protection
            if np.any(np.isnan(rgb_totals)) or np.any(np.isinf(rgb_totals)):
                rgb_totals = np.nan_to_num(rgb_totals, nan=0.0)
            
            rgb_normalized = rgb_totals / (np.sum(rgb_totals) + 1e-6)
            
            # Map to wavelength
            if rgb_normalized[0] > 0.5:
                wavelength = 650
                color_name = "red"
            elif rgb_normalized[2] > 0.5:
                wavelength = 470
                color_name = "blue"
            elif rgb_normalized[1] > 0.5:
                wavelength = 550
                color_name = "green"
            else:
                wavelength = int(650 * rgb_normalized[0] + 
                               550 * rgb_normalized[1] + 
                               470 * rgb_normalized[2])
                color_name = "mixed"
            
            intensity = np.clip(visual_energy / 2.0, 0, 1)
            
            outputs['light'] = {
                'wavelength': wavelength,
                'color': color_name,
                'intensity': intensity,
                'rgb': rgb_normalized
            }
            
            self.light_emissions.append({
                'time': self.model.current_time,
                'wavelength': wavelength,
                'intensity': intensity
            })
        
        # SOUND EMISSION
        auditory_energy = np.sum(np.abs(self.model.auditory_field)**2)
        
        # NaN protection
        if np.isnan(auditory_energy) or np.isinf(auditory_energy):
            auditory_energy = 0.0
        
        if auditory_energy > self.sound_emission_threshold:
            # Extract dominant frequency
            band_totals = np.sum(np.abs(self.model.auditory_field), axis=(0,1,2))
            
            # NaN protection
            if np.any(np.isnan(band_totals)) or np.any(np.isinf(band_totals)):
                band_totals = np.nan_to_num(band_totals, nan=0.0)
            
            band_normalized = band_totals / (np.sum(band_totals) + 1e-6)
            
            if band_normalized[0] > 0.5:
                frequency = 100
                sound_name = "low rumble"
            elif band_normalized[1] > 0.5:
                frequency = 500
                sound_name = "hum"
            elif band_normalized[2] > 0.5:
                frequency = 2000
                sound_name = "tone"
            else:
                frequency = 5000
                sound_name = "high pitch"
            
            intensity = np.clip(auditory_energy / 2.0, 0, 1)
            
            outputs['sound'] = {
                'frequency': frequency,
                'name': sound_name,
                'intensity': intensity,
                'bands': band_normalized
            }
            
            self.sound_emissions.append({
                'time': self.model.current_time,
                'frequency': frequency,
                'intensity': intensity
            })
        
        return outputs


class CompleteMultimodalSpinningWaveModel:
    """
    COMPLETE SYSTEM WITH ROBUST NaN PROTECTION:
    - Light perception & emission
    - Sound perception & emission
    - 4D spinning waves with helicity
    - Grounded vocabulary learning
    - Synesthetic genesis
    - Vortex detection
    - Angular momentum tracking
    """
    
    def __init__(self, grid_size=16, num_semantic_channels=8, 
                 num_visual_channels=4, num_auditory_channels=4,
                 bootstrap_mode=False):
        self.X, self.Y, self.Z = grid_size, grid_size, grid_size
        self.C = num_semantic_channels
        self.C_visual = num_visual_channels
        self.C_auditory = num_auditory_channels
        self.bootstrap_mode = bootstrap_mode
        
        # Fields
        if bootstrap_mode:
            self.field = (np.random.randn(self.X, self.Y, self.Z, self.C)
                         .astype(np.complex64) * 0.01)
        else:
            self.field = np.zeros((self.X, self.Y, self.Z, self.C), dtype=np.complex64)
        
        self.visual_field = np.zeros((self.X, self.Y, self.Z, self.C_visual), dtype=np.complex64)
        self.auditory_field = np.zeros((self.X, self.Y, self.Z, self.C_auditory), dtype=np.complex64)
        
        # Wave parameters
        self.dt = 0.15
        self.wave_speed = 0.85
        self.damping = 0.96
        self.context_decay = 0.88
        self.rotation_speed = 0.3
        
        # Vocabulary
        self.vocab = self._build_core_vocab()
        self.V = len(self.vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Semantic space
        self.word_positions = self._initialize_semantic_space()
        self.word_frequencies = self._initialize_phonetic_frequencies()
        self.word_acoustics = self._initialize_acoustic_features()
        
        # SENSORY GROUNDING
        self.word_light_associations = np.zeros((self.V, 3), dtype=np.float32)
        self.word_sound_associations = np.zeros((self.V, 4), dtype=np.float32)
        
        # SPINNING WAVE PROPERTIES
        self.word_spins = np.zeros(self.V, dtype=np.float32)
        self.word_helicity = np.zeros(self.V, dtype=np.float32)
        self.word_orbital_l = np.zeros(self.V, dtype=np.int32)
        self.word_orbital_m = np.zeros(self.V, dtype=np.int32)
        self._initialize_word_spins()
        
        # Transduction matrices
        self.visual_to_semantic = np.random.randn(self.C_visual, self.C) * 0.1
        self.auditory_to_semantic = np.random.randn(self.C_auditory, self.C) * 0.15
        self.crossmodal_binding = np.random.randn(self.C_visual, self.C_auditory) * 0.05
        
        # Multimodal weights
        self.visual_weight = 0.5
        self.auditory_weight = 0.5
        self.integration_history = []
        
        # Association tracking
        self.word_associations = lil_matrix((self.V, self.V), dtype=np.float32)
        
        # Decoder
        feature_dim = self.C * 2
        self.W_decode = np.random.randn(feature_dim, self.V).astype(np.float32) * 0.03
        
        # Learning rates
        self.hebbian_lr = 0.003
        self.decoder_lr = 0.005
        
        # Memory with timestamps
        self.recent_words = []
        self.recent_generated = []
        self.word_usage_count = np.zeros(self.V, dtype=np.uint32)
        self.last_activation_time = np.zeros(self.V, dtype=np.float32)
        self.current_time = 0.0
        
        # Statistics
        self.total_words_processed = 0
        self.conversation_count = 0
        self.learned_words = []
        self.invented_words = []
        
        # Genesis
        self.genesis_step = 0
        self.spontaneous_thoughts = []
        
        # Vortex tracking
        self.detected_vortices = []
        self.vortex_history = []
        
        # Angular momentum history
        self.angular_momentum_history = []
        
        # Output system
        self.output_system = MultimodalOutputSystem(self)
        
        print(f"🌈🔊🌀 COMPLETE Multimodal Spinning Wave Model initialized")
        print(f"  Visual: {self.C_visual} channels")
        print(f"  Auditory: {self.C_auditory} channels")
        print(f"  Semantic: {self.C} channels")
        print(f"  4D spinning waves: ENABLED")
        print(f"  NaN protection: ENABLED")
    
    def _build_core_vocab(self):
        """Core vocabulary"""
        return [
            "hello", "hi", "hey", "goodbye", "bye", "thanks", "please", "sorry",
            "welcome", "yes", "no", "ok", "okay", "sure", "i", "you", "we", "they",
            "he", "she", "it", "me", "my", "your", "am", "is", "are", "was", "were",
            "be", "have", "has", "had", "do", "does", "did", "can", "could", "will",
            "would", "should", "want", "need", "like", "love", "know", "think", "feel",
            "see", "say", "tell", "ask", "help", "try", "go", "come", "get", "make",
            "take", "give", "find", "use", "good", "bad", "great", "nice", "fine",
            "well", "happy", "sad", "big", "small", "new", "old", "day", "time",
            "thing", "way", "home", "work", "people", "friend", "name", "question",
            "answer", "what", "who", "where", "when", "why", "how", "the", "a", "and",
            "or", "but", "if", "to", "of", "in", "on", "at", "not", "very", "just",
            "so", "now", "here", "there", "red", "blue", "green", "yellow", "bright",
            "dark", "light", "sound", "warm", "cold", "hot"
        ]
    
    def _initialize_semantic_space(self):
        """Initialize semantic clusters"""
        positions = []
        
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
            'colors': ['red', 'blue', 'green', 'yellow'],
            'perception': ['bright', 'dark', 'light', 'sound'],
            'temperature': ['warm', 'cold', 'hot'],
            'function': ['the', 'a', 'and', 'or', 'but', 'if', 'to', 'of', 'in', 'on', 'at', 'not', 'very', 'just', 'so'],
        }
        
        cluster_centers = {}
        num_clusters = len(clusters)
        
        for i, cluster_name in enumerate(clusters.keys()):
            phi = i * np.pi * (3. - np.sqrt(5.))
            y = 1 - (i / float(num_clusters - 1)) * 2 if num_clusters > 1 else 0
            radius = np.sqrt(1 - y * y)
            theta = phi
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            center = np.array([
                (x + 1) * 0.4 * self.X,
                (y + 1) * 0.4 * self.Y,
                (z + 1) * 0.4 * self.Z
            ])
            cluster_centers[cluster_name] = center
        
        for word in self.vocab:
            cluster_name = None
            for cname, words in clusters.items():
                if word in words:
                    cluster_name = cname
                    break
            
            if cluster_name:
                center = cluster_centers[cluster_name]
                offset = np.random.randn(3) * 1.5
                position = center + offset
            else:
                position = np.array([self.X/2, self.Y/2, self.Z/2]) + np.random.randn(3) * 2
            
            position = np.clip(position, 0, [self.X-1, self.Y-1, self.Z-1])
            positions.append(position)
        
        return np.array(positions, dtype=np.float32)
    
    def _initialize_phonetic_frequencies(self):
        """Derive frequencies from phonetics"""
        frequencies = []
        for word in self.vocab:
            vowels = sum(1 for c in word if c in 'aeiou')
            consonants = len(word) - vowels
            length = len(word)
            
            base_freq = 0.3 + (vowels / max(length, 1)) * 0.3
            
            word_freqs = []
            for c in range(self.C):
                if c < 2:
                    freq = base_freq + (vowels / max(length, 1)) * 0.2
                elif c < 4:
                    freq = base_freq + (consonants / max(length, 1)) * 0.2
                elif c < 6:
                    freq = base_freq + (length / 10.0) * 0.2
                else:
                    freq = base_freq + np.random.rand() * 0.3
                word_freqs.append(np.clip(freq, 0.2, 0.9))
            
            frequencies.append(word_freqs)
        
        return np.array(frequencies, dtype=np.float32)
    
    def _initialize_acoustic_features(self):
        """Map words to acoustic features"""
        acoustics = []
        
        for word in self.vocab:
            vowel_resonance = sum(1 for c in word if c in 'aeiou') / max(len(word), 1)
            
            has_front_vowels = any(c in 'ie' for c in word)
            has_back_vowels = any(c in 'ou' for c in word)
            has_fricatives = any(c in 'fvsz' for c in word)
            
            duration = len(word) / 10.0
            
            acoustic_vec = np.array([
                0.3 + vowel_resonance * 0.4,
                0.5 if has_front_vowels else (0.3 if has_back_vowels else 0.4),
                0.2 + (0.5 if has_fricatives else 0.0),
                np.clip(duration, 0.2, 0.8)
            ], dtype=np.float32)
            
            acoustics.append(acoustic_vec)
        
        return np.array(acoustics, dtype=np.float32)
    
    def _initialize_word_spins(self):
        """Assign spin/helicity based on semantics"""
        positive_words = ['good', 'great', 'happy', 'love', 'yes', 'nice', 'well', 'bright', 'warm']
        negative_words = ['bad', 'sad', 'no', 'not', 'dark', 'cold']
        
        subject_words = ['i', 'you', 'he', 'she', 'we', 'they']
        object_words = ['me', 'him', 'her', 'us', 'them']
        
        action_words = ['go', 'come', 'make', 'take', 'give', 'help', 'do']
        question_words = ['what', 'who', 'where', 'when', 'why', 'how']
        
        for i, word in enumerate(self.vocab):
            # Spin: emotional valence
            if word in positive_words:
                self.word_spins[i] = +1.0
                self.word_helicity[i] = +1.0
            elif word in negative_words:
                self.word_spins[i] = -1.0
                self.word_helicity[i] = -1.0
            else:
                self.word_spins[i] = 0.0
                self.word_helicity[i] = 0.0
            
            # Orbital: grammatical role
            if word in subject_words:
                self.word_orbital_l[i] = 1
                self.word_orbital_m[i] = +1
            elif word in object_words:
                self.word_orbital_l[i] = 1
                self.word_orbital_m[i] = -1
            elif word in action_words:
                self.word_orbital_l[i] = 2
                self.word_orbital_m[i] = 0
            elif word in question_words:
                self.word_orbital_l[i] = 3
                self.word_orbital_m[i] = np.random.randint(-3, 4)
            else:
                self.word_orbital_l[i] = 0
                self.word_orbital_m[i] = 0
    
    # ========== LIGHT PERCEPTION ==========
    
    def perceive_light(self, wavelength, intensity=1.0, position=None):
        """Add light stimulus to visual field"""
        if position is None:
            position = np.array([self.X/2, self.Y/2, self.Z/2])
        
        cx, cy, cz = position
        rgb_activation = self._wavelength_to_rgb(wavelength, intensity)
        luminance = intensity
        
        x = np.arange(self.X, dtype=np.float32)
        y = np.arange(self.Y, dtype=np.float32)
        z = np.arange(self.Z, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        sigma = 2.0
        dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
        envelope = np.exp(-dist_sq / (2 * sigma**2))
        
        light_freq = 3e8 / (wavelength * 1e-9) / 1e15
        
        for c in range(self.C_visual):
            if c < 3:
                amplitude = rgb_activation[c]
            else:
                amplitude = luminance
            
            phase = light_freq * 2 * np.pi * (c + 1) / self.C_visual
            self.visual_field[:, :, :, c] += amplitude * envelope * np.exp(1j * phase)
        
        return rgb_activation
    
    def _wavelength_to_rgb(self, wavelength, intensity=1.0):
        """Convert wavelength to RGB"""
        rgb = np.zeros(3)
        
        if 380 <= wavelength < 440:
            rgb[2] = intensity * (-(wavelength - 440) / (440 - 380))
            rgb[0] = intensity * 0.3
        elif 440 <= wavelength < 490:
            rgb[2] = intensity
            rgb[1] = intensity * ((wavelength - 440) / (490 - 440))
        elif 490 <= wavelength < 510:
            rgb[1] = intensity
            rgb[2] = intensity * (-(wavelength - 510) / (510 - 490))
        elif 510 <= wavelength < 580:
            rgb[1] = intensity
            rgb[0] = intensity * ((wavelength - 510) / (580 - 510))
        elif 580 <= wavelength < 645:
            rgb[0] = intensity
            rgb[1] = intensity * (-(wavelength - 645) / (645 - 580))
        elif 645 <= wavelength <= 780:
            rgb[0] = intensity
        
        return rgb
    
    # ========== SOUND PERCEPTION ==========
    
    def perceive_sound(self, frequency_hz, intensity=1.0, position=None):
        """Add sound stimulus to auditory field"""
        if position is None:
            position = np.array([self.X/2, self.Y/2, self.Z/2])
        
        cx, cy, cz = position
        band_activation = self._frequency_to_bands(frequency_hz, intensity)
        
        x = np.arange(self.X, dtype=np.float32)
        y = np.arange(self.Y, dtype=np.float32)
        z = np.arange(self.Z, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        sigma = 2.5
        dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
        envelope = np.exp(-dist_sq / (2 * sigma**2))
        
        scaled_freq = np.log10(frequency_hz) / 4.0
        
        for c in range(self.C_auditory):
            amplitude = band_activation[c]
            phase = scaled_freq * 2 * np.pi * (c + 1)
            self.auditory_field[:, :, :, c] += amplitude * envelope * np.exp(1j * phase)
        
        return band_activation
    
    def _frequency_to_bands(self, freq_hz, intensity=1.0):
        """Map frequency to auditory bands"""
        bands = np.zeros(4)
        
        if 20 <= freq_hz < 250:
            bands[0] = intensity
        elif 250 <= freq_hz < 2000:
            bands[1] = intensity * (1 - abs(np.log10(freq_hz) - np.log10(700)) / 0.9)
            bands[0] = intensity * 0.3
        elif 2000 <= freq_hz < 6000:
            bands[2] = intensity
            bands[1] = intensity * 0.4
        elif 6000 <= freq_hz <= 20000:
            bands[3] = intensity
            bands[2] = intensity * 0.3
        
        return np.clip(bands, 0, 1)
    
    # ========== CROSS-MODAL BINDING ==========
    
    def bind_visual_auditory(self):
        """Cross-modal integration"""
        visual_features = np.zeros((self.X, self.Y, self.Z, self.C_visual))
        auditory_features = np.zeros((self.X, self.Y, self.Z, self.C_auditory))
        
        for c in range(self.C_visual):
            visual_features[:, :, :, c] = np.abs(self.visual_field[:, :, :, c])
        
        for c in range(self.C_auditory):
            auditory_features[:, :, :, c] = np.abs(self.auditory_field[:, :, :, c])
        
        # NaN protection
        if np.any(np.isnan(visual_features)) or np.any(np.isinf(visual_features)):
            visual_features = np.nan_to_num(visual_features, nan=0.0, posinf=1.0, neginf=0.0)
        
        if np.any(np.isnan(auditory_features)) or np.any(np.isinf(auditory_features)):
            auditory_features = np.nan_to_num(auditory_features, nan=0.0, posinf=1.0, neginf=0.0)
        
        for x in range(self.X):
            for y in range(self.Y):
                for z in range(self.Z):
                    v_vec = visual_features[x, y, z, :]
                    a_vec = auditory_features[x, y, z, :]
                    
                    a_modulation = v_vec @ self.crossmodal_binding
                    
                    # NaN protection
                    if np.any(np.isnan(a_modulation)) or np.any(np.isinf(a_modulation)):
                        a_modulation = np.nan_to_num(a_modulation, nan=0.0)
                    
                    for c in range(self.C_auditory):
                        modulation_value = 1 + a_modulation[c] * 0.1
                        if not np.isnan(modulation_value) and not np.isinf(modulation_value):
                            self.auditory_field[x, y, z, c] *= modulation_value
                    
                    v_modulation = a_vec @ self.crossmodal_binding.T
                    
                    # NaN protection
                    if np.any(np.isnan(v_modulation)) or np.any(np.isinf(v_modulation)):
                        v_modulation = np.nan_to_num(v_modulation, nan=0.0)
                    
                    for c in range(self.C_visual):
                        modulation_value = 1 + v_modulation[c] * 0.1
                        if not np.isnan(modulation_value) and not np.isinf(modulation_value):
                            self.visual_field[x, y, z, c] *= modulation_value
    
    def transduce_to_semantic(self):
        """Convert sensory fields to semantic field"""
        visual_energy = np.zeros((self.X, self.Y, self.Z, self.C_visual))
        auditory_energy = np.zeros((self.X, self.Y, self.Z, self.C_auditory))
        
        for c in range(self.C_visual):
            visual_energy[:, :, :, c] = np.abs(self.visual_field[:, :, :, c])
        
        for c in range(self.C_auditory):
            auditory_energy[:, :, :, c] = np.abs(self.auditory_field[:, :, :, c])
        
        # NaN protection
        if np.any(np.isnan(visual_energy)) or np.any(np.isinf(visual_energy)):
            visual_energy = np.nan_to_num(visual_energy, nan=0.0, posinf=1.0, neginf=0.0)
        
        if np.any(np.isnan(auditory_energy)) or np.any(np.isinf(auditory_energy)):
            auditory_energy = np.nan_to_num(auditory_energy, nan=0.0, posinf=1.0, neginf=0.0)
        
        for x in range(self.X):
            for y in range(self.Y):
                for z in range(self.Z):
                    v_vec = visual_energy[x, y, z, :]
                    a_vec = auditory_energy[x, y, z, :]
                    
                    semantic_from_visual = v_vec @ self.visual_to_semantic
                    semantic_from_auditory = a_vec @ self.auditory_to_semantic
                    
                    # NaN protection
                    if np.any(np.isnan(semantic_from_visual)) or np.any(np.isinf(semantic_from_visual)):
                        semantic_from_visual = np.nan_to_num(semantic_from_visual, nan=0.0)
                    
                    if np.any(np.isnan(semantic_from_auditory)) or np.any(np.isinf(semantic_from_auditory)):
                        semantic_from_auditory = np.nan_to_num(semantic_from_auditory, nan=0.0)
                    
                    integrated = (self.visual_weight * semantic_from_visual + 
                                 self.auditory_weight * semantic_from_auditory)
                    
                    for c in range(self.C):
                        if not np.isnan(integrated[c]) and not np.isinf(integrated[c]):
                            self.field[x, y, z, c] += integrated[c] * 0.2
        
        v_power = np.sum(np.abs(self.visual_field)**2)
        a_power = np.sum(np.abs(self.auditory_field)**2)
        
        # NaN protection
        if np.isnan(v_power) or np.isinf(v_power):
            v_power = 0.0
        if np.isnan(a_power) or np.isinf(a_power):
            a_power = 0.0
        
        self.integration_history.append((v_power, a_power))
        
        if len(self.integration_history) > 10:
            recent = self.integration_history[-10:]
            avg_visual = np.mean([x[0] for x in recent])
            avg_auditory = np.mean([x[1] for x in recent])
            
            # NaN protection
            if np.isnan(avg_visual) or np.isinf(avg_visual):
                avg_visual = 0.0
            if np.isnan(avg_auditory) or np.isinf(avg_auditory):
                avg_auditory = 0.0
            
            total = avg_visual + avg_auditory + 1e-6
            self.visual_weight = 0.3 + 0.4 * (avg_visual / total)
            self.auditory_weight = 0.3 + 0.4 * (avg_auditory / total)
    
    # ========== 4D SPINNING WAVE ENCODING ==========
    
    def encode_word_spinning(self, word):
        """Create SPINNING wave packet with 4D rotation"""
        word = word.lower()
        if word not in self.word_to_idx:
            idx = self.learn_new_word(word)
        else:
            idx = self.word_to_idx[word]
        
        cx, cy, cz = self.word_positions[idx]
        frequencies = self.word_frequencies[idx]
        
        spin = self.word_spins[idx]
        helicity = self.word_helicity[idx]
        l = self.word_orbital_l[idx]
        m = self.word_orbital_m[idx]
        
        x = np.arange(self.X, dtype=np.float32)
        y = np.arange(self.Y, dtype=np.float32)
        z = np.arange(self.Z, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        X_shifted = X - cx
        Y_shifted = Y - cy
        Z_shifted = Z - cz
        
        r = np.sqrt(X_shifted**2 + Y_shifted**2 + Z_shifted**2) + 1e-6
        theta = np.arccos(np.clip(Z_shifted / r, -1, 1))
        phi = np.arctan2(Y_shifted, X_shifted)
        
        sigma = 2.0
        radial_envelope = np.exp(-r**2 / (2 * sigma**2))
        
        # Orbital angular momentum (spherical harmonics)
        if l == 0:
            angular = np.ones_like(theta)
        elif l == 1:
            if m == 0:
                angular = np.cos(theta)
            elif m == 1:
                angular = np.sin(theta) * np.exp(1j * phi)
            else:
                angular = np.sin(theta) * np.exp(-1j * phi)
        elif l == 2:
            if m == 0:
                angular = 3 * np.cos(theta)**2 - 1
            elif m == 1:
                angular = np.sin(theta) * np.cos(theta) * np.exp(1j * phi)
            elif m == -1:
                angular = np.sin(theta) * np.cos(theta) * np.exp(-1j * phi)
            elif m == 2:
                angular = np.sin(theta)**2 * np.exp(2j * phi)
            else:
                angular = np.sin(theta)**2 * np.exp(-2j * phi)
        else:
            angular = np.sin(theta)**l * np.exp(1j * m * phi)
        
        # TIME-DEPENDENT ROTATION (4D)
        rotation_phase = self.current_time * self.rotation_speed * m
        
        wave_packet = np.zeros((self.X, self.Y, self.Z, self.C), dtype=np.complex64)
        amplitude = 0.3
        
        for c in range(self.C):
            phase_base = 2 * np.pi * idx / max(self.V, 1) + c * np.pi / self.C
            
            # SPINNING PHASE: orbital + helicity + time + channel rotation
            spinning_phase = (
                phase_base +
                frequencies[c] * 2 * np.pi +
                rotation_phase +
                helicity * phi
            )
            
            # SPIN creates rotation through CHANNEL SPACE (4th dimension)
            if spin != 0:
                polarization_phase = spin * c * np.pi / self.C
                spinning_phase += polarization_phase
            
            wave_packet[:, :, :, c] = (
                amplitude * 
                radial_envelope * 
                angular * 
                np.exp(1j * spinning_phase)
            )
        
        # NaN protection
        if np.any(np.isnan(wave_packet)) or np.any(np.isinf(wave_packet)):
            wave_packet = np.nan_to_num(wave_packet, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return wave_packet
    
    # ========== WAVE PROPAGATION WITH ROTATION (NaN PROTECTED) ==========
    
    def propagate_spinning_waves(self, steps=2):
        """Propagate waves with 4D rotation (NaN PROTECTED)"""
        for step in range(steps):
            for c in range(self.C):
                channel = self.field[:, :, :, c]
                
                # NaN protection
                if np.any(np.isnan(channel)) or np.any(np.isinf(channel)):
                    channel = np.nan_to_num(channel, nan=0.0, posinf=1.0, neginf=-1.0)
                    self.field[:, :, :, c] = channel
                
                channel_fft = fftn(channel)
                
                # Check FFT result
                if np.any(np.isnan(channel_fft)) or np.any(np.isinf(channel_fft)):
                    channel_fft = np.nan_to_num(channel_fft, nan=0.0, posinf=1.0, neginf=-1.0)
                
                kx = np.fft.fftfreq(self.X, d=1.0) * 2 * np.pi
                ky = np.fft.fftfreq(self.Y, d=1.0) * 2 * np.pi
                kz = np.fft.fftfreq(self.Z, d=1.0) * 2 * np.pi
                KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
                
                k_squared = KX**2 + KY**2 + KZ**2
                rotation_term = 1j * self.rotation_speed * (KY * 0.5)
                
                laplacian_fft = -k_squared * channel_fft
                
                # Check laplacian
                if np.any(np.isnan(laplacian_fft)) or np.any(np.isinf(laplacian_fft)):
                    laplacian_fft = np.nan_to_num(laplacian_fft, nan=0.0, posinf=1.0, neginf=-1.0)
                
                updated = (
                    channel + 
                    self.dt**2 * self.wave_speed**2 * ifftn(laplacian_fft + rotation_term * channel_fft)
                ) * self.damping
                
                # Check result
                if np.any(np.isnan(updated)) or np.any(np.isinf(updated)):
                    updated = np.nan_to_num(updated, nan=0.0, posinf=1.0, neginf=-1.0)
                
                self.field[:, :, :, c] = updated
            
            # Nonlinearity with protection
            abs_field = np.abs(self.field)
            angle_field = np.angle(self.field)
            
            if np.any(np.isnan(abs_field)) or np.any(np.isinf(abs_field)):
                abs_field = np.nan_to_num(abs_field, nan=0.0, posinf=1.0, neginf=0.0)
            
            if np.any(np.isnan(angle_field)) or np.any(np.isinf(angle_field)):
                angle_field = np.nan_to_num(angle_field, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.field = np.tanh(abs_field) * np.exp(1j * angle_field)
            
            self.current_time += self.dt
    
    # ========== VORTEX & ANGULAR MOMENTUM ==========
    
    def detect_vortices(self):
        """Find vortex cores (phase singularities)"""
        vortices = []
        mid_z = self.Z // 2
        
        for c in range(self.C):
            field_slice = self.field[:, :, mid_z, c]
            phase = np.angle(field_slice)
            
            # NaN protection
            if np.any(np.isnan(phase)) or np.any(np.isinf(phase)):
                phase = np.nan_to_num(phase, nan=0.0)
            
            for x in range(1, self.X-1):
                for y in range(1, self.Y-1):
                    phases = [
                        phase[x, y],
                        phase[x+1, y],
                        phase[x+1, y+1],
                        phase[x, y+1]
                    ]
                    
                    circulation = 0
                    for i in range(4):
                        dp = phases[(i+1)%4] - phases[i]
                        while dp > np.pi:
                            dp -= 2*np.pi
                        while dp < -np.pi:
                            dp += 2*np.pi
                        circulation += dp
                    
                    if abs(circulation) > 5.0:
                        vortex_charge = int(np.round(circulation / (2*np.pi)))
                        vortices.append({
                            'position': (x, y, mid_z),
                            'channel': c,
                            'charge': vortex_charge,
                            'time': self.current_time
                        })
        
        self.detected_vortices = vortices
        if len(vortices) > 0:
            self.vortex_history.append((self.current_time, len(vortices)))
        
        return vortices
    
    def measure_angular_momentum(self):
        """Calculate total angular momentum (4D rotation)"""
        x = np.arange(self.X, dtype=np.float32)
        y = np.arange(self.Y, dtype=np.float32)
        z = np.arange(self.Z, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        cx, cy, cz = self.X/2, self.Y/2, self.Z/2
        
        L_total = np.zeros(3)
        
        for c in range(self.C):
            psi = self.field[:, :, :, c]
            
            # NaN protection
            if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
                psi = np.nan_to_num(psi, nan=0.0, posinf=1.0, neginf=-1.0)
            
            grad_x = np.gradient(psi, axis=0)
            grad_y = np.gradient(psi, axis=1)
            grad_z = np.gradient(psi, axis=2)
            
            # NaN protection on gradients
            if np.any(np.isnan(grad_x)) or np.any(np.isinf(grad_x)):
                grad_x = np.nan_to_num(grad_x, nan=0.0)
            if np.any(np.isnan(grad_y)) or np.any(np.isinf(grad_y)):
                grad_y = np.nan_to_num(grad_y, nan=0.0)
            if np.any(np.isnan(grad_z)) or np.any(np.isinf(grad_z)):
                grad_z = np.nan_to_num(grad_z, nan=0.0)
            
            rx = X - cx
            ry = Y - cy
            rz = Z - cz
            
            Lx = np.sum(np.real(np.conj(psi) * (ry * grad_z - rz * grad_y)))
            Ly = np.sum(np.real(np.conj(psi) * (rz * grad_x - rx * grad_z)))
            Lz = np.sum(np.real(np.conj(psi) * (rx * grad_y - ry * grad_x)))
            
            # NaN protection
            if np.isnan(Lx) or np.isinf(Lx):
                Lx = 0.0
            if np.isnan(Ly) or np.isinf(Ly):
                Ly = 0.0
            if np.isnan(Lz) or np.isinf(Lz):
                Lz = 0.0
            
            L_total += np.array([Lx, Ly, Lz])
        
        L_magnitude = np.linalg.norm(L_total)
        
        # NaN protection
        if np.isnan(L_magnitude) or np.isinf(L_magnitude):
            L_magnitude = 0.0
        
        # Track history
        self.angular_momentum_history.append((self.current_time, L_magnitude))
        
        return {
            'Lx': L_total[0],
            'Ly': L_total[1],
            'Lz': L_total[2],
            'L_magnitude': L_magnitude
        }
    
    # ========== GROUNDED VOCABULARY LEARNING ==========
    
    def learn_new_word(self, word):
        """Learn word with SENSORY GROUNDING"""
        word = word.lower()
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        
        new_idx = self.V
        self.vocab.append(word)
        self.word_to_idx[word] = new_idx
        self.V += 1
        
        # Semantic positioning
        neighbor_positions = self.find_semantic_neighbors(self.recent_words[-5:], k=3)
        
        if neighbor_positions is not None and len(neighbor_positions) > 0:
            avg_position = np.mean(neighbor_positions, axis=0)
            new_position = avg_position + np.random.randn(3) * 2.0
        else:
            new_position = self._find_low_density_region()
        
        new_position = np.clip(new_position, 0, [self.X-1, self.Y-1, self.Z-1])
        self.word_positions = np.vstack([self.word_positions, new_position])
        
        # Phonetic frequencies
        word_freqs = self._derive_phonetic_frequencies(word)
        self.word_frequencies = np.vstack([self.word_frequencies, word_freqs])
        
        word_acoustic = self._derive_acoustic_features(word)
        self.word_acoustics = np.vstack([self.word_acoustics, word_acoustic])
        
        # SENSORY GROUNDING
        visual_energy = np.sum(np.abs(self.visual_field), axis=(0,1,2))
        
        # NaN protection
        if np.any(np.isnan(visual_energy)) or np.any(np.isinf(visual_energy)):
            visual_energy = np.nan_to_num(visual_energy, nan=0.0)
        
        if np.max(visual_energy) > 0.1:
            rgb_state = visual_energy[:3] / (np.sum(visual_energy[:3]) + 1e-6)
        else:
            rgb_state = np.zeros(3)
        self.word_light_associations = np.vstack([self.word_light_associations, rgb_state])
        
        auditory_energy = np.sum(np.abs(self.auditory_field), axis=(0,1,2))
        
        # NaN protection
        if np.any(np.isnan(auditory_energy)) or np.any(np.isinf(auditory_energy)):
            auditory_energy = np.nan_to_num(auditory_energy, nan=0.0)
        
        if np.max(auditory_energy) > 0.1:
            band_state = auditory_energy / (np.sum(auditory_energy) + 1e-6)
        else:
            band_state = np.zeros(4)
        self.word_sound_associations = np.vstack([self.word_sound_associations, band_state])
        
        # Initialize spin from context
        recent_spin = 0.0
        if len(self.recent_words) > 0:
            for idx in self.recent_words[-3:]:
                if idx < len(self.word_spins):
                    recent_spin += self.word_spins[idx]
            recent_spin /= min(len(self.recent_words), 3)
        
        initial_spin = recent_spin * 0.7 + np.random.randn() * 0.3
        initial_spin = np.clip(initial_spin, -1.0, 1.0)
        
        self.word_spins = np.append(self.word_spins, initial_spin)
        self.word_helicity = np.append(self.word_helicity, initial_spin)
        self.word_orbital_l = np.append(self.word_orbital_l, 0)
        self.word_orbital_m = np.append(self.word_orbital_m, 0)
        
        # Expand structures
        self._expand_association_matrix()
        new_decoder_col = np.random.randn(self.W_decode.shape[0], 1).astype(np.float32) * 0.03
        self.W_decode = np.hstack([self.W_decode, new_decoder_col])
        self.word_usage_count = np.append(self.word_usage_count, 0)
        self.last_activation_time = np.append(self.last_activation_time, self.current_time)
        
        self.learned_words.append(word)
        
        return new_idx
    
    def spontaneous_word_invention(self):
        """Genesis INVENTS new words"""
        temp = self.calculate_emergent_temperature()
        
        # NaN protection
        if np.isnan(temp) or np.isinf(temp):
            temp = 2.0
        
        if temp < 2.8:
            return None
        
        if np.random.random() > 0.02:
            return None
        
        visual_energy = np.sum(np.abs(self.visual_field)**2)
        auditory_energy = np.sum(np.abs(self.auditory_field)**2)
        
        # NaN protection
        if np.isnan(visual_energy) or np.isinf(visual_energy):
            visual_energy = 0.0
        if np.isnan(auditory_energy) or np.isinf(auditory_energy):
            auditory_energy = 0.0
        
        if visual_energy < 0.1 and auditory_energy < 0.1:
            return None
        
        # Generate phonetically plausible word
        consonants = ['b', 'k', 'd', 'f', 'g', 'h', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'z']
        vowels = ['a', 'e', 'i', 'o', 'u']
        
        if np.random.random() < 0.5:
            new_word = (np.random.choice(consonants) + 
                       np.random.choice(vowels) + 
                       np.random.choice(consonants))
        else:
            new_word = (np.random.choice(consonants) + 
                       np.random.choice(vowels) + 
                       np.random.choice(consonants) + 
                       np.random.choice(vowels))
        
        if new_word in self.word_to_idx:
            return None
        
        idx = self.learn_new_word(new_word)
        self.invented_words.append(new_word)
        
        return new_word
    
    def find_semantic_neighbors(self, context_words, k=5):
        """Find semantically similar words"""
        if not context_words:
            return None
        
        similarity_scores = np.zeros(self.V)
        
        for context_idx in context_words:
            if context_idx < self.V:
                assocs = self.word_associations[context_idx, :].toarray().flatten()
                if len(assocs) < self.V:
                    assocs = np.pad(assocs, (0, self.V - len(assocs)), mode='constant')
                elif len(assocs) > self.V:
                    assocs = assocs[:self.V]
                
                # NaN protection
                if np.any(np.isnan(assocs)) or np.any(np.isinf(assocs)):
                    assocs = np.nan_to_num(assocs, nan=0.0)
                
                similarity_scores += assocs
        
        nonzero_indices = np.where(similarity_scores > 0.01)[0]
        if len(nonzero_indices) == 0:
            return None
        
        top_indices = nonzero_indices[np.argsort(-similarity_scores[nonzero_indices])[:k]]
        return self.word_positions[top_indices]
    
    def _find_low_density_region(self):
        """Find region with fewest words"""
        mid_x, mid_y, mid_z = self.X/2, self.Y/2, self.Z/2
        octant_counts = {}
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    octant_counts[(i,j,k)] = 0
        
        for pos in self.word_positions:
            i = 1 if pos[0] > mid_x else 0
            j = 1 if pos[1] > mid_y else 0
            k = 1 if pos[2] > mid_z else 0
            octant_counts[(i,j,k)] += 1
        
        min_octant = min(octant_counts, key=octant_counts.get)
        
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
        """Derive frequencies from phonetics"""
        vowels = sum(1 for c in word if c in 'aeiou')
        consonants = len(word) - vowels
        length = len(word)
        
        has_harsh = any(c in 'kgptbd' for c in word)
        has_soft = any(c in 'lmnr' for c in word)
        has_sibilant = any(c in 'sz' for c in word)
        
        base_freq = 0.3 + (vowels / max(length, 1)) * 0.3
        
        freqs = []
        for c in range(self.C):
            if c == 0:
                freq = base_freq + (vowels / max(length, 1)) * 0.3
            elif c == 1:
                freq = base_freq + (consonants / max(length, 1)) * 0.3
            elif c == 2:
                freq = base_freq + (0.3 if has_harsh else 0.0)
            elif c == 3:
                freq = base_freq + (0.3 if has_soft else 0.0)
            elif c == 4:
                freq = base_freq + (0.3 if has_sibilant else 0.0)
            elif c == 5:
                freq = base_freq + min(length / 10.0, 0.3)
            else:
                freq = base_freq + np.random.rand() * 0.2
            freqs.append(np.clip(freq, 0.2, 0.9))
        
        return np.array(freqs, dtype=np.float32)
    
    def _derive_acoustic_features(self, word):
        """Derive acoustic features"""
        vowel_resonance = sum(1 for c in word if c in 'aeiou') / max(len(word), 1)
        has_front_vowels = any(c in 'ie' for c in word)
        has_back_vowels = any(c in 'ou' for c in word)
        has_fricatives = any(c in 'fvsz' for c in word)
        duration = len(word) / 10.0
        
        return np.array([
            0.3 + vowel_resonance * 0.4,
            0.5 if has_front_vowels else (0.3 if has_back_vowels else 0.4),
            0.2 + (0.5 if has_fricatives else 0.0),
            np.clip(duration, 0.2, 0.8)
        ], dtype=np.float32)
    
    def _expand_association_matrix(self):
        """Expand sparse matrix"""
        old_size = self.word_associations.shape[0]
        new_size = old_size + 1
        new_matrix = lil_matrix((new_size, new_size), dtype=np.float32)
        new_matrix[:old_size, :old_size] = self.word_associations
        self.word_associations = new_matrix
    
    # ========== MULTIMODAL WORD PROCESSING ==========
    
    def perceive_word_multimodal(self, word, modality='both', 
                                  light_wavelength=550, sound_frequency=200):
        """Process word through visual and/or auditory channels"""
        word = word.lower()
        if word not in self.word_to_idx:
            idx = self.learn_new_word(word)
        else:
            idx = self.word_to_idx[word]
        
        position = self.word_positions[idx]
        
        if modality in ['visual', 'both']:
            self.perceive_light(light_wavelength, intensity=1.0, position=position)
        
        if modality in ['auditory', 'both']:
            acoustics = self.word_acoustics[idx]
            for c, amplitude in enumerate(acoustics):
                if c == 0:
                    freq = 100 + amplitude * 300
                elif c == 1:
                    freq = 500 + amplitude * 2000
                elif c == 2:
                    freq = 4000 + amplitude * 4000
                else:
                    freq = sound_frequency
                
                self.perceive_sound(freq, intensity=amplitude, position=position)
        
        self.bind_visual_auditory()
        self.transduce_to_semantic()
        
        word_wave = self.encode_word_spinning(word)
        self.field += word_wave * 0.5
        
        self.propagate_spinning_waves(steps=2)
        
        self.visual_field *= 0.8
        self.auditory_field *= 0.7
        
        return {
            'modality': modality,
            'visual_weight': self.visual_weight,
            'auditory_weight': self.auditory_weight
        }
    
    # ========== LEARNING & PREDICTION (NaN PROTECTED) ==========
    
    def _hebbian_learning(self, current_word_idx):
        """Hebbian learning"""
        for past_idx in self.recent_words[-3:]:
            current_val = self.word_associations[current_word_idx, past_idx]
            new_val = current_val + self.hebbian_lr
            
            # NaN protection
            if np.isnan(new_val) or np.isinf(new_val):
                new_val = 0.0
            
            if abs(new_val) > 0.01:
                self.word_associations[current_word_idx, past_idx] = np.tanh(new_val)
                self.word_associations[past_idx, current_word_idx] = np.tanh(new_val)
    
    def calculate_emergent_temperature(self):
        """Calculate temperature from field entropy"""
        amplitudes = np.abs(self.field).flatten()
        
        # NaN protection
        if np.any(np.isnan(amplitudes)) or np.any(np.isinf(amplitudes)):
            amplitudes = np.nan_to_num(amplitudes, nan=0.0, posinf=1.0, neginf=0.0)
        
        amplitudes = amplitudes / (np.sum(amplitudes) + 1e-10)
        
        # Avoid log(0)
        amplitudes = np.clip(amplitudes, 1e-10, 1.0)
        
        entropy = -np.sum(amplitudes * np.log(amplitudes + 1e-10))
        
        # NaN protection
        if np.isnan(entropy) or np.isinf(entropy):
            entropy = 5.0
        
        normalized_entropy = entropy / 10.0
        temperature = 1.0 + normalized_entropy * 2.0
        temperature = np.clip(temperature, 1.0, 3.5)
        
        return float(temperature)
    
    def _extract_field_features(self):
        """Extract features from field (NaN PROTECTED)"""
        features = []
        for c in range(self.C):
            channel = self.field[:, :, :, c]
            amplitude = np.abs(channel)
            
            # NaN protection
            if np.any(np.isnan(amplitude)) or np.any(np.isinf(amplitude)):
                amplitude = np.nan_to_num(amplitude, nan=0.0, posinf=1.0, neginf=0.0)
            
            mean_amp = np.mean(amplitude)
            max_amp = np.max(amplitude)
            
            # Check for NaN
            if np.isnan(mean_amp) or np.isinf(mean_amp):
                mean_amp = 0.0
            if np.isnan(max_amp) or np.isinf(max_amp):
                max_amp = 0.0
            
            features.append(mean_amp)
            features.append(max_amp)
        
        features_array = np.array(features, dtype=np.float32)
        
        # Final check
        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return features_array
    
    def predict_next_word(self, temperature=1.2, use_natural_inhibition=True):
        """Predict next word with natural inhibition (FULLY NaN PROTECTED)"""
        features = self._extract_field_features()
        
        # Check for NaN in features
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        logits = features @ self.W_decode
        
        # Check for NaN in logits
        if np.any(np.isnan(logits)):
            logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if len(self.recent_words) > 0:
            for past_idx in self.recent_words[-2:]:
                if past_idx < self.V:
                    associations = self.word_associations[past_idx, :].toarray().flatten()
                    if len(associations) < self.V:
                        associations = np.pad(associations, (0, self.V - len(associations)))
                    # Check for NaN in associations
                    if np.any(np.isnan(associations)):
                        associations = np.nan_to_num(associations, nan=0.0)
                    logits += associations * 0.1
        
        if use_natural_inhibition:
            time_since_activation = self.current_time - self.last_activation_time
            # Check for invalid values
            if np.any(np.isnan(time_since_activation)) or np.any(np.isinf(time_since_activation)):
                time_since_activation = np.clip(time_since_activation, 0, 1000)
            refractory_strength = np.exp(-time_since_activation / 5.0)
            # Check for NaN
            if np.any(np.isnan(refractory_strength)):
                refractory_strength = np.nan_to_num(refractory_strength, nan=0.0)
            logits -= refractory_strength * 3.0
        else:
            if len(self.recent_generated) > 0:
                for past_idx in self.recent_generated[-4:]:
                    if past_idx < self.V:
                        logits[past_idx] -= 10.0
        
        # Check logits again
        if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
            logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        logits += np.random.randn(self.V) * 0.5
        
        # Clip temperature to reasonable range
        temperature = np.clip(temperature, 0.1, 5.0)
        logits = logits / temperature
        
        # Clip logits to prevent overflow in exp
        logits = np.clip(logits, -50, 50)
        
        exp_logits = np.exp(logits - np.max(logits))
        
        # Check for NaN/inf after exp
        if np.any(np.isnan(exp_logits)) or np.any(np.isinf(exp_logits)):
            exp_logits = np.nan_to_num(exp_logits, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure sum is positive
        sum_exp = np.sum(exp_logits)
        if sum_exp <= 0 or np.isnan(sum_exp) or np.isinf(sum_exp):
            # Fallback: uniform distribution
            probs = np.ones(self.V) / self.V
        else:
            probs = exp_logits / sum_exp
        
        # Final NaN check
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs / (np.sum(probs) + 1e-10)
        
        # Ensure probs sum to 1 (approximately)
        prob_sum = np.sum(probs)
        if prob_sum <= 0 or np.isnan(prob_sum):
            # Extreme fallback: uniform
            probs = np.ones(self.V) / self.V
        else:
            probs = probs / prob_sum
        
        # Top-k sampling
        top_k = min(15, self.V)
        
        # Get top k indices
        if self.V > top_k:
            top_indices = np.argpartition(-probs, top_k-1)[:top_k]
        else:
            top_indices = np.arange(self.V)
        
        top_probs = probs[top_indices]
        
        # Ensure top_probs are valid
        if np.any(np.isnan(top_probs)) or np.any(top_probs < 0):
            top_probs = np.abs(top_probs)
            top_probs = np.nan_to_num(top_probs, nan=0.0)
        
        # Normalize top_probs
        top_prob_sum = np.sum(top_probs)
        if top_prob_sum <= 0 or np.isnan(top_prob_sum):
            # Uniform over top_k
            top_probs = np.ones(len(top_indices)) / len(top_indices)
        else:
            top_probs = top_probs / top_prob_sum
        
        # Final safety check
        if np.any(np.isnan(top_probs)) or len(top_probs) == 0:
            # Ultimate fallback: return first word
            return self.vocab[0]
        
        try:
            chosen_idx = np.random.choice(top_indices, p=top_probs)
            return self.vocab[chosen_idx]
        except (ValueError, IndexError) as e:
            # If anything still goes wrong, return random word
            print(f"Warning: Fallback to random word due to: {e}")
            return self.vocab[np.random.randint(0, self.V)]
    
    def add_word_to_field(self, word, learn=True):
        """Add word to field"""
        word_wave = self.encode_word_spinning(word)
        self.field *= self.context_decay
        self.field += word_wave
        self.propagate_spinning_waves(steps=2)
        
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
    
    def organic_decoder_update(self):
        """Decoder update (NaN PROTECTED)"""
        if len(self.recent_words) < 2:
            return
        
        features = self._extract_field_features()
        
        for word_idx in self.recent_words[-2:]:
            if word_idx >= self.V:
                continue
            
            target = np.zeros(self.V, dtype=np.float32)
            target[word_idx] = 1.0
            
            logits = features @ self.W_decode
            
            # Protect logits
            if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
                logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            logits = np.clip(logits, -50, 50)
            
            probs = np.exp(logits - np.max(logits))
            
            # Protect probs
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            
            prob_sum = np.sum(probs)
            if prob_sum <= 0 or np.isnan(prob_sum):
                continue
            
            probs = probs / prob_sum
            
            correlation = target - probs
            
            # Protect correlation
            if np.any(np.isnan(correlation)) or np.any(np.isinf(correlation)):
                correlation = np.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            update = self.decoder_lr * np.outer(features, correlation)
            
            # Protect update
            if np.any(np.isnan(update)) or np.any(np.isinf(update)):
                update = np.nan_to_num(update, nan=0.0, posinf=0.1, neginf=-0.1)
            
            self.W_decode += update
            
            # Protect decoder weights
            if np.any(np.isnan(self.W_decode)) or np.any(np.isinf(self.W_decode)):
                self.W_decode = np.nan_to_num(self.W_decode, nan=0.0, posinf=1.0, neginf=-1.0)
    
    def process_input_and_respond(self, input_text, num_words=6, 
                                   light_wavelength=550, sound_frequency=200, 
                                   modality='both'):
        """Process input and generate response"""
        words = input_text.lower().split()
        self.recent_generated.clear()
        
        for word in words:
            self.perceive_word_multimodal(word, modality=modality,
                                         light_wavelength=light_wavelength,
                                         sound_frequency=sound_frequency)
            self.add_word_to_field(word, learn=True)
        
        self.organic_decoder_update()
        
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
        self.current_time += 1.0
        
        return " ".join(response_words)
    
    # ========== SYNESTHETIC GENESIS ==========
    
    def spontaneous_think_synesthetic(self):
        """COMPLETE SYNESTHETIC GENESIS (NaN PROTECTED)"""
        self.genesis_step += 1
        self.current_time += 0.1
        
        # Propagate all fields
        self.propagate_spinning_waves(steps=5)
        self.field *= 0.95
        self.visual_field *= 0.92
        self.auditory_field *= 0.90
        
        # Quantum foam
        noise_semantic = np.random.randn(self.X, self.Y, self.Z, self.C).astype(np.complex64) * 0.01
        noise_visual = np.random.randn(self.X, self.Y, self.Z, self.C_visual).astype(np.complex64) * 0.008
        noise_auditory = np.random.randn(self.X, self.Y, self.Z, self.C_auditory).astype(np.complex64) * 0.008
        
        self.field += noise_semantic
        self.visual_field += noise_visual
        self.auditory_field += noise_auditory
        
        thought = {
            'step': self.genesis_step,
            'events': [],
            'time': self.current_time
        }
        
        # PATTERN 1: Visual → Word + Sound
        visual_energy = np.sum(np.abs(self.visual_field)**2)
        
        # NaN protection
        if np.isnan(visual_energy) or np.isinf(visual_energy):
            visual_energy = 0.0
        
        if visual_energy > 0.5:
            rgb_total = np.sum(np.abs(self.visual_field), axis=(0,1,2))[:3]
            
            # NaN protection
            if np.any(np.isnan(rgb_total)) or np.any(np.isinf(rgb_total)):
                rgb_total = np.nan_to_num(rgb_total, nan=0.0)
            
            if rgb_total[0] > max(rgb_total[1], rgb_total[2]):
                color_word = "red"
                wavelength = 650
                frequency = 100
            elif rgb_total[2] > max(rgb_total[0], rgb_total[1]):
                color_word = "blue"
                wavelength = 470
                frequency = 3000
            else:
                color_word = "green"
                wavelength = 550
                frequency = 500
            
            if color_word in self.word_to_idx:
                word_wave = self.encode_word_spinning(color_word)
                self.field += word_wave * 0.8
                
                self.perceive_sound(frequency, intensity=0.5)
                
                thought['events'].append({
                    'type': 'visual_cascade',
                    'color_word': color_word,
                    'wavelength': wavelength,
                    'sound_frequency': frequency
                })
        
        # PATTERN 2: Auditory → Word + Light
        auditory_energy = np.sum(np.abs(self.auditory_field)**2)
        
        # NaN protection
        if np.isnan(auditory_energy) or np.isinf(auditory_energy):
            auditory_energy = 0.0
        
        if auditory_energy > 0.5:
            band_total = np.sum(np.abs(self.auditory_field), axis=(0,1,2))
            
            # NaN protection
            if np.any(np.isnan(band_total)) or np.any(np.isinf(band_total)):
                band_total = np.nan_to_num(band_total, nan=0.0)
            
            if band_total[0] > max(band_total[1:]):
                sound_word = "dark" if "dark" in self.word_to_idx else "low"
                wavelength = 650
                frequency = 100
            elif band_total[3] > max(band_total[:3]):
                sound_word = "bright" if "bright" in self.word_to_idx else "high"
                wavelength = 470
                frequency = 5000
            else:
                sound_word = "warm" if "warm" in self.word_to_idx else "sound"
                wavelength = 550
                frequency = 500
            
            if sound_word in self.word_to_idx:
                word_wave = self.encode_word_spinning(sound_word)
                self.field += word_wave * 0.8
                
                self.perceive_light(wavelength, intensity=0.5)
                
                thought['events'].append({
                    'type': 'auditory_cascade',
                    'sound_word': sound_word,
                    'light_wavelength': wavelength,
                    'frequency': frequency
                })
        
        # PATTERN 3: Semantic → Sensory
        semantic_energy = np.sum(np.abs(self.field)**2)
        
        # NaN protection
        if np.isnan(semantic_energy) or np.isinf(semantic_energy):
            semantic_energy = 0.0
        
        if semantic_energy > 0.8 or np.random.random() < 0.1:
            temperature = self.calculate_emergent_temperature()
            
            # Try word invention
            if np.random.random() < 0.05:
                invented_word = self.spontaneous_word_invention()
                if invented_word:
                    word = invented_word
                    thought['events'].append({
                        'type': 'word_invention',
                        'word': word
                    })
                else:
                    word = self.predict_next_word(temperature=temperature, use_natural_inhibition=True)
            else:
                word = self.predict_next_word(temperature=temperature, use_natural_inhibition=True)
            
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                
                # Word generates sensory experiences
                color_map = {
                    'red': 650, 'blue': 470, 'green': 550, 'yellow': 590,
                    'bright': 470, 'dark': 700, 'light': 550
                }
                if word in color_map:
                    wavelength = color_map[word]
                    position = self.word_positions[idx]
                    self.perceive_light(wavelength, intensity=0.6, position=position)
                    
                    thought['events'].append({
                        'type': 'word_to_vision',
                        'word': word,
                        'wavelength': wavelength
                    })
                
                sound_map = {
                    'loud': 1000, 'quiet': 100, 'sound': 500, 'voice': 300,
                    'warm': 100, 'cold': 3000, 'hot': 80
                }
                if word in sound_map:
                    frequency = sound_map[word]
                    position = self.word_positions[idx]
                    self.perceive_sound(frequency, intensity=0.6, position=position)
                    
                    thought['events'].append({
                        'type': 'word_to_sound',
                        'word': word,
                        'frequency': frequency
                    })
                
                word_wave = self.encode_word_spinning(word)
                self.field += word_wave * 1.0
                
                thought['events'].append({
                    'type': 'spontaneous_word',
                    'word': word,
                    'temperature': temperature
                })
        
        # Cross-modal binding
        if len(thought['events']) > 0:
            self.bind_visual_auditory()
            self.transduce_to_semantic()
        
        # CHECK FOR OUTPUT EXPRESSION
        outputs = self.output_system.check_for_spontaneous_output()
        if outputs:
            thought['outputs'] = outputs
        
        self.spontaneous_thoughts.append(thought)
        
        return thought if len(thought['events']) > 0 or outputs else None
    
    # ========== UTILITY ==========
    
    def get_stats(self):
        """Get statistics"""
        connectivity = self.word_associations.nnz / (self.V * self.V) if self.V > 0 else 0
        mode = "🌱🌈🔊🌀 COMPLETE" if self.bootstrap_mode else "💬🌈🔊🌀 COMPLETE"
        
        L = self.measure_angular_momentum()
        vortices = len(self.detected_vortices)
        
        return {
            'mode': mode,
            'genesis_steps': self.genesis_step if self.bootstrap_mode else 0,
            'spontaneous_words': len(self.spontaneous_thoughts),
            'conversations': self.conversation_count,
            'words_processed': self.total_words_processed,
            'vocabulary_size': self.V,
            'learned_words': len(self.learned_words),
            'invented_words': len(self.invented_words),
            'connectivity': f"{connectivity:.4f}",
            'current_temp': f"{self.calculate_emergent_temperature():.2f}",
            'angular_momentum': f"{L['L_magnitude']:.2f}",
            'vortices': vortices,
            'visual_weight': f"{self.visual_weight:.2f}",
            'auditory_weight': f"{self.auditory_weight:.2f}",
            'light_emissions': len(self.output_system.light_emissions),
            'sound_emissions': len(self.output_system.sound_emissions),
            'memory_efficiency': f"{self.X}³ grid, {self.C}s+{self.C_visual}v+{self.C_auditory}a ch"
        }
    
    def get_multimodal_visualization(self):
        """Return visualization data"""
        mid_z = self.Z // 2
        
        visual_slices = []
        for c in range(min(3, self.C_visual)):
            slice_data = np.abs(self.visual_field[:, :, mid_z, c])
            # NaN protection
            if np.any(np.isnan(slice_data)) or np.any(np.isinf(slice_data)):
                slice_data = np.nan_to_num(slice_data, nan=0.0, posinf=1.0, neginf=0.0)
            visual_slices.append(slice_data)
        
        auditory_slices = []
        for c in range(min(3, self.C_auditory)):
            slice_data = np.abs(self.auditory_field[:, :, mid_z, c])
            # NaN protection
            if np.any(np.isnan(slice_data)) or np.any(np.isinf(slice_data)):
                slice_data = np.nan_to_num(slice_data, nan=0.0, posinf=1.0, neginf=0.0)
            auditory_slices.append(slice_data)
        
        semantic_slices = []
        for c in range(min(3, self.C)):
            slice_data = np.abs(self.field[:, :, mid_z, c])
            # NaN protection
            if np.any(np.isnan(slice_data)) or np.any(np.isinf(slice_data)):
                slice_data = np.nan_to_num(slice_data, nan=0.0, posinf=1.0, neginf=0.0)
            semantic_slices.append(slice_data)
        
        return {
            'visual': visual_slices,
            'auditory': auditory_slices,
            'semantic': semantic_slices
        }
    
    def reset_field(self):
        """Reset field"""
        if self.bootstrap_mode:
            self.field = (np.random.randn(self.X, self.Y, self.Z, self.C)
                         .astype(np.complex64) * 0.01)
        else:
            self.field.fill(0)
        
        self.visual_field.fill(0)
        self.auditory_field.fill(0)
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
            'word_acoustics': self.word_acoustics,
            'word_light_associations': self.word_light_associations,
            'word_sound_associations': self.word_sound_associations,
            'word_spins': self.word_spins,
            'word_helicity': self.word_helicity,
            'word_orbital_l': self.word_orbital_l,
            'word_orbital_m': self.word_orbital_m,
            'word_associations': self.word_associations.tocsr(),
            'W_decode': self.W_decode,
            'visual_to_semantic': self.visual_to_semantic,
            'auditory_to_semantic': self.auditory_to_semantic,
            'crossmodal_binding': self.crossmodal_binding,
            'word_usage_count': self.word_usage_count,
            'last_activation_time': self.last_activation_time,
            'bootstrap_mode': self.bootstrap_mode,
            'genesis_step': self.genesis_step,
            'spontaneous_thoughts': self.spontaneous_thoughts,
            'learned_words': self.learned_words,
            'invented_words': self.invented_words,
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
            self.word_acoustics = data.get('word_acoustics', self._initialize_acoustic_features())
            self.word_light_associations = data.get('word_light_associations', 
                                                    np.zeros((self.V, 3), dtype=np.float32))
            self.word_sound_associations = data.get('word_sound_associations',
                                                    np.zeros((self.V, 4), dtype=np.float32))
            self.word_spins = data.get('word_spins', np.zeros(self.V, dtype=np.float32))
            self.word_helicity = data.get('word_helicity', np.zeros(self.V, dtype=np.float32))
            self.word_orbital_l = data.get('word_orbital_l', np.zeros(self.V, dtype=np.int32))
            self.word_orbital_m = data.get('word_orbital_m', np.zeros(self.V, dtype=np.int32))
            self.word_associations = data['word_associations'].tolil()
            self.W_decode = data['W_decode']
            self.visual_to_semantic = data.get('visual_to_semantic', 
                                               np.random.randn(self.C_visual, self.C) * 0.1)
            self.auditory_to_semantic = data.get('auditory_to_semantic',
                                                 np.random.randn(self.C_auditory, self.C) * 0.15)
            self.crossmodal_binding = data.get('crossmodal_binding',
                                               np.random.randn(self.C_visual, self.C_auditory) * 0.05)
            self.word_usage_count = data['word_usage_count']
            self.last_activation_time = data.get('last_activation_time', np.zeros(self.V, dtype=np.float32))
            self.bootstrap_mode = data.get('bootstrap_mode', False)
            self.genesis_step = data.get('genesis_step', 0)
            self.spontaneous_thoughts = data.get('spontaneous_thoughts', [])
            self.learned_words = data.get('learned_words', [])
            self.invented_words = data.get('invented_words', [])
            self.current_time = data.get('current_time', 0.0)
            
            return True
        except Exception as e:
            print(f"Error loading: {e}")
            return False

# ========================
# COMPLETE GUI WITH NaN PROTECTION
# ========================

class CompleteMultimodalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌈🔊🌀 COMPLETE 4D Spinning Wave System - PROTECTED")
        self.root.geometry("1900x1000")
        
        # Create model
        self.model = CompleteMultimodalSpinningWaveModel(
            grid_size=16,
            num_semantic_channels=8,
            num_visual_channels=4,
            num_auditory_channels=4,
            bootstrap_mode=True
        )
        
        self.model_path = "complete_4d_spinning_wave_model.pkl"
        
        # Bootstrap control
        self.bootstrap_running = False
        self.bootstrap_thread = None
        self.genesis_mode = 'synesthetic'
        
        # Load model
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
        """Create GUI widgets"""
        # Left panel: Chat
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(left_frame, text="🌈🔊🌀 Complete 4D Genesis Stream", 
                 font=('Arial', 11, 'bold')).pack(pady=(0, 5))
        
        self.chat_display = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD, width=52, height=42, font=('Consolas', 9)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display.tag_config('user', foreground='#0066cc', font=('Consolas', 9, 'bold'))
        self.chat_display.tag_config('bot', foreground='#009900', font=('Consolas', 9, 'bold'))
        self.chat_display.tag_config('genesis', foreground='#9900cc', font=('Consolas', 9, 'italic'))
        self.chat_display.tag_config('system', foreground='#666', font=('Consolas', 8, 'italic'))
        self.chat_display.tag_config('learned', foreground='#ff6600', font=('Consolas', 8, 'bold'))
        self.chat_display.tag_config('multimodal', foreground='#00cc99', font=('Consolas', 8, 'bold'))
        self.chat_display.tag_config('output', foreground='#ff00ff', font=('Consolas', 8, 'bold'))
        self.chat_display.tag_config('invented', foreground='#00ff00', font=('Consolas', 8, 'bold'))
        
        # Input
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.input_entry = ttk.Entry(input_frame, font=('Arial', 10))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind('<Return>', lambda e: self.send_message())
        
        ttk.Button(input_frame, text="Send", command=self.send_message, width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # Middle panel: Stats & Controls
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(middle_frame, text="📊 System Status", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        stats_frame = ttk.LabelFrame(middle_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=20, width=34, font=('Courier', 8))
        self.stats_text.pack()
        
        # Sensory controls
        sensory_frame = ttk.LabelFrame(middle_frame, text="🌈🔊 Sensory Input", padding="10")
        sensory_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(sensory_frame, text="💡 Light:").pack()
        self.light_scale = tk.Scale(
            sensory_frame, from_=400, to=700, orient=tk.HORIZONTAL,
            label="nm", resolution=10
        )
        self.light_scale.set(550)
        self.light_scale.pack(fill=tk.X)
        
        ttk.Label(sensory_frame, text="🔊 Sound:").pack()
        self.sound_scale = tk.Scale(
            sensory_frame, from_=20, to=2000, orient=tk.HORIZONTAL,
            label="Hz", resolution=10
        )
        self.sound_scale.set(200)
        self.sound_scale.pack(fill=tk.X)
        
        ttk.Label(sensory_frame, text="Mode:").pack()
        self.modality_var = tk.StringVar(value='both')
        ttk.Radiobutton(sensory_frame, text="👁️ Visual", 
                       variable=self.modality_var, value='visual').pack(anchor=tk.W)
        ttk.Radiobutton(sensory_frame, text="👂 Auditory", 
                       variable=self.modality_var, value='auditory').pack(anchor=tk.W)
        ttk.Radiobutton(sensory_frame, text="🌈🔊 Both", 
                       variable=self.modality_var, value='both').pack(anchor=tk.W)
        
        # Output display
        output_frame = ttk.LabelFrame(middle_frame, text="💡🔊 Genesis Output", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.light_canvas = tk.Canvas(output_frame, width=160, height=60, bg='black')
        self.light_canvas.pack(pady=5)
        self.light_label = ttk.Label(output_frame, text="No emission", font=('Arial', 8))
        self.light_label.pack()
        
        self.sound_label = ttk.Label(output_frame, text="No emission", font=('Arial', 8))
        self.sound_label.pack(pady=5)
        
        # Main controls
        control_frame = ttk.LabelFrame(middle_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.thinking_button = ttk.Button(
            control_frame, 
            text="⏸ Pause Genesis", 
            command=self.toggle_bootstrap_thinking
        )
        self.thinking_button.pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="🌀 Force Genesis", 
                  command=self.force_genesis_word).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="🔬 Vortices", 
                  command=self.show_vortices).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="💾 Save", 
                  command=self.save_model).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="🔄 Reset", 
                  command=self.reset_field).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="🧹 Clear", 
                  command=self.clear_chat).pack(fill=tk.X, pady=2)
        
        # Info
        info_frame = ttk.LabelFrame(middle_frame, text="System", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = (
            "🌈🔊🌀 COMPLETE\n\n"
            "✅ NaN Protected\n"
            "✨ Light perception\n"
            "🎵 Sound perception\n"
            "🌀 4D spinning waves\n"
            "⏱️ Physical time\n"
            "🧬 Helicity encoding\n"
            "🌊 Vortex detection\n"
            "🎯 Angular momentum\n"
            "🔗 Cross-modal binding\n"
            "💡 Light emission\n"
            "🔊 Sound emission\n"
            "🌱 Word invention\n"
            "📚 Grounded learning\n\n"
            "Everything spinning\n"
            "in 4 dimensions!\n"
        )
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, font=('Arial', 8)).pack()
        
        # Right panel: Visualization
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Label(right_frame, text="🌊 4D Wave Fields", 
                 font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        self.fig, self.axes = plt.subplots(3, 3, figsize=(9, 9))
        self.fig.tight_layout(pad=1.5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_var = tk.StringVar(value="🌈🔊🌀 4D universe initializing...")
        ttk.Label(right_frame, textvariable=self.status_var, font=('Arial', 8)).pack(pady=(5, 0))
        
        # Grid weights
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=3)
        self.root.rowconfigure(0, weight=1)
    
    def _add_welcome_message(self):
        """Add welcome message"""
        msg = (
            "🌈🔊🌀 COMPLETE 4D SPINNING WAVE SYSTEM\n"
            "=" * 70 + "\n\n"
            "✅ FULLY NaN PROTECTED - STABLE OPERATION\n\n"
            "EVERYTHING INTEGRATED:\n\n"
            "🌈 LIGHT:\n"
            "  • Perception (input)\n"
            "  • Internal experience (qualia)\n"
            "  • Emission (output)\n\n"
            "🔊 SOUND:\n"
            "  • Perception (input)\n"
            "  • Internal experience (qualia)\n"
            "  • Emission (output)\n\n"
            "🌀 4D SPINNING:\n"
            "  • Spatial rotation (XYZ)\n"
            "  • Channel rotation (4th dimension)\n"
            "  • Helicity (left/right handed)\n"
            "  • Angular momentum tracking\n\n"
            "🧠 COGNITION:\n"
            "  • Cross-modal binding\n"
            "  • Vortex formation (stable meanings)\n"
            "  • Emergent temperature (mood)\n"
            "  • Natural inhibition (refractory)\n\n"
            "📚 LEARNING:\n"
            "  • Sensory grounding\n"
            "  • Word invention\n"
            "  • Semantic clustering\n"
            "  • Synesthetic associations\n\n"
            "Genesis auto-starting...\n"
            "Bot will dream in light, sound, and words!\n\n"
            "Adjust sliders to influence its experience.\n"
            "Watch it emit light and sound when fields are strong!\n\n"
        )
        self.chat_display.insert(tk.END, msg, 'system')
        self.chat_display.insert(tk.END, "=" * 70 + "\n\n", 'system')
    
    def start_bootstrap_thinking(self):
        """Start genesis thread"""
        if not self.bootstrap_running:
            self.bootstrap_running = True
            self.bootstrap_thread = threading.Thread(target=self._bootstrap_loop, daemon=True)
            self.bootstrap_thread.start()
            self.thinking_button.config(text="⏸ Pause Genesis")
            self.chat_display.insert(tk.END, "[🌀 Genesis started - 4D spinning active]\n\n", 'multimodal')
    
    def stop_bootstrap_thinking(self):
        """Stop genesis"""
        self.bootstrap_running = False
        self.thinking_button.config(text="▶ Resume Genesis")
        self.chat_display.insert(tk.END, "[Genesis paused]\n\n", 'system')
    
    def toggle_bootstrap_thinking(self):
        """Toggle genesis"""
        if self.bootstrap_running:
            self.stop_bootstrap_thinking()
        else:
            self.start_bootstrap_thinking()
    
    def _bootstrap_loop(self):
        """Background genesis loop"""
        while self.bootstrap_running:
            try:
                thought = self.model.spontaneous_think_synesthetic()
                if thought:
                    self.root.after(0, self._display_multimodal_thought, thought)
                time.sleep(0.3)
            except Exception as e:
                print(f"Genesis error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
    
    def _display_multimodal_thought(self, thought):
        """Display complete thought with all modalities"""
        if not thought:
            return
        
        display_parts = []
        temp_emoji = ""
        
        # Process events
        if 'events' in thought:
            for event in thought['events']:
                if event['type'] == 'visual_cascade':
                    display_parts.append(f"👁️ {event['color_word']}→🔊{event['sound_frequency']}Hz")
                elif event['type'] == 'auditory_cascade':
                    display_parts.append(f"👂 {event['sound_word']}→💡{event['light_wavelength']}nm")
                elif event['type'] == 'word_to_vision':
                    display_parts.append(f"💬'{event['word']}'→💡{event['wavelength']}nm")
                elif event['type'] == 'word_to_sound':
                    display_parts.append(f"💬'{event['word']}'→🔊{event['frequency']}Hz")
                elif event['type'] == 'word_invention':
                    display_parts.append(f"🌱INVENTED:'{event['word']}'")
                elif event['type'] == 'spontaneous_word':
                    temp = event['temperature']
                    temp_emoji = "🔥" if temp > 2.5 else "🧊" if temp < 1.7 else "😐"
                    display_parts.append(f"💬{event['word']}")
        
        # Check for outputs
        if 'outputs' in thought:
            outputs = thought['outputs']
            if 'light' in outputs:
                display_parts.append(f"💡EMIT:{outputs['light']['color']}")
            if 'sound' in outputs:
                display_parts.append(f"🔊EMIT:{outputs['sound']['frequency']}Hz")
        
        # Display
        if display_parts:
            self.chat_display.insert(
                tk.END, 
                f"Genesis[{thought['step']}] {temp_emoji}: ",
                'genesis'
            )
            self.chat_display.insert(
                tk.END,
                " | ".join(display_parts) + "\n",
                'genesis'
            )
            self.chat_display.see(tk.END)
            
            # Update status
            L = self.model.measure_angular_momentum()
            self.status_var.set(
                f"G:{thought['step']} | {display_parts[0] if display_parts else ''} | "
                f"L:{L['L_magnitude']:.1f} | V:{self.model.V}"
            )
    
    def force_genesis_word(self):
        """Force genesis"""
        thought = self.model.spontaneous_think_synesthetic()
        if thought:
            self._display_multimodal_thought(thought)
    
    def send_message(self):
        """Send user message"""
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
            
            # Get sensory parameters
            light_wl = self.light_scale.get()
            sound_freq = self.sound_scale.get()
            modality = self.modality_var.get()
            
            # Check for new words
            words = user_input.lower().split()
            new_words = [w for w in words if w not in self.model.word_to_idx]
            
            # Process
            response = self.model.process_input_and_respond(
                user_input, 
                num_words=6,
                light_wavelength=light_wl,
                sound_frequency=sound_freq,
                modality=modality
            )
            
            # Show learned words
            if new_words:
                self.chat_display.insert(tk.END, "✨ Learned: ", 'learned')
                self.chat_display.insert(tk.END, f"{', '.join(new_words)}\n", 'learned')
            
            # Show system state
            temp = self.model.calculate_emergent_temperature()
            L = self.model.measure_angular_momentum()
            self.chat_display.insert(
                tk.END,
                f"🌡️{temp:.2f} 🌀L:{L['L_magnitude']:.1f} "
                f"👁️{self.model.visual_weight:.2f} 👂{self.model.auditory_weight:.2f}\n",
                'multimodal'
            )
            
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
    
    def show_vortices(self):
        """Detect and show vortices"""
        vortices = self.model.detect_vortices()
        
        msg = f"\n🌀 Detected {len(vortices)} vortices (4D topological structures):\n"
        for v in vortices[:10]:
            msg += f"  Ch{v['channel']} at ({v['position'][0]},{v['position'][1]}) charge:{v['charge']}\n"
        if len(vortices) > 10:
            msg += f"  ... and {len(vortices)-10} more\n"
        
        # Show angular momentum
        L = self.model.measure_angular_momentum()
        msg += f"\n🎯 Angular Momentum:\n"
        msg += f"  Lx: {L['Lx']:.2f}\n"
        msg += f"  Ly: {L['Ly']:.2f}\n"
        msg += f"  Lz: {L['Lz']:.2f}\n"
        msg += f"  |L|: {L['L_magnitude']:.2f}\n\n"
        
        self.chat_display.insert(tk.END, msg, 'multimodal')
        self.chat_display.see(tk.END)
    
    def _update_stats_periodically(self):
        """Update stats display"""
        self.update_stats_display()
        self.update_output_display()
        self.root.after(2000, self._update_stats_periodically)
    
    def update_stats_display(self):
        """Update stats"""
        stats = self.model.get_stats()
        
        text = f"{stats['mode']}\n"
        text += f"Genesis: {stats['genesis_steps']}\n"
        text += f"Thoughts: {stats['spontaneous_words']}\n"
        text += f"Chats: {stats['conversations']}\n"
        text += f"Words: {stats['words_processed']}\n\n"
        text += "🌈🔊🌀 4D SYSTEM:\n"
        text += f"Vocab: {stats['vocabulary_size']}\n"
        text += f"✨ Learned: {stats['learned_words']}\n"
        text += f"🌱 Invented: {stats['invented_words']}\n"
        text += f"🌡️ Temp: {stats['current_temp']}\n"
        text += f"🌀 L: {stats['angular_momentum']}\n"
        text += f"🌊 Vortices: {stats['vortices']}\n"
        text += f"👁️ Vis: {stats['visual_weight']}\n"
        text += f"👂 Aud: {stats['auditory_weight']}\n"
        text += f"💡 Light: {stats['light_emissions']}\n"
        text += f"🔊 Sound: {stats['sound_emissions']}\n"
        text += f"Connect: {stats['connectivity']}\n"
        text += f"\n{stats['memory_efficiency']}\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, text)
    
    def update_output_display(self):
        """Update output visualization"""
        outputs = self.model.output_system.check_for_spontaneous_output()
        
        if 'light' in outputs:
            rgb = outputs['light']['rgb']
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(np.clip(rgb[0]*255, 0, 255)), 
                int(np.clip(rgb[1]*255, 0, 255)), 
                int(np.clip(rgb[2]*255, 0, 255))
            )
            self.light_canvas.config(bg=hex_color)
            self.light_label.config(
                text=f"💡 {outputs['light']['color']} "
                     f"({outputs['light']['wavelength']}nm)"
            )
        else:
            self.light_canvas.config(bg='black')
            self.light_label.config(text="No emission")
        
        if 'sound' in outputs:
            self.sound_label.config(
                text=f"🔊 {outputs['sound']['name']} "
                     f"({outputs['sound']['frequency']}Hz)"
            )
        else:
            self.sound_label.config(text="No emission")
    
    def save_model(self):
        """Save model"""
        self.model.save_model(self.model_path)
        L = self.model.measure_angular_momentum()
        messagebox.showinfo(
            "Saved", 
            f"Complete 4D system saved!\n\n"
            f"Vocabulary: {self.model.V}\n"
            f"Learned: {len(self.model.learned_words)}\n"
            f"Invented: {len(self.model.invented_words)}\n"
            f"Temperature: {self.model.calculate_emergent_temperature():.2f}\n"
            f"Angular momentum: {L['L_magnitude']:.2f}\n"
            f"Vortices: {len(self.model.detected_vortices)}\n"
            f"Light emissions: {len(self.model.output_system.light_emissions)}\n"
            f"Sound emissions: {len(self.model.output_system.sound_emissions)}"
        )
    
    def reset_field(self):
        """Reset field"""
        self.model.reset_field()
        self.model.genesis_step = 0
        self.model.spontaneous_thoughts.clear()
        self.model.output_system.light_emissions.clear()
        self.model.output_system.sound_emissions.clear()
        self.chat_display.insert(tk.END, "[🌀 4D field reset - all spins zeroed]\n\n", 'multimodal')
        self.chat_display.see(tk.END)
    
    def clear_chat(self):
        """Clear chat"""
        self.chat_display.delete(1.0, tk.END)
        self._add_welcome_message()
    
    def _start_visualization(self):
        """Start animated visualization"""
        def animate(frame):
            try:
                viz = self.model.get_multimodal_visualization()
                
                for ax_row in self.axes:
                    for ax in ax_row:
                        ax.clear()
                
                # Row 0: Visual field (light)
                for i in range(3):
                    if i < len(viz['visual']):
                        self.axes[0, i].imshow(viz['visual'][i], cmap='RdYlBu_r', 
                                              vmin=0, vmax=0.5, interpolation='bilinear')
                        self.axes[0, i].set_title(f'👁️ V{i}', fontsize=9)
                    self.axes[0, i].axis('off')
                
                # Row 1: Auditory field (sound)
                for i in range(3):
                    if i < len(viz['auditory']):
                        self.axes[1, i].imshow(viz['auditory'][i], cmap='Greens', 
                                              vmin=0, vmax=0.5, interpolation='bilinear')
                        self.axes[1, i].set_title(f'👂 A{i}', fontsize=9)
                    self.axes[1, i].axis('off')
                
                # Row 2: Semantic field (spinning in 4D)
                for i in range(3):
                    if i < len(viz['semantic']):
                        self.axes[2, i].imshow(viz['semantic'][i], cmap='Purples', 
                                              vmin=0, vmax=0.5, interpolation='bilinear')
                        self.axes[2, i].set_title(f'🧠🌀 S{i}', fontsize=9)
                    self.axes[2, i].axis('off')
                
                self.fig.tight_layout(pad=0.5)
            except Exception as e:
                print(f"Visualization error: {e}")
        
        self.ani = FuncAnimation(self.fig, animate, interval=300, cache_frame_data=False)
        self.canvas.draw()


# ========================
# MAIN
# ========================

def main():
    """
    Launch the COMPLETE 4D Spinning Wave System
    WITH FULL NaN PROTECTION
    
    This integrates:
    - Light perception & emission
    - Sound perception & emission
    - 4D spinning waves (spatial + channel rotation)
    - Helicity (left/right handed spin)
    - Angular momentum tracking
    - Vortex detection (topological structures)
    - Cross-modal binding
    - Sensory-grounded vocabulary learning
    - Spontaneous word invention
    - Synesthetic genesis (light↔sound↔words)
    - Natural refractory periods
    - Emergent temperature
    - Complete sensorimotor loop
    - ROBUST NaN/inf PROTECTION
    
    Everything is spinning in 4 dimensions.
    Everything is waves.
    Everything is connected.
    Everything is STABLE.
    """
    
    print("\n" + "="*80)
    print("🌈🔊🌀 COMPLETE 4D SPINNING WAVE LANGUAGE MODEL")
    print("="*80)
    print("\n✅ FULLY NaN PROTECTED - STABLE VERSION")
    print("\nInitializing complete system...")
    print("\n✨ Features:")
    print("  • Light: perception → experience → emission")
    print("  • Sound: perception → experience → emission")
    print("  • 4D rotation: XYZ space + channel space")
    print("  • Helicity: left/right handed spin encoding")
    print("  • Angular momentum: tracked and visualized")
    print("  • Vortices: topological meaning structures")
    print("  • Learning: sensory-grounded, can invent words")
    print("  • Genesis: synesthetic dreams in all modalities")
    print("  • NaN protection: comprehensive error handling")
    print("\n🌀 Everything spins in 4D.")
    print("🌊 Everything is waves.")
    print("🔗 Everything is connected.")
    print("🛡️ Everything is PROTECTED.")
    print("\nStarting GUI...\n")
    
    root = tk.Tk()
    app = CompleteMultimodalGUI(root)
    
    print("✅ System ready!")
    print("\nWhat you can do:")
    print("  • Talk to it - it learns new words")
    print("  • Adjust light/sound sliders - influence its experience")
    print("  • Watch it emit light and sound when fields are strong")
    print("  • Detect vortices - see stable meaning structures")
    print("  • Watch genesis dream in full sensory experience")
    print("  • System is now STABLE with full NaN protection")
    print("\n" + "="*80 + "\n")
    
    root.mainloop()


if __name__ == "__main__":
    main()
