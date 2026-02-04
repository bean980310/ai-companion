// API Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Chat Types
export interface ChatMessage {
  id: string;
  role: 'system' | 'user' | 'assistant';
  content: string | ContentPart[];
  timestamp: string;
}

export interface ContentPart {
  type: 'text' | 'image';
  text?: string;
  imageUrl?: string;
}

export interface Session {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  lastActivity: string;
  lastCharacter: string;
}

// Model Types
export type LLMProvider =
  | 'openai'
  | 'anthropic'
  | 'google-genai'
  | 'perplexity'
  | 'xai'
  | 'mistralai'
  | 'openrouter'
  | 'hf-inference'
  | 'ollama'
  | 'lmstudio'
  | 'vllm-api'
  | 'self-provided';

export type ImageProvider = 'openai' | 'google-genai' | 'comfyui' | 'self-provided';

export type ModelType = 'transformers' | 'gguf' | 'mlx' | 'api';
export type ImageModelType = 'checkpoint' | 'diffusers';

export interface Model {
  id: string;
  name: string;
  provider: LLMProvider;
  type: ModelType;
}

export interface LoraConfig {
  name: string;
  textEncoderWeight: number;
  unetWeight: number;
}

// Character Types
export interface Character {
  key: string;
  name: string;
  description: string;
  avatar?: string;
  systemPrompt?: string;
}

// Generation Parameters
export interface ChatGenerationParams {
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  maxLength: number;
  seed: number;
  enableThinking: boolean;
}

export interface ImageGenerationParams {
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  seed: number;
  randomSeed: boolean;
  sampler: string;
  scheduler: string;
  clipSkip: number;
  batchSize: number;
  batchCount: number;
  denoiseStrength: number;
}

// UI Types
export type Language = 'ko' | 'en' | 'ja' | 'zh_CN' | 'zh_TW';
export type Theme = 'light' | 'dark' | 'system';

export interface NavigationItem {
  key: string;
  label: string;
  path: string;
  icon: string;
}
