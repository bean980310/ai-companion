import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { LLMProvider } from '../types';

interface ModelState {
  // Available options
  providers: LLMProvider[];
  modelsByProvider: Record<string, string[]>;

  // Current selection
  selectedProvider: LLMProvider;
  selectedModel: string;
  apiKey: string;

  // UI state
  showApiKey: boolean;
  isLoading: boolean;

  // Actions
  setProviders: (providers: LLMProvider[]) => void;
  setModelsByProvider: (models: Record<string, string[]>) => void;
  setSelectedProvider: (provider: LLMProvider) => void;
  setSelectedModel: (model: string) => void;
  setApiKey: (key: string) => void;
  setLoading: (loading: boolean) => void;

  // Computed
  models: string[];
}

// Providers that require API keys
const API_KEY_PROVIDERS: LLMProvider[] = [
  'openai',
  'anthropic',
  'google-genai',
  'perplexity',
  'xai',
  'mistralai',
  'openrouter',
  'hf-inference',
];

const DEFAULT_PROVIDERS: LLMProvider[] = [
  'openai',
  'anthropic',
  'google-genai',
  'ollama',
  'lmstudio',
];

const DEFAULT_MODELS: Record<string, string[]> = {
  openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
  anthropic: ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'],
  'google-genai': ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'],
  ollama: ['llama3.2', 'mistral', 'codellama'],
  lmstudio: ['local-model'],
};

export const useModelStore = create<ModelState>()(
  persist(
    (set, get) => ({
      providers: DEFAULT_PROVIDERS,
      modelsByProvider: DEFAULT_MODELS,
      selectedProvider: 'openai',
      selectedModel: 'gpt-4o',
      apiKey: '',
      showApiKey: true,
      isLoading: false,

      setProviders: (providers) => set({ providers }),

      setModelsByProvider: (models) => set({ modelsByProvider: models }),

      setSelectedProvider: (provider) => {
        const { modelsByProvider } = get();
        const models = modelsByProvider[provider] || [];
        set({
          selectedProvider: provider,
          selectedModel: models[0] || '',
          showApiKey: API_KEY_PROVIDERS.includes(provider),
        });
      },

      setSelectedModel: (model) => set({ selectedModel: model }),

      setApiKey: (key) => set({ apiKey: key }),

      setLoading: (loading) => set({ isLoading: loading }),

      get models() {
        const { selectedProvider, modelsByProvider } = get();
        return modelsByProvider[selectedProvider] || [];
      },
    }),
    {
      name: 'ai-companion-models',
      partialize: (state) => ({
        selectedProvider: state.selectedProvider,
        selectedModel: state.selectedModel,
        // Note: API key is NOT persisted for security
      }),
    }
  )
);

export default useModelStore;
