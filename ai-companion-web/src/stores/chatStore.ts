import { create } from 'zustand';
import type { ChatMessage, ChatGenerationParams } from '../types';

interface ChatState {
  // Messages
  messages: ChatMessage[];
  isStreaming: boolean;
  streamingContent: string;

  // System message
  systemMessage: string;
  selectedCharacter: string;

  // Generation parameters
  params: ChatGenerationParams;

  // Actions
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  updateLastMessage: (content: string) => void;
  clearMessages: () => void;

  // Streaming
  setStreaming: (isStreaming: boolean) => void;
  appendStreamContent: (content: string) => void;
  clearStreamContent: () => void;

  // Settings
  setSystemMessage: (message: string) => void;
  setSelectedCharacter: (character: string) => void;
  setParams: (params: Partial<ChatGenerationParams>) => void;
  resetParams: () => void;
}

const DEFAULT_PARAMS: ChatGenerationParams = {
  temperature: 0.7,
  topK: 50,
  topP: 0.9,
  repetitionPenalty: 1.1,
  maxLength: 2048,
  seed: -1,
  enableThinking: false,
};

const generateMessageId = () => `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

export const useChatStore = create<ChatState>((set) => ({
  // Initial state
  messages: [],
  isStreaming: false,
  streamingContent: '',
  systemMessage: '',
  selectedCharacter: 'AI Assistant',
  params: { ...DEFAULT_PARAMS },

  // Messages actions
  setMessages: (messages) => set({ messages }),

  addMessage: (message) => {
    const messageWithId = {
      ...message,
      id: message.id || generateMessageId(),
      timestamp: message.timestamp || new Date().toISOString(),
    };
    set((state) => ({
      messages: [...state.messages, messageWithId],
    }));
  },

  updateLastMessage: (content) => {
    set((state) => {
      const messages = [...state.messages];
      if (messages.length > 0) {
        messages[messages.length - 1] = {
          ...messages[messages.length - 1],
          content,
        };
      }
      return { messages };
    });
  },

  clearMessages: () => set({ messages: [], streamingContent: '' }),

  // Streaming actions
  setStreaming: (isStreaming) => set({ isStreaming }),

  appendStreamContent: (content) => {
    set((state) => ({
      streamingContent: state.streamingContent + content,
    }));
  },

  clearStreamContent: () => set({ streamingContent: '' }),

  // Settings actions
  setSystemMessage: (systemMessage) => set({ systemMessage }),

  setSelectedCharacter: (selectedCharacter) => set({ selectedCharacter }),

  setParams: (newParams) => {
    set((state) => ({
      params: { ...state.params, ...newParams },
    }));
  },

  resetParams: () => set({ params: { ...DEFAULT_PARAMS } }),
}));

export default useChatStore;
