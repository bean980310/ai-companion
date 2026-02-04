// Gradio API Endpoints
export const API_ENDPOINTS = {
  // Chat
  CHAT: '/api/chat',
  LIST_MODELS: '/api/list_models',
  LIST_SESSIONS: '/api/list_sessions',
  GET_HISTORY: '/api/get_history',
  GENERATE_TITLE: '/api/generate_title',

  // Image Generation
  GENERATE_IMAGE: '/api/generations',

  // Utilities
  TRANSLATE: '/api/translate',
  SUMMARIZE: '/api/summarize',
  ANALYZE_IMAGE: '/api/analyze_image',

  // MCP
  MCP_SSE: '/gradio_api/mcp/sse',
} as const;

export type ApiEndpoint = (typeof API_ENDPOINTS)[keyof typeof API_ENDPOINTS];
