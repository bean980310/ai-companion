// Gradio API Endpoints
// Gradio auto-generates these from api_name in gr.Interface()
// Format: POST /api/{api_name} with body { "data": [...args] }
export const GRADIO_API_NAMES = {
  // Chat (from MCP Tools tab)
  CHAT: 'chat',
  LIST_MODELS: 'list_models',
  LIST_SESSIONS: 'list_sessions',
  GET_HISTORY: 'get_history',
  GENERATE_TITLE: 'generate_title',


  // Image Generation
  GENERATE_IMAGE: 'image',

  // Utilities
  TRANSLATE: 'translate',
  SUMMARIZE: 'summarize',
  ANALYZE_IMAGE: 'analyze_image',

  // MCP
  MCP_SSE: '/gradio_api/mcp/sse',
} as const;

export type GradioApiName = (typeof GRADIO_API_NAMES)[keyof typeof GRADIO_API_NAMES];
