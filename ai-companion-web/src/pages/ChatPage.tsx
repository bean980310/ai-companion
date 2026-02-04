import { useCallback } from 'react';
import { ChatWindow, ChatInput } from '../components/chat';
import { useChatStore } from '../stores/chatStore';
import { useSessionStore } from '../stores/sessionStore';
import { useModelStore } from '../stores/modelStore';

export function ChatPage() {
  const {
    messages,
    isStreaming,
    streamingContent,
    addMessage,
    setStreaming,
    appendStreamContent,
    clearStreamContent,
  } = useChatStore();

  const { currentSessionId, createSession, updateSession } = useSessionStore();
  const { selectedProvider, selectedModel } = useModelStore();

  const handleSendMessage = useCallback(
    async (content: string, _images?: File[]) => {
      // Create session if none exists
      let sessionId = currentSessionId;
      if (!sessionId) {
        createSession();
        sessionId = currentSessionId;
      }

      // Add user message
      addMessage({
        id: `msg_${Date.now()}`,
        role: 'user',
        content,
        timestamp: new Date().toISOString(),
      });

      // Start streaming
      setStreaming(true);
      clearStreamContent();

      try {
        // TODO: Replace with actual API call
        // Simulated streaming response for demo
        const response = `This is a simulated response from **${selectedModel}** (${selectedProvider}).

Your message was: "${content}"

Here's some example formatted content:

\`\`\`javascript
function greet(name) {
  return \`Hello, \${name}!\`;
}

console.log(greet('World'));
\`\`\`

- Feature 1: Real-time streaming
- Feature 2: Markdown support
- Feature 3: Code highlighting

The AI Companion web app is now ready for development!`;

        // Simulate streaming character by character
        for (let i = 0; i < response.length; i++) {
          await new Promise((resolve) => setTimeout(resolve, 10));
          appendStreamContent(response[i]);
        }

        // Add assistant message
        addMessage({
          id: `msg_${Date.now()}`,
          role: 'assistant',
          content: response,
          timestamp: new Date().toISOString(),
        });

        // Update session name if it's the first message
        if (messages.length === 0 && sessionId) {
          const title = content.slice(0, 30) + (content.length > 30 ? '...' : '');
          updateSession(sessionId, { name: title });
        }
      } catch (error) {
        console.error('Failed to send message:', error);
        addMessage({
          id: `msg_${Date.now()}`,
          role: 'assistant',
          content: 'Sorry, an error occurred. Please try again.',
          timestamp: new Date().toISOString(),
        });
      } finally {
        setStreaming(false);
        clearStreamContent();
      }
    },
    [
      currentSessionId,
      createSession,
      addMessage,
      setStreaming,
      appendStreamContent,
      clearStreamContent,
      updateSession,
      messages.length,
      selectedModel,
      selectedProvider,
    ]
  );

  return (
    <div className="h-full flex flex-col bg-bg-primary">
      <ChatWindow
        messages={messages}
        streamingContent={streamingContent}
        isStreaming={isStreaming}
      />
      <ChatInput
        onSend={handleSendMessage}
        isLoading={isStreaming}
        placeholder="Type your message..."
      />
    </div>
  );
}

export default ChatPage;
