import { useEffect, useRef } from 'react';
import { MessageSquare } from 'lucide-react';
import { ChatMessage } from './ChatMessage';
import type { ChatMessage as ChatMessageType } from '../../types';

interface ChatWindowProps {
  messages: ChatMessageType[];
  streamingContent?: string;
  isStreaming?: boolean;
}

export function ChatWindow({ messages, streamingContent, isStreaming }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  // Empty state
  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-text-muted p-8">
        <div className="w-16 h-16 bg-bg-tertiary rounded-2xl flex items-center justify-center mb-4">
          <MessageSquare size={32} className="text-accent-primary" />
        </div>
        <h3 className="text-lg font-medium text-text-primary mb-2">Start a conversation</h3>
        <p className="text-sm text-center max-w-md">
          Send a message to begin chatting with the AI assistant.
          You can ask questions, get help with tasks, or just have a conversation.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto">
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}

        {/* Streaming message */}
        {isStreaming && streamingContent && (
          <ChatMessage
            message={{
              id: 'streaming',
              role: 'assistant',
              content: streamingContent,
              timestamp: new Date().toISOString(),
            }}
            isStreaming
          />
        )}

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

export default ChatWindow;
