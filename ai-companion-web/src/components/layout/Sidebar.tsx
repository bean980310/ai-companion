import { Plus, Trash2, ChevronDown, ChevronRight, X } from 'lucide-react';
import { useState } from 'react';
import type { Session, LLMProvider } from '../../types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  sessions: Session[];
  currentSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  onDeleteSession: (sessionId: string) => void;
  // Model selection
  providers: string[];
  selectedProvider: LLMProvider;
  onProviderChange: (provider: LLMProvider) => void;
  models: string[];
  selectedModel: string;
  onModelChange: (model: string) => void;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
  showApiKey: boolean;
}

export function Sidebar({
  isOpen,
  onClose,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  providers,
  selectedProvider,
  onProviderChange,
  models,
  selectedModel,
  onModelChange,
  apiKey,
  onApiKeyChange,
  showApiKey,
}: SidebarProps) {
  const [sessionsExpanded, setSessionsExpanded] = useState(true);
  const [modelsExpanded, setModelsExpanded] = useState(true);

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed top-14 left-0 h-[calc(100vh-3.5rem)] w-72 bg-bg-secondary border-r border-border-subtle
          flex flex-col z-50 transition-transform duration-300
          lg:translate-x-0 lg:static lg:z-auto
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        {/* Mobile close button */}
        <button
          onClick={onClose}
          className="absolute top-2 right-2 p-2 hover:bg-bg-tertiary rounded-lg lg:hidden"
        >
          <X size={18} />
        </button>

        {/* Sessions Section */}
        <div className="flex-1 overflow-hidden flex flex-col">
          <button
            onClick={() => setSessionsExpanded(!sessionsExpanded)}
            className="flex items-center justify-between px-4 py-3 hover:bg-bg-tertiary transition-default"
          >
            <span className="font-medium text-sm">Sessions</span>
            {sessionsExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </button>

          {sessionsExpanded && (
            <div className="flex-1 overflow-y-auto px-2">
              {/* New Session Button */}
              <button
                onClick={onNewSession}
                className="w-full flex items-center gap-2 px-3 py-2 mb-2 rounded-lg
                           border border-dashed border-border-default
                           hover:bg-bg-tertiary hover:border-accent-primary
                           text-text-secondary hover:text-text-primary transition-default"
              >
                <Plus size={16} />
                <span className="text-sm">New Chat</span>
              </button>

              {/* Session List */}
              <div className="space-y-1">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={`
                      group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer
                      transition-default
                      ${currentSessionId === session.id
                        ? 'bg-accent-primary/20 text-accent-primary'
                        : 'hover:bg-bg-tertiary text-text-secondary hover:text-text-primary'
                      }
                    `}
                    onClick={() => onSelectSession(session.id)}
                  >
                    <span className="text-sm truncate flex-1">{session.name}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(session.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-status-error/20 rounded transition-default"
                    >
                      <Trash2 size={14} className="text-status-error" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Models Section */}
        <div className="border-t border-border-subtle">
          <button
            onClick={() => setModelsExpanded(!modelsExpanded)}
            className="flex items-center justify-between w-full px-4 py-3 hover:bg-bg-tertiary transition-default"
          >
            <span className="font-medium text-sm">Model</span>
            {modelsExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </button>

          {modelsExpanded && (
            <div className="px-4 pb-4 space-y-3">
              {/* Provider Select */}
              <div>
                <label className="block text-xs text-text-muted mb-1">Provider</label>
                <select
                  value={selectedProvider}
                  onChange={(e) => onProviderChange(e.target.value as LLMProvider)}
                  className="input-base text-sm"
                >
                  {providers.map((provider) => (
                    <option key={provider} value={provider}>
                      {provider}
                    </option>
                  ))}
                </select>
              </div>

              {/* Model Select */}
              <div>
                <label className="block text-xs text-text-muted mb-1">Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => onModelChange(e.target.value)}
                  className="input-base text-sm"
                >
                  {models.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>

              {/* API Key Input */}
              {showApiKey && (
                <div>
                  <label className="block text-xs text-text-muted mb-1">API Key</label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => onApiKeyChange(e.target.value)}
                    placeholder="Enter API key..."
                    className="input-base text-sm"
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
