import { useState, useCallback } from 'react';
import {
  Settings,
  Palette,
  Globe,
  Key,
  Database,
  Bell,
  Info,
  ChevronRight,
  Moon,
  Sun,
  Monitor,
  Check,
  Eye,
  EyeOff,
  Trash2,
  Download,
  Upload,
} from 'lucide-react';
import type { Theme, Language } from '../types';

type SettingsSection = 'general' | 'appearance' | 'api' | 'data' | 'notifications' | 'about';

const SECTIONS = [
  { id: 'general' as const, name: 'General', icon: Settings },
  { id: 'appearance' as const, name: 'Appearance', icon: Palette },
  { id: 'api' as const, name: 'API Keys', icon: Key },
  { id: 'data' as const, name: 'Data & Storage', icon: Database },
  { id: 'notifications' as const, name: 'Notifications', icon: Bell },
  { id: 'about' as const, name: 'About', icon: Info },
];

const LANGUAGES: { id: Language; name: string; nativeName: string }[] = [
  { id: 'en', name: 'English', nativeName: 'English' },
  { id: 'ko', name: 'Korean', nativeName: '한국어' },
  { id: 'ja', name: 'Japanese', nativeName: '日本語' },
  { id: 'zh_CN', name: 'Chinese (Simplified)', nativeName: '简体中文' },
  { id: 'zh_TW', name: 'Chinese (Traditional)', nativeName: '繁體中文' },
];

const THEMES: { id: Theme; name: string; icon: typeof Sun }[] = [
  { id: 'light', name: 'Light', icon: Sun },
  { id: 'dark', name: 'Dark', icon: Moon },
  { id: 'system', name: 'System', icon: Monitor },
];

interface ApiKeyConfig {
  provider: string;
  name: string;
  key: string;
  isSet: boolean;
}

const DEFAULT_API_KEYS: ApiKeyConfig[] = [
  { provider: 'openai', name: 'OpenAI', key: '', isSet: false },
  { provider: 'anthropic', name: 'Anthropic', key: '', isSet: false },
  { provider: 'google', name: 'Google AI', key: '', isSet: false },
  { provider: 'elevenlabs', name: 'ElevenLabs', key: '', isSet: false },
];

export function SettingsPage() {
  const [activeSection, setActiveSection] = useState<SettingsSection>('general');
  const [theme, setTheme] = useState<Theme>('dark');
  const [language, setLanguage] = useState<Language>('ko');
  const [apiKeys, setApiKeys] = useState<ApiKeyConfig[]>(DEFAULT_API_KEYS);
  const [showApiKey, setShowApiKey] = useState<Record<string, boolean>>({});
  const [notifications, setNotifications] = useState({
    enabled: true,
    sound: true,
    desktop: false,
  });

  const handleApiKeyChange = useCallback((provider: string, value: string) => {
    setApiKeys((prev) =>
      prev.map((key) =>
        key.provider === provider ? { ...key, key: value, isSet: value.length > 0 } : key
      )
    );
  }, []);

  const toggleShowApiKey = useCallback((provider: string) => {
    setShowApiKey((prev) => ({ ...prev, [provider]: !prev[provider] }));
  }, []);

  const handleExportData = useCallback(() => {
    const data = {
      settings: { theme, language, notifications },
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'ai-companion-settings.json';
    link.click();
    URL.revokeObjectURL(url);
  }, [theme, language, notifications]);

  const handleClearData = useCallback(() => {
    if (window.confirm('Are you sure you want to clear all local data? This cannot be undone.')) {
      localStorage.clear();
      window.location.reload();
    }
  }, []);

  const renderSection = () => {
    switch (activeSection) {
      case 'general':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-4">General Settings</h3>

              {/* Language */}
              <div className="space-y-3">
                <label className="block text-sm font-medium text-text-secondary">Language</label>
                <div className="grid grid-cols-1 gap-2">
                  {LANGUAGES.map((lang) => (
                    <button
                      key={lang.id}
                      onClick={() => setLanguage(lang.id)}
                      className={`flex items-center justify-between px-4 py-3 rounded-lg border transition-colors ${
                        language === lang.id
                          ? 'border-accent-primary bg-accent-primary/10'
                          : 'border-border-primary bg-bg-secondary hover:bg-bg-tertiary'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <Globe size={20} className="text-text-muted" />
                        <div className="text-left">
                          <p className="text-text-primary">{lang.name}</p>
                          <p className="text-xs text-text-muted">{lang.nativeName}</p>
                        </div>
                      </div>
                      {language === lang.id && (
                        <Check size={20} className="text-accent-primary" />
                      )}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Default Character */}
            <div className="space-y-3 pt-4 border-t border-border-primary">
              <label className="block text-sm font-medium text-text-secondary">
                Default Character
              </label>
              <select className="w-full px-4 py-3 bg-bg-secondary border border-border-primary rounded-lg text-text-primary focus:outline-none focus:border-accent-primary">
                <option value="assistant">AI Assistant</option>
                <option value="aria">Aria</option>
                <option value="custom">Custom</option>
              </select>
            </div>
          </div>
        );

      case 'appearance':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-4">Appearance</h3>

              {/* Theme */}
              <div className="space-y-3">
                <label className="block text-sm font-medium text-text-secondary">Theme</label>
                <div className="grid grid-cols-3 gap-3">
                  {THEMES.map((t) => {
                    const Icon = t.icon;
                    return (
                      <button
                        key={t.id}
                        onClick={() => setTheme(t.id)}
                        className={`flex flex-col items-center gap-2 px-4 py-4 rounded-lg border transition-colors ${
                          theme === t.id
                            ? 'border-accent-primary bg-accent-primary/10'
                            : 'border-border-primary bg-bg-secondary hover:bg-bg-tertiary'
                        }`}
                      >
                        <Icon
                          size={24}
                          className={theme === t.id ? 'text-accent-primary' : 'text-text-muted'}
                        />
                        <span
                          className={
                            theme === t.id ? 'text-accent-primary' : 'text-text-secondary'
                          }
                        >
                          {t.name}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Font Size */}
            <div className="space-y-3 pt-4 border-t border-border-primary">
              <label className="block text-sm font-medium text-text-secondary">
                Chat Font Size
              </label>
              <div className="flex items-center gap-4">
                <span className="text-sm text-text-muted">Small</span>
                <input
                  type="range"
                  min={12}
                  max={20}
                  defaultValue={16}
                  className="flex-1 accent-accent-primary"
                />
                <span className="text-sm text-text-muted">Large</span>
              </div>
            </div>

            {/* Message Density */}
            <div className="space-y-3 pt-4 border-t border-border-primary">
              <label className="block text-sm font-medium text-text-secondary">
                Message Density
              </label>
              <select className="w-full px-4 py-3 bg-bg-secondary border border-border-primary rounded-lg text-text-primary focus:outline-none focus:border-accent-primary">
                <option value="compact">Compact</option>
                <option value="comfortable">Comfortable</option>
                <option value="spacious">Spacious</option>
              </select>
            </div>
          </div>
        );

      case 'api':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-2">API Keys</h3>
              <p className="text-sm text-text-muted mb-4">
                Configure API keys for external services. Keys are stored locally.
              </p>

              <div className="space-y-4">
                {apiKeys.map((apiKey) => (
                  <div
                    key={apiKey.provider}
                    className="p-4 bg-bg-secondary rounded-lg border border-border-primary"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Key size={16} className="text-text-muted" />
                        <span className="font-medium text-text-primary">{apiKey.name}</span>
                      </div>
                      {apiKey.isSet && (
                        <span className="text-xs px-2 py-1 bg-accent-secondary/20 text-accent-secondary rounded">
                          Configured
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type={showApiKey[apiKey.provider] ? 'text' : 'password'}
                        value={apiKey.key}
                        onChange={(e) => handleApiKeyChange(apiKey.provider, e.target.value)}
                        placeholder={`Enter ${apiKey.name} API key...`}
                        className="flex-1 px-3 py-2 bg-bg-tertiary border border-border-primary rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-accent-primary"
                      />
                      <button
                        onClick={() => toggleShowApiKey(apiKey.provider)}
                        className="p-2 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                      >
                        {showApiKey[apiKey.provider] ? (
                          <EyeOff size={18} />
                        ) : (
                          <Eye size={18} />
                        )}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case 'data':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-4">Data & Storage</h3>

              {/* Storage Info */}
              <div className="p-4 bg-bg-secondary rounded-lg border border-border-primary mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-text-secondary">Local Storage Used</span>
                  <span className="text-text-primary font-medium">2.4 MB</span>
                </div>
                <div className="w-full h-2 bg-bg-tertiary rounded-full overflow-hidden">
                  <div className="h-full w-1/4 bg-accent-primary rounded-full" />
                </div>
              </div>

              {/* Export Data */}
              <div className="space-y-3">
                <button
                  onClick={handleExportData}
                  className="w-full flex items-center justify-between px-4 py-3 bg-bg-secondary border border-border-primary rounded-lg hover:bg-bg-tertiary transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Download size={20} className="text-text-muted" />
                    <div className="text-left">
                      <p className="text-text-primary">Export Settings</p>
                      <p className="text-xs text-text-muted">Download your settings as JSON</p>
                    </div>
                  </div>
                  <ChevronRight size={20} className="text-text-muted" />
                </button>

                <button className="w-full flex items-center justify-between px-4 py-3 bg-bg-secondary border border-border-primary rounded-lg hover:bg-bg-tertiary transition-colors">
                  <div className="flex items-center gap-3">
                    <Upload size={20} className="text-text-muted" />
                    <div className="text-left">
                      <p className="text-text-primary">Import Settings</p>
                      <p className="text-xs text-text-muted">Restore from backup file</p>
                    </div>
                  </div>
                  <ChevronRight size={20} className="text-text-muted" />
                </button>
              </div>

              {/* Danger Zone */}
              <div className="pt-6 mt-6 border-t border-border-primary">
                <h4 className="text-sm font-medium text-status-error mb-3">Danger Zone</h4>
                <button
                  onClick={handleClearData}
                  className="w-full flex items-center justify-between px-4 py-3 bg-status-error/10 border border-status-error/30 rounded-lg hover:bg-status-error/20 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Trash2 size={20} className="text-status-error" />
                    <div className="text-left">
                      <p className="text-status-error">Clear All Data</p>
                      <p className="text-xs text-status-error/70">
                        Delete all local data including chat history
                      </p>
                    </div>
                  </div>
                </button>
              </div>
            </div>
          </div>
        );

      case 'notifications':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-4">Notifications</h3>

              <div className="space-y-4">
                <label className="flex items-center justify-between p-4 bg-bg-secondary rounded-lg border border-border-primary">
                  <div className="flex items-center gap-3">
                    <Bell size={20} className="text-text-muted" />
                    <div>
                      <p className="text-text-primary">Enable Notifications</p>
                      <p className="text-xs text-text-muted">Receive alerts for new messages</p>
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notifications.enabled}
                    onChange={(e) =>
                      setNotifications((prev) => ({ ...prev, enabled: e.target.checked }))
                    }
                    className="w-5 h-5 rounded accent-accent-primary"
                  />
                </label>

                <label className="flex items-center justify-between p-4 bg-bg-secondary rounded-lg border border-border-primary">
                  <div className="flex items-center gap-3">
                    <Bell size={20} className="text-text-muted" />
                    <div>
                      <p className="text-text-primary">Sound</p>
                      <p className="text-xs text-text-muted">Play sound for notifications</p>
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notifications.sound}
                    onChange={(e) =>
                      setNotifications((prev) => ({ ...prev, sound: e.target.checked }))
                    }
                    className="w-5 h-5 rounded accent-accent-primary"
                  />
                </label>

                <label className="flex items-center justify-between p-4 bg-bg-secondary rounded-lg border border-border-primary">
                  <div className="flex items-center gap-3">
                    <Monitor size={20} className="text-text-muted" />
                    <div>
                      <p className="text-text-primary">Desktop Notifications</p>
                      <p className="text-xs text-text-muted">Show system notifications</p>
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notifications.desktop}
                    onChange={(e) =>
                      setNotifications((prev) => ({ ...prev, desktop: e.target.checked }))
                    }
                    className="w-5 h-5 rounded accent-accent-primary"
                  />
                </label>
              </div>
            </div>
          </div>
        );

      case 'about':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-4">About</h3>

              <div className="p-6 bg-bg-secondary rounded-lg border border-border-primary text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-accent-primary to-accent-secondary rounded-2xl mx-auto mb-4 flex items-center justify-center">
                  <span className="text-3xl">🤖</span>
                </div>
                <h2 className="text-xl font-bold text-text-primary mb-1">AI Companion</h2>
                <p className="text-sm text-text-muted mb-4">Version 1.0.0</p>
                <p className="text-sm text-text-secondary max-w-md mx-auto">
                  A Gradio-based application combining chatbot functionality with AI-powered
                  content generation.
                </p>
              </div>

              <div className="space-y-2 mt-4">
                <a
                  href="https://github.com/bean980310/ai-companion"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between px-4 py-3 bg-bg-secondary border border-border-primary rounded-lg hover:bg-bg-tertiary transition-colors"
                >
                  <span className="text-text-primary">GitHub Repository</span>
                  <ChevronRight size={20} className="text-text-muted" />
                </a>
                <a
                  href="#"
                  className="flex items-center justify-between px-4 py-3 bg-bg-secondary border border-border-primary rounded-lg hover:bg-bg-tertiary transition-colors"
                >
                  <span className="text-text-primary">Documentation</span>
                  <ChevronRight size={20} className="text-text-muted" />
                </a>
                <a
                  href="#"
                  className="flex items-center justify-between px-4 py-3 bg-bg-secondary border border-border-primary rounded-lg hover:bg-bg-tertiary transition-colors"
                >
                  <span className="text-text-primary">License</span>
                  <ChevronRight size={20} className="text-text-muted" />
                </a>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="h-full flex bg-bg-primary">
      {/* Sidebar Navigation */}
      <div className="w-64 flex-shrink-0 border-r border-border-primary overflow-y-auto">
        <div className="p-4">
          <h2 className="text-xl font-bold text-text-primary mb-4">Settings</h2>
          <nav className="space-y-1">
            {SECTIONS.map((section) => {
              const Icon = section.icon;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                    activeSection === section.id
                      ? 'bg-accent-primary/10 text-accent-primary'
                      : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary'
                  }`}
                >
                  <Icon size={20} />
                  <span>{section.name}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto p-6">{renderSection()}</div>
      </div>
    </div>
  );
}

export default SettingsPage;
