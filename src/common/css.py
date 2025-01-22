css="""
/* Updated style.css */

:root {
    --color-background-primary: #f8fafc;
    --color-background-secondary: white;
    
    --color-background-primary-dark: #1a1a1a;
    --color-background-secondary-dark: #2d2d2d;
}

.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1.5rem;
    background-color: #f8fafc;
    min-height: 100vh;
}

.header-container {
    background-color: var(--color-background-secondary);
    padding: 1rem 1.5rem;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}

.main-title {
    margin: 0;
    color: #1a1a1a;
}

.language-selector {
    background-color: var(--color-background-secondary);
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem;
}

.session-container {
    background-color: var(--color-background-secondary);
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.session-dropdown {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem;
}

.icon-button {
    padding: 0.5rem;
    border-radius: 8px;
    border: none;
    background-color: #9ca3af;
    cursor: pointer;
    transition: all 0.2s;
}

.icon-button-delete {
    padding: 0.5rem;
    border-radius: 8px;
    border: none;
    background-color: #ef4444;
    cursor: pointer;
    transition: all 0.2s;
}

.icon-button:hover {
    background-color: #4b5563;
}

.icon-button-delete:hover {
    background-color: #7f1d1d;
    color: #dc2626;
}

.model-container {
    background-color: var(--color-background-secondary);
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.chat-interface {
    background-color: var(--color-background-secondary);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    min-height: 600px;
}

.system-message {
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 1rem;
}

.chat-window {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background-color: #ffffff;
    padding: 1rem;
    height: 500px;
    overflow-y: auto;
}

.chat-window .user-message {
    background-color: #f0f9ff;
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
}

.chat-window .assistant-message {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
}

.input-area {
    margin-top: 1rem;
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 0.5rem;
}

.message-input {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.75rem;
}

.send-button {
    background-color: #3b82f6;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.75rem 1.5rem;
    cursor: pointer;
    transition: background-color 0.2s;
}

.send-button:hover {
    background-color: #2563eb;
}

.side-panel {
    padding-left: 1rem;
}

.profile-image {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}

.character-dropdown, .preset-dropdown {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
}

.status-bar {
    margin-top: 1rem;
    padding: 0.5rem;
    background-color: #f8fafc;
    border-radius: 8px;
    color: #6b7280;
    font-size: 0.875rem;
}

/* 다크 모드 지원 */
@media (prefers-color-scheme: dark) {
    .main-container {
        background-color: var(--color-background-primary-dark);
    }
    
    .header-container, .session-container, .model-container, .chat-interface {
        background-color: var(--color-background-secondary-dark);
        border-color: #404040;
    }
    
    .main-title {
        color: #ffffff;
    }
    
    .chat-window {
        background-color: var(--color-background-secondary-dark);
        border-color: #404040;
    }
    
    .chat-window .user-message {
        background-color: #3b4252;
    }
    
    .chat-window .assistant-message {
        background-color: #2e3440;
    }
    
    .system-message, .input-area {
        background-color: var(--color-background-secondary-dark);
        border-color: #404040;
    }
    
    .message-input {
        background-color: var(--color-background-primary-dark);
        border-color: #404040;
        color: #ffffff;
    }
    
    .status-bar {
        background-color: var(--color-background-secondary-dark);
        color: #9ca3af;
    }
}
/* css 변수에 추가 */
.settings-popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
    z-index: 1000;
    width: 80%;
    max-width: 800px;
    max-height: 80vh;
    overflow-y: auto;
}

.popup-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.close-button {
    padding: 5px 10px;
    cursor: pointer;
    background: none;
    border: none;
    font-size: 20px;
}

/* 팝업이 표시될 때 배경을 어둡게 처리 */
.settings-popup::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: -1;
}
.confirm-dialog {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    z-index: 1100;  /* settings-popup보다 더 위에 표시 */
    width: 400px;
    text-align: center;
}

.confirm-dialog .buttons {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 20px;
}

.reset-confirm-modal {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 1000 !important;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    width: 400px;
    max-width: 90%;
}

.reset-confirm-modal::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: -1;
}

.reset-confirm-title {
    margin-bottom: 15px;
    font-weight: bold;
    color: #374151;
}

.reset-confirm-message {
    margin-bottom: 20px;
    color: #6B7280;
}

.reset-confirm-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
}

/* 다크 모드 지원 */
@media (prefers-color-scheme: dark) {
    .reset-confirm-modal {
        background: #2d2d2d;
    }
    
    .reset-confirm-title {
        color: #e5e7eb;
    }
    
    .reset-confirm-message {
        color: #9ca3af;
    }
}
.delete-session-modal {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 1000 !important;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    width: 400px;
    max-width: 90%;
}

.delete-session-modal::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: -1;
}

.delete-session-title {
    font-size: 1.25rem;
    font-weight: bold;
    color: #374151;
    margin-bottom: 1rem;
}

.delete-session-message {
    color: #6B7280;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
}

.delete-session-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
}

/* 다크모드 지원 */
@media (prefers-color-scheme: dark) {
    .delete-session-modal {
        background: #2d2d2d;
    }
    
    .delete-session-title {
        color: #e5e7eb;
    }
    
    .delete-session-message {
        color: #9ca3af;
    }
}
"""