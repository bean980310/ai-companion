import { Menu, Settings, MessageSquare, Image, Mic, BookOpen } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

interface HeaderProps {
  onMenuClick?: () => void;
}

const navItems = [
  { path: '/', label: 'Chat', icon: MessageSquare },
  { path: '/images', label: 'Images', icon: Image },
  { path: '/audio', label: 'Audio', icon: Mic },
  { path: '/story', label: 'Story', icon: BookOpen },
];

export function Header({ onMenuClick }: HeaderProps) {
  const location = useLocation();

  return (
    <header className="h-14 bg-bg-elevated border-b border-border-subtle flex items-center justify-between px-4 shrink-0">
      {/* Left: Menu button and Logo */}
      <div className="flex items-center gap-4">
        <button
          onClick={onMenuClick}
          className="p-2 hover:bg-bg-tertiary rounded-lg transition-default lg:hidden"
          aria-label="Toggle menu"
        >
          <Menu size={20} />
        </button>
        <Link to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-accent-primary rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">AI</span>
          </div>
          <span className="font-semibold text-lg hidden sm:block">AI Companion</span>
        </Link>
      </div>

      {/* Center: Navigation */}
      <nav className="hidden md:flex items-center gap-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg transition-default
                ${isActive
                  ? 'bg-accent-primary text-white'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
                }
              `}
            >
              <Icon size={18} />
              <span className="text-sm font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Right: Settings */}
      <div className="flex items-center gap-2">
        <Link
          to="/settings"
          className="p-2 hover:bg-bg-tertiary rounded-lg transition-default text-text-secondary hover:text-text-primary"
          aria-label="Settings"
        >
          <Settings size={20} />
        </Link>
      </div>
    </header>
  );
}

export default Header;
