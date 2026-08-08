import { NavLink } from "react-router-dom";
import { SIDEBAR_LINKABLE_SECTIONS } from "@/constants/navigation";

interface SidebarProps {
  open: boolean;
}

export function Sidebar({ open }: SidebarProps) {
  return (
    <aside
      className={`fixed inset-y-0 left-0 z-30 w-64 shrink-0 transform border-r border-base-700 bg-base-900 transition-transform duration-200 lg:static lg:translate-x-0 ${
        open ? "translate-x-0" : "-translate-x-full"
      }`}
    >
      <div className="flex h-16 items-center gap-2 border-b border-base-700 px-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-accent-500/15">
          <div className="h-2.5 w-2.5 rounded-full bg-accent-400" />
        </div>
        <div>
          <p className="text-sm font-semibold tracking-wide text-slate-100">OrbitShield</p>
          <p className="text-[10px] uppercase tracking-widest text-slate-500">Satellite Security</p>
        </div>
      </div>

      <nav className="h-[calc(100%-4rem)] overflow-y-auto px-3 py-4">
        {SIDEBAR_LINKABLE_SECTIONS.map((section) => (
          <div key={section.title} className="mb-5">
            <p className="mb-1.5 px-2 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
              {section.title}
            </p>
            <div className="space-y-0.5">
              {section.items.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === "/"}
                  className={({ isActive }) =>
                    `block rounded-md px-2.5 py-1.5 text-sm transition-colors ${
                      isActive
                        ? "bg-accent-500/10 text-accent-400"
                        : "text-slate-400 hover:bg-base-800 hover:text-slate-200"
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>
    </aside>
  );
}
