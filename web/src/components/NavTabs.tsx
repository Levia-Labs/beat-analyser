import { Link, useLocation } from "react-router";
import { ListIcon, PlusIcon } from "./icons";

// Both bars are always rendered; CSS shows the top tabs on desktop and the bottom bar on phones.
function Bar({ className, label }: { className: string; label: string }) {
  const { pathname } = useLocation();
  const onAdd = pathname === "/add";
  const onProjects = pathname === "/" || pathname.startsWith("/p/");
  return (
    <nav className={className} aria-label={label}>
      <Link
        to="/"
        className={onProjects ? "nav-tab on" : "nav-tab"}
        aria-current={onProjects ? "page" : undefined}
      >
        <ListIcon size={20} />
        <span>Projects</span>
      </Link>
      <Link
        to="/add"
        className={onAdd ? "nav-tab on" : "nav-tab"}
        aria-current={onAdd ? "page" : undefined}
      >
        <PlusIcon size={20} />
        <span>Add</span>
      </Link>
    </nav>
  );
}

export default function NavTabs() {
  return (
    <>
      <Bar className="nav-tabs" label="Sections" />
      <Bar className="bottom-bar" label="Sections, bottom bar" />
    </>
  );
}
