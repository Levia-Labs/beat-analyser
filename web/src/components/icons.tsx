// One icon set on saber-tooth's grid: 24px viewBox, 1.75 stroke (in CSS, `.ic`), round caps and
// joins, currentColor. Names that saber-tooth also has use its drawings.
import type { ReactNode, SVGProps } from "react";

const ICONS = {
  search: (
    <>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="M15.5 15.5L20.5 20.5" />
    </>
  ),
  filter: <path d="M4 6h16M7 12h10M10 18h4" />,
  "filter-x": (
    <>
      <path d="M4 6h10M7 12h5M10 18h2" />
      <path d="M15 12l6 6M21 12l-6 6" />
    </>
  ),
  "sort-asc": <path d="M12 19V5M6 11l6-6 6 6" />,
  "sort-desc": <path d="M12 5v14M6 13l6 6 6-6" />,
  download: <path d="M12 4v12M7 11l5 5 5-5M4 17v3h16v-3" />,
  upload: <path d="M12 16V4M7 9l5-5 5 5M4 17v3h16v-3" />,
  trash: <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />,
  pencil: <path d="M4 20l1-5L16 4l4 4L9 19zM14 6l4 4" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  // saber-tooth's Generate-map note with a plus: it adds a track here too.
  generate: (
    <>
      <circle cx="5.5" cy="16" r="2.5" />
      <circle cx="13.5" cy="10" r="2.5" />
      <path d="M8 16V5l8-1.5V10M19 15v6M16 18h6" />
    </>
  ),
  // saber-tooth's ok icon, and its cross in the same circle for a project that is not built.
  ok: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M7.8 12.4l3 3 5.4-6" />
    </>
  ),
  "close-circle": (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M9 9l6 6M15 9l-6 6" />
    </>
  ),
  "arrow-left": <path d="M19 12H5M11 6l-6 6 6 6" />,
  refresh: (
    <>
      <path d="M4.5 12a7.5 7.5 0 1 0 2.3-5.4L4 9.2" />
      <path d="M4 4.5v5h5" />
    </>
  ),
  alert: (
    <>
      <path d="M12 3.5L21.5 20h-19z" />
      <path d="M12 10v4.5M12 17.2v.1" />
    </>
  ),
  list: (
    <>
      <path d="M9 6h11M9 12h11M9 18h11" />
      <circle cx="4.5" cy="6" r="1" />
      <circle cx="4.5" cy="12" r="1" />
      <circle cx="4.5" cy="18" r="1" />
    </>
  ),
  play: <path d="M8 5l11 7-11 7z" />,
  link: (
    <path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3A4 4 0 0 0 11 18.7l1-1" />
  ),
  "zoom-in": (
    <>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="M15.5 15.5L20.5 20.5M10.5 7.5v6M7.5 10.5h6" />
    </>
  ),
  "zoom-out": (
    <>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="M15.5 15.5L20.5 20.5M7.5 10.5h6" />
    </>
  ),
  lock: (
    <>
      <rect x="5" y="11" width="14" height="9" rx="2" />
      <path d="M8 11V8a4 4 0 0 1 8 0v3" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5.5M12 7.8v.1" />
    </>
  ),
  "external-link": (
    <path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />
  ),
  logout: <path d="M10 4H5v16h5M15 8l4 4-4 4M19 12H9" />,
} satisfies Record<string, ReactNode>;

export type IconName = keyof typeof ICONS;

interface IconProps extends Omit<SVGProps<SVGSVGElement>, "name"> {
  name: IconName;
  filled?: boolean;
}

// data-icon names the drawing, so a test (or a stylesheet) can tell which icon a button carries.
export function Icon({ name, filled = false, className, ...rest }: IconProps) {
  const cls = ["ic", filled ? "ic-filled" : "", className ?? ""].filter(Boolean).join(" ");
  return (
    <svg
      viewBox="0 0 24 24"
      className={cls}
      data-icon={name}
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {ICONS[name]}
    </svg>
  );
}
