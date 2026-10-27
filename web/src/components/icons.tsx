import type { ReactNode } from "react";

// Inline stroke icons: currentColor follows the text, aria-hidden because a label always sits beside them.
function icon(paths: ReactNode) {
  return function Icon({ size = 18 }: { size?: number }) {
    return (
      <svg
        className="icon"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        focusable="false"
      >
        {paths}
      </svg>
    );
  };
}

export const UploadIcon = icon(<path d="M12 16V4m0 0L7 9m5-5 5 5M4 20h16" />);
export const LinkIcon = icon(
  <path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3A4 4 0 0 0 11 18.7l1-1" />,
);
export const AnalyseIcon = icon(<path d="M7 4v16l13-8z" />);
export const DownloadIcon = icon(<path d="M12 4v12m0 0 5-5m-5 5-5-5M4 20h16" />);
export const TrashIcon = icon(<path d="M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3" />);
export const PencilIcon = icon(<path d="M4 20l1-5L16 4l4 4L9 19zM14 6l4 4" />);
export const CheckIcon = icon(<path d="M5 12.5l4.5 4.5L19 7" />);
export const XIcon = icon(<path d="M6 6l12 12M18 6L6 18" />);
export const PlusIcon = icon(<path d="M12 5v14M5 12h14" />);
export const ListIcon = icon(<path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01" />);
export const SearchIcon = icon(
  <>
    <circle cx="11" cy="11" r="6" />
    <path d="M16 16l4 4" />
  </>,
);
export const FilterIcon = icon(<path d="M4 5h16l-6 7v6l-4 2v-8z" />);
export const SortAscIcon = icon(<path d="M12 19V5m0 0-5 5m5-5 5 5" />);
export const SortDescIcon = icon(<path d="M12 5v14m0 0-5-5m5 5 5-5" />);
export const ZoomInIcon = icon(
  <>
    <circle cx="11" cy="11" r="6" />
    <path d="M16 16l4 4M11 8v6M8 11h6" />
  </>,
);
export const ZoomOutIcon = icon(
  <>
    <circle cx="11" cy="11" r="6" />
    <path d="M16 16l4 4M8 11h6" />
  </>,
);
export const ArrowLeftIcon = icon(<path d="M19 12H5m0 0 6-6m-6 6 6 6" />);
export const RefreshIcon = icon(<path d="M20 12a8 8 0 1 1-2.5-5.8M20 4v5h-5" />);
export const AlertIcon = icon(
  <>
    <path d="M12 4 3 20h18z" />
    <path d="M12 10v4m0 3h.01" />
  </>,
);
export const LockIcon = icon(
  <>
    <rect x="5" y="11" width="14" height="9" rx="2" />
    <path d="M8 11V8a4 4 0 0 1 8 0v3" />
  </>,
);
export const LogoutIcon = icon(<path d="M10 4H5v16h5M15 8l4 4-4 4M19 12H9" />);
export const SelectIcon = icon(
  <>
    <rect x="4" y="4" width="16" height="16" rx="3" />
    <path d="M8 12.5l3 3 5-6" />
  </>,
);
