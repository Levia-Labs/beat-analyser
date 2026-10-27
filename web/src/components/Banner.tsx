import type { ReactNode } from "react";

export default function Banner({ message, action }: { message: string; action?: ReactNode }) {
  return (
    <div className="banner" role="alert">
      <span>{message}</span>
      {action}
    </div>
  );
}
