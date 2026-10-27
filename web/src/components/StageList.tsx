import type { Progress } from "../api";

const LABELS: Record<string, string> = {
  download: "Download",
  decode: "Decode",
  rhythm: "Rhythm",
  notes: "Notes",
  key: "Key",
  package: "Package",
};

export default function StageList({ progress }: { progress: Progress }) {
  return (
    <ol className="stages">
      {progress.stages.map((stage, index) => {
        const state =
          index < progress.done
            ? "done"
            : stage === progress.stage && !progress.finished
              ? "now"
              : "todo";
        return (
          <li key={stage} className={state} aria-current={state === "now" ? "step" : undefined}>
            {LABELS[stage] ?? stage}
            {state === "now" && progress.fraction !== null && (
              <span className="num">{Math.round(progress.fraction * 100)}%</span>
            )}
          </li>
        );
      })}
    </ol>
  );
}
