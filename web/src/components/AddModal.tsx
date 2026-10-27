import * as Dialog from "@radix-ui/react-dialog";
import { useState } from "react";
import { useNavigate } from "react-router";
import { getConfig } from "../api";
import { useLoad } from "../useLoad";
import AnalyseCard from "./AnalyseCard";
import { Icon } from "./icons";

// Loaded only while the modal is open: Radix unmounts the content when it closes.
function Body({
  onDone,
  onBusy,
}: {
  onDone: (name: string) => void;
  onBusy: (b: boolean) => void;
}) {
  const config = useLoad(getConfig);
  return <AnalyseCard config={config.data} onDone={onDone} onBusy={onBusy} />;
}

// The Add button and its dialog. A run in flight locks every way out: closing would hide an
// upload that still completes and navigates away.
export default function AddModal() {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();
  const keepOpen = (event: Event) => busy && event.preventDefault();

  return (
    <Dialog.Root open={open} onOpenChange={(next) => (next || !busy) && setOpen(next)}>
      <Dialog.Trigger className="btn primary">
        <Icon name="generate" />
        <span className="label-hide-mobile">Add</span>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="overlay" />
        <Dialog.Content
          className="modal"
          aria-describedby={undefined}
          onEscapeKeyDown={keepOpen}
          onPointerDownOutside={keepOpen}
          onInteractOutside={keepOpen}
        >
          <div className="modal-head">
            <Dialog.Title className="modal-title">Add project</Dialog.Title>
            <Dialog.Close className="btn plain icon" aria-label="Close" disabled={busy}>
              <Icon name="x" />
            </Dialog.Close>
          </div>
          <Body
            onBusy={setBusy}
            onDone={(name) => {
              setBusy(false);
              setOpen(false);
              navigate(`/p/${encodeURIComponent(name)}`);
            }}
          />
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
