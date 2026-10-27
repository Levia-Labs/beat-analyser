import { Link } from "react-router";
import { ListIcon } from "../components/icons";
import StatusPage from "../components/StatusPage";

export default function NotFound() {
  return (
    <StatusPage title="Not found" detail="Nothing lives at this address.">
      <Link className="chip" to="/">
        <ListIcon /> Projects
      </Link>
    </StatusPage>
  );
}
