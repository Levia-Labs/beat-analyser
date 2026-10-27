import { Link } from "react-router";
import { Icon } from "../components/icons";
import StatusPage from "../components/StatusPage";

export default function NotFound() {
  return (
    <StatusPage title="Not found" detail="Nothing lives at this address.">
      <Link className="btn" to="/">
        <Icon name="list" /> Projects
      </Link>
    </StatusPage>
  );
}
