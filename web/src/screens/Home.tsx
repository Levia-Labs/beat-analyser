import { useNavigate } from "react-router";
import { getConfig, getProjects } from "../api";
import AnalyseCard from "../components/AnalyseCard";
import ProjectList from "../components/ProjectList";
import { message, useLoad } from "../useLoad";

export default function Home() {
  const navigate = useNavigate();
  const config = useLoad(getConfig);
  const projects = useLoad(getProjects);
  return (
    <>
      <AnalyseCard
        config={config.data}
        onDone={(name) => navigate(`/p/${encodeURIComponent(name)}`)}
      />
      <ProjectList
        projects={projects.data}
        error={message(projects.error)}
        loading={projects.loading}
      />
    </>
  );
}
