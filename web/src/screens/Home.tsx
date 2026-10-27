import { getProjects } from "../api";
import ProjectList from "../components/ProjectList";
import { message, useLoad } from "../useLoad";

export default function Home() {
  const projects = useLoad(getProjects);
  return (
    <ProjectList
      projects={projects.data}
      error={message(projects.error)}
      loading={projects.loading}
    />
  );
}
