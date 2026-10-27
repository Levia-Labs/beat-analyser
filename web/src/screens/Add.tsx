import { useNavigate } from "react-router";
import { getConfig } from "../api";
import AnalyseCard from "../components/AnalyseCard";
import { useLoad } from "../useLoad";

export default function Add() {
  const navigate = useNavigate();
  const config = useLoad(getConfig);
  return (
    <AnalyseCard
      config={config.data}
      onDone={(name) => navigate(`/p/${encodeURIComponent(name)}`)}
    />
  );
}
