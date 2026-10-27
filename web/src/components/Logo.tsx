import logoUrl from "../assets/logo.svg";

export default function Logo({ size = 40 }: { size?: number }) {
  return <img className="logo" src={logoUrl} width={size} height={size} alt="" />;
}
