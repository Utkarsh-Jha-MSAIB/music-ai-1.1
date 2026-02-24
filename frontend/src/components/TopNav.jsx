import { NavLink } from "react-router-dom";
import "./TopNav.css";

export default function TopNav() {
  return (
    <header className="topNav">
      <div className="topNavBrand">
        <span className="topNavLogo">♫</span>
        <span className="topNavTitle">Music AI</span>
      </div>

      <nav className="topNavLinks">
        <NavLink to="/perform" className={({ isActive }) => `topLink ${isActive ? "active" : ""}`}>
          Perform
        </NavLink>
        <NavLink to="/rag" className={({ isActive }) => `topLink ${isActive ? "active" : ""}`}>
          RAG
        </NavLink>
      </nav>
    </header>
  );
}