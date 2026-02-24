import { Routes, Route, Navigate } from "react-router-dom";
import App from "./App.jsx";
import Rag from "./pages/Rag.jsx";

export default function RouterApp() {
  return (
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/rag" element={<Rag />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
