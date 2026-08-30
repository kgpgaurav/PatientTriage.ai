import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import AddPatient from "./pages/AddPatient";
import SurgeDashboard from "./pages/SurgeDashboard";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/add-patient" element={<AddPatient />} />
        <Route path="/surge" element={<SurgeDashboard />} />
      </Route>
    </Routes>
  );
}
