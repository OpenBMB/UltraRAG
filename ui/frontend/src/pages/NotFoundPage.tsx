import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <main className="missing-page">
      <h1>404</h1>
      <p>The requested page does not exist.</p>
      <Link to="/settings">Go to Builder</Link>
    </main>
  );
}
